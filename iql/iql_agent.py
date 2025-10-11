import copy
from typing import Dict, List, Optional, Type, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import Config
from rl_autoschedular.actions import ActionSpace
from rl_autoschedular.observation import Observation, ObservationPart, OpFeatures, ActionHistory

# ---- bring in the updated models we defined earlier ----
from iql.value_function import IQLValueModel
from iql.policy import IQLPolicyModel
from iql.q_functions import IQLTwinQ


class IQLAgent(nn.Module):
    """
    IQL agent adapted to the PPO-aligned architecture and hierarchical action space.
    - Uses Observation.get_parts(obs, *obs_parts)
    - Shared 3×512 backbone across policy/value/Q
    - Hierarchical heads (action + per-action params)
    """
    def __init__(self, cfg: Config, device: Union[torch.device, str], obs_parts=None, param_dims=None):
        super().__init__()
        self.obs_parts = obs_parts or [OpFeatures, ActionHistory]

        # ---- device handling ----
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # Use config hyperparameters
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.beta = cfg.beta
        self.alpha = cfg.alpha

        # Networks (move to device)
        self.value_model = IQLValueModel(self.obs_parts, tau=self.tau).to(self.device)
        self.policy_model = IQLPolicyModel(self.obs_parts).to(self.device)
        self.q_model = IQLTwinQ(self.obs_parts).to(self.device)

        # Target Q
        self.q_target = copy.deepcopy(self.q_model).to(self.device)
        for p in self.q_target.parameters():
            p.requires_grad = False

        # Optimizers with cfg.lr dict (after models are on device)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=cfg.lr["value"])
        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=cfg.lr["q"])
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=cfg.lr["policy"])
        self.policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer,
            T_max=600000,
            eta_min=1e-5
        )

    # --------- helpers to move inputs to device ----------
    def _to_device_tensor(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return x.to(self.device, non_blocking=True)

    def _to_device_tensor_list(
        self,
        xs: Optional[List[Optional[torch.Tensor]]]
    ) -> Optional[List[Optional[torch.Tensor]]]:
        if xs is None:
            return None
        out: List[Optional[torch.Tensor]] = []
        for t in xs:
            out.append(self._to_device_tensor(t) if isinstance(t, torch.Tensor) else None if t is None else t)
        return out

    # ------------------------
    # Action selection (hierarchical)
    # ------------------------
    @torch.no_grad()
    def sample(
        self,
        obs: torch.Tensor,
        greedy: bool = False,
        eps: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample hierarchical action indices using the same API style as PPO.
        Returns:
            actions_index: packed hierarchical indices (ActionSpace format)
            actions_log_p: log-prob of sampled action under current policy
            entropies: per-head entropies (aggregated by ActionSpace)
        """

        # Build distributions from policy
        dists = self.policy_model(obs)
        eps_dists = ActionSpace.uniform_distributions(obs)

        # Hierarchical sample
        use_uniform = (eps is not None) and (torch.rand((), device=self.device).item() < eps)
        actions_index = ActionSpace.sample(
            obs,
            dists,
            eps_dists,
            uniform=use_uniform,
            greedy=greedy,
        )

        # Stats for the sampled actions
        actions_log_p, entropies = ActionSpace.distributions_stats(
            dists,
            actions_index,
            eps_distributions=eps_dists if eps is not None else None,
            eps=eps,
        )
        return actions_index, actions_log_p, entropies

    # ------------------------
    # Value update (expectile regression using target twin-Q)
    # ------------------------
    def update_value(
        self,
        obs: torch.Tensor,
        action_idx: torch.LongTensor,
        *,
        param_indices: Optional[List[Optional[torch.LongTensor]]] = None,
        param_values: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Updates V(s) by regressing towards min(Q1, Q2) from the *target* Q network.
        """

        with torch.no_grad():
            q1_t, q2_t = self.q_target(obs, action_idx)
            q_min_t = torch.min(q1_t, q2_t)  # [B]
            
        self.value_optimizer.zero_grad(set_to_none=True)
        loss_v = self.value_model.loss(obs, q_min_t)
        loss_v.backward()
        self.value_optimizer.step()
        return loss_v

    # ------------------------
    # Q update (TD with V(s'))
    # ------------------------
    def update_q(
        self,
        obs: torch.Tensor,
        action_idx: torch.LongTensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Update twin Q networks with TD target:
            target_q = r + gamma * (1 - done) * V_target(s')
        If target_v is not provided, it is computed from the current value_model.
        """


        with torch.no_grad():
            target_v = self.value_model(next_obs).to(self.device)  # [B]

        target_q = rewards + self.gamma * (1.0 - dones) * target_v  # [B]
        
        self.q_optimizer.zero_grad(set_to_none=True)


        loss_q = self.q_model.loss(
            obs,
            action_idx,
            target_q
        )
        loss_q.backward()
        self.q_optimizer.step()
        return loss_q

    # ------------------------
    # Policy update (advantage-weighted BC)
    # ------------------------
    def update_policy(
        self,
        obs: torch.Tensor,
        actions_index: torch.Tensor,  # packed hierarchical indices (as stored by dataset)
        *,
        action_idx: Optional[torch.LongTensor] = None,
        param_indices: Optional[List[Optional[torch.LongTensor]]] = None,
        param_values: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Update policy with advantage-weighted log-likelihood:
            weights = exp(A / beta), A = min(Q1, Q2) - V(s)

        - actions_index is used to compute log π(a|s) via ActionSpace.distributions_stats(...)
        - Q needs decomposed (action_idx, param_indices/values).
        """

        # 1) log π(a|s) from hierarchical distributions
        dists = self.policy_model(obs)
        actions_log_p, _ = ActionSpace.distributions_stats(dists, actions_index)

        # 2) advantages = Q_min(s,a) - V(s)
        assert action_idx is not None, "action_idx (top-level) is required for Q evaluation"
        with torch.no_grad():
            q_min = self.q_model.q_values(obs, action_idx)  # [B]
            v = self.value_model(obs)                                          # [B]
            advantages = q_min - v                                             # [B]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        # 3) loss (AWAC/IQL style)
        
        # 1. zero gradients
        self.policy_optimizer.zero_grad(set_to_none=True)
        # 2. compute loss
        loss_pi = self.policy_model.loss(
            actions_log_p=actions_log_p,
            advantages=advantages,
            beta=self.beta,
        )
        
        # 3. backpropagate
        loss_pi.backward()

        # 4. clip gradients (to avoid instability)
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=5.0)

        
        self.policy_optimizer.step()
        self.policy_lr_scheduler.step()
        return loss_pi

    # ------------------------
    # Soft update of target Q
    # ------------------------
    @torch.no_grad()
    def soft_update_q_target(self):
        """
        θ_target ← α θ + (1-α) θ_target
        """
        for p, tp in zip(self.q_model.parameters(), self.q_target.parameters()):
            tp.data.copy_(self.alpha * p.data + (1.0 - self.alpha) * tp.data)

    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """
        One full IQL update step:
        1. Update Q-functions
        2. Update value function
        3. Update policy (AWAC/IQL style)
        4. Soft update target Q
        Returns dict of losses for logging.
        """
        obs, actions_index, rewards, next_obs, dones = (t.to(self.device, non_blocking=True) for t in batch)


        # ---- 1) Update Q ----
        loss_q = self.update_q(
            obs=obs,
            action_idx=actions_index,  # top-level index
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )
        
        # ---- 2) Update Value ----
        loss_v = self.update_value(obs, actions_index)


        # ---- 3) Update Policy ----
        loss_pi = self.update_policy(
            obs=obs,
            actions_index=actions_index,
            action_idx=actions_index,  # required for Q evaluation
        )



        # ---- 4) Soft update Q target ----
        self.soft_update_q_target()

        return {
            "q": float(loss_q.item()),
            "policy": float(loss_pi.item()),
            "value": float(loss_v.item()),
        }



