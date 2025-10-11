from dotenv import load_dotenv
load_dotenv(override=True)


import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Type

# reuse same Observation/ActionSpace imports you have in repo
from rl_autoschedular.observation import Observation, ObservationPart, OpFeatures, ActionHistory
from rl_autoschedular.actions import ActionSpace
from rl_autoschedular.state import OperationState
from rl_autoschedular import config as cfg

# Keep same activation used elsewhere
ACTIVATION = nn.ReLU  # replace with your actual ACTIVATION if different


class _DiscreteQHead(nn.Module):
    """A head that outputs Q-values for each discrete option (flat logits -> Q-values)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # single linear mapping from embedding -> out_dim Q-values
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns [B, out_dim]
        return self.net(x)


class _TwinHiearchicalQNetwork(nn.Module):
    """
    One Q network:
      - backbone: embed observation -> [B, embed_dim]
      - head_action: Q for selecting each action type [B, |A|]
      - param_heads: for each action, a head producing Q-values for that action's network_output_size
        NOTE: For multi-slot actions (e.g. Tiling), param_head outputs are flat: a concatenation
              of per-slot Q-values. We'll reshape inside q_contribs when computing param-contribution.
    """
    def __init__(self, obs_parts: List[Type[ObservationPart]], embed_dim: int = 512):
        super().__init__()
        self.obs_parts = obs_parts
        in_size = sum(p.size() for p in obs_parts)

        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(in_size, embed_dim),
            ACTIVATION(),
            nn.Linear(embed_dim, embed_dim),
            ACTIVATION(),
            nn.Linear(embed_dim, embed_dim),
            ACTIVATION(),
        )

        # Action selection Q head
        self.head_action = _DiscreteQHead(embed_dim, ActionSpace.size())

        # Parameter heads (one per supported action)
        self.param_heads = nn.ModuleList()
        # Save meta per action for convenient reshaping later
        self._param_meta: List[Optional[dict]] = []
        for action_cls in ActionSpace.supported_actions:
            out_dim = action_cls.network_output_size()
            params_size = action_cls.params_size()
            if out_dim and out_dim > 0:
                head = _DiscreteQHead(embed_dim, out_dim)
                self.param_heads.append(head)
                # if multi-slot, compute classes_per_slot = out_dim // params_size (integer)
                if params_size > 0:
                    classes_per_slot = None
                    if params_size > 0:
                        classes_per_slot = out_dim // params_size if params_size != 0 else None
                    self._param_meta.append({
                        "params_size": params_size, # number of slots for this action
                        "out_dim": out_dim,
                        "classes_per_slot": classes_per_slot # number of classes per slot (None if single slot)
                        # Example : Interchange -> params_size=1 , out_dim= 7 , classes_per_slot= 7 , 7 loop choices for the current interchange 
                    })
                else:
                    self._param_meta.append({"params_size": 0, "out_dim": out_dim, "classes_per_slot": None})
            else:
                # no parameters -> placeholder (we'll treat as None)
                # ( NT | V )
                self.param_heads.append(nn.Identity())
                self._param_meta.append(None)
                
    def forward(
        self,
        obs: torch.Tensor,
        action_idx: torch.LongTensor,
        param_indices: Optional[List[Optional[torch.LongTensor]]] = None
    ) -> torch.Tensor:
        emb = self._embed(obs)
        return self.q_contribs(emb, action_idx, param_indices)

    def _embed(self, obs: torch.Tensor) -> torch.Tensor:
        parts = Observation.get_parts(obs, *self.obs_parts)  # returns [B, in_size]
        return self.backbone(parts)  # [B, embed_dim]

    def q_contribs(
        self,
        emb: torch.Tensor,
        action_idx: torch.LongTensor,                     # [B]
        param_indices: Optional[List[Optional[torch.LongTensor]]] = None  # list of length B
    ) -> torch.Tensor:
        """
        Compute Q(s, a, params) as: Q_action(s)[a] + Q_params(s, a, params).
        - emb: [B, embed_dim]
        - action_idx: [B] integers in [0..|A|-1]
        - param_indices: list of length B (each either None or [params_size])

        Returns:
            q_total: [B] - scalar Q for each sample
        """
        B = emb.size(0)
        device = emb.device

        # ---- top-level action contribution ----
        act_qs = self.head_action(emb)                                    # [B, |A|]
        act_q = act_qs.gather(1, action_idx.view(-1, 1)).squeeze(1)        # [B]

        # ---- parameter contribution ----
        param_q = torch.zeros(B, device=device)

        # group samples by chosen action to do batched head computation
        for k, head in enumerate(self.param_heads):
            meta = self._param_meta[k]
            if isinstance(head, nn.Identity) or (meta is None):
                continue

            # mask = all samples where chosen action == k
            mask = (action_idx == k)
            if not mask.any():
                continue

            # get embeddings and their chosen param indices for this action
            emb_masked = emb[mask]
            head_out = head(emb_masked)  # [N_mask, out_dim_k]

            psize = meta["params_size"] # 
            out_dim = meta["out_dim"]
            cps = meta["classes_per_slot"]

            # collect just the param indices for the masked samples
            masked_params = [param_indices[i] for i in range(B) if mask[i]] # [N_mask, Optional[LongTensor] of consistet size psize]
            # they should all be not None if this action has params
            assert all((p is not None) for p in masked_params) or psize == 0

            if psize == 0:
                continue

            if psize == 1:
                # single-slot
                idx_tensor = torch.stack([p.view(-1)[0] for p in masked_params]).view(-1, 1)  # [N_mask, 1]
                q_k = head_out.gather(1, idx_tensor).squeeze(1)                               # [N_mask]
                param_q[mask] = q_k
            else:
                # multi-slot
                assert cps is not None and cps > 0, "classes_per_slot unknown for multi-slot head"

                reshaped = head_out.view(-1, psize, cps)      # [N_mask, psize, cps]
                idx_tensor = torch.stack(masked_params).long()  # [N_mask, psize]
                idx_exp = idx_tensor.unsqueeze(-1)            # [N_mask, psize, 1]
                gathered = torch.gather(reshaped, dim=2, index=idx_exp).squeeze(-1)  # [N_mask, psize]
                q_k = gathered.sum(dim=1)                     # [N_mask]
                param_q[mask] = q_k

        return act_q + param_q





# ---------------- hierarchical double Q network ----------------
class IQLTwinQ(nn.Module):
    """
    Top-level twin Q network that matches style of PolicyModel:
     - builds two Q-branches (Q1 and Q2), each using _TwinHiearchicalQNetwork
     - provides helpers to split flat action-tensor into action_idx + param slices
    """
    def __init__(self, obs_parts: List[Type[ObservationPart]], embed_dim: int = 512):
        super().__init__()
        self.obs_parts = obs_parts
        # instantiate two Q heads (Q1, Q2)
        self.q1 = _TwinHiearchicalQNetwork(obs_parts, embed_dim=embed_dim)
        self.q2 = _TwinHiearchicalQNetwork(obs_parts, embed_dim=embed_dim)
        
    
    def forward(self, obs: torch.Tensor, index: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q1(s, a, params) and Q2(s, a, params) for a batch.
        - obs: observation tensor [B, ...]
        - index: flat index tensor from ActionSpace.sample() -> [B, 1 + sum(params)]
        Returns:
          (q1_vals, q2_vals) each shaped [B]
        """
        action_idx, params_list = self._split_action_tensor(index)
        q1_vals = self.q1(obs, action_idx, params_list)
        q2_vals = self.q2(obs, action_idx, params_list)
        return q1_vals, q2_vals
    
    def q_values(self, obs: torch.Tensor, index: torch.LongTensor) -> torch.Tensor:
        """
        Compute min(Q1(s, a, params), Q2(s, a, params)) for a batch.
        - obs: observation tensor [B, ...]
        - index: flat index tensor from ActionSpace.sample() -> [B, 1 + sum(params)]
        Returns:
          q_vals shaped [B]
        """
        action_idx, params_list = self._split_action_tensor(index)
        q1_vals = self.q1(obs, action_idx, params_list)
        q2_vals = self.q2(obs, action_idx, params_list)
        return torch.min(q1_vals, q2_vals)
    
    
    def loss(self, obs: torch.Tensor, index: torch.LongTensor, target_q: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between Q1, Q2 and target_q.
        - obs: observation tensor [B, ...]
        - index: flat index tensor from ActionSpace.sample() -> [B, 1 + sum(params)]
        - target_q: target Q-values [B]
        Returns:
          scalar loss
        """
        q1_vals, q2_vals = self.forward(obs, index)  # each [B]
        loss_fn = nn.MSELoss()
        loss1 = loss_fn(q1_vals, target_q)
        loss2 = loss_fn(q2_vals, target_q)
        return loss1 + loss2

    @staticmethod
    def _split_action_tensor(index: torch.LongTensor) -> Tuple[torch.LongTensor, List[Optional[torch.LongTensor]]]:
        """
        Split the  `index` tensor returned by ActionSpace.sample() into:
        - action_idx: [B]
        - params: list of length B, each either None (no params) or LongTensor [params_size] for that action
        """
        B = index.size(0)
        device = index.device

        action_idx = index[:, 0].long()  # [B]
        cum = ActionSpace.cumulative_params_sizes()

        params: List[Optional[torch.LongTensor]] = []

        for i in range(B):
            a_idx = action_idx[i].item()
            action_type = ActionSpace.supported_actions[a_idx]
            if action_type.params_size() == 0:
                params.append(None)
            else:
                start, end = cum[a_idx], cum[a_idx + 1]
                # extract just that sample's params for its chosen action
                params.append(index[i, start:end].long())

        return action_idx, params



def main():
    model = IQLTwinQ([OpFeatures, ActionHistory])

    _model = _TwinHiearchicalQNetwork([OpFeatures, ActionHistory])

    x = torch.tensor([[2, 1, 5, 7, 0, 0, 0, 0, 3, 4, 2, 0, 0, 0, 0, 2]]).float()
    
    
    action_idx , param_idx = model._split_action_tensor(x)
    
    
    obs = torch.zeros([1, 2152])
    
    
    q = _model(obs, action_idx, param_idx)  
    
    print("Q-values:", q)
 
    
    
if __name__ == "__main__":
    
    main()