"""Behavior-only action masking for MaskablePPO.

Standard MaskablePPO masks disallowed actions to a large negative logit for BOTH the
sampled/argmax *behavior* AND the training *objective* (log_prob / entropy). When the
schedule graph leaves a small or singleton allowed set, the masked categorical collapses
(zero entropy, degenerate log-prob), so the policy gradient is noisy/uninformative at most
steps.

This module provides a "behavior-only" variant: the executed action is still constrained to
the allowed set (``sample()`` / ``mode()`` stay masked), but ``log_prob()`` and ``entropy()``
are computed on the UNMASKED full distribution, so gradients and entropy regularization stay
well-conditioned over all actions. In rollout the behavior samples within the allowed set;
in greedy eval it is the argmax over the allowed set ("max of the available set").

Caveat (intentional): the PPO ratio uses the unmasked density for an action drawn from the
masked behavior policy — a deliberate behavior/objective mismatch. The policy is never
directly penalized for placing probability mass on disallowed actions. This is an
experimental knob (``--policy-mask-mode behavior-only``); the default (``hard``) reproduces
stock MaskablePPO exactly.
"""
import torch as th
from gymnasium import spaces
from torch.distributions import Categorical

from sb3_contrib.common.maskable.distributions import (
    MaskableCategorical,
    MaskableCategoricalDistribution,
    MaskableMultiCategoricalDistribution,
)
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class BehaviorMaskedCategorical(MaskableCategorical):
    """MaskableCategorical whose ``log_prob`` / ``entropy`` use the UNMASKED logits.

    ``sample()`` / ``mode()`` rely on ``self.probs`` (masked) and are inherited unchanged, so
    the behavior stays within the allowed set; only the learning signal sees the full
    distribution. ``_original_logits`` is the pre-mask (normalized) logit tensor stored by the
    base ``MaskableCategorical.__init__``.
    """

    def log_prob(self, value: th.Tensor) -> th.Tensor:
        return Categorical(logits=self._original_logits).log_prob(value)

    def entropy(self) -> th.Tensor:
        return Categorical(logits=self._original_logits).entropy()


class BehaviorMaskedCategoricalDistribution(MaskableCategoricalDistribution):
    """Discrete distribution that builds a :class:`BehaviorMaskedCategorical`."""

    def proba_distribution(self, action_logits: th.Tensor) -> "BehaviorMaskedCategoricalDistribution":
        reshaped_logits = action_logits.view(-1, self.action_dim)
        self.distribution = BehaviorMaskedCategorical(logits=reshaped_logits)
        return self


class BehaviorMaskedMultiCategoricalDistribution(MaskableMultiCategoricalDistribution):
    """MultiDiscrete distribution whose per-slot categoricals are behavior-masked."""

    def proba_distribution(self, action_logits: th.Tensor) -> "BehaviorMaskedMultiCategoricalDistribution":
        reshaped_logits = action_logits.view(-1, sum(self.action_dims))
        self.distributions = [
            BehaviorMaskedCategorical(logits=split)
            for split in th.split(reshaped_logits, list(self.action_dims), dim=1)
        ]
        return self


def make_behavior_masked_distribution(action_space: spaces.Space):
    """Behavior-masked analogue of ``make_masked_proba_distribution`` (same dims/handling)."""
    if isinstance(action_space, spaces.Discrete):
        return BehaviorMaskedCategoricalDistribution(int(action_space.n))
    if isinstance(action_space, spaces.MultiDiscrete):
        return BehaviorMaskedMultiCategoricalDistribution(list(action_space.nvec))
    raise NotImplementedError(
        "BehaviorMasked policy supports Discrete / MultiDiscrete action spaces, "
        f"got {type(action_space)}."
    )


class BehaviorMaskedActorCriticPolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy that masks only behavior, not the objective.

    The ``action_net`` built by ``super().__init__`` depends only on the action dimensions,
    which are identical between the stock and behavior-masked distributions, so swapping the
    (stateless) ``action_dist`` after construction is safe — ``proba_distribution()`` is
    re-invoked on every forward pass and now builds :class:`BehaviorMaskedCategorical`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dist = make_behavior_masked_distribution(self.action_space)
