import torch


def sample_from_dist(dist: torch.Tensor) -> int:
    """Sample from a distribution.

    Args:
        dist (torch.Tensor): The distribution to sample from.

    Returns:
        int: The sampled value.
    """
    dist_cum = torch.cumsum(dist, dim=0)
    return torch.searchsorted(dist_cum, torch.rand(1)).squeeze().item()
