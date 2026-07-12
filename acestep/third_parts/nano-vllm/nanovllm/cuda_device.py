"""Resolve physical CUDA device indices for nano-vllm workers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def resolve_cuda_device_id(cuda_device: int, rank: int) -> int:
    """Map tensor-parallel rank onto a physical CUDA device index.

    ``cuda_device`` is the base GPU for rank 0. Rank ``r`` uses
    ``cuda_device + r`` so multi-GPU tensor parallel can start at a
    non-zero device while single-GPU stays on the mapped index.

    Args:
        cuda_device: Physical CUDA index for rank 0 (must be >= 0).
        rank: Tensor-parallel rank (must be >= 0).

    Returns:
        Physical CUDA device index for this worker.

    Raises:
        ValueError: If ``cuda_device`` or ``rank`` is negative.
    """
    if cuda_device < 0:
        raise ValueError(f"cuda_device must be >= 0, got {cuda_device}")
    if rank < 0:
        raise ValueError(f"rank must be >= 0, got {rank}")
    return int(cuda_device) + int(rank)


def to_runner_cuda_device(tensor: "torch.Tensor", device_id: int) -> "torch.Tensor":
    """Move *tensor* onto the runner's physical CUDA device.

    Bare ``tensor.cuda()`` uses ``torch.cuda.current_device()``, which may be
    another GPU after DiT/VAE work. Always pin to the mapped LM device.
    """
    return tensor.to(device=device_id, non_blocking=True)
