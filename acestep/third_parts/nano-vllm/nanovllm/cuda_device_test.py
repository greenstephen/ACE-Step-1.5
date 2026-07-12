"""Unit tests for nano-vllm physical CUDA device resolution."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from nanovllm.cuda_device import resolve_cuda_device_id, to_runner_cuda_device


class TestResolveCudaDeviceId(unittest.TestCase):
    """resolve_cuda_device_id must separate TP rank from physical GPU index."""

    def test_default_rank0_on_gpu0(self) -> None:
        """Backward-compatible default: rank 0 on cuda:0."""
        self.assertEqual(resolve_cuda_device_id(0, 0), 0)

    def test_mapped_lm_on_gpu3(self) -> None:
        """ACE-Step lm:3 must place single-GPU LM on physical device 3."""
        self.assertEqual(resolve_cuda_device_id(3, 0), 3)

    def test_tensor_parallel_offset(self) -> None:
        """TP ranks are offsets from the base cuda_device."""
        self.assertEqual(resolve_cuda_device_id(2, 1), 3)

    def test_rejects_negative_cuda_device(self) -> None:
        """Negative base device is invalid."""
        with self.assertRaises(ValueError):
            resolve_cuda_device_id(-1, 0)

    def test_rejects_negative_rank(self) -> None:
        """Negative TP rank is invalid."""
        with self.assertRaises(ValueError):
            resolve_cuda_device_id(0, -1)


class TestToRunnerCudaDevice(unittest.TestCase):
    """Transfers must target the mapped device, not current_device."""

    def test_pins_to_explicit_device_id(self) -> None:
        """to_runner_cuda_device must call tensor.to(device=device_id)."""
        tensor = MagicMock()
        moved = MagicMock(name="moved")
        tensor.to.return_value = moved

        result = to_runner_cuda_device(tensor, 3)

        tensor.to.assert_called_once_with(device=3, non_blocking=True)
        self.assertIs(result, moved)


if __name__ == "__main__":
    unittest.main()
