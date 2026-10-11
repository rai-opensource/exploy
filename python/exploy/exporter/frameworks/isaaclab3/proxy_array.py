# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Torch-only stand-in for Isaac Lab 3's ``ProxyArray``, and quaternion convention helpers."""

from __future__ import annotations

import torch

from exploy.exporter.core.tensor_proxy import TensorProxy, to_tensor


class TorchProxyArray:
    """Expose an exploy-managed tensor through the ``ProxyArray`` interface of Isaac Lab 3.

    Isaac Lab 3 asset and sensor data return ``ProxyArray`` objects, read by observation terms via
    ``.torch``. During export, the data sources return this class instead, whose ``.torch`` is the
    exploy-managed tensor (or ``TensorProxy``) itself, so that the tensors read during tracing are the
    exact objects held by the exporter inputs.

    There is no ``.warp`` view: warp kernels cannot be traced into an ONNX graph.
    """

    __slots__ = ("_value",)

    def __init__(self, value: torch.Tensor | TensorProxy):
        self._value = value

    @property
    def torch(self) -> torch.Tensor | TensorProxy:
        """The exploy-managed tensor."""
        return self._value

    @property
    def warp(self):
        raise RuntimeError(
            "[exploy] Warp views of Isaac Lab data are not available during ONNX export. Use `.torch` in"
            " the observation and action terms of exported environments."
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(to_tensor(self._value).shape)

    @property
    def device(self) -> torch.device:
        return to_tensor(self._value).device

    @property
    def dtype(self) -> torch.dtype:
        return to_tensor(self._value).dtype

    def __repr__(self) -> str:
        return f"TorchProxyArray({self._value!r})"


def as_torch(value) -> torch.Tensor | TensorProxy:
    """Return the torch view of an Isaac Lab ``ProxyArray`` or a ``TorchProxyArray``.

    Plain tensors are passed through unchanged.
    """
    return value if isinstance(value, torch.Tensor | TensorProxy) else value.torch


def wxyz_to_xyzw(quat: torch.Tensor | TensorProxy) -> torch.Tensor:
    """Convert quaternions from (w, x, y, z), the convention of exploy's ONNX interface, to the
    (x, y, z, w) convention of Isaac Lab 3."""
    quat = to_tensor(quat)
    return torch.cat([quat[..., 1:4], quat[..., 0:1]], dim=-1)


def xyzw_to_wxyz(quat: torch.Tensor | TensorProxy) -> torch.Tensor:
    """Convert quaternions from the (x, y, z, w) convention of Isaac Lab 3 to (w, x, y, z), the
    convention of exploy's ONNX interface."""
    quat = to_tensor(quat)
    return torch.cat([quat[..., 3:4], quat[..., 0:3]], dim=-1)
