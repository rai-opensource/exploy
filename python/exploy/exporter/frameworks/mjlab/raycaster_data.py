# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

from typing import Any

import torch
from mjlab.sensor.raycast_sensor import ObjRef, RayCastSensor

from exploy.exporter.core.tensor_proxy import TensorProxy


class RayCasterDataSource:
    """Proxy for raycaster sensor data that holds ONNX-traceable tensors.

    Replaces ``sensor._data`` during ONNX export so that raycaster height
    observations use managed tensors that become ONNX graph inputs.
    """

    def __init__(self, sensor: RayCastSensor, entities: dict[str, Any]):
        frames = sensor.cfg.frame
        frame: ObjRef = frames[0] if isinstance(frames, tuple) else frames
        self._robot = entities[frame.entity]

        ray_hits = sensor.data.hit_pos_w  # [B, N, 3]
        self._hit_pos_w = TensorProxy(ray_hits, split_dim=ray_hits.ndim - 1)

    @property
    def hit_pos_w(self) -> torch.Tensor:
        return self._hit_pos_w

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass
