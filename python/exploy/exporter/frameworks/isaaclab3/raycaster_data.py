# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Export data source mimicking the data of Isaac Lab 3 ray casters."""

from __future__ import annotations

from typing import Any

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster

from exploy.exporter.core.tensor_proxy import TensorProxy
from exploy.exporter.frameworks.isaaclab3 import utils
from exploy.exporter.frameworks.isaaclab3.proxy_array import TorchProxyArray, as_torch


class RayCasterDataSource:
    """Mimic the interface of an Isaac Lab 3 ``RayCasterData``, but manage its own tensor data.

    The ray hits are stored, split by coordinate so that the heights are an input of the ONNX graph.
    The sensor pose is computed from the pose of the body the sensor is attached to, and the pose of the
    sensor relative to that body, measured when the data source is created. The physics backends of
    Isaac Lab 3 differ in whether the sensor pose includes the configured offset (e.g. Newton applies it
    to the ray starts only), so the relative pose is measured rather than taken from the configuration.
    """

    def __init__(self, sensor: BaseRayCaster, articulations: dict[str, Articulation]):
        sensor_data = sensor.data
        self._source_data = sensor_data
        self._articulation, body_ids = utils.prim_path_to_articulation_and_body_ids(
            sensor.cfg.prim_path, articulations
        )
        assert len(body_ids) == 1, "RayCaster must be attached to exactly one body."
        self._body_id = body_ids[0]

        # Pose of the sensor relative to the body it is attached to.
        body_pos_w, body_quat_w = self._body_pose_w()
        self._offset_pos, self._offset_quat = math_utils.subtract_frame_transforms(
            body_pos_w, body_quat_w, as_torch(sensor_data.pos_w), as_torch(sensor_data.quat_w)
        )

        # Split ray_hits_w by its last dimension (XYZ).
        ray_hits = as_torch(sensor_data.ray_hits_w).clone()
        self._ray_hits_w = TensorProxy(ray_hits, split_dim=ray_hits.ndim - 1)

    def __getattr__(self, name: str) -> Any:
        if name == "_source_data":
            raise AttributeError(name)
        return getattr(self._source_data, name)

    def _body_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._articulation.data
        # Index the bodies without concatenating them, so that the pose only depends on the inputs of this body.
        body_pos_w = as_torch(data.body_link_pos_w)[:, self._body_id]
        body_quat_w = as_torch(data.body_link_quat_w)[:, self._body_id]
        return body_pos_w, body_quat_w

    def _pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        body_pos_w, body_quat_w = self._body_pose_w()
        return math_utils.combine_frame_transforms(
            body_pos_w, body_quat_w, self._offset_pos, self._offset_quat
        )

    @property
    def pos_w(self) -> TorchProxyArray:
        """Sensor positions in world frame, computed from the pose of the body it is attached to."""
        return TorchProxyArray(self._pose_w()[0])

    @property
    def quat_w(self) -> TorchProxyArray:
        """Sensor orientations (x, y, z, w) in world frame, computed from the pose of the body it is
        attached to."""
        return TorchProxyArray(self._pose_w()[1])

    @property
    def ray_hits_w(self) -> TorchProxyArray:
        """Ray hit positions in world frame."""
        return TorchProxyArray(self._ray_hits_w)

    def update(self, *args, **kwargs):
        """Empty update call to implement the interface of RayCasterData."""
