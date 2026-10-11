# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Export data source mimicking the data of Isaac Lab 3 articulations."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation

from exploy.exporter.frameworks.isaaclab3.body_data import BodyDataSource
from exploy.exporter.frameworks.isaaclab3.proxy_array import TorchProxyArray, as_torch


class ArticulationDataSource(BodyDataSource):
    """Mimic the interface of an Isaac Lab 3 ``ArticulationData``, but manage its own tensor data.

    In addition to the rigid-body state of :class:`BodyDataSource`, the joint state is stored, as well
    as the joint targets. Isaac Lab 3 writes joint targets with warp kernels, which cannot be traced:
    during export, the targets written by action terms are recorded into the torch buffers of this data
    source instead (see :class:`exploy.exporter.frameworks.isaaclab3.writes.ArticulationWriteRecorder`).
    """

    def __init__(self, articulation: Articulation):
        data = articulation.data
        super().__init__(data)

        self._joint_pos = as_torch(data.joint_pos).clone()
        self._joint_vel = as_torch(data.joint_vel).clone()
        self._joint_acc = as_torch(data.joint_acc).clone()

        self._joint_pos_target = as_torch(data.joint_pos_target).clone()
        self._joint_vel_target = as_torch(data.joint_vel_target).clone()
        self._joint_effort_target = as_torch(data.joint_effort_target).clone()

    ##
    # Joint state.
    ##

    @property
    def joint_pos(self) -> TorchProxyArray:
        return TorchProxyArray(self._joint_pos)

    @property
    def joint_vel(self) -> TorchProxyArray:
        return TorchProxyArray(self._joint_vel)

    @property
    def joint_acc(self) -> TorchProxyArray:
        return TorchProxyArray(self._joint_acc)

    ##
    # Joint targets.
    ##

    @property
    def joint_pos_target(self) -> TorchProxyArray:
        return TorchProxyArray(self._joint_pos_target)

    @property
    def joint_vel_target(self) -> TorchProxyArray:
        return TorchProxyArray(self._joint_vel_target)

    @property
    def joint_effort_target(self) -> TorchProxyArray:
        return TorchProxyArray(self._joint_effort_target)

    def joint_target_buffer(self, kind: str) -> torch.Tensor:
        """Return the stored joint target buffer of a kind (``position``, ``velocity`` or ``effort``)."""
        return {
            "position": self._joint_pos_target,
            "velocity": self._joint_vel_target,
            "effort": self._joint_effort_target,
        }[kind]
