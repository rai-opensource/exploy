# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Record the joint targets written by Isaac Lab 3 action terms into traceable torch buffers.

Isaac Lab 3 writes joint targets into the actuator command buffers with warp kernels, which are not part
of the computational graph traced by the ONNX exporter. During export, the asset of each action term is
replaced by an :class:`ArticulationWriteRecorder`, which writes the targets with torch indexing into the
target buffers of the articulation's :class:`ArticulationDataSource`, from which the exporter's outputs
are read. This mirrors the write interception of Isaac Lab's own LEAPP export.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.assets import Articulation

from exploy.exporter.frameworks.isaaclab3.articulation_data import ArticulationDataSource

Indices = Sequence[int] | torch.Tensor | slice | None


def _as_index(ids: Any, device: torch.device) -> torch.Tensor | slice:
    if ids is None or isinstance(ids, slice):
        return slice(None)
    if hasattr(ids, "warp"):
        ids = ids.torch
    return torch.as_tensor(ids, device=device, dtype=torch.long)


def _as_mask(mask: Any, size: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(size, dtype=torch.bool, device=device)
    if hasattr(mask, "warp"):
        mask = mask.torch
    return torch.as_tensor(mask, device=device, dtype=torch.bool)


def _write_index(
    buffer: torch.Tensor,
    target: torch.Tensor,
    joint_ids: Indices,
    env_ids: Indices,
    full_data: bool,
) -> None:
    env_index = _as_index(env_ids, buffer.device)
    joint_index = _as_index(joint_ids, buffer.device)
    if full_data:
        # The target holds all instances and joints: select the entries to write.
        target = target[env_index][:, joint_index]
    if isinstance(env_index, slice):
        buffer[:, joint_index] = target
    elif isinstance(joint_index, slice):
        buffer[env_index] = target
    else:
        buffer[env_index[:, None], joint_index[None, :]] = target


def _write_mask(buffer: torch.Tensor, target: torch.Tensor, joint_mask: Any, env_mask: Any) -> None:
    env_mask = _as_mask(env_mask, buffer.shape[0], buffer.device)
    joint_mask = _as_mask(joint_mask, buffer.shape[1], buffer.device)
    mask = env_mask[:, None] & joint_mask[None, :]
    buffer.copy_(torch.where(mask, target, buffer))


class _TargetCommandRecorder:
    """Stand-in for ``ActuatorCollection.target_command`` that records the targets."""

    def __init__(self, data_source: ArticulationDataSource, target_command: Any):
        self._data_source = data_source
        self._target_command = target_command

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target_command, name)

    def _set_index(self, kind, *, value, joint_ids=None, env_ids=None, full_data=False):
        _write_index(
            self._data_source.joint_target_buffer(kind), value, joint_ids, env_ids, full_data
        )

    def _set_mask(self, kind, *, value, joint_mask=None, env_mask=None):
        _write_mask(self._data_source.joint_target_buffer(kind), value, joint_mask, env_mask)

    def set_position_index(self, **kwargs):
        self._set_index("position", **kwargs)

    def set_velocity_index(self, **kwargs):
        self._set_index("velocity", **kwargs)

    def set_effort_index(self, **kwargs):
        self._set_index("effort", **kwargs)

    def set_position_mask(self, **kwargs):
        self._set_mask("position", **kwargs)

    def set_velocity_mask(self, **kwargs):
        self._set_mask("velocity", **kwargs)

    def set_effort_mask(self, **kwargs):
        self._set_mask("effort", **kwargs)

    @property
    def position(self):
        return self._data_source.joint_pos_target

    @property
    def velocity(self):
        return self._data_source.joint_vel_target

    @property
    def effort(self):
        return self._data_source.joint_effort_target


class _ActuatorsRecorder:
    """Stand-in for ``Articulation.actuators`` whose ``target_command`` records the targets."""

    def __init__(self, target_command: _TargetCommandRecorder, actuators: Any):
        self._recorder = target_command
        self._actuators = actuators

    def __getattr__(self, name: str) -> Any:
        return getattr(self._actuators, name)

    def __getitem__(self, name: str) -> Any:
        return self._actuators[name]

    def __iter__(self):
        return iter(self._actuators)

    def __len__(self) -> int:
        return len(self._actuators)

    @property
    def target_command(self) -> _TargetCommandRecorder:
        return self._recorder


class ArticulationWriteRecorder:
    """Proxy of an articulation, for action terms, recording their joint targets during export.

    The joint target setters of the articulation (``set_joint_*_target_index``/``_mask``) and of its
    actuator collection (``actuators.target_command.set_*``) write into the target buffers of the
    articulation's data source. All other attributes are forwarded to the articulation, whose ``data`` is
    the data source while it is swapped in.

    Note that the targets are not written to the simulation: the recorder is only installed while the
    environment is traced.
    """

    def __init__(self, articulation: Articulation):
        data_source = articulation.data
        assert isinstance(data_source, ArticulationDataSource), (
            "ArticulationWriteRecorder requires the articulation's data to be swapped for an ArticulationDataSource."
        )
        self._articulation = articulation
        self._target_command = _TargetCommandRecorder(
            data_source, articulation.actuators.target_command
        )
        self._actuators = _ActuatorsRecorder(self._target_command, articulation.actuators)

    @property
    def articulation(self) -> Articulation:
        return self._articulation

    def __getattr__(self, name: str) -> Any:
        return getattr(self._articulation, name)

    @property
    def actuators(self) -> _ActuatorsRecorder:
        return self._actuators

    def set_joint_position_target_index(
        self, *, target, joint_ids=None, env_ids=None, full_data=False
    ):
        self._target_command.set_position_index(
            value=target, joint_ids=joint_ids, env_ids=env_ids, full_data=full_data
        )

    def set_joint_velocity_target_index(
        self, *, target, joint_ids=None, env_ids=None, full_data=False
    ):
        self._target_command.set_velocity_index(
            value=target, joint_ids=joint_ids, env_ids=env_ids, full_data=full_data
        )

    def set_joint_effort_target_index(
        self, *, target, joint_ids=None, env_ids=None, full_data=False
    ):
        self._target_command.set_effort_index(
            value=target, joint_ids=joint_ids, env_ids=env_ids, full_data=full_data
        )

    def set_joint_position_target_mask(self, *, target, joint_mask=None, env_mask=None):
        self._target_command.set_position_mask(
            value=target, joint_mask=joint_mask, env_mask=env_mask
        )

    def set_joint_velocity_target_mask(self, *, target, joint_mask=None, env_mask=None):
        self._target_command.set_velocity_mask(
            value=target, joint_mask=joint_mask, env_mask=env_mask
        )

    def set_joint_effort_target_mask(self, *, target, joint_mask=None, env_mask=None):
        self._target_command.set_effort_mask(value=target, joint_mask=joint_mask, env_mask=env_mask)
