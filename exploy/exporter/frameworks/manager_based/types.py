# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import torch


class ActionTermProtocol(Protocol):
    """Protocol for action terms from manager based environments."""

    processed_actions: torch.Tensor
    _processed_actions: torch.Tensor


class ActionManagerProtocol(Protocol):
    """Protocol for action managers from manager based environments."""

    _action: torch.Tensor
    active_terms: Iterable[str]

    def get_term(self, name: str) -> ActionTermProtocol: ...

    def process_action(self, actions: torch.Tensor) -> None: ...

    def apply_action(self) -> None: ...


class CommandTermProtocol(Protocol):
    """Protocol for command terms."""

    command: torch.Tensor
    cfg: Any


class CommandManagerProtocol(Protocol):
    """Protocol for command managers."""

    active_terms: Iterable[str]
    _terms: dict[str, Any]

    def get_term(self, name: str) -> CommandTermProtocol: ...

    def compute(self, dt: float) -> None: ...

    def reset(self, env_ids: Sequence[int]) -> dict[str, float]: ...


class ObservationManagerProtocol(Protocol):
    """Protocol for observation managers."""

    active_terms: dict[str, Any]
    group_obs_concatenate: dict[str, bool]
    group_obs_dim: dict[str, dict[int, int]]
    _group_obs_term_names: dict[str, list[str]]
    _group_obs_term_dim: dict[str, list[tuple[int, ...]]]
    _group_obs_term_cfgs: dict[str, list[Any]]
    _obs_buffer: dict[str, torch.Tensor]

    def compute(self, **kwargs: Any) -> dict[str, torch.Tensor]: ...


class EntityDataProtocol(Protocol):
    """Protocol for articulation/rigid_body/entity data."""

    body_pos_w: torch.Tensor | Any
    body_link_pos_w: torch.Tensor | Any
    body_quat_w: torch.Tensor | Any
    body_link_quat_w: torch.Tensor | Any
    root_lin_vel_b: torch.Tensor
    root_com_lin_vel_b: torch.Tensor
    root_ang_vel_b: torch.Tensor
    root_com_ang_vel_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_pos_target: torch.Tensor
    joint_vel_target: torch.Tensor
    joint_effort_target: torch.Tensor


class EntityProtocol(Protocol):
    """Protocol for Articulation/RigidBody/Entity."""

    _data: EntityDataProtocol
    joint_names: list[str]
    body_names: list[str]

    def find_joints(self, name_expr: str) -> tuple[list[int], list[str]]: ...
    def find_bodies(self, name_expr: str) -> tuple[list[int], list[str]]: ...


class SensorDataProtocol(Protocol):
    """Protocol for sensor data."""

    ray_hits_w: torch.Tensor
    pos_w: torch.Tensor
    data: torch.Tensor


class SensorProtocol(Protocol):
    """Protocol for sensors."""

    _data: SensorDataProtocol
    cfg: Any


class SceneProtocol(Protocol):
    """Protocol for scene."""

    articulations: dict[str, EntityProtocol]
    entities: dict[str, EntityProtocol]
    rigid_objects: dict[str, Any]
    sensors: dict[str, SensorProtocol]
    _sensors: dict[str, Any]
    cfg: Any
    _cfg: Any


class ManagerBasedEnvProtocol(Protocol):
    """Protocol for manager-based environments from both isaaclab and mjlab."""

    action_manager: ActionManagerProtocol
    command_manager: CommandManagerProtocol
    observation_manager: ObservationManagerProtocol
    scene: SceneProtocol
    obs_buf: dict[str, torch.Tensor]
    cfg: Any
    physics_dt: float

    def step(
        self, actions: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]: ...

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]: ...
