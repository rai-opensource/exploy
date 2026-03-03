# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.


import torch

from exploy.exporter.core.context_manager import ContextManager, Group, Input
from exploy.exporter.frameworks.manager_based.types import (
    EntityProtocol,
    ManagerBasedEnvProtocol,
    SensorProtocol,
)

OBJ_PREFIX = "obj"
SENSOR_PREFIX = "sensor"
CMD_PREFIX = "cmd"


def add_command(
    env: ManagerBasedEnvProtocol,
    context_manager: ContextManager,
    command_name: str,
    command_type: str,
):
    """Add a single command input to the context manager.

    Args:
        env: The environment
        context_manager: The context manager to add components to
        command_name: Name of the command term to add
        command_type: Type of command - either "se2_velocity" or "se3_pose"
    """
    command_manager = env.command_manager
    command = command_manager.get_term(name=command_name)
    name = f"{CMD_PREFIX}.{command_type}.{command_name}"
    metadata = {}

    if command_type == "se2_velocity":
        metadata = (
            {
                "ranges": {
                    "lin_vel_x": command.cfg.ranges.lin_vel_x,
                    "lin_vel_y": command.cfg.ranges.lin_vel_y,
                    "ang_vel_z": command.cfg.ranges.ang_vel_z,
                },
            },
        )
    elif command_type == "se3_pose":
        pass
    else:
        raise ValueError(f"Unknown command type: {command_type}.")

    def getter() -> torch.Tensor:
        term = command_manager.get_term(name=command_name)
        return term.command

    context_manager.add_component(Input(name=name, get_from_env_cb=getter, metadata=metadata))


def add_body_pos_and_quat(entities: dict[str, EntityProtocol], context_manager: ContextManager):
    """Add body position and orientation inputs for all entities.

    For each entity, this function adds inputs for the position and quaternion
    of each body in the world frame.

    Args:
        entities: Dictionary mapping object names to EntityProtocol instances.
        context_manager: The context manager to add body pose inputs to.
    """
    # Add inputs for all body positions and quaternions in world frame
    for obj_name, entity in entities.items():
        for i, body_name in enumerate(entity.body_names):
            pos_b_rt_w_in_w = Input(
                name=f"{OBJ_PREFIX}.{obj_name}.{body_name}.pos_b_rt_w_in_w",
                get_from_env_cb=lambda r=entity, idx=i: r.data.body_link_pos_w[:, idx],
            )
            w_Q_b = Input(
                name=f"{OBJ_PREFIX}.{obj_name}.{body_name}.w_Q_b",
                get_from_env_cb=lambda r=entity, idx=i: r.data.body_link_quat_w[:, idx],
            )
            context_manager.add_component(pos_b_rt_w_in_w)
            context_manager.add_component(w_Q_b)


def add_base_vel(entities: dict[str, EntityProtocol], context_manager: ContextManager):
    """Add base velocity inputs for all entities.

    For each entity, this function adds inputs for the base linear and angular
    velocities in the base frame.

    Args:
        entities: Dictionary mapping object names to EntityProtocol instances.
        context_manager: The context manager to add base velocity inputs to.
    """
    for obj_name, entity in entities.items():
        input_name_prefix = f"{OBJ_PREFIX}.{obj_name}"

        # Add base orientation and velocities
        base_lin_vel_b_rt_w_in_b = Input(
            name=f"{input_name_prefix}.base.lin_vel_b_rt_w_in_b",
            get_from_env_cb=lambda r=entity: r.data.root_com_lin_vel_b,
        )
        base_ang_vel_b_rt_w_in_b = Input(
            name=f"{input_name_prefix}.base.ang_vel_b_rt_w_in_b",
            get_from_env_cb=lambda r=entity: r.data.root_com_ang_vel_b,
        )
        context_manager.add_component(base_lin_vel_b_rt_w_in_b)
        context_manager.add_component(base_ang_vel_b_rt_w_in_b)


def add_joint_pos_and_vel(entities: dict[str, EntityProtocol], context_manager: ContextManager):
    """Add joint position and velocity inputs for all entities.

    For each entity, this function creates a group containing joint positions
    and velocities, along with metadata about joint names.

    Args:
        entities: Dictionary mapping object names to EntityProtocol instances.
        context_manager: The context manager to add joint state inputs to.
    """
    for obj_name, entity in entities.items():
        input_name_prefix = f"{OBJ_PREFIX}.{obj_name}"

        joint_group = Group(
            name=f"{input_name_prefix}.joints",
            items=[
                Input(
                    name="pos",
                    get_from_env_cb=lambda r=entity: r.data.joint_pos,
                ),
                Input(
                    name="vel",
                    get_from_env_cb=lambda r=entity: r.data.joint_vel,
                ),
            ],
            metadata={
                "joint_names": entity.joint_names,
            },
        )

        context_manager.add_group(joint_group)


def add_sensor_input(
    sensor_name: str,
    sensor: SensorProtocol,
    context_manager: ContextManager,
):
    """Add a single sensor input to the context manager.

    Args:
        sensor_name: Name of the sensor
        sensor: The sensor object
        context_manager: The context manager to add components to
    """
    # Ray caster sensor
    if "RayCast" in type(sensor).__name__:
        pattern_cfg = sensor.cfg.pattern_cfg

        context_manager.add_group(
            Group(
                name=f"{SENSOR_PREFIX}.ray_caster.{sensor_name}",
                metadata={
                    "pattern_type": "grid_pattern",
                    "offset_x": sensor.cfg.offset.pos[0],
                    "offset_y": sensor.cfg.offset.pos[1],
                    "resolution": pattern_cfg.resolution,
                    "size_x": pattern_cfg.size[0],
                    "size_y": pattern_cfg.size[1],
                },
                items=[
                    Input(
                        name="height",
                        get_from_env_cb=lambda s=sensor: s._data.ray_hits_w[..., 2],
                    ),
                ],
            )
        )
    # Builtin sensor
    elif "imu" in sensor_name.lower():
        body_name = "imu_torso" if "torso" in sensor_name.lower() else "imu_pelvis"
        if "ang_vel" in sensor_name.lower():
            quantity_name = "ang_vel_b_rt_w_in_b"
        elif "lin_vel" in sensor_name.lower():
            quantity_name = "lin_vel_b_rt_w_in_b"
        else:
            return
        context_manager.add_component(
            Input(
                name=f"{SENSOR_PREFIX}.imu.{body_name}.{quantity_name}",
                get_from_env_cb=lambda s=sensor: s.data,
            )
        )
