# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Any

from exploy.exporter.frameworks.manager_based.types import (
    EntityProtocol,
    ObservationManagerProtocol,
)


def get_robot_actuator_gains(robot: EntityProtocol) -> dict:
    """Get a dictionary of actuator gains.

    Args:
        robot: The robot (Articulation or Entity)
    """
    robot_cfg = robot.cfg
    if hasattr(robot_cfg, "actuators"):  # isaaclab
        actuators_access = "cfg.actuators"
        joint_names_attr = "joint_names_expr"
    elif hasattr(robot_cfg, "articulation"):  # mjlab
        actuators_access = "cfg.articulation.actuators"
        joint_names_attr = "target_names_expr"
    else:
        raise AttributeError(
            "Could not infer manager type from robot.cfg. Expected either "
            "'cfg.actuators' (isaaclab) or 'cfg.articulation.actuators' (mjlab)."
        )

    gains = {}

    # Navigate to actuators config
    obj = robot
    for attr in actuators_access.split("."):
        obj = getattr(obj, attr)
    actuators = obj

    def _update_dict(gain_cfg: dict | float, gain_name: str, actuator_cfg: Any):
        if isinstance(gain_cfg, (float, int)):
            joint_names_expr = getattr(actuator_cfg, joint_names_attr)
            _, joint_names = robot.find_joints(joint_names_expr)
            for name in joint_names:
                if name not in gains:
                    gains[name] = {}
                gains[name][gain_name] = float(gain_cfg)
        elif isinstance(gain_cfg, dict):
            for expr, val in gain_cfg.items():
                _, joint_names = robot.find_joints(expr)
                for name in joint_names:
                    if name not in gains:
                        gains[name] = {}
                    gains[name][gain_name] = float(val)

    # Handle both dict (isaaclab) and list (mjlab) of actuators
    actuator_list = actuators.values() if isinstance(actuators, dict) else actuators

    for actuator_cfg in actuator_list:
        _update_dict(
            gain_cfg=actuator_cfg.stiffness, gain_name="stiffness", actuator_cfg=actuator_cfg
        )
        _update_dict(gain_cfg=actuator_cfg.damping, gain_name="damping", actuator_cfg=actuator_cfg)

    return gains


def get_observation_names(
    observation_manager: ObservationManagerProtocol,
    group_name: str = "policy",
) -> list[str]:
    """Compute a list of observation names.

    Given an `ObservationManager` and an observation group name, create a list of names for each entry in
    the manager's observation group buffer.

    Args:
        observation_manager: An environment's observation manager.
        group_name: The observation group for which we want to generate names.

    Returns:
        A list of names for each entry in the observation manager's group buffer.
    """
    term_names = observation_manager._group_obs_term_names[group_name]
    term_dims = observation_manager._group_obs_term_dim[group_name]

    names = []
    for i_name, name in enumerate(term_names):
        assert len(term_dims[i_name]) == 1, (
            "Only 1D observation terms are supported when generating names."
        )
        dim = term_dims[i_name][0]
        for i in range(dim):
            names.append(f"{name}_{i:02}")
    return names
