# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import functools

from exploy.exporter.core.context_manager import ContextManager, Group, Output
from exploy.exporter.frameworks.manager_based.types import EntityProtocol, ManagerBasedEnvProtocol
from exploy.exporter.frameworks.manager_based.utils import get_robot_actuator_gains

OUTPUT_PREFIX = "output"


def add_output(
    env: ManagerBasedEnvProtocol, context_manager: ContextManager, action_term_name: str
):
    """Add output targets for a single action term to the context manager.

    Args:
        env: The environment
        context_manager: The context manager to add components to
        action_term_name: Name of the action term to add outputs for
    """

    action_manager = env.action_manager
    action_term = action_manager.get_term(action_term_name)
    if hasattr(action_term, "_asset"):  # isaaclab
        robot = action_term._asset
        joint_names_expr = action_term.cfg.joint_names
        joint_ids, joint_names = robot.find_joints(joint_names_expr)
    elif hasattr(action_term, "_entity"):  # mjlab
        robot = action_term._entity
        joint_names = robot.joint_names
        joint_ids = list(range(len(joint_names)))
    else:
        raise AttributeError(
            f"Could not infer manager type for action term '{action_term_name}'. "
            "Expected action term to expose either '_asset' (isaaclab) or '_entity' (mjlab)."
        )

    # Make getter functions for joint states.
    def get_joint_pos_target(entity: EntityProtocol, joint_ids: list[int]):
        return entity.data.joint_pos_target[..., joint_ids]

    def get_joint_vel_target(entity: EntityProtocol, joint_ids: list[int]):
        return entity.data.joint_vel_target[..., joint_ids]

    def get_joint_eff_target(entity: EntityProtocol, joint_ids: list[int]):
        return entity.data.joint_effort_target[..., joint_ids]

    # Update metadata.
    actuator_gains = get_robot_actuator_gains(robot)

    onnx_joint_outputs = Group(
        name=f"{OUTPUT_PREFIX}.joint_targets.{action_term_name}",
        metadata={
            "type": "joint_targets",
            "names": joint_names,
            "stiffness": [actuator_gains[name]["stiffness"] for name in joint_names],
            "damping": [actuator_gains[name]["damping"] for name in joint_names],
        },
        items=[
            Output(
                name="pos",
                get_from_env_cb=functools.partial(get_joint_pos_target, robot, joint_ids.copy()),
            ),
            Output(
                name="vel",
                get_from_env_cb=functools.partial(get_joint_vel_target, robot, joint_ids.copy()),
            ),
            Output(
                name="effort",
                get_from_env_cb=functools.partial(get_joint_eff_target, robot, joint_ids.copy()),
            ),
        ],
    )

    context_manager.add_group(group=onnx_joint_outputs)
