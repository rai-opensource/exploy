# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch

from exploy.exporter.core.context_manager import ContextManager, Memory
from exploy.exporter.frameworks.manager_based.types import ManagerBasedEnvProtocol


def add_memory(
    env: ManagerBasedEnvProtocol,
    context_manager: ContextManager,
    attr_name: str,
    action_term_name: str | None = None,
):
    """Add a single memory component for an action term attribute.

    Args:
        env: The environment (ManagerBasedRLEnv or ManagerBasedRlEnv)
        context_manager: The context manager to add components to
        action_term_name: Name of the action term, or None to use action_manager directly
        attr_name: The attribute name to get from the action term or action_manager
    """
    if action_term_name is None:
        # Get from action_manager directly
        source = env.action_manager
        memory_name = attr_name.lstrip("_")
    else:
        # Get from specific action term
        source = env.action_manager.get_term(action_term_name)
        clean_attr_name = attr_name.lstrip("_")
        memory_name = f"{action_term_name}.{clean_attr_name}"

    def getter_func() -> torch.Tensor:
        return getattr(source, attr_name)

    context_manager.add_component(Memory(name=memory_name, get_from_env_cb=getter_func))
