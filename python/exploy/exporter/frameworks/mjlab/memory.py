# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch

from exploy.exporter.core.context_manager import ContextManager, Memory


def add_memory(
    env,
    context_manager: ContextManager,
) -> None:
    """Add memory components for the action manager.

    Registers the latest actions and processed actions (one per active action term)
    as Memory components so they are available as ONNX graph inputs.

    Args:
        env: The MjLab ManagerBasedRlEnv.
        context_manager: The context manager to add components to.
    """
    action_manager = env.action_manager

    def get_action() -> torch.Tensor:
        return action_manager._action

    context_manager.add_component(Memory(name="actions", get_from_env_cb=get_action))

    def make_getter(term):
        def getter() -> torch.Tensor:
            return term._processed_actions

        return getter

    for term_name in action_manager.active_terms:
        term = action_manager.get_term(term_name)
        context_manager.add_component(
            Memory(name=f"{term_name}.processed_actions", get_from_env_cb=make_getter(term))
        )


def add_term_memory(
    action_manager,
    context_manager: ContextManager,
    action_term_name: str,
    attr_name: str,
) -> None:
    """Register an arbitrary attribute of an action term as a Memory component.

    Args:
        action_manager: The MjLab action manager.
        context_manager: The context manager to add components to.
        action_term_name: Name of the action term in the action manager.
        attr_name: Attribute name on the action term to expose as memory.
    """
    term = action_manager.get_term(action_term_name)

    def getter() -> torch.Tensor:
        return getattr(term, attr_name)

    context_manager.add_component(
        Memory(name=f"{action_term_name}.{attr_name}", get_from_env_cb=getter)
    )
