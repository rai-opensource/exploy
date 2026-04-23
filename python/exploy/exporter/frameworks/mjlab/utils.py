# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.managers import ObservationManager


def get_observation_names(
    observation_manager: ObservationManager,
    group_name: str = "actor",
) -> list[str]:
    """Compute a list of observation names for the flattened observation buffer.

    Args:
        observation_manager: The MjLab observation manager.
        group_name: The observation group (default: ``"actor"``).

    Returns:
        A list of names for each scalar in the flattened observation buffer.
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


def get_entity_actuator_gains(entity: Entity) -> dict[str, dict[str, float]]:
    """Extract stiffness and damping gains from a MjLab entity's actuator config.

    Iterates the entity's actuators and returns a dict mapping joint name to
    ``{"stiffness": float, "damping": float}``.  Only actuators that expose
    ``cfg.stiffness`` / ``cfg.damping`` (e.g. ``IdealPdActuator``) contribute
    gains; other actuator types map to zeros.

    Args:
        entity: A MjLab entity (robot) with actuators.

    Returns:
        Dict mapping joint/target name to gain values.
    """
    gains: dict[str, dict[str, float]] = {}

    for actuator in entity._actuators:
        cfg = actuator.cfg
        stiffness = float(getattr(cfg, "stiffness", 0.0))
        damping = float(getattr(cfg, "damping", 0.0))
        for name in actuator._target_names:
            gains[name] = {"stiffness": stiffness, "damping": damping}

    return gains
