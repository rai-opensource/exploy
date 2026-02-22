# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import pytest
from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import inspect

import torch
from isaaclab.assets import Articulation
from isaaclab_tasks.utils import parse_env_cfg

from exploy.exporter.frameworks.isaaclab.articulation_data import ArticulationDataSource

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


def test_articulation_data_interface():
    """Test the implementation of the `ArticulationDataSource` class by comparing it against each
    property of an `ArticulationData` instance.
    """
    import isaaclab_tasks.manager_based.locomotion.velocity.config.g1  # noqa: F401

    task_name = "Isaac-Velocity-Rough-G1-Play-v0"
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(task_name, num_envs=10, device="cpu")
    env_cfg.seed = 42
    env: ManagerBasedRLEnv = gym.make(
        task_name,
        cfg=env_cfg,
        render_mode=None,
    ).unwrapped

    # Step the environment for a few steps to populate the articulation data with a non-default state.
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    num_steps = 10
    for _ in range(num_steps):
        env.step(actions)

    articulation: Articulation = env.scene.articulations["robot"]
    articulation_data_source = ArticulationDataSource(articulation=articulation)

    # Cycle through every property available in `ArticulationDataSource` and compare it with the same property
    # from `ArticulationData`.
    for name, _ in inspect.getmembers(
        ArticulationDataSource, predicate=lambda o: isinstance(o, property)
    ):
        if "tendon" in name:
            continue

        expected_val = getattr(articulation.data, name)
        source_val = getattr(articulation_data_source, name)
        # For a discussion on how to set tolerances, see:
        #   https://docs.pytorch.org/docs/stable/testing.html
        assert torch.allclose(expected_val, source_val, rtol=1.0e-6, atol=1.0e-5)

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
