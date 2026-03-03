# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import inspect

import pytest
import torch
from mjlab.entity import Entity, EntityData
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg

from exploy.exporter.frameworks.mjlab.entity_data import EntityDataSource


def test_articulation_data_interface():
    """Test the implementation of the `ArticulationDataSource` class by comparing it against each
    property of an `ArticulationData` instance.
    """
    task_name = "Mjlab-Velocity-Flat-Unitree-G1"
    env_cfg: ManagerBasedRlEnvCfg = load_env_cfg(task_name, play=True)
    env_cfg.seed = 42
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
    env.reset()

    # Step the environment for a few steps to populate the articulation data with a non-default state.
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    num_steps = 10
    for _ in range(num_steps):
        env.step(actions)

    entity: Entity = env.scene.entities["robot"]
    entity_data_source = EntityDataSource(entity=entity)

    # Cycle through every property available in `EntityData` and compare it with the same property
    # from `EntityDataSource`.
    for name, _ in inspect.getmembers(EntityData, predicate=lambda o: isinstance(o, property)):
        if "joint_torques" in name:
            continue

        expected_val = getattr(entity.data, name)
        source_val = getattr(entity_data_source, name)
        # For a discussion on how to set tolerances, see:
        #   https://docs.pytorch.org/docs/stable/testing.html
        assert torch.allclose(expected_val, source_val, rtol=1.0e-6, atol=1.0e-5), (
            f"Mismatch found in property '{name}'"
        )

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
