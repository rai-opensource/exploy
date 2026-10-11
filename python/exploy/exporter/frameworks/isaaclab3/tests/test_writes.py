# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations


def test_recorded_joint_targets_match_simulation(make_env):
    """Test that the joint targets recorded by `ArticulationWriteRecorder` during export match the targets
    that the action terms write to the actuators of the simulation."""
    import torch

    from exploy.exporter.frameworks.isaaclab3.env import IsaacLabExportableEnvironment
    from exploy.exporter.frameworks.isaaclab3.writes import ArticulationWriteRecorder

    env = make_env("Isaac-Velocity-Flat-G1", num_envs=2)
    torch.manual_seed(0)
    actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    articulation = env.scene.articulations["robot"]

    # The targets written to the simulation.
    env.action_manager.process_action(actions)
    env.action_manager.apply_action()
    expected_pos_target = articulation.actuators.target_command.position.torch.clone()

    exportable_env = IsaacLabExportableEnvironment(env)
    term = env.action_manager.get_term("joint_pos")
    assert isinstance(term._asset, ArticulationWriteRecorder)

    # Recorded targets start from the snapshot, and are overwritten by the action terms.
    source = articulation.data
    source.joint_pos_target.torch.zero_()
    exportable_env.process_actions(actions)
    exportable_env.apply_actions()
    joint_ids = articulation.find_joints(term.cfg.joint_names)[0]
    torch.testing.assert_close(
        source.joint_pos_target.torch[:, joint_ids], expected_pos_target[:, joint_ids]
    )

    # The simulation targets are left untouched while the writes are recorded.
    exportable_env.process_actions(torch.zeros_like(actions))
    exportable_env.apply_actions()
    exportable_env.cleanup()
    torch.testing.assert_close(
        articulation.actuators.target_command.position.torch, expected_pos_target
    )
    assert not isinstance(term._asset, ArticulationWriteRecorder)
