# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from __future__ import annotations

import argparse
import pathlib
from dataclasses import asdict

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.runners import OnPolicyRunner

from exploy.exporter.core.evaluator import evaluate
from exploy.exporter.core.exporter import export_environment_as_onnx
from exploy.exporter.core.session_wrapper import SessionWrapper
from exploy.exporter.frameworks.manager_based import inputs, memory, outputs
from exploy.exporter.frameworks.manager_based.actor import make_exportable_actor
from exploy.exporter.frameworks.mjlab.env import MjlabExportableEnvironment


def get_args() -> argparse.Namespace:
    # Create argument parser for headless mode
    parser = argparse.ArgumentParser(description="Export mjlab environment to ONNX")

    # Add custom arguments
    parser.add_argument(
        "--task",
        type=str,
        default="Mjlab-Velocity-Flat-Unitree-G1",
        help="Name of the mjlab task to export (default: Mjlab-Velocity-Flat-Unitree-G1)",
    )
    parser.add_argument(
        "--pause-on-failure",
        action="store_true",
        default=False,
        help="Pause on evaluation failure (default: False, useful for debugging)",
    )

    args_cli = parser.parse_args()
    args_cli.headless = True
    args_cli.num_envs = 1

    return args_cli


args = get_args()


def export(task_name: str = "Mjlab-Velocity-Flat-Unitree-G1", pause_on_failure: bool = False):
    """Test mjlab ONNX export and evaluation pipeline."""
    test_dir = pathlib.Path(__file__).parent / "exporter_tests"

    task_device = "cpu"

    env_cfg: ManagerBasedRlEnvCfg = load_env_cfg(task_name, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.events.pop("encoder_bias", None)
    agent_cfg: RslRlOnPolicyRunnerCfg = load_rl_cfg(task_name)

    # create mjlab environment
    # wrap around environment for rsl-rl
    env = ManagerBasedRlEnv(cfg=env_cfg, device=task_device, render_mode=None)
    env.reset()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_name) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), log_dir=test_dir, device=task_device)
    runner.logger_type = "disabled"

    # Export to ONNX.
    onnx_export_dir = test_dir
    onnx_export_file = "test_export.onnx"

    unwrapped_env = env.unwrapped
    exportable_env = MjlabExportableEnvironment(unwrapped_env)

    # Get the policy and its normalizer.
    alg: PPO = runner.alg
    assert isinstance(alg, PPO), f"Expected PPO algorithm, got: {type(alg).__name__}"
    actor = make_exportable_actor(exportable_env, alg.policy, device=task_device)

    entities = unwrapped_env.scene.entities
    context_manager = exportable_env.context_manager()

    inputs.add_command(unwrapped_env, context_manager, "twist", "se2_velocity")

    inputs.add_base_vel(entities, context_manager)

    inputs.add_body_pos_and_quat(entities, context_manager)

    inputs.add_joint_pos_and_vel(entities, context_manager)

    for sensor_name, sensor in unwrapped_env.scene.sensors.items():
        inputs.add_sensor_input(sensor_name, sensor, context_manager)

    # Memory and outputs
    memory.add_memory(unwrapped_env, context_manager, attr_name="_action")
    for action_term_name in unwrapped_env.action_manager.active_terms:
        memory.add_memory(
            unwrapped_env,
            context_manager,
            attr_name="_processed_actions",
            action_term_name=action_term_name,
        )
        outputs.add_output(
            unwrapped_env,
            context_manager,
            action_term_name=action_term_name,
        )

    export_environment_as_onnx(
        env=exportable_env,
        actor=actor,
        path=onnx_export_dir,
        filename=onnx_export_file,
        verbose=False,
    )

    exportable_env.cleanup()

    # Make a session wrapper.
    session_wrapper = SessionWrapper(
        onnx_folder=onnx_export_dir,
        onnx_file_name=onnx_export_file,
        actor=actor,
        optimize=True,
    )

    # Evaluate.
    evaluate_steps = 200
    with torch.inference_mode():
        export_ok, _ = evaluate(
            env=exportable_env,
            context_manager=exportable_env.context_manager(),
            session_wrapper=session_wrapper,
            num_steps=evaluate_steps,
            verbose=True,
            pause_on_failure=pause_on_failure,
        )

    env.close()

    assert export_ok, "ONNX export validation failed"


if __name__ == "__main__":
    import sys

    try:
        export(task_name=args.task, pause_on_failure=args.pause_on_failure)
    except Exception as e:
        print(f"❌ Test ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
