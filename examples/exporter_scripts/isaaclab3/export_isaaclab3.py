# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Export an Isaac Lab 3 environment and an (untrained) RSL-RL policy to ONNX, and evaluate the export.

Usage:
    python examples/exporter_scripts/isaaclab3/export_isaaclab3.py --task Isaac-Velocity-Flat-G1 physics=newton_mjwarp

Arguments of the form ``key=value`` (e.g. ``physics=isaacsim_physx``, ``presets=...``) are Isaac Lab task
configuration overrides.
"""

from __future__ import annotations

import argparse
import pathlib
from importlib import metadata

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import resolve_task_config
from rsl_rl.runners import OnPolicyRunner

from exploy.exporter.core.evaluator import evaluate
from exploy.exporter.core.exporter import export_environment_as_onnx
from exploy.exporter.core.session_wrapper import SessionWrapper
from exploy.exporter.frameworks.isaaclab3 import (
    environments,  # noqa: F401
    inputs,
    memory,
    outputs,
)
from exploy.exporter.frameworks.isaaclab3.actor import make_exportable_actor
from exploy.exporter.frameworks.isaaclab3.env import IsaacLabExportableEnvironment


def export_isaaclab3(
    task_name: str,
    overrides: list[str],
    device: str = "cpu",
    pause_on_failure: bool = False,
    export_dir: pathlib.Path | None = None,
) -> bool:
    """Export an Isaac Lab 3 task to ONNX and evaluate the exported graph against the environment."""
    export_dir = export_dir or pathlib.Path(__file__).parent / "exporter_tests"
    onnx_export_file = "test_export.onnx"

    env_cfg, agent_cfg = resolve_task_config(
        task_name, "rsl_rl_cfg_entry_point", play_mode=True, overrides=overrides
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = 1

    with launch_simulation(env_cfg, {}):
        # Create the environment, wrapped for RSL-RL.
        env = RslRlVecEnvWrapper(gym.make(task_name, cfg=env_cfg, render_mode=None))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)

        exportable_env = IsaacLabExportableEnvironment(env.unwrapped)
        actor = make_exportable_actor(exportable_env, runner.alg.actor, device=device)

        articulations = env.unwrapped.scene.articulations
        context_manager = exportable_env.context_manager()

        inputs.add_body_pos_and_quat(articulations=articulations, context_manager=context_manager)
        inputs.add_body_vel(articulations=articulations, context_manager=context_manager)
        inputs.add_commands(
            command_manager=env.unwrapped.command_manager, context_manager=context_manager
        )
        inputs.add_joint_pos_and_vel(articulations=articulations, context_manager=context_manager)
        inputs.add_sensor_inputs(
            sensors=env.unwrapped.scene.sensors, context_manager=context_manager
        )
        memory.add_memory(env=env.unwrapped, context_manager=context_manager)
        outputs.add_outputs(
            action_manager=env.unwrapped.action_manager, context_manager=context_manager
        )

        export_environment_as_onnx(
            env=exportable_env,
            actor=actor,
            path=export_dir,
            filename=onnx_export_file,
            verbose=False,
            ir_version=10,
        )

        exportable_env.cleanup()

        session_wrapper = SessionWrapper(
            onnx_folder=export_dir,
            onnx_file_name=onnx_export_file,
            actor=actor,
            optimize=True,
        )

        with torch.inference_mode():
            export_ok, _ = evaluate(
                env=exportable_env,
                context_manager=exportable_env.context_manager(),
                session_wrapper=session_wrapper,
                num_episodes=2,
                max_episode_steps=100,
                verbose=True,
                pause_on_failure=pause_on_failure,
            )

        env.close()

    return export_ok


def main():
    parser = argparse.ArgumentParser(description="Export an Isaac Lab 3 environment to ONNX.")
    parser.add_argument(
        "--task", type=str, default="Isaac-Velocity-Flat-G1", help="Isaac Lab task to export."
    )
    parser.add_argument("--device", type=str, default="cpu", help="Simulation and policy device.")
    parser.add_argument(
        "--pause-on-failure",
        action="store_true",
        default=False,
        help="Pause on evaluation failure (useful for debugging).",
    )
    args, overrides = parser.parse_known_args()

    export_ok = export_isaaclab3(
        task_name=args.task,
        overrides=overrides,
        device=args.device,
        pause_on_failure=args.pause_on_failure,
    )
    assert export_ok, "ONNX export validation failed"


if __name__ == "__main__":
    main()
