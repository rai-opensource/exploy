# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import importlib.util
import os
import pathlib
from collections.abc import Callable

import pytest

# Detect Isaac Lab 3 by its API: the version of a source installation is the version of the `isaaclab`
# extension, not of the release.
HAS_ISAACLAB3 = (
    importlib.util.find_spec("isaaclab") is not None
    and importlib.util.find_spec("isaaclab.utils.warp.proxy_array") is not None
)

# Isaac Lab 3 runs kit-less: the tests create their environments with the Newton physics backend on the
# CPU by default. Set these to e.g. `cuda:0` and `physics=isaacsim_physx` to test other backends.
DEVICE = os.environ.get("EXPLOY_ISAACLAB3_DEVICE", "cpu")
OVERRIDES = os.environ.get("EXPLOY_ISAACLAB3_OVERRIDES", "physics=newton_mjwarp").split()


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Mark the tests of this folder with `isaaclab3`, and skip them if Isaac Lab 3 is not found."""
    for item in items:
        if str(pathlib.Path(__file__).parent) in str(item.path):
            item.add_marker(marker="isaaclab3")

    if HAS_ISAACLAB3:
        return

    skip_isaaclab3 = pytest.mark.skip(
        reason="Optional dependency 'isaaclab>=3' not found, skipping test"
    )
    for item in items:
        if "isaaclab3" in item.keywords:
            item.add_marker(skip_isaaclab3)


@pytest.fixture
def make_env() -> Callable:
    """Return a factory of Isaac Lab environments (in the play mode of their task, as exported), closing
    them at the end of the test."""
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import resolve_task_config

    envs = []

    def _make_env(task_name: str, num_envs: int, seed: int = 42):
        env_cfg, _ = resolve_task_config(task_name, None, play_mode=True, overrides=OVERRIDES)
        env_cfg.sim.device = DEVICE
        env_cfg.scene.num_envs = num_envs
        env_cfg.seed = seed
        env = gym.make(task_name, cfg=env_cfg, render_mode=None).unwrapped
        envs.append(env)
        return env

    yield _make_env

    for env in envs:
        env.close()
