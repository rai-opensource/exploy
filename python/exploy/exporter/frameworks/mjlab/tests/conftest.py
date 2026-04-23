# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
import pathlib

import pytest

try:
    import mjlab  # noqa: F401

    HAS_MJLAB = True
except ImportError:
    HAS_MJLAB = False

logger = logging.getLogger(__name__)


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Modify collected test items to add custom markers based on their file paths.

    Note:
        MjLab tests are skipped in the global test suite if ``mjlab`` is not found.
    """
    for item in items:
        if str(pathlib.Path(__file__).parent) in str(item.path):
            item.add_marker(marker="mjlab")

    if HAS_MJLAB:
        return

    skip_mjlab = pytest.mark.skip(
        reason="Optional dependency 'mjlab' not found, skipping test",
    )

    for item in items:
        if "mjlab" in item.keywords:
            item.add_marker(skip_mjlab)


@pytest.fixture(scope="session")
def mjlab_env():
    """Session-level fixture that creates a MjLab environment.

    The environment is shared across all tests in the session to avoid the overhead
    of re-creating it for each test.
    """

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    task_name = "Mjlab-Velocity-Flat-Unitree-G1"

    env_cfg = load_env_cfg(task_name, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.events.pop("encoder_bias", None)

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
    env.reset()

    yield env

    logger.info("Closing MjLab environment...")
    env.close()


@pytest.fixture(scope="function")
def env(mjlab_env):
    """Function-level fixture providing the environment to individual tests."""
    return mjlab_env
