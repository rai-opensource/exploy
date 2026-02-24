# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging

import pytest

# 1. The Launcher must be initialized before any other omni imports
from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def sim_app():
    """
    Session-level fixture to start the SimulationApp.
    Isaac Sim cannot be restarted in the same process, so we keep it alive.
    """
    # Initialize the launcher with AppLauncher arguments
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    yield simulation_app

    # Clean teardown at the end of the test session
    logger.info("Closing Simulation App...")
    simulation_app.close()


@pytest.fixture(scope="function")
def sim_setup(sim_app):
    """
    Function-level fixture to ensure app is ready for tests.
    Each test handles its own stage setup/teardown as needed.
    """
    return sim_app
