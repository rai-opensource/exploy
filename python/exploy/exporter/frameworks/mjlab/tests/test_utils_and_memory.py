# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch

from exploy.exporter.core.context_manager import ContextManager
from exploy.exporter.frameworks.mjlab.memory import add_memory
from exploy.exporter.frameworks.mjlab.utils import get_observation_names


class _DummyObservationManager:
    def __init__(self):
        self._group_obs_term_names = {
            "actor": ["base", "joints"],
        }
        self._group_obs_term_dim = {
            "actor": [(2,), (3,)],
        }


def test_get_observation_names_default_group_is_actor():
    obs_manager = _DummyObservationManager()

    names = get_observation_names(observation_manager=obs_manager)

    assert names == ["base_00", "base_01", "joints_00", "joints_01", "joints_02"]


class _DummyActionTerm:
    def __init__(self, processed_actions: torch.Tensor):
        self._processed_actions = processed_actions


class _DummyActionManager:
    def __init__(self):
        self._action = torch.zeros(1, 4)
        self._terms = {
            "walk": _DummyActionTerm(processed_actions=torch.ones(1, 2)),
        }
        self.active_terms = list(self._terms.keys())

    def get_term(self, name: str) -> _DummyActionTerm:
        return self._terms[name]


class _DummyEnv:
    def __init__(self):
        self.action_manager = _DummyActionManager()


def test_add_memory_registers_actions_memory_name():
    env = _DummyEnv()
    context_manager = ContextManager()

    add_memory(env=env, context_manager=context_manager)

    input_names = context_manager.get_input_names()
    output_names = context_manager.get_output_names()

    assert "memory.actions.in" in input_names
    assert "memory.action.in" not in input_names
    assert "memory.actions.out" in output_names
    assert "memory.walk.processed_actions.in" in input_names
    assert "memory.walk.processed_actions.out" in output_names
