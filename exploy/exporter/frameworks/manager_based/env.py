# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import torch

from exploy.exporter.core.exportable_environment import ExportableEnvironment
from exploy.exporter.frameworks.manager_based.types import ManagerBasedEnvProtocol


class ExportableEnvironment(ExportableEnvironment, ABC):
    """Base class for exportable environments that provides common functionality."""

    def __init__(
        self,
        env: ManagerBasedEnvProtocol,
        policy_obs_group_name: str = "policy",
    ):
        super().__init__()
        self._env = env
        self._policy_obs_group_name = policy_obs_group_name

    @property
    def env(self) -> ManagerBasedEnvProtocol:
        return self._env

    @property
    def decimation(self) -> int:
        return self._env.cfg.decimation

    def prepare_export(self):
        # We need to call compute on the commands to populate the command buffers if sensors are used.
        # This is not required for other terms, as this is already done in IsaacLab.
        self._env.command_manager.compute(dt=0.0)

    @abstractmethod
    def cleanup(self):
        """Clean up resources (framework-specific)."""
        pass

    @abstractmethod
    def _get_robots_dict(self) -> dict[str, Any]:
        """Get the dictionary of robots (articulations or entities)."""
        pass

    @abstractmethod
    def _get_physics_dt(self) -> float:
        """Get the physics timestep."""
        pass

    @abstractmethod
    def _get_scene_cfg_attr(self) -> str:
        """Get the scene config attribute name ('cfg' or '_cfg')."""
        pass

    def _create_onnx_evaluator_sensor_class(
        self,
        update: Callable[[], None],
        reset: Callable[[], None],
        evaluate_substep: Callable[[int], None],
    ) -> type:
        class ONNXEvaluatorSensor:
            def __init__(self):
                self.sub_step_ctr = 0

            def update(self, dt: float, force_recompute: bool = False):
                evaluate_substep(self.sub_step_ctr)
                self.sub_step_ctr += 1
                update()

            def reset(self, env_ids: Sequence[int]):
                reset()

        return ONNXEvaluatorSensor

    @abstractmethod
    def _compute_observations_kwargs(self) -> dict[str, Any]:
        """Get kwargs for observation_manager.compute()."""
        pass

    def compute_observations(self) -> torch.Tensor:
        """Compute and return the observations of the environment."""
        obs_dict = self._env.observation_manager.compute(**self._compute_observations_kwargs())
        observations = obs_dict[self._policy_obs_group_name].view(1, -1)
        return observations

    def process_actions(self, actions: torch.Tensor):
        """Process actions."""
        self._env.action_manager.process_action(actions)

    def apply_actions(self):
        """Apply processed actions (e.g., joint targets) to the environment"""
        self._env.action_manager.apply_action()

    def validate(self) -> bool:
        """Validate that the environment conforms to the ExportableEnvironment interface."""

        # Check that the observation manager has a policy group.
        if self._policy_obs_group_name not in self._env.observation_manager.active_terms:
            raise RuntimeError(
                f"[OnnxEnvironmentExporter] Could not find observation group named {self._policy_obs_group_name} in"
                f" observation manager. Active terms are: {self._env.observation_manager.active_terms}"
            )

        # Check if observation terms are concatenated.
        if not self._env.observation_manager.group_obs_concatenate[self._policy_obs_group_name]:
            raise RuntimeError("[OnnxEnvironmentExporter] Observation terms must be concatenated.")

        # Check that observation noise is disabled, else it will be part of the computational graph.
        for group_cfg in self._env.observation_manager._group_obs_term_cfgs.values():
            for term_cfg in group_cfg:
                if term_cfg.noise is not None:
                    raise RuntimeError(
                        "[OnnxEnvironmentExporter] While trying to convert to ONNX, found an observation term with"
                        " noise enabled."
                        "\n[OnnxEnvironmentExporter] Hint: turn off observation noise, or use a `Play` task."
                    )

        # Check if encoder bias is zero
        if hasattr(self._env.scene, "entities"):
            for entity_name, entity in self._env.scene.entities.items():
                if hasattr(entity._data, "encoder_bias") and not torch.allclose(
                    entity._data.encoder_bias, torch.zeros_like(entity._data.encoder_bias)
                ):
                    raise RuntimeError(
                        "[OnnxEnvironmentExporter] Encoder bias must be zero for export. Found non-zero encoder bias"
                        f" in entity '{entity_name}'."
                    )

        return True

    def empty_actor_observations(self) -> torch.Tensor:
        return torch.zeros_like(
            self._env.observation_manager._obs_buffer[self._policy_obs_group_name]
        )

    def empty_actions(self) -> torch.Tensor:
        return torch.zeros_like(self._env.action_manager._action)

    def metadata(self) -> dict[str, str]:
        metadata = {}

        # Observation names.
        metadata["obs_term_names"] = json.dumps(
            self._env.observation_manager._group_obs_term_names[self._policy_obs_group_name]
        )

        # Decimation info.
        metadata["decimation"] = str(self._env.cfg.decimation)

        # Sim_dt info.
        physics_dt = self._get_physics_dt()
        metadata["sim_dt"] = str(physics_dt)

        # Update rate.
        metadata["update_rate"] = str(1.0 / physics_dt)

        return metadata

    def register_evaluation_hooks(
        self,
        update: Callable[[], None],
        reset: Callable[[], None],
        evaluate_substep: Callable[[int], None],
    ):
        """Register evaluation hooks for this environment."""

        # Disable lazy sensor update
        scene_cfg_attr = self._get_scene_cfg_attr()
        setattr(self._env.scene, scene_cfg_attr, getattr(self._env.scene, scene_cfg_attr))
        getattr(self._env.scene, scene_cfg_attr).lazy_sensor_update = False

        # Create framework-specific sensor class
        ONNXEvaluatorSensor = self._create_onnx_evaluator_sensor_class(
            update, reset, evaluate_substep
        )
        self._env.scene._sensors["onnx"] = ONNXEvaluatorSensor()

        class ONNXEvaluatorCommand:
            def compute(self, dt: float):
                update()

            def reset(self, env_ids: Sequence[int]) -> dict[str, float]:
                reset()
                return {}

        self._env.command_manager._terms["onnx"] = ONNXEvaluatorCommand()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Step the environment forward by one step."""
        self._env.scene._sensors["onnx"].sub_step_ctr = 0
        obs, _, dones, timeouts, _ = self._env.step(actions)
        is_reset: bool = torch.logical_or(dones, timeouts).any()
        obs = obs[self._policy_obs_group_name]
        return obs, is_reset

    @abstractmethod
    def get_observation_names(self) -> list[str]:
        """Get observation names (framework-specific)."""
        pass

    def observations_reset(self) -> torch.Tensor:
        with torch.inference_mode():
            obs_buf, _ = self._env.reset()
            return obs_buf[self._policy_obs_group_name].clone()
