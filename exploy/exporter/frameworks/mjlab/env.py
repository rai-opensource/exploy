# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Any

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import RayCastSensor

from exploy.exporter.frameworks.manager_based.env import ExportableEnvironment
from exploy.exporter.frameworks.manager_based.raycaster_data import RayCasterDataSource
from exploy.exporter.frameworks.manager_based.utils import get_observation_names
from exploy.exporter.frameworks.mjlab.entity_data import EntityDataSource


class MjlabExportableEnvironment(ExportableEnvironment):
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        policy_obs_group_name: str = "policy",
    ):
        super().__init__(env=env, policy_obs_group_name=policy_obs_group_name)

        # Replace entity data.
        self._entity_data_list = []
        for entity in self._env.scene.entities.values():
            self._entity_data_list.append(entity._data)
            entity._data = EntityDataSource(entity=entity)

        # Replace raycaster sensor data.
        self._raycaster_data_list = []
        for sensor_name, sensor in self._env.scene.sensors.items():
            if isinstance(sensor, RayCastSensor):
                self._raycaster_data_list.append((sensor_name, sensor._data))
                sensor._data = RayCasterDataSource(sensor, self._env.scene.entities)

        assert self.validate()

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        for i_entity, entity in enumerate(self._env.scene.entities.values()):
            entity._data = self._entity_data_list[i_entity]

        for sensor_name, original_data in self._raycaster_data_list:
            self._env.scene.sensors[sensor_name]._data = original_data

    def _get_robots_dict(self) -> dict[str, Any]:
        return self._env.scene.entities

    def _get_physics_dt(self) -> float:
        return self._env.physics_dt

    def _get_scene_cfg_attr(self) -> str:
        return "_cfg"

    def _compute_observations_kwargs(self) -> dict[str, Any]:
        return {"update_history": True}

    def get_observation_names(self) -> list[str]:
        return get_observation_names(
            observation_manager=self._env.observation_manager,
            group_name=self._policy_obs_group_name,
        )
