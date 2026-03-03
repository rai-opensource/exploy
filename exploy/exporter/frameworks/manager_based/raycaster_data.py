# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Any

import torch

from exploy.exporter.core.tensor_proxy import TensorProxy
from exploy.exporter.frameworks.manager_based.types import EntityProtocol, SensorProtocol


class RayCasterDataSource:
    """Base class for raycaster sensor data sources.

    This class contains common logic for mimicking raycaster sensor data interfaces
    while managing separate tensor data for ONNX export.
    """

    def __init__(self, sensor: SensorProtocol, robots: dict[str, EntityProtocol]):
        """Initialize the base raycaster data source.

        Args:
            sensor: The raycaster sensor instance (RayCaster or RayCastSensor)
            robots: Dictionary of robot instances (Articulation or Entity)
        """
        self._sensor = sensor
        self._robot, body_ids = prim_path_to_robot_and_body_ids(sensor.cfg.prim_path, robots)
        assert len(body_ids) == 1, "RayCaster must be attached to exactly one body."
        self._body_id = body_ids[0]

        # Split ray_hits_w by last dimension (XYZ) using TensorProxy
        ray_hits = sensor.data.ray_hits_w
        self._ray_hits_w = TensorProxy(ray_hits, split_dim=ray_hits.ndim - 1)

    def _get_body_id_from_sensor(self, sensor: SensorProtocol, robot: EntityProtocol) -> int:
        """Get the body ID that the sensor is attached to.

        Args:
            sensor: The raycaster sensor instance
            robot: The robot instance

        Returns:
            The body ID (index) of the body the sensor is attached to
        """
        body_expr = sensor.cfg.prim_path.split("/")[-1]
        return robot.find_bodies(body_expr)[0][0]

    def _get_body_pos_w(self) -> torch.Tensor:
        """Get the world position of the body the sensor is attached to.

        Returns:
            Body position in world frame. Shape: (num_instances, 3)
        """
        return self._robot.data.body_pos_w[:, self._body_id]

    @property
    def pos_w(self) -> torch.Tensor:
        """Ray origin positions in world frame, computed from robot body position."""
        return self._get_body_pos_w()

    @property
    def ray_hits_w(self) -> torch.Tensor:
        """Ray hit positions in world frame."""
        return self._ray_hits_w

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Empty update call to implement the sensor data interface."""
        pass


def prim_path_to_robot_and_body_ids(
    prim_path: str, robots: dict[str, EntityProtocol]
) -> tuple[EntityProtocol, list[int]]:
    """Convert a primitive path to an robot and body ids.

    Args:
        prim_path (str): The primitive path to convert.
        robots (dict[str, EntityProtocol]): A dictionary of robots in the scene, keyed by their primitive paths.
    Returns:
        A tuple containing the robot and a list of body ids corresponding to the primitive path.
    """
    # Split the primitive path into robot primitive path and body expression.
    body_expr = prim_path.split("/")[-1]
    robot_prim_path = "/".join(prim_path.split("/")[:-1])

    # Find the robot associated with the primitive path. Raise if no robot is found.
    robot_dict = {robot.cfg.prim_path: robot for robot in robots.values()}
    try:
        robot = robot_dict[robot_prim_path]
    except KeyError as e:
        raise KeyError(
            f"Could not find robot with primitive path `{robot_prim_path}` in any of the robot in the scene. "
            f"Available primitive paths are: {robot_dict.keys()}. "
            f"Raised exception: {e}"
        ) from e

    # The the body ids related to the body expression.
    body_ids, _ = robot.find_bodies(body_expr)
    return robot, body_ids
