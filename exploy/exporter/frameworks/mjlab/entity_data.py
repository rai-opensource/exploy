# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.


import mjlab.utils.lab_api.math as math_utils
import torch
from mjlab.entity import Entity, EntityData

from exploy.exporter.frameworks.manager_based.physics_data import PhysicsDataSource


class EntityDataSource(PhysicsDataSource):
    """Mimic the interface of an `EntityData`, but manage its own tensor data.

    This class is an adaptor for an `EntityData` class, mimicking its full interface.
    However, it holds its own tensor data. The main use case for this class is to be used
    in the context of exporting an MDP to ONNX.
    """

    def __init__(self, entity: Entity):
        entity_data: EntityData = entity.data
        super().__init__(entity, entity_data, math_utils)

        # mjlab-specific: actuator force
        self._actuator_force = entity_data.actuator_force.clone()

    ##
    # Getters to initialize base class with correct data
    ##

    def _get_gravity_vec(self, robot_data: EntityData) -> torch.Tensor:
        return robot_data.gravity_vec_w

    def _get_forward_vec(self, robot_data: EntityData) -> torch.Tensor:
        return robot_data.forward_vec_b

    def _get_com_poses(self, robot_data: EntityData) -> tuple[torch.Tensor, torch.Tensor]:
        body_ipos = robot_data.model.body_ipos[:, robot_data.indexing.body_ids]
        body_iquat = robot_data.model.body_iquat[:, robot_data.indexing.body_ids]
        return body_ipos, body_iquat

    def _get_body_link_poses(self, robot_data: EntityData) -> tuple[torch.Tensor, torch.Tensor]:
        """Get body link positions and quaternions in world frame."""
        return (
            robot_data.body_link_pos_w,
            robot_data.body_link_quat_w,
        )

    def _get_body_com_velocities_b(
        self, robot_data: EntityData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get body COM linear and angular velocities in body frame.

        mjlab stores velocities in world frame, so we need to transform them to body frame.
        """
        # Transform world frame velocities to body frame for each body
        body_lin_vel_b = math_utils.quat_apply_inverse(
            robot_data.body_link_quat_w, robot_data.body_com_lin_vel_w
        )
        body_ang_vel_b = math_utils.quat_apply_inverse(
            robot_data.body_link_quat_w, robot_data.body_com_ang_vel_w
        )
        return body_lin_vel_b, body_ang_vel_b

    ##
    # mjlab-specific properties
    ##

    @property
    def body_names(self) -> list[str]:
        """Names of all bodies/links in the entity."""
        return self._robot.body_names

    @property
    def actuator_force(self) -> torch.Tensor:
        """Actuator forces of all actuators. Shape is (num_instances, num_actuators)."""
        return self._actuator_force

    @property
    def site_pose_w(self) -> torch.Tensor:
        """Site poses in world frame."""
        return self._data.site_pose_w

    @property
    def site_pos_w(self) -> torch.Tensor:
        """Site positions in world frame."""
        return self._data.site_pos_w

    @property
    def site_quat_w(self) -> torch.Tensor:
        """Site orientations in world frame."""
        return self._data.site_quat_w

    @property
    def site_vel_w(self) -> torch.Tensor:
        """Site velocities in world frame."""
        return self._data.site_vel_w

    @property
    def site_lin_vel_w(self) -> torch.Tensor:
        """Site linear velocities in world frame."""
        return self._data.site_lin_vel_w

    @property
    def site_ang_vel_w(self) -> torch.Tensor:
        """Site angular velocities in world frame."""
        return self._data.site_ang_vel_w

    @property
    def geom_pose_w(self) -> torch.Tensor:
        """Geom poses in world frame."""
        return self._data.geom_pose_w

    @property
    def geom_pos_w(self) -> torch.Tensor:
        """Geom positions in world frame."""
        return self._data.geom_pos_w

    @property
    def geom_quat_w(self) -> torch.Tensor:
        """Geom orientations in world frame."""
        return self._data.geom_quat_w

    @property
    def geom_vel_w(self) -> torch.Tensor:
        """Geom velocities in world frame."""
        return self._data.geom_vel_w

    @property
    def geom_lin_vel_w(self) -> torch.Tensor:
        """Geom linear velocities in world frame."""
        return self._data.geom_lin_vel_w

    @property
    def geom_ang_vel_w(self) -> torch.Tensor:
        """Geom angular velocities in world frame."""
        return self._data.geom_ang_vel_w

    @property
    def tendon_len(self) -> torch.Tensor:
        """Tendon lengths."""
        return self._data.tendon_len

    @property
    def tendon_vel(self) -> torch.Tensor:
        """Tendon velocities."""
        return self._data.tendon_vel

    @property
    def joint_torques(self) -> torch.Tensor:
        """Joint torques."""
        return self._data.joint_torques

    @property
    def joint_pos_biased(self) -> torch.Tensor:
        """Joint positions with encoder bias applied."""
        return self._data.joint_pos_biased

    @property
    def generalized_force(self) -> torch.Tensor:
        """Generalized forces."""
        return self._data.generalized_force

    @property
    def body_external_force(self) -> torch.Tensor:
        """External forces on bodies."""
        return self._data.body_external_force

    @property
    def body_external_torque(self) -> torch.Tensor:
        """External torques on bodies."""
        return self._data.body_external_torque

    @property
    def body_external_wrench(self) -> torch.Tensor:
        """External wrenches on bodies."""
        return self._data.body_external_wrench

    def write_ctrl(
        self,
        ctrl: torch.Tensor,
        ctrl_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """Write control values to the entity data."""
        if not self._data.is_actuated:
            return

        env_ids = self._data._resolve_env_ids(env_ids)
        local_ctrl_ids = ctrl_ids if ctrl_ids is not None else slice(None)
        global_ctrl_ids = self._data.indexing.ctrl_ids[local_ctrl_ids]
        self._data.data.ctrl[env_ids, global_ctrl_ids] = ctrl
