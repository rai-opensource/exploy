# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, ArticulationData

from exploy.exporter.frameworks.manager_based.physics_data import PhysicsDataSource


class ArticulationDataSource(PhysicsDataSource):
    """Mimic the interface of an `ArticulationData`, but manage its own tensor data.

    This class is an adaptor for an `ArticulationData` class, mimicking its full interface. However,
    it holds its own tensor data. The main use case for this class is to be used in the context of
    exporting an MDP to ONNX.
    """

    def __init__(self, articulation: Articulation):
        articulation_data: ArticulationData = articulation.data
        super().__init__(articulation, articulation_data, math_utils)

        # IsaacLab-specific: Initialize history for finite differencing
        self._previous_joint_vel = articulation_data._previous_joint_vel.clone()

        # IsaacLab-specific: body acceleration and joint wrench
        self._body_acc_w = articulation_data.body_acc_w.clone()
        self._body_incoming_joint_wrench_b = articulation_data.body_incoming_joint_wrench_b.clone()

    ##
    # Getters to initialize base class with correct data
    ##

    def _get_gravity_vec(self, robot_data: ArticulationData) -> torch.Tensor:
        return robot_data.GRAVITY_VEC_W

    def _get_forward_vec(self, robot_data: ArticulationData) -> torch.Tensor:
        return robot_data.FORWARD_VEC_B

    def _get_com_poses(self, robot_data: ArticulationData) -> tuple[torch.Tensor, torch.Tensor]:
        coms_pos_b = robot_data.body_com_pose_b[:, :, :3].clone()
        coms_quat_b = robot_data.body_com_pose_b[:, :, 3:7].clone()
        return coms_pos_b, coms_quat_b

    def _get_body_link_poses(
        self, robot_data: ArticulationData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get body link positions and quaternions in world frame."""
        return (
            robot_data.body_pos_w,
            robot_data.body_quat_w,
        )

    def _get_body_com_velocities_b(
        self, robot_data: ArticulationData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get body COM linear and angular velocities in body frame.

        IsaacLab stores velocities in world frame, so we need to transform them to body frame.
        """
        # Transform world frame velocities to body frame for each body
        body_lin_vel_b = math_utils.quat_apply_inverse(
            robot_data.body_quat_w, robot_data.body_lin_vel_w
        )
        body_ang_vel_b = math_utils.quat_apply_inverse(
            robot_data.body_quat_w, robot_data.body_ang_vel_w
        )
        return body_lin_vel_b, body_ang_vel_b

    ##
    # IsaacLab-specific properties
    ##

    @property
    def body_incoming_joint_wrench_b(self) -> torch.Tensor:
        """Joint reaction wrench applied from body parent to child body in parent body frame."""
        return self._body_incoming_joint_wrench_b
