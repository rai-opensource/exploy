# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject, RigidObjectData

from exploy.exporter.frameworks.manager_based.physics_data import PhysicsDataSource


class RigidObjectDataSource(PhysicsDataSource):
    """Mimic the interface of a `RigidObjectData`, but manage its own tensor data.

    This class is an adaptor for a `RigidObjectData` class, mimicking its full interface. However,
    it holds its own tensor data. The main use case for this class is to be used in the context of
    exporting an MDP to ONNX.
    """

    def __init__(self, rigid_object: RigidObject):
        rigid_object_data: RigidObjectData = rigid_object.data

        # For rigid objects, create empty joint tensors since they have no joints
        num_instances = rigid_object_data.root_pos_w.shape[0]
        device = rigid_object_data.root_pos_w.device
        empty_joint_data = torch.zeros((num_instances, 0), device=device)

        # Temporarily add joint data to rigid_object_data for base class initialization
        rigid_object_data.joint_pos = empty_joint_data
        rigid_object_data.joint_vel = empty_joint_data
        rigid_object_data.joint_acc = empty_joint_data

        super().__init__(rigid_object, rigid_object_data, math_utils)

        # RigidObject-specific: body acceleration
        self._body_acc_w = rigid_object_data.body_acc_w.clone()

    ##
    # Getters to initialize base class with correct data
    ##

    def _get_gravity_vec(self, robot_data: RigidObjectData) -> torch.Tensor:
        return robot_data.GRAVITY_VEC_W

    def _get_forward_vec(self, robot_data: RigidObjectData) -> torch.Tensor:
        return robot_data.FORWARD_VEC_B

    def _get_com_poses(self, robot_data: RigidObjectData) -> tuple[torch.Tensor, torch.Tensor]:
        pos = robot_data.body_com_pose_b[..., :3].clone()
        quat = robot_data.body_com_pose_b[..., 3:7].clone()
        return pos, quat

    def _get_body_link_poses(
        self, robot_data: RigidObjectData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get body link positions and quaternions in world frame.

        For rigid objects, body data is just root link data expanded to (num_instances, 1, dim).
        """
        return (
            robot_data.root_link_pos_w.unsqueeze(1),
            robot_data.root_link_quat_w.unsqueeze(1),
        )

    def _get_body_com_velocities_b(
        self, robot_data: RigidObjectData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get body COM linear and angular velocities in body frame.

        For rigid objects, use the root COM velocity in body frame.
        """
        return (
            robot_data.root_com_lin_vel_b.unsqueeze(1),
            robot_data.root_com_ang_vel_b.unsqueeze(1),
        )

    ##
    # RigidObject-specific overrides for body properties
    # (Override to return simple views since rigid objects have only 1 body)
    ##
