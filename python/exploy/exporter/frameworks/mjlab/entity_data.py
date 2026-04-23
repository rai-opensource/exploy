# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import copy

import mjlab.utils.lab_api.math as math_utils
import torch
from mjlab.entity import Entity, EntityData

from exploy.exporter.core.tensor_proxy import TensorProxy


class EntityDataSource:
    """Mimic the interface of an ``EntityData``, but manage its own ONNX-traceable tensor data.

    Replaces ``entity._data`` during ONNX export so that observation functions
    access managed tensors instead of live simulation buffers.  These managed
    tensors become ONNX graph inputs because they are the same Python objects
    returned by ``entity.data.*`` during tracing.

    After export, the original ``entity._data`` is restored by
    ``MjlabExportableEnvironment.cleanup()``.
    """

    def __init__(self, entity: Entity):
        entity_data: EntityData = entity.data

        self._robot = entity
        self._data = entity_data

        # Initialize constants.
        self.GRAVITY_VEC_W = entity_data.gravity_vec_w.clone()
        self.FORWARD_VEC_B = entity_data.forward_vec_b.clone()

        # Initialize COM offset data (in body frame).
        body_ipos = entity_data.model.body_ipos[:, entity_data.indexing.body_ids]
        body_iquat = entity_data.model.body_iquat[:, entity_data.indexing.body_ids]
        self._coms_pos_b = body_ipos.clone()
        self._coms_quat_b = body_iquat.clone()

        # Initialize body link data (positions and orientations in world frame).
        self._root_body_id = 0
        self._body_link_pos_w = TensorProxy(entity_data.body_link_pos_w.clone(), split_dim=1)
        self._body_link_quat_w = TensorProxy(entity_data.body_link_quat_w.clone(), split_dim=1)

        # Initialize body COM velocities in body frame.
        # mjlab stores COM velocities in world frame, so transform them and keep
        # clones of the computed values as ONNX-traceable managed tensors.
        body_com_lin_vel_b = math_utils.quat_apply_inverse(
            entity_data.body_link_quat_w, entity_data.body_com_lin_vel_w
        )
        body_com_ang_vel_b = math_utils.quat_apply_inverse(
            entity_data.body_link_quat_w, entity_data.body_com_ang_vel_w
        )
        self._body_com_lin_vel_b = TensorProxy(body_com_lin_vel_b.clone(), split_dim=1)
        self._body_com_ang_vel_b = TensorProxy(body_com_ang_vel_b.clone(), split_dim=1)

        # Initialize joint data.
        self._joint_pos = entity_data.joint_pos.clone()
        self._joint_vel = entity_data.joint_vel.clone()
        self._joint_acc = entity_data.joint_acc.clone()

        # mjlab-specific: actuator force.
        self._actuator_force = entity_data.actuator_force.clone()

        # Clone all remaining tensors from source class.
        for k, v in vars(entity_data).items():
            # Skip if this attribute is already defined as a property in this class.
            if hasattr(type(self), k) and isinstance(getattr(type(self), k), property):
                continue
            if v is None:
                setattr(self, k, v)
            elif isinstance(v, torch.Tensor):
                setattr(self, k, v.clone())
            elif isinstance(v, list):
                setattr(self, k, copy.copy(v))

    def update(self, *args, **kwargs) -> None:
        """No-op update to mimic simulator data-source interfaces."""
        return

    def reset(self, *args, **kwargs) -> None:
        """No-op reset to mimic simulator data-source interfaces."""
        return

    ##
    # Root state properties (combined pose + velocity)
    ##

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        """Root link pose [pos, quat] in simulation world frame. Shape is (num_instances, 7)."""
        return torch.cat([self.root_link_pos_w, self.root_link_quat_w], dim=-1)

    @property
    def root_com_pose_w(self) -> torch.Tensor:
        """Root COM pose [pos, quat] in simulation world frame. Shape is (num_instances, 7)."""
        return torch.cat([self.root_com_pos_w, self.root_com_quat_w], dim=-1)

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Projection of gravity direction on base frame. Shape is (num_instances, 3)."""
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self) -> torch.Tensor:
        """Yaw heading of the base frame (in radians). Shape is (num_instances,)."""
        forward_w = math_utils.quat_apply(self.root_link_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    ##
    # Joint properties
    ##

    @property
    def joint_pos(self) -> torch.Tensor:
        """Joint positions. Shape is (num_instances, num_joints)."""
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        """Joint velocities. Shape is (num_instances, num_joints)."""
        return self._joint_vel

    @property
    def joint_acc(self) -> torch.Tensor:
        """Joint accelerations. Shape is (num_instances, num_joints)."""
        return self._joint_acc

    ##
    # Root link frame properties (actor frame)
    ##

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position in world frame. Shape is (num_instances, 3).

        This is the base stored data - position of the root body's link frame.
        """
        return self._body_link_pos_w[:, self._root_body_id]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation (w, x, y, z) in world frame. Shape is (num_instances, 4).

        This is the base stored data - orientation of the root body's link frame.
        """
        return self._body_link_quat_w[:, self._root_body_id]

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root link velocity in world frame. Shape is (num_instances, 6)."""
        return torch.cat([self.root_link_lin_vel_w, self.root_link_ang_vel_w], dim=-1)

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root link linear velocity in world frame. Shape is (num_instances, 3).

        Computed from COM velocity by subtracting the velocity offset due to COM offset.
        """
        # v_link = v_com - omega x r_com (where r_com is vector from link to COM)
        return self.root_com_lin_vel_w - torch.linalg.cross(
            self.root_com_ang_vel_w,
            math_utils.quat_apply(self.root_link_quat_w, self._coms_pos_b[:, 0, :]),
            dim=-1,
        )

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity in world frame. Shape is (num_instances, 3).

        Angular velocity is the same for link and COM frames.
        """
        return self.root_com_ang_vel_w

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in base frame. Shape is (num_instances, 3)."""
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base frame. Shape is (num_instances, 3)."""
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    ##
    # Root COM properties (center of mass frame)
    ##

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root COM position in world frame. Shape is (num_instances, 3)."""
        pos, _ = math_utils.combine_frame_transforms(
            self.root_link_pos_w,
            self.root_link_quat_w,
            self._coms_pos_b[:, 0, :],
            self._coms_quat_b[:, 0, :],
        )
        return pos

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root COM orientation (w, x, y, z) in world frame. Shape is (num_instances, 4)."""
        _, quat = math_utils.combine_frame_transforms(
            self.root_link_pos_w,
            self.root_link_quat_w,
            self._coms_pos_b[:, 0, :],
            self._coms_quat_b[:, 0, :],
        )
        return quat

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root COM velocity in world frame. Shape is (num_instances, 6)."""
        return torch.cat([self.root_com_lin_vel_w, self.root_com_ang_vel_w], dim=-1)

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root COM linear velocity in world frame. Shape is (num_instances, 3).

        Computed by transforming from body frame to world frame.
        """
        return math_utils.quat_apply(
            self.root_link_quat_w, self._body_com_lin_vel_b[:, self._root_body_id]
        )

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root COM angular velocity in world frame. Shape is (num_instances, 3).

        Computed by transforming from body frame to world frame.
        """
        return math_utils.quat_apply(
            self.root_link_quat_w, self._body_com_ang_vel_b[:, self._root_body_id]
        )

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root COM linear velocity in base frame. Shape is (num_instances, 3).

        This is the base stored data - linear velocity of the root body's COM in body frame.
        """
        return self._body_com_lin_vel_b[:, self._root_body_id]

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root COM angular velocity in base frame. Shape is (num_instances, 3).

        This is the base stored data - angular velocity of the root body's COM in body frame.
        """
        return self._body_com_ang_vel_b[:, self._root_body_id]

    ##
    # Body link properties (actor frame)
    ##

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        """Poses [pos, quat] of all body links in world frame. Shape is (num_instances, num_bodies, 7)."""
        return torch.cat([self.body_link_pos_w, self.body_link_quat_w], dim=-1)

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """Positions of all body links in world frame.

        This is the base stored data - positions of all bodies' link frames.
        """
        return self._body_link_pos_w

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """Orientations (w, x, y, z) of all body links in world frame.

        This is the base stored data - orientations of all bodies' link frames.
        """
        return self._body_link_quat_w

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """Velocities of all body links in world frame."""
        return torch.cat([self.body_link_lin_vel_w, self.body_link_ang_vel_w], dim=-1)

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Linear velocities of all body links in world frame.

        Computed from COM velocity by subtracting the velocity offset due to COM offset.
        """
        # v_link = v_com - omega x r_com (where r_com is vector from link to COM)
        return self.body_com_lin_vel_w - torch.linalg.cross(
            self.body_com_ang_vel_w,
            math_utils.quat_apply(self._body_link_quat_w.to_tensor(), self._coms_pos_b),
            dim=-1,
        )

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Angular velocities of all body links in world frame.

        Angular velocity is the same for link and COM frames.
        """
        return self.body_com_ang_vel_w

    ##
    # Body COM properties (center of mass frame)
    ##

    @property
    def body_com_pose_w(self) -> torch.Tensor:
        """Poses [pos, quat] of all body COMs in world frame. Shape is (num_instances, num_bodies, 7)."""
        return torch.cat([self.body_com_pos_w, self.body_com_quat_w], dim=-1)

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """Positions of all body COMs in world frame."""
        pos, _ = math_utils.combine_frame_transforms(
            self._body_link_pos_w.to_tensor(),
            self._body_link_quat_w.to_tensor(),
            self._coms_pos_b,
            self._coms_quat_b,
        )
        return pos

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Orientations (w, x, y, z) of all body COMs in world frame."""
        _, quat = math_utils.combine_frame_transforms(
            self._body_link_pos_w.to_tensor(),
            self._body_link_quat_w.to_tensor(),
            self._coms_pos_b,
            self._coms_quat_b,
        )
        return quat

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """Velocities of all body COMs in world frame."""
        return torch.cat([self.body_com_lin_vel_w, self.body_com_ang_vel_w], dim=-1)

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Linear velocities of all body COMs in world frame.

        Computed by transforming from body frame to world frame.
        """
        return math_utils.quat_apply(
            self._body_link_quat_w.to_tensor(), self._body_com_lin_vel_b.to_tensor()
        )

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Angular velocities of all body COMs in world frame.

        Computed by transforming from body frame to world frame.
        """
        return math_utils.quat_apply(
            self._body_link_quat_w.to_tensor(), self._body_com_ang_vel_b.to_tensor()
        )

    ##
    # mjlab-specific properties
    ##

    @property
    def actuator_force(self) -> torch.Tensor:
        return self._actuator_force

    @property
    def site_pose_w(self) -> torch.Tensor:
        return self._data.site_pose_w

    @property
    def site_pos_w(self) -> torch.Tensor:
        return self._data.site_pos_w

    @property
    def site_quat_w(self) -> torch.Tensor:
        return self._data.site_quat_w

    @property
    def site_vel_w(self) -> torch.Tensor:
        return self._data.site_vel_w

    @property
    def site_lin_vel_w(self) -> torch.Tensor:
        return self._data.site_lin_vel_w

    @property
    def site_ang_vel_w(self) -> torch.Tensor:
        return self._data.site_ang_vel_w

    @property
    def geom_pose_w(self) -> torch.Tensor:
        return self._data.geom_pose_w

    @property
    def geom_pos_w(self) -> torch.Tensor:
        return self._data.geom_pos_w

    @property
    def geom_quat_w(self) -> torch.Tensor:
        return self._data.geom_quat_w

    @property
    def geom_vel_w(self) -> torch.Tensor:
        return self._data.geom_vel_w

    @property
    def geom_lin_vel_w(self) -> torch.Tensor:
        return self._data.geom_lin_vel_w

    @property
    def geom_ang_vel_w(self) -> torch.Tensor:
        return self._data.geom_ang_vel_w

    @property
    def tendon_len(self) -> torch.Tensor:
        return self._data.tendon_len

    @property
    def tendon_vel(self) -> torch.Tensor:
        return self._data.tendon_vel

    @property
    def joint_torques(self) -> torch.Tensor:
        return self._data.joint_torques

    @property
    def joint_pos_biased(self) -> torch.Tensor:
        return self._data.joint_pos_biased

    @property
    def generalized_force(self) -> torch.Tensor:
        return self._data.generalized_force

    @property
    def body_external_force(self) -> torch.Tensor:
        return self._data.body_external_force

    @property
    def body_external_torque(self) -> torch.Tensor:
        return self._data.body_external_torque

    @property
    def body_external_wrench(self) -> torch.Tensor:
        return self._data.body_external_wrench
