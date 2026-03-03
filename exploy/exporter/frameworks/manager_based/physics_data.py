# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import copy
from abc import ABC, abstractmethod
from typing import Any

import torch

from exploy.exporter.core.tensor_proxy import TensorProxy


class PhysicsDataSource(ABC):
    """Base class for physics data sources (ArticulationDataSource, EntityDataSource, and RigidObjectDataSource).

    This class contains all the common logic for mimicking physics data interfaces while
    managing separate tensor data for ONNX export.
    """

    def __init__(self, robot: Any, robot_data: Any, math_utils: Any):
        """Initialize the base robot data source.

        Args:
            robot: The robot instance (Articulation or Entity)
            robot_data: The robot's data instance
            math_utils: The math utils module (isaaclab.utils.math or mjlab.utils.lab_api.math)
        """
        self._robot = robot
        self._data = robot_data
        self._math_utils = math_utils

        # Initialize constants
        self.GRAVITY_VEC_W = self._get_gravity_vec(robot_data).clone()
        self.FORWARD_VEC_B = self._get_forward_vec(robot_data).clone()

        # Initialize COM offset data (in body frame)
        coms_pos_b, coms_quat_b = self._get_com_poses(robot_data)
        self._coms_pos_b = coms_pos_b.clone()
        self._coms_quat_b = coms_quat_b.clone()

        # Initialize body link data (this is the base data - positions and orientations in world frame)
        self._root_body_id = 0
        body_link_pos_w, body_link_quat_w = self._get_body_link_poses(robot_data)
        self._body_link_pos_w = TensorProxy(body_link_pos_w.clone(), split_dim=1)
        self._body_link_quat_w = TensorProxy(body_link_quat_w.clone(), split_dim=1)

        # Initialize body COM velocities (in body frame - this is what IsaacLab stores)
        body_com_lin_vel_b, body_com_ang_vel_b = self._get_body_com_velocities_b(robot_data)
        self._body_com_lin_vel_b = TensorProxy(body_com_lin_vel_b.clone(), split_dim=1)
        self._body_com_ang_vel_b = TensorProxy(body_com_ang_vel_b.clone(), split_dim=1)

        # Initialize joint data
        self._joint_pos = robot_data.joint_pos.clone()
        self._joint_vel = robot_data.joint_vel.clone()
        self._joint_acc = robot_data.joint_acc.clone()

        # Clone all tensors from source class
        for k, v in vars(robot_data).items():
            # Skip if this attribute is already defined as a property in this class
            if hasattr(type(self), k) and isinstance(getattr(type(self), k), property):
                continue
            if v is None:
                setattr(self, k, v)
            elif isinstance(v, torch.Tensor):
                setattr(self, k, v.clone())
            elif isinstance(v, list):
                setattr(self, k, copy.copy(v))

    @abstractmethod
    def _get_gravity_vec(self, robot_data: Any) -> torch.Tensor:
        """Get gravity vector from robot data."""
        pass

    @abstractmethod
    def _get_forward_vec(self, robot_data: Any) -> torch.Tensor:
        """Get forward vector from robot data."""
        pass

    @abstractmethod
    def _get_com_poses(self, robot_data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Get COM positions and quaternions from robot data."""
        pass

    @abstractmethod
    def _get_body_link_poses(self, robot_data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Get body link positions and quaternions in world frame from robot data.

        Returns:
            Tuple of (positions, quaternions) where each has shape (num_instances, num_bodies, 3/4)
        """
        pass

    @abstractmethod
    def _get_body_com_velocities_b(self, robot_data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Get body COM linear and angular velocities in body frame from robot data.

        Returns:
            Tuple of (linear_velocities, angular_velocities) in body frame,
            where each has shape (num_instances, num_bodies, 3)
        """
        pass

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
    def root_pose_w(self) -> torch.Tensor:
        """Root pose [pos, quat] in simulation world frame. Shape is (num_instances, 7).

        For backward compatibility: Same as :attr:`root_link_pose_w`.
        """
        return self.root_link_pose_w

    @property
    def root_link_state_w(self) -> torch.Tensor:
        """Root link state [pos, quat, lin_vel, ang_vel] in simulation world frame. Shape is (num_instances, 13).

        This is the state of the root link frame (actor frame).
        """
        return torch.cat(
            [
                self.root_link_pos_w,
                self.root_link_quat_w,
                self.root_link_lin_vel_w,
                self.root_link_ang_vel_w,
            ],
            dim=-1,
        )

    @property
    def root_com_state_w(self) -> torch.Tensor:
        """Root COM state [pos, quat, lin_vel, ang_vel] in simulation world frame. Shape is (num_instances, 13).

        This is the state of the root center of mass frame.
        """
        return torch.cat(
            [
                self.root_com_pos_w,
                self.root_com_quat_w,
                self.root_com_lin_vel_w,
                self.root_com_ang_vel_w,
            ],
            dim=-1,
        )

    @property
    def root_state_w(self) -> torch.Tensor:
        """Root state [pos, quat, lin_vel, ang_vel] in simulation world frame. Shape is (num_instances, 13).

        For backward compatibility with IsaacLab:
        - Position and orientation are from the link frame
        - Velocities are from the COM frame
        """
        return torch.cat(
            [
                self.root_link_pos_w,
                self.root_link_quat_w,
                self.root_com_lin_vel_w,
                self.root_com_ang_vel_w,
            ],
            dim=-1,
        )

    @property
    def body_link_state_w(self) -> torch.Tensor:
        """State of all bodies' link frame [pos, quat, lin_vel, ang_vel] in world frame.

        This is the state of each body's link frame (actor frame).
        """
        return torch.cat(
            [
                self.body_link_pos_w,
                self.body_link_quat_w,
                self.body_link_lin_vel_w,
                self.body_link_ang_vel_w,
            ],
            dim=-1,
        )

    @property
    def body_com_state_w(self) -> torch.Tensor:
        """State of all bodies COM [pos, quat, lin_vel, ang_vel] in world frame.

        This is the state of each body's center of mass frame.
        """
        return torch.cat(
            [
                self.body_com_pos_w,
                self.body_com_quat_w,
                self.body_com_lin_vel_w,
                self.body_com_ang_vel_w,
            ],
            dim=-1,
        )

    @property
    def body_state_w(self) -> torch.Tensor:
        """State of all bodies [pos, quat, lin_vel, ang_vel] in simulation world frame.

        For backward compatibility with IsaacLab:
        - Position and orientation are from the link frame
        - Velocities are from the COM frame
        """
        return torch.cat(
            [
                self.body_link_pos_w,
                self.body_link_quat_w,
                self.body_com_lin_vel_w,
                self.body_com_ang_vel_w,
            ],
            dim=-1,
        )

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Projection of gravity direction on base frame. Shape is (num_instances, 3)."""
        return self._math_utils.quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self) -> torch.Tensor:
        """Yaw heading of the base frame (in radians). Shape is (num_instances,)."""
        forward_w = self._math_utils.quat_apply(self.root_link_quat_w, self.FORWARD_VEC_B)
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
    # Backward compatibility properties (IsaacLab convention)
    # Position/orientation from link frame, velocities from COM frame
    ##

    @property
    def body_pose_w(self) -> torch.Tensor:
        """Poses [pos, quat] of all bodies in world frame. Shape is (num_instances, num_bodies, 7).

        For backward compatibility: Same as :attr:`body_link_pose_w`.
        """
        return self.body_link_pose_w

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in world frame. Shape is (num_instances, 3).

        For backward compatibility: Same as :attr:`root_link_pos_w`.
        """
        return self.root_link_pos_w

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in world frame. Shape is (num_instances, 4).

        For backward compatibility: Same as :attr:`root_link_quat_w`.
        """
        return self.root_link_quat_w

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity in world frame. Shape is (num_instances, 6).

        For backward compatibility: Same as :attr:`root_com_vel_w`.
        """
        return self.root_com_vel_w

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in world frame. Shape is (num_instances, 3).

        For backward compatibility: Same as :attr:`root_com_lin_vel_w`.
        """
        return self.root_com_lin_vel_w

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in world frame. Shape is (num_instances, 3).

        For backward compatibility: Same as :attr:`root_com_ang_vel_w`.
        """
        return self.root_com_ang_vel_w

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in base frame. Shape is (num_instances, 3).

        For backward compatibility: Same as :attr:`root_com_lin_vel_b`.
        """
        return self.root_com_lin_vel_b

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in base frame. Shape is (num_instances, 3).

        For backward compatibility: Same as :attr:`root_com_ang_vel_b`.
        """
        return self.root_com_ang_vel_b

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
        # This is because v_com = v_link + omega x r_com
        return self.root_com_lin_vel_w - torch.linalg.cross(
            self.root_com_ang_vel_w,
            self._math_utils.quat_apply(self.root_link_quat_w, self.com_pos_b[:, 0, :]),
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
        return self._math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base frame. Shape is (num_instances, 3)."""
        return self._math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    ##
    # Root COM properties (center of mass frame)
    ##

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root COM position in world frame. Shape is (num_instances, 3).

        Computed from link position + COM offset.
        """
        pos, _ = self._math_utils.combine_frame_transforms(
            self.root_link_pos_w,
            self.root_link_quat_w,
            self.com_pos_b[:, 0, :],
            self.com_quat_b[:, 0, :],
        )
        return pos

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root COM orientation (w, x, y, z) in world frame. Shape is (num_instances, 4).

        Computed from link orientation + COM orientation offset.
        """
        _, quat = self._math_utils.combine_frame_transforms(
            self.root_link_pos_w,
            self.root_link_quat_w,
            self.com_pos_b[:, 0, :],
            self.com_quat_b[:, 0, :],
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
        return self._math_utils.quat_apply(self.root_link_quat_w, self.root_com_lin_vel_b)

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root COM angular velocity in world frame. Shape is (num_instances, 3).

        Computed by transforming from body frame to world frame.
        """
        return self._math_utils.quat_apply(self.root_link_quat_w, self.root_com_ang_vel_b)

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
    # Body backward compatibility properties (IsaacLab convention)
    # Position/orientation from link frame, velocities from COM frame
    ##

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in world frame.

        For backward compatibility: Same as :attr:`body_link_pos_w`.
        """
        return self.body_link_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Orientations (w, x, y, z) of all bodies in world frame.

        For backward compatibility: Same as :attr:`body_link_quat_w`.
        """
        return self.body_link_quat_w

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Velocities of all bodies in world frame.

        For backward compatibility: Same as :attr:`body_com_vel_w`.
        """
        return self.body_com_vel_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Linear velocities of all bodies in world frame.

        For backward compatibility: Same as :attr:`body_com_lin_vel_w`.
        """
        return self.body_com_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Angular velocities of all bodies in world frame.

        For backward compatibility: Same as :attr:`body_com_ang_vel_w`.
        """
        return self.body_com_ang_vel_w

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
        # This is because v_com = v_link + omega x r_com
        return self.body_com_lin_vel_w - torch.linalg.cross(
            self.body_com_ang_vel_w,
            self._math_utils.quat_apply(self._body_link_quat_w.to_tensor(), self.com_pos_b),
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
        """Positions of all body COMs in world frame.

        Computed from link positions + COM offsets.
        """
        pos, _ = self._math_utils.combine_frame_transforms(
            self._body_link_pos_w.to_tensor(),
            self._body_link_quat_w.to_tensor(),
            self.com_pos_b,
            self.com_quat_b,
        )
        return pos

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Orientations (w, x, y, z) of all body COMs in world frame.

        Computed from link orientations + COM orientation offsets.
        """
        _, quat = self._math_utils.combine_frame_transforms(
            self._body_link_pos_w.to_tensor(),
            self._body_link_quat_w.to_tensor(),
            self.com_pos_b,
            self.com_quat_b,
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
        # Transform each body's velocity from its body frame to world frame
        return self._math_utils.quat_apply(
            self._body_link_quat_w.to_tensor(), self._body_com_lin_vel_b.to_tensor()
        )

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Angular velocities of all body COMs in world frame.

        Computed by transforming from body frame to world frame.
        """
        # Transform each body's velocity from its body frame to world frame
        return self._math_utils.quat_apply(
            self._body_link_quat_w.to_tensor(), self._body_com_ang_vel_b.to_tensor()
        )

    @property
    def body_com_lin_vel_b(self) -> torch.Tensor:
        """Linear velocities of all body COMs in body frame.

        This is the base stored data - linear velocities of all bodies' COM in body frame.
        """
        return self._body_com_lin_vel_b

    @property
    def body_com_ang_vel_b(self) -> torch.Tensor:
        """Angular velocities of all body COMs in body frame.

        This is the base stored data - angular velocities of all bodies' COM in body frame.
        """
        return self._body_com_ang_vel_b

    @property
    def body_com_acc_w(self) -> torch.Tensor:
        """Accelerations of all body COMs in world frame.

        This property returns the stored acceleration data if available (IsaacLab specific).
        Shape is (num_instances, num_bodies, 6).
        """
        if hasattr(self, "_body_acc_w"):
            return self._body_acc_w
        raise AttributeError(f"{self.__class__.__name__} does not support body_com_acc_w")

    @property
    def body_com_lin_acc_w(self) -> torch.Tensor:
        """Linear accelerations of all body COMs in world frame. Shape is (num_instances, num_bodies, 3)."""
        return self.body_com_acc_w[..., :3]

    @property
    def body_com_ang_acc_w(self) -> torch.Tensor:
        """Angular accelerations of all body COMs in world frame. Shape is (num_instances, num_bodies, 3)."""
        return self.body_com_acc_w[..., 3:6]

    @property
    def body_acc_w(self) -> torch.Tensor:
        """Accelerations of all bodies in world frame.

        For backward compatibility: Same as :attr:`body_com_acc_w`.
        """
        return self.body_com_acc_w

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Linear accelerations of all bodies in world frame.

        For backward compatibility: Same as :attr:`body_com_lin_acc_w`.
        """
        return self.body_com_lin_acc_w

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Angular accelerations of all bodies in world frame.

        For backward compatibility: Same as :attr:`body_com_ang_acc_w`.
        """
        return self.body_com_ang_acc_w

    @property
    def com_pos_b(self) -> torch.Tensor:
        """COM positions in body frames. Shape is (num_instances, num_bodies, 3)."""
        return self._coms_pos_b

    @property
    def com_quat_b(self) -> torch.Tensor:
        """COM orientations in body frames. Shape is (num_instances, num_bodies, 4)."""
        return self._coms_quat_b

    @property
    def body_com_pose_b(self) -> torch.Tensor:
        """Center of mass pose [pos, quat] of all bodies in their respective body's link frames.
        Shape is (num_instances, num_bodies, 7).

        This is a concatenation of com_pos_b and com_quat_b.
        """
        return torch.cat([self.com_pos_b, self.com_quat_b], dim=-1)

    @property
    def body_com_pos_b(self) -> torch.Tensor:
        """Center of mass position of all bodies in their respective link frames.
        Shape is (num_instances, num_bodies, 3).

        For backward compatibility: Same as :attr:`com_pos_b`.
        """
        return self.com_pos_b

    @property
    def body_com_quat_b(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        For backward compatibility: Same as :attr:`com_quat_b`.
        """
        return self.com_quat_b

    ##
    # Deprecated aliases for backward compatibility
    ##

    @property
    def joint_limits(self) -> torch.Tensor:
        """Deprecated alias for :attr:`joint_pos_limits`."""
        return self.joint_pos_limits if hasattr(self, "joint_pos_limits") else None

    @property
    def default_joint_limits(self) -> torch.Tensor:
        """Deprecated alias for :attr:`default_joint_pos_limits`."""
        return self.default_joint_pos_limits if hasattr(self, "default_joint_pos_limits") else None

    @property
    def joint_velocity_limits(self) -> torch.Tensor:
        """Deprecated alias for :attr:`joint_vel_limits`."""
        return self.joint_vel_limits if hasattr(self, "joint_vel_limits") else None

    @property
    def joint_friction(self) -> torch.Tensor:
        """Deprecated alias for :attr:`joint_friction_coeff`."""
        return self.joint_friction_coeff if hasattr(self, "joint_friction_coeff") else None

    @property
    def default_joint_friction(self) -> torch.Tensor:
        """Deprecated alias for :attr:`default_joint_friction_coeff`."""
        return (
            self.default_joint_friction_coeff
            if hasattr(self, "default_joint_friction_coeff")
            else None
        )

    @property
    def fixed_tendon_limit(self) -> torch.Tensor:
        """Deprecated alias for :attr:`fixed_tendon_pos_limits`."""
        return self.fixed_tendon_pos_limits if hasattr(self, "fixed_tendon_pos_limits") else None

    @property
    def default_fixed_tendon_limit(self) -> torch.Tensor:
        """Deprecated alias for :attr:`default_fixed_tendon_pos_limits`."""
        return (
            self.default_fixed_tendon_pos_limits
            if hasattr(self, "default_fixed_tendon_pos_limits")
            else None
        )
