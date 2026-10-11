# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Export data sources mimicking the rigid-body data of Isaac Lab 3 assets."""

from __future__ import annotations

from typing import Any

import isaaclab.utils.math as math_utils
import torch

from exploy.exporter.core.tensor_proxy import TensorProxy, to_tensor
from exploy.exporter.frameworks.isaaclab3.proxy_array import (
    TorchProxyArray,
    as_torch,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def _snapshot(value: Any) -> Any:
    """Freeze an Isaac Lab data value: tensors and ``ProxyArray`` objects are cloned into a
    ``TorchProxyArray``, everything else (names, counts, ...) is returned unchanged."""
    if isinstance(value, torch.Tensor):
        return TorchProxyArray(value.clone())
    if hasattr(value, "torch") and hasattr(value, "warp"):
        return TorchProxyArray(value.torch.clone())
    return value


class BodyDataSource:
    """Mimic the rigid-body interface of Isaac Lab 3 asset data, but manage its own tensor data.

    The data source replaces an asset's ``_data`` during export, so that observation terms read
    exploy-managed tensors instead of the simulation buffers: the managed tensors become the inputs of
    the ONNX graph. The state of the bodies is held by a small set of stored tensors (the world-frame
    link poses, the world-frame center-of-mass velocities, and the root velocities in the base frame),
    from which all other quantities are derived with Isaac Lab's math utilities.

    The body orientations are stored in the (w, x, y, z) convention of exploy's ONNX interface (shared
    with the other frameworks and exploy's C++ controller), and converted to the (x, y, z, w) convention
    of Isaac Lab 3 when read.

    Properties that are not state (e.g. default states, limits, masses) are snapshots of the asset data
    when the data source was created, returned on first access.
    """

    def __init__(self, data: Any):
        self._source_data = data
        self._snapshots: dict[str, Any] = {}

        # Pose of the center of mass of each body wrt the link frame of the body.
        self._body_com_pos_b = as_torch(data.body_com_pos_b).clone()
        self._body_com_quat_b = as_torch(data.body_com_quat_b).clone()

        # Constants. Newton binds the gravity vector of each environment in [m/s^2]: keep its direction.
        gravity = as_torch(data.GRAVITY_VEC_W).clone()
        self._gravity_dir_w = gravity / torch.linalg.norm(gravity, dim=-1, keepdim=True).clamp_min(
            1.0e-9
        )
        self._forward_vec_b = as_torch(data.FORWARD_VEC_B).clone()

        # Stored state. Body tensors are split into one tensor per body, so that each body is an
        # independent input of the ONNX graph.
        self._root_body_id = 0
        self._body_link_pos_w = TensorProxy(as_torch(data.body_link_pos_w).clone(), split_dim=1)
        self._body_link_quat_w_wxyz = TensorProxy(
            xyzw_to_wxyz(as_torch(data.body_link_quat_w)).clone(), split_dim=1
        )
        self._body_com_lin_vel_w = TensorProxy(
            as_torch(data.body_com_lin_vel_w).clone(), split_dim=1
        )
        self._body_com_ang_vel_w = TensorProxy(
            as_torch(data.body_com_ang_vel_w).clone(), split_dim=1
        )
        self._body_com_acc_w = as_torch(data.body_com_acc_w).clone()

        # Like the 2.x data source, the root velocities are stored separately in the base frame.
        self._root_com_lin_vel_b = as_torch(data.root_com_lin_vel_b).clone()
        self._root_com_ang_vel_b = as_torch(data.root_com_ang_vel_b).clone()

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes not defined on the data source. Private attributes are forwarded to
        # the asset data, as they are only used by the asset's own implementation.
        if name.startswith("_"):
            if name in ("_source_data", "_snapshots"):
                raise AttributeError(name)
            return getattr(self._source_data, name)
        if name not in self._snapshots:
            self._snapshots[name] = _snapshot(getattr(self._source_data, name))
        return self._snapshots[name]

    def update(self, *args, **kwargs):
        """Empty ``update`` call to implement the interface of the asset data."""

    def reset(self, *args, **kwargs):
        """Empty ``reset`` call to implement the interface of the asset data."""

    ##
    # Stored quantities, used as ONNX inputs.
    ##

    def body_link_quat_w_wxyz(self) -> TensorProxy:
        """Orientation (w, x, y, z) of all bodies' link frames. Shape is (num_instances, num_bodies, 4)."""
        return self._body_link_quat_w_wxyz

    ##
    # Root link.
    ##

    @property
    def root_link_pos_w(self) -> TorchProxyArray:
        return TorchProxyArray(self._body_link_pos_w[:, self._root_body_id])

    @property
    def root_link_quat_w(self) -> TorchProxyArray:
        return TorchProxyArray(wxyz_to_xyzw(self._body_link_quat_w_wxyz[:, self._root_body_id]))

    @property
    def root_link_pose_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            torch.cat([self.root_link_pos_w.torch, self.root_link_quat_w.torch], dim=-1)
        )

    @property
    def root_link_lin_vel_b(self) -> TorchProxyArray:
        return TorchProxyArray(
            self._root_com_lin_vel_b
            + torch.linalg.cross(self._root_com_ang_vel_b, -self._body_com_pos_b[:, 0], dim=-1)
        )

    @property
    def root_link_ang_vel_b(self) -> TorchProxyArray:
        return TorchProxyArray(self._root_com_ang_vel_b)

    @property
    def root_link_lin_vel_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            math_utils.quat_apply(self.root_link_quat_w.torch, self.root_link_lin_vel_b.torch)
        )

    @property
    def root_link_ang_vel_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            math_utils.quat_apply(self.root_link_quat_w.torch, self.root_link_ang_vel_b.torch)
        )

    @property
    def root_link_vel_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            torch.cat([self.root_link_lin_vel_w.torch, self.root_link_ang_vel_w.torch], dim=-1)
        )

    @property
    def root_link_state_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            torch.cat([self.root_link_pose_w.torch, self.root_link_vel_w.torch], dim=-1)
        )

    ##
    # Root center of mass.
    ##

    def _root_com_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return math_utils.combine_frame_transforms(
            self.root_link_pos_w.torch,
            self.root_link_quat_w.torch,
            self._body_com_pos_b[:, 0],
            self._body_com_quat_b[:, 0],
        )

    @property
    def root_com_pos_w(self) -> TorchProxyArray:
        return TorchProxyArray(self._root_com_pose()[0])

    @property
    def root_com_quat_w(self) -> TorchProxyArray:
        return TorchProxyArray(self._root_com_pose()[1])

    @property
    def root_com_pose_w(self) -> TorchProxyArray:
        return TorchProxyArray(torch.cat(self._root_com_pose(), dim=-1))

    @property
    def root_com_lin_vel_b(self) -> TorchProxyArray:
        return TorchProxyArray(self._root_com_lin_vel_b)

    @property
    def root_com_ang_vel_b(self) -> TorchProxyArray:
        return TorchProxyArray(self._root_com_ang_vel_b)

    @property
    def root_com_lin_vel_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            math_utils.quat_apply(self.root_link_quat_w.torch, self._root_com_lin_vel_b)
        )

    @property
    def root_com_ang_vel_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            math_utils.quat_apply(self.root_link_quat_w.torch, self._root_com_ang_vel_b)
        )

    @property
    def root_com_vel_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            torch.cat([self.root_com_lin_vel_w.torch, self.root_com_ang_vel_w.torch], dim=-1)
        )

    @property
    def root_com_state_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            torch.cat([self.root_com_pose_w.torch, self.root_com_vel_w.torch], dim=-1)
        )

    @property
    def root_state_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            torch.cat([self.root_link_pose_w.torch, self.root_com_vel_w.torch], dim=-1)
        )

    ##
    # Root shorthands.
    ##

    @property
    def root_pose_w(self) -> TorchProxyArray:
        return self.root_link_pose_w

    @property
    def root_pos_w(self) -> TorchProxyArray:
        return self.root_link_pos_w

    @property
    def root_quat_w(self) -> TorchProxyArray:
        return self.root_link_quat_w

    @property
    def root_vel_w(self) -> TorchProxyArray:
        return self.root_com_vel_w

    @property
    def root_lin_vel_w(self) -> TorchProxyArray:
        return self.root_com_lin_vel_w

    @property
    def root_ang_vel_w(self) -> TorchProxyArray:
        return self.root_com_ang_vel_w

    @property
    def root_lin_vel_b(self) -> TorchProxyArray:
        return self.root_com_lin_vel_b

    @property
    def root_ang_vel_b(self) -> TorchProxyArray:
        return self.root_com_ang_vel_b

    ##
    # Derived root quantities.
    ##

    @property
    def projected_gravity_b(self) -> TorchProxyArray:
        return TorchProxyArray(
            math_utils.quat_apply_inverse(self.root_link_quat_w.torch, self._gravity_dir_w)
        )

    @property
    def heading_w(self) -> TorchProxyArray:
        forward_w = math_utils.quat_apply(self.root_link_quat_w.torch, self._forward_vec_b)
        return TorchProxyArray(torch.atan2(forward_w[:, 1], forward_w[:, 0]))

    ##
    # Body links.
    ##

    @property
    def body_link_pos_w(self) -> TorchProxyArray:
        return TorchProxyArray(self._body_link_pos_w)

    @property
    def body_link_quat_w(self) -> TorchProxyArray:
        # Convert each body's quaternion separately, so that reading a body only depends on its own input.
        return TorchProxyArray(
            TensorProxy.from_tensors(
                [wxyz_to_xyzw(quat) for quat in self._body_link_quat_w_wxyz.tensors], split_dim=1
            )
        )

    @property
    def body_link_pose_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            torch.cat(
                [to_tensor(self._body_link_pos_w), to_tensor(self.body_link_quat_w.torch)], dim=-1
            )
        )

    @property
    def body_link_lin_vel_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            to_tensor(self._body_com_lin_vel_w)
            + torch.linalg.cross(
                to_tensor(self._body_com_ang_vel_w),
                math_utils.quat_apply(
                    to_tensor(self.body_link_quat_w.torch), -self._body_com_pos_b
                ),
                dim=-1,
            )
        )

    @property
    def body_link_ang_vel_w(self) -> TorchProxyArray:
        return TorchProxyArray(self._body_com_ang_vel_w)

    @property
    def body_link_vel_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            torch.cat([self.body_link_lin_vel_w.torch, to_tensor(self._body_com_ang_vel_w)], dim=-1)
        )

    @property
    def body_link_state_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            torch.cat([self.body_link_pose_w.torch, self.body_link_vel_w.torch], dim=-1)
        )

    ##
    # Body centers of mass.
    ##

    def _body_com_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return math_utils.combine_frame_transforms(
            to_tensor(self._body_link_pos_w),
            to_tensor(self.body_link_quat_w.torch),
            self._body_com_pos_b,
            self._body_com_quat_b,
        )

    @property
    def body_com_pos_w(self) -> TorchProxyArray:
        return TorchProxyArray(self._body_com_pose()[0])

    @property
    def body_com_quat_w(self) -> TorchProxyArray:
        return TorchProxyArray(self._body_com_pose()[1])

    @property
    def body_com_pose_w(self) -> TorchProxyArray:
        return TorchProxyArray(torch.cat(self._body_com_pose(), dim=-1))

    @property
    def body_com_lin_vel_w(self) -> TorchProxyArray:
        return TorchProxyArray(self._body_com_lin_vel_w)

    @property
    def body_com_ang_vel_w(self) -> TorchProxyArray:
        return TorchProxyArray(self._body_com_ang_vel_w)

    @property
    def body_com_vel_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            torch.cat(
                [to_tensor(self._body_com_lin_vel_w), to_tensor(self._body_com_ang_vel_w)], dim=-1
            )
        )

    @property
    def body_com_state_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            torch.cat([self.body_com_pose_w.torch, self.body_com_vel_w.torch], dim=-1)
        )

    @property
    def body_state_w(self) -> TorchProxyArray:
        return TorchProxyArray(
            torch.cat([self.body_link_pose_w.torch, self.body_com_vel_w.torch], dim=-1)
        )

    @property
    def body_com_acc_w(self) -> TorchProxyArray:
        return TorchProxyArray(self._body_com_acc_w)

    @property
    def body_com_lin_acc_w(self) -> TorchProxyArray:
        return TorchProxyArray(self._body_com_acc_w[..., 0:3])

    @property
    def body_com_ang_acc_w(self) -> TorchProxyArray:
        return TorchProxyArray(self._body_com_acc_w[..., 3:6])

    @property
    def body_com_pos_b(self) -> TorchProxyArray:
        return TorchProxyArray(self._body_com_pos_b)

    @property
    def body_com_quat_b(self) -> TorchProxyArray:
        return TorchProxyArray(self._body_com_quat_b)

    @property
    def body_com_pose_b(self) -> TorchProxyArray:
        return TorchProxyArray(torch.cat([self._body_com_pos_b, self._body_com_quat_b], dim=-1))

    ##
    # Body shorthands.
    ##

    @property
    def body_pose_w(self) -> TorchProxyArray:
        return self.body_link_pose_w

    @property
    def body_pos_w(self) -> TorchProxyArray:
        return self.body_link_pos_w

    @property
    def body_quat_w(self) -> TorchProxyArray:
        return self.body_link_quat_w

    @property
    def body_vel_w(self) -> TorchProxyArray:
        return self.body_com_vel_w

    @property
    def body_lin_vel_w(self) -> TorchProxyArray:
        return self.body_com_lin_vel_w

    @property
    def body_ang_vel_w(self) -> TorchProxyArray:
        return self.body_com_ang_vel_w

    @property
    def body_acc_w(self) -> TorchProxyArray:
        return self.body_com_acc_w

    @property
    def body_lin_acc_w(self) -> TorchProxyArray:
        return self.body_com_lin_acc_w

    @property
    def body_ang_acc_w(self) -> TorchProxyArray:
        return self.body_com_ang_acc_w

    @property
    def com_pos_b(self) -> TorchProxyArray:
        return self.body_com_pos_b

    @property
    def com_quat_b(self) -> TorchProxyArray:
        return self.body_com_quat_b


class RigidObjectDataSource(BodyDataSource):
    """Mimic the interface of an Isaac Lab 3 ``RigidObjectData``, but manage its own tensor data."""


def rigid_object_data_to_dict(
    object_name: str,
    source: RigidObjectDataSource,
    env_id: int,
) -> dict[str, torch.Tensor]:
    """Read the stored root state of a rigid object data source. The quaternion is (w, x, y, z)."""
    assert isinstance(source, RigidObjectDataSource)
    return {
        f"body.{object_name}.pos": source.root_pos_w.torch[env_id],
        f"body.{object_name}.quat": source.body_link_quat_w_wxyz()[env_id, source._root_body_id],
        f"body.{object_name}.lin_vel": source.root_lin_vel_b.torch[env_id],
        f"body.{object_name}.ang_vel": source.root_ang_vel_b.torch[env_id],
    }


def dict_to_rigid_object_data(
    data: dict[str, torch.Tensor],
    object_name: str,
    target: RigidObjectDataSource,
    env_id: int,
) -> None:
    """Write the stored root state of a rigid object data source. The quaternion is (w, x, y, z)."""
    assert isinstance(target, RigidObjectDataSource)
    target._body_link_pos_w[env_id, target._root_body_id] = data[f"body.{object_name}.pos"]
    target._body_link_quat_w_wxyz[env_id, target._root_body_id] = data[f"body.{object_name}.quat"]
    target._root_com_lin_vel_b[env_id] = data[f"body.{object_name}.lin_vel"]
    target._root_com_ang_vel_b[env_id] = data[f"body.{object_name}.ang_vel"]
