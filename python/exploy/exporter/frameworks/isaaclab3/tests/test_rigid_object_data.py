# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import inspect


def _property_names(cls: type) -> set[str]:
    return {
        name for name, _ in inspect.getmembers(cls, predicate=lambda o: isinstance(o, property))
    }


def test_rigid_object_data_interface(make_env):
    """Compare each property of `RigidObjectDataSource` with the same property of the rigid object data."""
    import torch

    from exploy.exporter.core.tensor_proxy import to_tensor
    from exploy.exporter.frameworks.isaaclab3.body_data import RigidObjectDataSource

    env = make_env("Isaac-Lift-Franka", num_envs=4)

    # Step the environment with random actions so that the robot pushes the object around.
    torch.manual_seed(0)
    for _ in range(20):
        env.step(torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device))

    rigid_object = env.scene.rigid_objects["object"]
    source = RigidObjectDataSource(rigid_object.data)

    source_property_names = _property_names(RigidObjectDataSource)
    for name in sorted(source_property_names):
        expected_val = getattr(rigid_object.data, name).torch
        source_val = to_tensor(getattr(source, name).torch)
        torch.testing.assert_close(source_val, expected_val, rtol=1.0e-5, atol=1.0e-4, msg=name)

    extra_properties = source_property_names - _property_names(type(rigid_object.data))
    assert not extra_properties, (
        f"RigidObjectDataSource has properties not in RigidObjectData: {extra_properties}"
    )
