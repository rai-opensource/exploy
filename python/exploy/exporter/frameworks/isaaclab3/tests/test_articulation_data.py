# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import inspect


def _property_names(cls: type) -> set[str]:
    return {
        name for name, _ in inspect.getmembers(cls, predicate=lambda o: isinstance(o, property))
    }


def test_articulation_data_interface(make_env):
    """Compare each property of `ArticulationDataSource` with the same property of the articulation data."""
    import torch

    from exploy.exporter.core.tensor_proxy import to_tensor
    from exploy.exporter.frameworks.isaaclab3.articulation_data import ArticulationDataSource
    from exploy.exporter.frameworks.isaaclab3.proxy_array import TorchProxyArray

    env = make_env("Isaac-Velocity-Flat-G1", num_envs=4)

    # Step the environment with random actions to populate the articulation data with a non-default state.
    torch.manual_seed(0)
    for _ in range(10):
        env.step(
            0.5 * torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        )

    articulation = env.scene.articulations["robot"]
    articulation_data_source = ArticulationDataSource(articulation=articulation)

    source_property_names = _property_names(ArticulationDataSource)
    for name in sorted(source_property_names):
        expected_val = getattr(articulation.data, name).torch
        source_val = getattr(articulation_data_source, name)
        assert isinstance(source_val, TorchProxyArray), name
        # For a discussion on how to set tolerances, see:
        #   https://docs.pytorch.org/docs/stable/testing.html
        torch.testing.assert_close(
            to_tensor(source_val.torch), expected_val, rtol=1.0e-5, atol=1.0e-4, msg=name
        )

    # Properties that are not state fall back to snapshots of the articulation data.
    torch.testing.assert_close(
        articulation_data_source.default_joint_pos.torch, articulation.data.default_joint_pos.torch
    )
    assert (
        articulation_data_source.default_joint_pos.torch
        is not articulation.data.default_joint_pos.torch
    )
    assert articulation_data_source.joint_names == articulation.data.joint_names

    # The data source must not define properties that the articulation data does not have.
    extra_properties = source_property_names - _property_names(type(articulation.data))
    assert not extra_properties, (
        f"ArticulationDataSource has properties not in ArticulationData: {extra_properties}"
    )
