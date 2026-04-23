# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations


def test_entity_data_source_interface(env):
    """Test the implementation of the ``EntityDataSource`` class by comparing it against each
    matching property of an ``EntityData`` instance.

    Properties that exist only on ``EntityDataSource`` but not on ``EntityData``, and
    body-level ``TensorProxy`` attributes are excluded from the comparison.  Properties that raise
    ``AttributeError`` (e.g. optional data not available for this robot) are also skipped.
    """
    import inspect

    import torch

    from exploy.exporter.frameworks.mjlab.entity_data import EntityDataSource

    robot = env.scene.entities["robot"]

    # Step for a few steps to populate data with a non-default state.
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    num_steps = 10
    for _ in range(num_steps):
        env.step(actions)

    entity_data_source = EntityDataSource(robot)

    unimplemented_method_names = [
        "joint_torques",
    ]

    # Cycle through every property available in ``EntityData`` and, if the same property
    # exists on the live ``EntityDataSource``, compare the two values.
    for name, _ in inspect.getmembers(
        type(robot.data), predicate=lambda o: isinstance(o, property)
    ):
        if name in unimplemented_method_names:
            # Skip methods that are not yet implemented in EntityData.
            continue

        source_val = getattr(entity_data_source, name)
        expected_val = getattr(robot.data, name)

        assert torch.allclose(expected_val, source_val, rtol=1.0e-6, atol=1.0e-5), (
            f"Property '{name}' mismatch between EntityData and EntityDataSource.\n"
            f"  expected: {expected_val}\n"
            f"  got:      {source_val}"
        )

    entity_data_source_properties = inspect.getmembers(
        EntityDataSource, predicate=lambda o: isinstance(o, property)
    )
    entity_data_properties = inspect.getmembers(
        type(robot.data), predicate=lambda o: isinstance(o, property)
    )

    # Check if entity_data_source_properties has names that are not in entity_data_source.
    entity_data_source_property_names = {name for name, _ in entity_data_source_properties}
    entity_data_property_names = {name for name, _ in entity_data_properties}
    extra_properties = entity_data_source_property_names - entity_data_property_names

    # Assert that extra_properties is empty.
    assert not extra_properties, (
        f"EntityDataSource has properties that are not in EntityData: {extra_properties}"
    )
