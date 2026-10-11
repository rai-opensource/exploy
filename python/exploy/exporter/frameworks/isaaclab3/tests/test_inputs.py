# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations


def test_add_body_vel(make_env):
    """Test that `add_body_vel` adds body-frame velocity inputs for every body.

    The body-frame velocities are derived and materialized against the `ArticulationDataSource` as
    stored per-body leaves when `IsaacLabExportableEnvironment` swaps the data sources in. The inputs
    must hold the exact stored slice objects (required for ONNX tracing), and must fall back to
    computing from live data once the original articulation data is restored.

    The root body is special-cased: `ArticulationDataSource` keeps `root_lin_vel_b`/`root_ang_vel_b`
    as separate stored leaves, so the root inputs must bind to those.
    """
    import isaaclab.utils.math as math_utils
    import torch

    from exploy.exporter.core.context_manager import ContextManager
    from exploy.exporter.core.tensor_proxy import TensorProxy
    from exploy.exporter.frameworks.isaaclab3 import derived_tensors, inputs
    from exploy.exporter.frameworks.isaaclab3.env import IsaacLabExportableEnvironment

    env = make_env("Isaac-Velocity-Flat-G1", num_envs=2)
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(5):
        env.step(actions)

    exportable_env = IsaacLabExportableEnvironment(env)
    articulations = env.scene.articulations

    context_manager = ContextManager()
    inputs.add_body_vel(articulations=articulations, context_manager=context_manager)
    comp_by_name = {comp.name: comp for comp in context_manager.get_input_components()}

    expected_names = {
        f"obj.{obj_name}.{body_name}.{quantity}"
        for obj_name, articulation in articulations.items()
        for body_name in articulation.data.body_names
        for quantity in ("lin_vel_b_rt_w_in_b", "ang_vel_b_rt_w_in_b")
    }
    assert set(comp_by_name) == expected_names

    # The root inputs must bind to the separately stored root velocity leaves.
    for obj_name, articulation in articulations.items():
        root_prefix = f"obj.{obj_name}.{articulation.body_names[0]}"
        root_lin_vel_comp = comp_by_name[f"{root_prefix}.lin_vel_b_rt_w_in_b"]
        root_ang_vel_comp = comp_by_name[f"{root_prefix}.ang_vel_b_rt_w_in_b"]
        assert root_lin_vel_comp.data is articulation.data.root_lin_vel_b.torch
        assert root_ang_vel_comp.data is articulation.data.root_ang_vel_b.torch
        assert root_lin_vel_comp.get_from_env_cb() is root_lin_vel_comp.data
        assert root_ang_vel_comp.get_from_env_cb() is root_ang_vel_comp.data

    def expected_body_vel(articulation, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        body_quat_w = articulation.data.body_link_quat_w.torch[:, idx]
        lin_vel_w = articulation.data.body_link_lin_vel_w.torch[:, idx]
        ang_vel_w = articulation.data.body_link_ang_vel_w.torch[:, idx]
        return (
            math_utils.quat_apply_inverse(body_quat_w, lin_vel_w),
            math_utils.quat_apply_inverse(body_quat_w, ang_vel_w),
        )

    # The non-root inputs must hold the materialized per-body leaves, matching the derived values.
    for obj_name, articulation in articulations.items():
        assert isinstance(derived_tensors.body_link_lin_vel_b(articulation.data), TensorProxy)
        assert isinstance(derived_tensors.body_link_ang_vel_b(articulation.data), TensorProxy)
        for body_name in articulation.data.body_names[1:]:
            idx = articulation.find_bodies(body_name)[0][0]
            prefix = f"obj.{obj_name}.{body_name}"
            lin_vel_comp = comp_by_name[f"{prefix}.lin_vel_b_rt_w_in_b"]
            ang_vel_comp = comp_by_name[f"{prefix}.ang_vel_b_rt_w_in_b"]
            assert (
                lin_vel_comp.data is derived_tensors.body_link_lin_vel_b(articulation.data)[:, idx]
            )
            assert (
                ang_vel_comp.data is derived_tensors.body_link_ang_vel_b(articulation.data)[:, idx]
            )
            assert lin_vel_comp.get_from_env_cb() is lin_vel_comp.data

            expected_lin_vel, expected_ang_vel = expected_body_vel(articulation, idx)
            torch.testing.assert_close(lin_vel_comp.data, expected_lin_vel)
            torch.testing.assert_close(ang_vel_comp.data, expected_ang_vel)

    # After the original data is restored (as during evaluation), the inputs read the live data.
    exportable_env.cleanup()
    for obj_name, articulation in articulations.items():
        for body_name in articulation.data.body_names[1:]:
            idx = articulation.find_bodies(body_name)[0][0]
            prefix = f"obj.{obj_name}.{body_name}"
            expected_lin_vel, expected_ang_vel = expected_body_vel(articulation, idx)
            torch.testing.assert_close(
                comp_by_name[f"{prefix}.lin_vel_b_rt_w_in_b"].get_from_env_cb(), expected_lin_vel
            )
            torch.testing.assert_close(
                comp_by_name[f"{prefix}.ang_vel_b_rt_w_in_b"].get_from_env_cb(), expected_ang_vel
            )

        root_prefix = f"obj.{obj_name}.{articulation.body_names[0]}"
        torch.testing.assert_close(
            comp_by_name[f"{root_prefix}.lin_vel_b_rt_w_in_b"].get_from_env_cb(),
            articulation.data.root_lin_vel_b.torch,
        )


def test_body_pose_inputs_are_wxyz_leaves(make_env):
    """Test that the body pose inputs are per-body leaves, with quaternions in the (w, x, y, z) convention
    of exploy's ONNX interface, while the data source exposes Isaac Lab 3's (x, y, z, w) convention."""
    import torch

    from exploy.exporter.core.context_manager import ContextManager
    from exploy.exporter.frameworks.isaaclab3 import inputs
    from exploy.exporter.frameworks.isaaclab3.env import IsaacLabExportableEnvironment
    from exploy.exporter.frameworks.isaaclab3.proxy_array import xyzw_to_wxyz

    env = make_env("Isaac-Velocity-Flat-G1", num_envs=2)
    for _ in range(5):
        env.step(torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device))

    articulation = env.scene.articulations["robot"]
    live_quat_xyzw = articulation.data.body_link_quat_w.torch.clone()
    live_pos = articulation.data.body_link_pos_w.torch.clone()

    exportable_env = IsaacLabExportableEnvironment(env)
    source = articulation.data

    context_manager = ContextManager()
    inputs.add_body_pos_and_quat(
        articulations={"robot": articulation}, context_manager=context_manager
    )
    comp_by_name = {comp.name: comp for comp in context_manager.get_input_components()}

    for body_name in articulation.data.body_names:
        idx = articulation.find_bodies(body_name)[0][0]
        pos_comp = comp_by_name[f"obj.robot.{body_name}.pos_b_rt_w_in_w"]
        quat_comp = comp_by_name[f"obj.robot.{body_name}.w_Q_b"]

        # The inputs are the stored per-body leaves.
        assert pos_comp.data is source.body_link_pos_w.torch[:, idx]
        assert quat_comp.data is source.body_link_quat_w_wxyz()[:, idx]
        assert quat_comp.get_from_env_cb() is quat_comp.data

        # The input quaternion is (w, x, y, z), the data source exposes (x, y, z, w).
        torch.testing.assert_close(quat_comp.data, xyzw_to_wxyz(live_quat_xyzw[:, idx]))
        torch.testing.assert_close(source.body_link_quat_w.torch[:, idx], live_quat_xyzw[:, idx])
        torch.testing.assert_close(pos_comp.data, live_pos[:, idx])

    # The root quaternion is derived from the root body's input only.
    root_quat_comp = comp_by_name[f"obj.robot.{articulation.body_names[0]}.w_Q_b"]
    root_quat_comp.data.copy_(torch.tensor([0.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(
        source.root_link_quat_w.torch, torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
    )

    exportable_env.cleanup()
    torch.testing.assert_close(root_quat_comp.get_from_env_cb(), xyzw_to_wxyz(live_quat_xyzw[:, 0]))


def test_add_body_inputs_rigid_object(make_env):
    """Test that the body pose/velocity inputs also work for `RigidObject` assets, binding to the tensors
    stored on the swapped-in `RigidObjectDataSource`."""
    import torch

    from exploy.exporter.core.context_manager import ContextManager
    from exploy.exporter.frameworks.isaaclab3 import inputs
    from exploy.exporter.frameworks.isaaclab3.body_data import (
        RigidObjectDataSource,
        dict_to_rigid_object_data,
        rigid_object_data_to_dict,
    )
    from exploy.exporter.frameworks.isaaclab3.proxy_array import xyzw_to_wxyz

    env = make_env("Isaac-Lift-Franka", num_envs=3)
    torch.manual_seed(0)
    for _ in range(20):
        env.step(torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device))

    obj_name = "cube"
    cube = env.scene.rigid_objects["object"]
    body_name = cube.body_names[0]
    prefix = f"obj.{obj_name}.{body_name}"

    original_data = cube.data
    expected_pos = original_data.body_link_pos_w.torch[:, 0].clone()
    expected_quat = xyzw_to_wxyz(original_data.body_link_quat_w.torch[:, 0])
    expected_lin_vel = original_data.root_lin_vel_b.torch.clone()
    expected_ang_vel = original_data.root_ang_vel_b.torch.clone()

    source = RigidObjectDataSource(original_data)
    cube._data = source
    try:
        context_manager = ContextManager()
        inputs.add_body_pos_and_quat(
            articulations={obj_name: cube}, context_manager=context_manager
        )
        inputs.add_body_vel(articulations={obj_name: cube}, context_manager=context_manager)
        comp_by_name = {comp.name: comp for comp in context_manager.get_input_components()}

        assert set(comp_by_name) == {
            f"{prefix}.pos_b_rt_w_in_w",
            f"{prefix}.w_Q_b",
            f"{prefix}.lin_vel_b_rt_w_in_b",
            f"{prefix}.ang_vel_b_rt_w_in_b",
        }
        pos_comp = comp_by_name[f"{prefix}.pos_b_rt_w_in_w"]
        quat_comp = comp_by_name[f"{prefix}.w_Q_b"]
        lin_vel_comp = comp_by_name[f"{prefix}.lin_vel_b_rt_w_in_b"]
        ang_vel_comp = comp_by_name[f"{prefix}.ang_vel_b_rt_w_in_b"]

        # Identity: each input holds the exact tensor stored on the data source.
        assert pos_comp.data is source.root_pos_w.torch
        assert quat_comp.data is source.body_link_quat_w_wxyz()[:, 0]
        assert lin_vel_comp.data is source.root_lin_vel_b.torch
        assert ang_vel_comp.data is source.root_ang_vel_b.torch
        for comp in (pos_comp, quat_comp, lin_vel_comp, ang_vel_comp):
            assert comp.get_from_env_cb() is comp.data

        torch.testing.assert_close(pos_comp.data, expected_pos)
        torch.testing.assert_close(quat_comp.data, expected_quat)
        torch.testing.assert_close(lin_vel_comp.data, expected_lin_vel)
        torch.testing.assert_close(ang_vel_comp.data, expected_ang_vel)

        # Writing new state into the data source is visible through the registered inputs.
        new_state = {
            f"body.{obj_name}.pos": torch.tensor([1.0, 2.0, 3.0]),
            f"body.{obj_name}.quat": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            f"body.{obj_name}.lin_vel": torch.tensor([4.0, 5.0, 6.0]),
            f"body.{obj_name}.ang_vel": torch.tensor([7.0, 8.0, 9.0]),
        }
        dict_to_rigid_object_data(data=new_state, object_name=obj_name, target=source, env_id=0)
        torch.testing.assert_close(pos_comp.data[0], new_state[f"body.{obj_name}.pos"])
        torch.testing.assert_close(quat_comp.data[0], new_state[f"body.{obj_name}.quat"])
        torch.testing.assert_close(lin_vel_comp.data[0], new_state[f"body.{obj_name}.lin_vel"])
        torch.testing.assert_close(ang_vel_comp.data[0], new_state[f"body.{obj_name}.ang_vel"])
        read_back = rigid_object_data_to_dict(object_name=obj_name, source=source, env_id=0)
        for key, value in new_state.items():
            torch.testing.assert_close(read_back[key], value)
    finally:
        cube._data = original_data

    # Once the original data is restored, the callbacks read the live rigid object data.
    torch.testing.assert_close(pos_comp.get_from_env_cb(), expected_pos)
    torch.testing.assert_close(quat_comp.get_from_env_cb(), expected_quat)
    torch.testing.assert_close(lin_vel_comp.get_from_env_cb(), expected_lin_vel)
    torch.testing.assert_close(ang_vel_comp.get_from_env_cb(), expected_ang_vel)


def test_height_scan_matches_sensor(make_env):
    """Test that the height scan observation computed from the `RayCasterDataSource` matches the one
    computed from the live sensor data. The physics backends differ in whether the sensor position
    includes the configured offset, so the data source measures it."""
    import torch

    from exploy.exporter.frameworks.isaaclab3.env import IsaacLabExportableEnvironment

    env = make_env("Isaac-Velocity-Rough-AnymalD", num_envs=2)
    for _ in range(5):
        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))

    expected_obs = env.observation_manager.compute()["policy"].clone()
    exportable_env = IsaacLabExportableEnvironment(env)
    torch.testing.assert_close(env.observation_manager.compute()["policy"], expected_obs)
    exportable_env.cleanup()
