# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations


def test_rigid_object_data_interface(sim_setup):
    """Test the implementation of the `RigidObjectDataSource` class by comparing it against each property of a `RigidObjectData` instance."""
    # Import after AppLauncher is initialized
    import inspect

    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.prims as prim_utils
    import torch
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import build_simulation_context

    from exploy.exporter.frameworks.isaaclab.rigid_object_data import RigidObjectDataSource

    def generate_cubes_scene(num_cubes: int, device: str) -> tuple[RigidObject, torch.Tensor]:
        """Generate a scene with the provided number of cubes.

        Args:
            num_cubes: Number of cubes to generate.
            device: Device to use for the simulation.

        Returns:
            A tuple containing the rigid object representing the cubes and the origins of the cubes.
        """
        height = 1.0
        origins = torch.tensor([(i * 2.0, 0, height) for i in range(num_cubes)]).to(device)
        # Create Top-level Xforms, one for each cube
        for i, origin in enumerate(origins):
            prim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=origin)

        # Create spawn configuration using primitive cubes instead of USD files
        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )

        # Create rigid object
        cube_object_cfg = RigidObjectCfg(
            prim_path="/World/Table_.*/Object",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
        cube_object = RigidObject(cfg=cube_object_cfg)

        return cube_object, origins

    device = "cpu"
    num_cubes = 3
    torch.manual_seed(42)

    with build_simulation_context(
        sim_utils.SimulationCfg(), device=device, auto_add_lighting=True
    ) as sim:
        # Disable app control callback (from Isaac Lab test pattern)
        sim._app_control_on_stop_handle = None

        # Generate scene with cubes
        rigid_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

        # Reset simulation to initialize physics
        sim.reset()

        # Verify object is initialized
        assert rigid_object.is_initialized

        # Step the simulation for a few steps to populate the rigid object data with a non-default state.
        # Apply some random forces to make the objects move
        num_steps = 10
        for step in range(num_steps):
            # Apply random forces to make objects move
            if step % 5 == 0:
                # Create forces and torques with proper shape (num_envs, num_bodies, 3)
                # For RigidObject, num_bodies is always 1
                forces = torch.randn(num_cubes, 1, 3, device=device) * 100.0
                torques = torch.randn(num_cubes, 1, 3, device=device) * 10.0
                rigid_object.set_external_force_and_torque(forces, torques)

            # Write the forces to simulation before stepping
            rigid_object.write_data_to_sim()

            # Step simulation
            sim.step()
            # Update the rigid object
            rigid_object.update(dt=sim.cfg.dt)

        # Create the data source
        rigid_object_data_source = RigidObjectDataSource(rigid_object=rigid_object)

        # Cycle through every property available in `RigidObjectDataSource` and compare it with the same property
        # from `RigidObjectData`.
        for name, _ in inspect.getmembers(
            RigidObjectDataSource, predicate=lambda o: isinstance(o, property)
        ):
            expected_val = getattr(rigid_object.data, name)
            source_val = getattr(rigid_object_data_source, name)
            # For a discussion on how to set tolerances, see:
            #   https://docs.pytorch.org/docs/stable/testing.html
            assert torch.allclose(expected_val, source_val, rtol=1.0e-6, atol=1.0e-5)
