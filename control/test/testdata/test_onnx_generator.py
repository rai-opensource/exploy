# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import json
import os
import sys

import onnx
import torch


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        joint_pos,
        joint_vel,
        base_pos_in_w,
        world_Q_base,
        lin_vel_base_in_base,
        ang_vel_base_in_base,
        cmd_se2_vel,
        cmd_se2_vel_with_range,
        cmd_se3_pose,
        cmd_boolean,
        cmd_float,
        imu_data_quat,
        imu_data_ang_vel,
        heightscan,
        another_heightscan,
        range_image,
        trail_scan_height,
        trail_scan_r,
        trail_scan_g,
        trail_scan_b,
        depth_image,
        body_pos,
        body_quat,
        memory,
        step_count,
    ):
        # Collect all inputs (excluding self) and ensure they are not optimized away
        inputs = [v for k, v in locals().items() if k != "self"]

        # Identity operations to keep them in the graph
        processed = [i * 1.0 for i in inputs]

        concatenated = torch.cat(processed, dim=1)

        joint_pos_targets = concatenated[:, 0:2]
        joint_vel_targets = concatenated[:, 2:4]
        joint_effort_targets = concatenated[:, 4:6]
        se2_base_velocity_target = processed[6]  # cmd_se2_vel
        actions = concatenated[:, 6:8]
        joint_targets_mem = concatenated[:, 8:10]

        return (
            joint_pos_targets,
            joint_vel_targets,
            joint_effort_targets,
            se2_base_velocity_target,
            actions,
            joint_targets_mem,
        )


def add_metadata(path: str, metadata: dict):
    onnx_model = onnx.load(path)
    for key, val in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = json.dumps(val)
    onnx.save(onnx_model, path)


def main():
    # Define dummy inputs
    joint_pos = torch.rand((1, 3), dtype=torch.float32)
    joint_vel = torch.rand((1, 3), dtype=torch.float32)

    base_pos = torch.rand((1, 3), dtype=torch.float32)
    base_orientation = torch.rand((1, 4), dtype=torch.float32)
    base_lin_vel = torch.rand((1, 3), dtype=torch.float32)
    base_ang_vel = torch.rand((1, 3), dtype=torch.float32)

    se2_velocity_command = torch.rand((1, 3), dtype=torch.float32)
    se3_pose_command = torch.rand((1, 7), dtype=torch.float32)
    boolean_command = torch.tensor([[True]], dtype=torch.bool)
    float_command = torch.tensor([[3.14]], dtype=torch.float32)

    imu_data_quat = torch.rand((1, 4), dtype=torch.float32)
    imu_data_ang_vel = torch.rand((1, 3), dtype=torch.float32)

    heightscan = torch.rand((1, 4), dtype=torch.float32)
    range_image = torch.rand((1, 4), dtype=torch.float32)
    depth_image = torch.rand((1, 4), dtype=torch.float32)
    trail_scan = torch.rand((1, 8), dtype=torch.float32)

    body_pos = torch.rand((1, 3), dtype=torch.float32)
    body_quat = torch.rand((1, 4), dtype=torch.float32)

    memory = torch.rand((1, 2), dtype=torch.float32)
    step_count = torch.tensor([[42]], dtype=torch.int32)

    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Store model
    output_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(data_dir, "test.onnx")
    model = TestModel()
    torch.onnx.export(
        model,
        (
            # joints
            joint_pos,
            joint_vel,
            # base
            base_pos,
            base_orientation,
            base_lin_vel,
            base_ang_vel,
            # commands
            se2_velocity_command,
            se2_velocity_command,
            se3_pose_command,
            boolean_command,
            float_command,
            # IMU
            imu_data_quat,
            imu_data_ang_vel,
            # sensors
            heightscan,
            heightscan,
            range_image,
            trail_scan,  # height
            trail_scan,  # r
            trail_scan,  # g
            trail_scan,  # b
            depth_image,
            # body
            body_pos,
            body_quat,
            # memory
            memory,
            # step count
            step_count,
        ),
        output_path,
        input_names=[
            # joints
            "obj.robot1.joint.pos",
            "obj.robot1.joint.vel",
            # base
            "obj.robot1.base.body_pos_in_w",
            "obj.robot1.base.world_Q_body",
            "obj.robot1.base.lin_vel_body_in_body",
            "obj.robot1.base.ang_vel_body_in_body",
            # commands
            "cmd.se2_velocity.vel",
            "cmd.se2_velocity.vel_with_range",
            "cmd.se3_pose.pose",
            "cmd.boolean.selector",
            "cmd.float.value",
            # IMU
            "sensor.imu.torso.world_Q_body",
            "sensor.imu.pelvis.ang_vel_body",
            # sensors
            "sensor.height_scanner.one.height",
            "sensor.height_scanner.two.height",
            "sensor.range_image.one",
            "sensor.height_scanner.trail.height",
            "sensor.height_scanner.trail.r",
            "sensor.height_scanner.trail.g",
            "sensor.height_scanner.trail.b",
            "sensor.depth_image.one",
            # body
            "obj.box1.bodies.box.pos_body_in_w",
            "obj.box1.bodies.box.world_Q_body",
            # memory
            "memory.output.joint_targets.pos.in",
            # step count
            "ctx.step_count",
        ],
        output_names=[
            "output.joint_targets.pos",
            "output.joint_targets.vel",
            "output.joint_targets.effort",
            "output.se2_velocity",
            "actions",
            "memory.output.joint_targets.pos.out",
        ],
    )

    output_metadata = {
        "output.joint_targets": {
            "names": ["j1", "j2"],
            "stiffness": [1.0, 2.0],
            "damping": [0.1, 0.2],
        },
        "output.se2_velocity": {
            "target_frame": "base_frame",
        },
    }

    sensor_metadata = {
        "sensor.height_scanner.one": {
            "pattern_type": "grid_pattern",
            "resolution": 0.1,
            "size_x": 1.6,
            "size_y": 1.0,
            "offset_x": 1.0,
            "offset_y": 0.0,
        },
        "sensor.height_scanner.two": {
            "pattern_type": "grid_pattern",
            "resolution": 0.1,
            "size_x": 1.6,
            "size_y": 1.0,
        },
        "sensor.range_image.one": {
            "pattern_type": "lidar_pattern",
            "v_res": 128,
            "h_res": 1024,
            "v_fov_min_deg": -45.0,
            "v_fov_max_deg": 45.0,
            "unobserved_value": -2.0,
        },
        "sensor.depth_image.one": {
            "pattern_type": "grid_pattern",
            "height": 1,
            "width": 1,
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.5,
            "cy": 0.5,
        },
        "sensor.height_scanner.trail": {
            "pattern_type": "grid_pattern",
            "resolution": 0.1,
            "size_x": 1.6,
            "size_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
    }

    command_metadata = {
        "cmd.se2_velocity.vel": {},
        "cmd.se2_velocity.vel_with_range": {
            "ranges": {
                "lin_vel_x": [-1.5, 1.5],
                "lin_vel_y": [-0.75, 0.75],
                "ang_vel_z": [-2.5, 2.5],
            },
        },
    }

    env_metadata = {
        "update_rate": 10.0,
    }

    articulation_metadata = {
        "obj.robot1.joint.pos": {
            "names": ["j1", "j2", "j3"],
        },
        "obj.robot1.joint.vel": {
            "names": ["j1", "j2", "j3"],
        },
    }

    add_metadata(
        output_path,
        env_metadata | output_metadata | command_metadata | sensor_metadata | articulation_metadata,
    )


if __name__ == "__main__":
    main()
