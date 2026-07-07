// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

/**
 * @file loopback_state_interface.hpp
 * @brief RobotStateInterface that feeds commanded values back as measured state.
 *
 * This is intentionally minimal: every init*() succeeds, every getter returns a
 * sensible default (zeros / identity), and the set*() methods store the commanded
 * value so that it is returned the following cycle — modelling a perfect,
 * zero-delay actuator.
 */

#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <utility>

#include "exploy/state_interface.hpp"

namespace exploy::control::examples {

/**
 * @brief Per-joint loopback state (position, velocity, effort).
 */
struct JointState {
  double position{0.0};
  double velocity{0.0};
  double effort{0.0};
};

/**
 * @brief Concrete RobotStateInterface with perfect loopback semantics.
 *
 * Commanded joint targets written via setJoint*() are returned by
 * jointPosition() / jointVelocity() / jointEffort() on the next query,
 * modelling a zero-latency, perfectly tracking actuator.
 *
 * All body/IMU/base state values are fixed at identity / zero so that an
 * ONNX policy can run without a physics simulation.
 */
class LoopbackRobotStateInterface : public RobotStateInterface {
 public:
  bool initBasePosW(const BasePosWInfo& /*info*/) override { return true; }
  bool initBaseQuatW(const BaseQuatWInfo& /*info*/) override { return true; }

  std::optional<Position> basePosW(const BasePosWInfo& /*info*/) const override {
    return Position{0.0, 0.0, 0.8};
  }
  std::optional<Quaternion> baseQuatW(const BaseQuatWInfo& /*info*/) const override {
    return Quaternion::Identity();
  }

  bool initBaseLinVelB(const BaseLinVelBInfo& /*info*/) override { return true; }
  bool initBaseAngVelB(const BaseAngVelBInfo& /*info*/) override { return true; }

  std::optional<LinearVelocity> baseLinVelB(const BaseLinVelBInfo& /*info*/) const override {
    return LinearVelocity::Zero();
  }
  std::optional<AngularVelocity> baseAngVelB(const BaseAngVelBInfo& /*info*/) const override {
    return AngularVelocity::Zero();
  }

  bool initJointPosition(const JointPositionInfo& info) override {
    joint_states_.emplace(info.joint_name, JointState{});
    return true;
  }
  bool initJointVelocity(const JointVelocityInfo& info) override {
    joint_states_.emplace(info.joint_name, JointState{});
    return true;
  }
  bool initJointEffort(const JointEffortInfo& info) override {
    joint_states_.emplace(info.joint_name, JointState{});
    return true;
  }
  bool initJointOutput(const JointOutputInfo& /*info*/) override {
    return true;
  }

  std::optional<double> jointPosition(const JointPositionInfo& info) const override {
    if (joint_states_.contains(info.joint_name)) {
      return joint_states_.at(info.joint_name).position;
    }
    return std::nullopt;
  }
  std::optional<double> jointVelocity(const JointVelocityInfo& info) const override {
    if (joint_states_.contains(info.joint_name)) {
      return joint_states_.at(info.joint_name).velocity;
    }
    return std::nullopt;
  }
  std::optional<double> jointEffort(const JointEffortInfo& info) const override {
    if (joint_states_.contains(info.joint_name)) {
      return joint_states_.at(info.joint_name).effort;
    }
    return std::nullopt;
  }

  /// Loopback: store the commanded position so it is returned next cycle.
  bool setJointPosition(const SetJointPositionInfo& info) override {
    joint_states_[info.joint_name].position = info.position;
    return true;
  }
  bool setJointVelocity(const SetJointVelocityInfo& info) override {
    joint_states_[info.joint_name].velocity = info.velocity;
    return true;
  }
  bool setJointEffort(const SetJointEffortInfo& info) override {
    joint_states_[info.joint_name].effort = info.effort;
    return true;
  }
  bool setJointPGain(const SetJointPGainInfo& /*info*/) override {
    return true;
  }
  bool setJointDGain(const SetJointDGainInfo& /*info*/) override {
    return true;
  }

  // Read-only access to the full joint state map (used by main.cpp for logging).
  const std::unordered_map<std::string, JointState>& jointStates() const {
    return joint_states_;
  }

  bool initSe2Velocity(const Se2VelocityInfo& info) override {
    se2_velocities_.emplace(info.frame_name, SE2Velocity::Zero());
    return true;
  }
  bool setSe2Velocity(const Se2VelocityInfo& info) override {
    se2_velocities_[info.frame_name] = info.velocity;
    return true;
  }

  // Read-only access to all SE(2) velocity outputs (used by main.cpp for logging).
  const std::unordered_map<std::string, SE2Velocity>& se2VelocityOutputs() const {
    return se2_velocities_;
  }

  bool initImuLinearVelocityImu(const ImuLinearVelocityImuInfo& /*info*/) override {
    return true;
  }

  std::optional<LinearVelocity> imuLinearVelocityImu(
      const ImuLinearVelocityImuInfo& /*info*/) const override {
    return LinearVelocity::Zero();
  }

  bool initImuAngularVelocityImu(const ImuAngularVelocityImuInfo& /*info*/) override {
    return true;
  }

  std::optional<AngularVelocity> imuAngularVelocityImu(
      const ImuAngularVelocityImuInfo& /*info*/) const override {
    return AngularVelocity::Zero();
  }

  bool initImuOrientationW(const ImuOrientationWInfo& /*info*/) override { return true; }

  std::optional<Quaternion> imuOrientationW(const ImuOrientationWInfo& /*info*/) const override {
    return Quaternion::Identity();
  }

  bool initBodyPositionW(const BodyPositionWInfo& /*info*/) override {
    return true;
  }
  bool initBodyOrientationW(const BodyOrientationWInfo& /*info*/) override {
    return true;
  }

  std::optional<Position> bodyPositionW(const BodyPositionWInfo& /*info*/) const override {
    return Position{0.0, 0.0, 0.8};
  }
  std::optional<Quaternion> bodyOrientationW(const BodyOrientationWInfo& /*info*/) const override {
    return Quaternion::Identity();
  }

  bool initHeightScan(const HeightScanInfo& info) override {
    // Pre-allocate zero-filled layers sized by the grid formula.
    const auto& p = info.pattern;
    const std::size_t nx = static_cast<std::size_t>(std::round(p.size.x() / p.resolution)) + 1;
    const std::size_t ny = static_cast<std::size_t>(std::round(p.size.y() / p.resolution)) + 1;
    const std::size_t n = nx * ny;

    HeightScan scan;
    for (const auto& layer : info.layer_names) {
      scan.float_layers[layer] = std::vector<float>(n, 0.0);
    }
    height_scans_[info.sensor_name] = std::move(scan);
    return true;
  }

  std::optional<const HeightScan*> heightScan(const HeightScanInfo& info,
                                              const Position& /*base_pos_w*/,
                                              const Quaternion& /*base_quat_w*/) override {
    if (!height_scans_.contains(info.sensor_name)) {
      return std::nullopt;
    }
    return &height_scans_.at(info.sensor_name);
  }

 private:
  std::unordered_map<std::string, JointState> joint_states_;
  std::unordered_map<std::string, SE2Velocity> se2_velocities_;
  std::unordered_map<std::string, HeightScan> height_scans_;
};

}  // namespace exploy::control::examples
