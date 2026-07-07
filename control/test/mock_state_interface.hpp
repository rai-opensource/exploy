// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <gmock/gmock.h>
#include <unordered_set>
#include "exploy/state_interface.hpp"

namespace exploy::control {

class MockRobotStateInterface : public RobotStateInterface {
 public:
  MOCK_METHOD(bool, initBasePosW, (const BasePosWInfo& info), (override));
  MOCK_METHOD(bool, initBaseQuatW, (const BaseQuatWInfo& info), (override));
  MOCK_METHOD(bool, initBaseLinVelB, (const BaseLinVelBInfo& info), (override));
  MOCK_METHOD(bool, initBaseAngVelB, (const BaseAngVelBInfo& info), (override));
  MOCK_METHOD(std::optional<Position>, basePosW, (const BasePosWInfo& info), (const, override));
  MOCK_METHOD(std::optional<Quaternion>, baseQuatW, (const BaseQuatWInfo& info), (const, override));
  MOCK_METHOD(std::optional<LinearVelocity>, baseLinVelB, (const BaseLinVelBInfo& info),
              (const, override));
  MOCK_METHOD(std::optional<AngularVelocity>, baseAngVelB, (const BaseAngVelBInfo& info),
              (const, override));
  MOCK_METHOD(bool, initJointOutput, (const JointOutputInfo& info), (override));
  MOCK_METHOD(bool, initJointPosition, (const JointPositionInfo& info), (override));
  MOCK_METHOD(bool, initJointVelocity, (const JointVelocityInfo& info), (override));
  MOCK_METHOD(bool, initJointEffort, (const JointEffortInfo& info), (override));
  MOCK_METHOD(std::optional<double>, jointPosition, (const JointPositionInfo& info),
              (const, override));
  MOCK_METHOD(std::optional<double>, jointVelocity, (const JointVelocityInfo& info),
              (const, override));
  MOCK_METHOD(std::optional<double>, jointEffort, (const JointEffortInfo& info), (const, override));
  MOCK_METHOD(bool, setJointPosition, (const SetJointPositionInfo& info), (override));
  MOCK_METHOD(bool, setJointVelocity, (const SetJointVelocityInfo& info), (override));
  MOCK_METHOD(bool, setJointEffort, (const SetJointEffortInfo& info), (override));
  MOCK_METHOD(bool, setJointPGain, (const SetJointPGainInfo& info), (override));
  MOCK_METHOD(bool, setJointDGain, (const SetJointDGainInfo& info), (override));
  MOCK_METHOD(bool, initSe2Velocity, (const Se2VelocityInfo& info), (override));
  MOCK_METHOD(bool, setSe2Velocity, (const Se2VelocityInfo& info), (override));
  MOCK_METHOD(bool, initImuLinearVelocityImu, (const ImuLinearVelocityImuInfo& info), (override));
  MOCK_METHOD(bool, initImuAngularVelocityImu, (const ImuAngularVelocityImuInfo& info), (override));
  MOCK_METHOD(bool, initImuOrientationW, (const ImuOrientationWInfo& info), (override));
  MOCK_METHOD(std::optional<LinearVelocity>, imuLinearVelocityImu,
              (const ImuLinearVelocityImuInfo& info), (const, override));
  MOCK_METHOD(std::optional<AngularVelocity>, imuAngularVelocityImu,
              (const ImuAngularVelocityImuInfo& info), (const, override));
  MOCK_METHOD(std::optional<Quaternion>, imuOrientationW, (const ImuOrientationWInfo& info),
              (const, override));
  MOCK_METHOD(bool, initBodyOrientationW, (const BodyOrientationWInfo& info), (override));
  MOCK_METHOD(bool, initBodyPositionW, (const BodyPositionWInfo& info), (override));
  MOCK_METHOD(bool, initBodyLinearVelocityB, (const BodyLinearVelocityBInfo& info), (override));
  MOCK_METHOD(bool, initBodyAngularVelocityB, (const BodyAngularVelocityBInfo& info), (override));
  MOCK_METHOD(std::optional<Position>, bodyPositionW, (const BodyPositionWInfo& info),
              (const override));
  MOCK_METHOD(std::optional<Quaternion>, bodyOrientationW, (const BodyOrientationWInfo& info),
              (const override));
  MOCK_METHOD(std::optional<LinearVelocity>, bodyLinearVelocityB,
              (const BodyLinearVelocityBInfo& info), (const override));
  MOCK_METHOD(std::optional<AngularVelocity>, bodyAngularVelocityB,
              (const BodyAngularVelocityBInfo& info), (const override));
  MOCK_METHOD(bool, initHeightScan, (const HeightScanInfo& info), (override));
  MOCK_METHOD(std::optional<const HeightScan*>, heightScan,
              (const HeightScanInfo& info, const Position& base_pos_w,
               const Quaternion& base_quat_w),
              (override));
  MOCK_METHOD(bool, initSphericalImage, (const SphericalImageInfo& info), (override));
  MOCK_METHOD(std::optional<const MultiChannelImage*>, sphericalImage,
              (const SphericalImageInfo& info), (override));
  MOCK_METHOD(bool, initPinholeImage, (const PinholeImageInfo& info), (override));
  MOCK_METHOD(std::optional<const MultiChannelImage*>, pinholeImage, (const PinholeImageInfo& info),
              (override));
};

/// @brief Matcher for any info struct with an articulation_name field.
template <typename T>
auto ArticulationIs(const std::string& articulation_name) {
  return ::testing::Field(&T::articulation_name, articulation_name);
}

/// @brief Matcher for any info struct with articulation_name and joint_name fields.
template <typename T>
auto JointIs(const std::string& articulation_name, const std::string& joint_name) {
  return ::testing::AllOf(::testing::Field(&T::articulation_name, articulation_name),
                          ::testing::Field(&T::joint_name, joint_name));
}

/// @brief Matcher for any info struct with articulation_name and body_name fields.
template <typename T>
auto BodyIs(const std::string& articulation_name, const std::string& body_name) {
  return ::testing::AllOf(::testing::Field(&T::articulation_name, articulation_name),
                          ::testing::Field(&T::body_name, body_name));
}

/// @brief Matcher for any info struct with an imu_name field.
template <typename T>
auto ImuIs(const std::string& imu_name) {
  return ::testing::Field(&T::imu_name, imu_name);
}

/// @brief Matcher for any info struct with a sensor_name field.
template <typename T>
auto SensorIs(const std::string& sensor_name) {
  return ::testing::Field(&T::sensor_name, sensor_name);
}

/// @brief Matcher for any info struct with a frame_name field.
template <typename T>
auto FrameIs(const std::string& frame_name) {
  return ::testing::Field(&T::frame_name, frame_name);
}

}  // namespace exploy::control
