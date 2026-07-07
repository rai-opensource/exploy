// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <gmock/gmock.h>
#include "exploy/command_interface.hpp"

namespace exploy::control {

class MockCommandInterface : public CommandInterface {
 public:
  MOCK_METHOD(bool, initSe2Velocity, (const Se2VelocityCommandInfo& info), (override));
  MOCK_METHOD(std::optional<SE2Velocity>, se2Velocity, (const Se2VelocityCommandInfo& info),
              (override));
  MOCK_METHOD(bool, initSe3Pose, (const Se3PoseCommandInfo& info), (override));
  MOCK_METHOD(std::optional<SE3Pose>, se3Pose, (const Se3PoseCommandInfo& info), (const, override));
  MOCK_METHOD(bool, initBooleanSelector, (const BooleanSelectorCommandInfo& info), (override));
  MOCK_METHOD(std::optional<bool>, booleanSelector, (const BooleanSelectorCommandInfo& info),
              (const override));
  MOCK_METHOD(bool, initFloatValue, (const FloatValueCommandInfo& info), (override));
  MOCK_METHOD(std::optional<float>, floatValue, (const FloatValueCommandInfo& info),
              (const override));
  MOCK_METHOD(bool, initJointPosition, (const JointPositionCommandInfo& info), (override));
  MOCK_METHOD(std::optional<float>, jointPosition, (const JointPositionCommandInfo& info),
              (const, override));
};

/// @brief Matcher for a JointPositionCommandInfo with the given command and joint name.
inline auto JointPositionCommandIs(const std::string& command_name, const std::string& joint_name) {
  return ::testing::AllOf(::testing::Field(&JointPositionCommandInfo::command_name, command_name),
                          ::testing::Field(&JointPositionCommandInfo::joint_name, joint_name));
}

}  // namespace exploy::control
