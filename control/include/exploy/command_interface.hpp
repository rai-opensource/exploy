// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <optional>
#include <string>

#include "exploy/interfaces.hpp"
#include "exploy/logging_utils.hpp"

namespace exploy::control {

/// @brief Arguments for CommandInterface::initSe2Velocity and CommandInterface::se2Velocity.
struct Se2VelocityCommandInfo {
  std::string command_name;                   ///< The name of the command.
  std::optional<SE2VelocityRanges> ranges{};  ///< Optional ranges for the commanded se(2) velocity.
};

/// @brief Arguments for CommandInterface::initSe3Pose and CommandInterface::se3Pose.
struct Se3PoseCommandInfo {
  std::string command_name;  ///< The name of the command.
};

/// @brief Arguments for CommandInterface::initBooleanSelector and
/// CommandInterface::booleanSelector.
struct BooleanSelectorCommandInfo {
  std::string command_name;  ///< The name of the command.
};

/// @brief Arguments for CommandInterface::initFloatValue and CommandInterface::floatValue.
struct FloatValueCommandInfo {
  std::string command_name;      ///< The name of the command.
  std::optional<Range> range{};  ///< Optional range for the commanded float scalar.
};

/// @brief Arguments for CommandInterface::initJointPosition and CommandInterface::jointPosition.
struct JointPositionCommandInfo {
  std::string command_name;  ///< The name of the command.
  std::string joint_name;    ///< The name of the joint.
};

/**
 * @class CommandInterface
 *
 * @brief Interface which provides methods to send commands to the controllers.
 *
 */
class CommandInterface {
 public:
  virtual ~CommandInterface() = default;
  /**
   * @brief Initialize data source of commanded se2 velocity.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info The command info, including the command name and optional ranges.
   * @return True if initialization succeeded, false otherwise.
   */
  virtual bool initSe2Velocity(const Se2VelocityCommandInfo& info) {
    LOG_STREAM(ERROR, "initSe2Velocity() not implemented for command: " << info.command_name);
    return false;
  }
  /**
   * @brief Get commanded se2 velocity.
   *
   * @param info The command info, including the command name.
   * @return The commanded se2 velocity (lin x, lin y, ang z).
   */
  virtual std::optional<SE2Velocity> se2Velocity(const Se2VelocityCommandInfo& info) {
    LOG_STREAM(ERROR, "se2Velocity() not implemented for command: " << info.command_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of commanded SE3 pose.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info The command info, including the command name.
   * @return True if initialization succeeded, false otherwise.
   */
  virtual bool initSe3Pose(const Se3PoseCommandInfo& info) {
    LOG_STREAM(ERROR, "initSe3Pose() not implemented for command: " << info.command_name);
    return false;
  }
  /**
   * @brief Get commanded SE3 some pose.
   *
   * @param info The command info, including the command name.
   * @return The commanded SE3 pose.
   */
  virtual std::optional<SE3Pose> se3Pose(const Se3PoseCommandInfo& info) const {
    LOG_STREAM(ERROR, "se3Pose() not implemented for command: " << info.command_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of a boolean.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info The command info, including the command name.
   * @return True if initialization succeeded, false otherwise.
   */
  virtual bool initBooleanSelector(const BooleanSelectorCommandInfo& info) {
    LOG_STREAM(ERROR, "initBooleanSelector() not implemented for command: " << info.command_name);
    return false;
  }
  /**
   * @brief Get commanded boolean selector.
   *
   * @param info The command info, including the command name.
   * @return The commanded bool.
   */
  virtual std::optional<bool> booleanSelector(const BooleanSelectorCommandInfo& info) const {
    LOG_STREAM(ERROR, "booleanSelector() not implemented for command: " << info.command_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of a float.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info The command info, including the command name and optional range.
   * @return True if initialization succeeded, false otherwise.
   */
  virtual bool initFloatValue(const FloatValueCommandInfo& info) {
    LOG_STREAM(ERROR, "initFloatValue() not implemented for command: " << info.command_name);
    return false;
  }
  /**
   * @brief Get commanded float value.
   *
   * @param info The command info, including the command name.
   * @return The commanded float.
   */
  virtual std::optional<float> floatValue(const FloatValueCommandInfo& info) const {
    LOG_STREAM(ERROR, "floatValue() not implemented for command: " << info.command_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of a commanded joint position.
   *
   * Called once per joint during initialization (usually non real-time).
   *
   * @param info The command info, including the command name and joint name.
   * @return True if initialization succeeded, false otherwise.
   */
  virtual bool initJointPosition(const JointPositionCommandInfo& info) {
    LOG_STREAM(ERROR, "initJointPosition() not implemented for command: "
                          << info.command_name << ", joint: " << info.joint_name);
    return false;
  }
  /**
   * @brief Get the commanded position for a single joint.
   *
   * @param info The command info, including the command name and joint name.
   * @return The commanded joint position, or std::nullopt if unavailable.
   */
  virtual std::optional<float> jointPosition(const JointPositionCommandInfo& info) const {
    LOG_STREAM(ERROR, "jointPosition() not implemented for command: "
                          << info.command_name << ", joint: " << info.joint_name);
    return std::nullopt;
  }
};

}  // namespace exploy::control
