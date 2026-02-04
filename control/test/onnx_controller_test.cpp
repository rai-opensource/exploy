// Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "onnx_controller.hpp"
#include "mock_command_interface.hpp"
#include "mock_data_collection_interface.hpp"
#include "mock_state_interface.hpp"
#include "onnx_components.hpp"
#include "onnx_context.hpp"
#include "onnx_matcher.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <regex>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <filesystem>

namespace rai::cs::control::common::onnx {

using ::testing::_;
using ::testing::AllOf;
using ::testing::Eq;
using ::testing::Field;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

using operation::common::data_collection::MockDataCollectionInterface;

MATCHER_P2(DoubleRangeIs, min_val, max_val, "") {
  return arg.min == min_val && arg.max == max_val;
}

MATCHER_P(HasRanges, expected_ranges, "") {
  if (!arg.ranges) {
    *result_listener << "config ranges are not set";
    return false;
  }

  const auto ranges_matcher =
      AllOf(Field(&SE2VelocityRanges::lin_vel_x,
                  DoubleRangeIs(expected_ranges.lin_vel_x.min, expected_ranges.lin_vel_x.max)),
            Field(&SE2VelocityRanges::lin_vel_y,
                  DoubleRangeIs(expected_ranges.lin_vel_y.min, expected_ranges.lin_vel_y.max)),
            Field(&SE2VelocityRanges::ang_vel_z,
                  DoubleRangeIs(expected_ranges.ang_vel_z.min, expected_ranges.ang_vel_z.max)));

  if (!Matches(ranges_matcher)(arg.ranges.value())) {
    *result_listener << "ranges do not match";
    return false;
  }

  return true;
}

const std::string model_path = (std::filesystem::path(TEST_DATA_DIR) / "test_export.onnx").string();

HeightScan createTestHeightScan(int size) {
  return HeightScan{
      .layers =
          std::unordered_map<std::string, std::vector<double>>{
              {"height", std::vector<double>(size)},
              {"r", std::vector<double>(size)},
              {"g", std::vector<double>(size)},
              {"b", std::vector<double>(size)},
          },
  };
};
auto kHeightScanData = createTestHeightScan(4);
auto kTrailScanData = createTestHeightScan(8);
auto kRangeImageData = std::vector<double>{0, 0, 0, 0};
auto kDepthImageData = std::vector<double>{0, 0, 0, 0};
const auto kPositionData = std::make_optional(Position{0, 0, 0});
const auto kQuaternionData = std::make_optional(Quaternion{1, 0, 0, 0});  // w x y z
const auto kLinearVelocityData = std::make_optional(LinearVelocity{0, 0, 0});
const auto kAngularVelocityData = std::make_optional(AngularVelocity{0, 0, 0});
const auto kSE3PoseData = std::make_optional(SE3Pose{
    .position{0, 0, 0},
    .orientation{1, 0, 0, 0},
});

const SE2VelocityRanges kRanges{
    .lin_vel_x = {.min = -1.5, .max = 1.5},
    .lin_vel_y = {.min = -0.75, .max = 0.75},
    .ang_vel_z = {.min = -2.5, .max = 2.5},
};

class OnnxControllerTest : public ::testing::Test {
 protected:
  NiceMock<MockRobotStateInterface> state_mock_;
  NiceMock<MockCommandInterface> command_mock_;
  NiceMock<MockDataCollectionInterface> data_collection_mock_;
  OnnxRLController oc_{state_mock_, command_mock_, data_collection_mock_};

  void SetUp() override {}

  void ExpectInitBase() { ExpectInitBase(state_mock_); }

  void ExpectInitBase(MockRobotStateInterface& state_mock) {
    // base position and three height scans
    EXPECT_CALL(state_mock, initBasePosW()).Times(4).WillRepeatedly(Return(true));
    // base quat and three height scans
    EXPECT_CALL(state_mock, initBaseQuatW()).Times(4).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock, initBaseLinVelB()).WillOnce(Return(true));
    EXPECT_CALL(state_mock, initBaseAngVelB()).WillOnce(Return(true));
  }

  void ExpectInitOutput() { ExpectInitOutput(state_mock_); }

  void ExpectInitOutput(MockRobotStateInterface& state_mock) {
    // Joint outputs from the output joint targets
    EXPECT_CALL(state_mock, initJointOutput("j1")).WillOnce(Return(true));
    EXPECT_CALL(state_mock, initJointOutput("j2")).WillOnce(Return(true));
    // Joint position and velocity inputs
    EXPECT_CALL(state_mock, initJointPosition("j1")).WillOnce(Return(true));
    EXPECT_CALL(state_mock, initJointPosition("j2")).WillOnce(Return(true));
    EXPECT_CALL(state_mock, initJointPosition("j3")).WillOnce(Return(true));
    EXPECT_CALL(state_mock, initJointVelocity("j1")).WillOnce(Return(true));
    EXPECT_CALL(state_mock, initJointVelocity("j2")).WillOnce(Return(true));
    EXPECT_CALL(state_mock, initJointVelocity("j3")).WillOnce(Return(true));

    EXPECT_CALL(state_mock, initSe2Velocity("base_frame")).WillOnce(Return(true));
  }

  void ExpectInitSensors() { ExpectInitSensors(state_mock_); }

  void ExpectInitSensors(MockRobotStateInterface& state_mock) {
    EXPECT_CALL(state_mock, initHeightScan(_, _)).Times(3).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock, initRangeImage(_)).WillOnce(Return(true));
    EXPECT_CALL(state_mock, initDepthImage(_)).WillOnce(Return(true));
    EXPECT_CALL(state_mock, initImuAngularVelocityImu("pelvis")).WillOnce(Return(true));
    EXPECT_CALL(state_mock, initImuOrientationW("torso")).WillOnce(Return(true));
  }

  void ExpectInitBody() { ExpectInitBody(state_mock_); }

  void ExpectInitBody(MockRobotStateInterface& state_mock) {
    EXPECT_CALL(state_mock, initBodyPositionW("box")).WillOnce(Return(true));
    EXPECT_CALL(state_mock, initBodyOrientationW("box")).WillOnce(Return(true));
  }

  void ExpectInitCommands() { ExpectInitCommands(command_mock_); }

  template <typename CommandMock>
  void ExpectInitCommands(CommandMock& command_mock) {
    // SE2 velocity commands from inputs - check range configuration
    EXPECT_CALL(command_mock,
                initSe2Velocity("vel", Field(&SE2VelocityConfig::ranges, Eq(std::nullopt))))
        .WillOnce(Return(true));
    EXPECT_CALL(command_mock, initSe2Velocity("vel_with_range", HasRanges(kRanges)))
        .WillOnce(Return(true));
    EXPECT_CALL(command_mock, initSe3Pose("pose")).WillOnce(Return(true));
    EXPECT_CALL(command_mock, initBooleanSelector("selector")).WillOnce(Return(true));
    EXPECT_CALL(command_mock, initFloatValue("value")).WillOnce(Return(true));
  }

  void ExpectSetOutput() { ExpectSetOutput(state_mock_); }

  void ExpectSetOutput(MockRobotStateInterface& state_mock) {
    EXPECT_CALL(state_mock, setJointPosition("j1", _)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock, setJointPosition("j2", _)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock, setJointVelocity("j1", _)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock, setJointVelocity("j2", _)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock, setJointEffort("j1", _)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock, setJointEffort("j2", _)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock, setJointPGain("j1", 1)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock, setJointPGain("j2", 2)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock, setJointDGain("j1", 0.1)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock, setJointDGain("j2", 0.2)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock, setSe2Velocity("base_frame", _)).WillRepeatedly(Return(true));
  }

  void ExpectReadJointState() { ExpectReadJointState(state_mock_); }

  void ExpectReadJointState(MockRobotStateInterface& state_mock) {
    EXPECT_CALL(state_mock, jointPosition("j1")).WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock, jointPosition("j2")).WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock, jointPosition("j3")).WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock, jointVelocity("j1")).WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock, jointVelocity("j2")).WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock, jointVelocity("j3")).WillRepeatedly(Return(std::make_optional(0)));
  }

  void ExpectReadState() { ExpectReadState(state_mock_); }

  void ExpectReadState(MockRobotStateInterface& state_mock) {
    EXPECT_CALL(state_mock, imuAngularVelocityImu("pelvis")).WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(state_mock, imuOrientationW("torso")).WillRepeatedly(Return(kQuaternionData));
    EXPECT_CALL(state_mock, bodyPositionW("box")).WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(state_mock, bodyOrientationW("box")).WillRepeatedly(Return(kQuaternionData));
    EXPECT_CALL(state_mock, heightScan("trail", _, _, _))
        .WillRepeatedly(Return(std::make_optional(&kTrailScanData)));
    EXPECT_CALL(state_mock, heightScan("one", _, _, _))
        .WillRepeatedly(Return(std::make_optional(&kHeightScanData)));
    EXPECT_CALL(state_mock, heightScan("two", _, _, _))
        .WillRepeatedly(Return(std::make_optional(&kHeightScanData)));
    EXPECT_CALL(state_mock, basePosW()).WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(state_mock, baseQuatW()).WillRepeatedly(Return(kQuaternionData));
    EXPECT_CALL(state_mock, baseLinVelB()).WillRepeatedly(Return(kLinearVelocityData));
    EXPECT_CALL(state_mock, baseAngVelB()).WillRepeatedly(Return(kAngularVelocityData));
    EXPECT_CALL(state_mock, rangeImage())
        .WillRepeatedly(Return(std::make_optional(&kRangeImageData)));
    EXPECT_CALL(state_mock, depthImage())
        .WillRepeatedly(Return(std::make_optional(&kDepthImageData)));
  }

  void ExpectReadCommands() { ExpectReadCommands(command_mock_); }

  template <typename CommandMock>
  void ExpectReadCommands(CommandMock& command_mock) {
    EXPECT_CALL(command_mock, se2Velocity("vel")).WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(command_mock, se2Velocity("vel_with_range")).WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(command_mock, se3Pose("pose")).WillRepeatedly(Return(kSE3PoseData));
    EXPECT_CALL(command_mock, booleanSelector("selector"))
        .WillRepeatedly(Return(std::make_optional(true)));
    EXPECT_CALL(command_mock, floatValue("value"))
        .WillRepeatedly(Return(std::make_optional(1.23f)));
  }
};

TEST_F(OnnxControllerTest, LoadFails) {
  StrictMock<MockRobotStateInterface> strict_state_mock;
  StrictMock<MockCommandInterface> strict_command_mock;
  NiceMock<MockDataCollectionInterface> nice_data_collection_mock;
  OnnxRLController oc(strict_state_mock, strict_command_mock, nice_data_collection_mock);

  ASSERT_FALSE(oc.load(""));
}

TEST_F(OnnxControllerTest, InitWithoutLoad) {
  StrictMock<MockRobotStateInterface> strict_state_mock;
  StrictMock<MockCommandInterface> strict_command_mock;
  NiceMock<MockDataCollectionInterface> nice_data_collection_mock;
  OnnxRLController oc(strict_state_mock, strict_command_mock, nice_data_collection_mock);

  EXPECT_FALSE(oc.init(false));
}

TEST_F(OnnxControllerTest, Update) {
  ExpectInitBase();
  ExpectInitOutput();
  ExpectInitSensors();
  ExpectInitCommands();
  ExpectInitBody();

  // Initialize
  ASSERT_TRUE(oc_.load(model_path));
  ASSERT_TRUE(oc_.init(false));
  EXPECT_EQ(oc_.updateRate(), 10);

  ExpectReadJointState();
  ExpectReadState();
  ExpectReadCommands();

  ExpectSetOutput();

  const uint64_t t0_us = 0;
  ASSERT_TRUE(oc_.update(t0_us));
}

TEST_F(OnnxControllerTest, ReadJointFailure) {
  ExpectInitBase();
  ExpectInitOutput();
  ExpectInitSensors();
  ExpectInitCommands();
  ExpectInitBody();

  ASSERT_TRUE(oc_.load(model_path));
  ASSERT_TRUE(oc_.init(false));

  EXPECT_CALL(state_mock_, jointPosition("j1")).WillRepeatedly(Return(std::nullopt));
  EXPECT_CALL(state_mock_, jointPosition("j2")).WillRepeatedly(Return(std::make_optional(0)));
  EXPECT_CALL(state_mock_, jointPosition("j3")).WillRepeatedly(Return(std::make_optional(0)));
  EXPECT_CALL(state_mock_, jointVelocity("j1")).WillRepeatedly(Return(std::make_optional(0)));
  EXPECT_CALL(state_mock_, jointVelocity("j2")).WillRepeatedly(Return(std::make_optional(0)));
  EXPECT_CALL(state_mock_, jointVelocity("j3")).WillRepeatedly(Return(std::make_optional(0)));

  ExpectReadState();
  ExpectReadCommands();

  ExpectSetOutput();

  EXPECT_FALSE(oc_.update(0));
}

TEST_F(OnnxControllerTest, WriteJointFailure) {
  ExpectInitBase();
  ExpectInitOutput();
  ExpectInitSensors();
  ExpectInitCommands();
  ExpectInitBody();

  ASSERT_TRUE(oc_.load(model_path));
  ASSERT_TRUE(oc_.init(false));

  ExpectReadJointState();
  ExpectReadState();
  ExpectReadCommands();

  // j1 write fails
  EXPECT_CALL(state_mock_, setJointPosition("j1", _)).WillOnce(Return(false));
  EXPECT_CALL(state_mock_, setJointPosition("j2", _)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setJointVelocity(_, _)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setJointEffort(_, _)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setJointPGain(_, _)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setJointDGain(_, _)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setSe2Velocity(_, _)).WillRepeatedly(Return(true));

  EXPECT_FALSE(oc_.update(0));
}

// ========== Extensibility Test Classes ==========

// Simple mock interface for testing extensibility pattern
class CustomInterface {
 public:
  MOCK_METHOD(bool, initExtensibleCommand, (const std::string& command_name), ());
  MOCK_METHOD(std::optional<std::vector<double>>, extensibleCommand,
              (const std::string& command_name), (const));
};

// Custom input component that calls the extended command interface
class CustomInput : public Input {
 public:
  CustomInput(const std::string& key, const std::string& command_name,
              CustomInterface* custom_interface)
      : key_(key), command_name_(command_name), custom_interface_(*custom_interface) {}

  bool init(RobotStateInterface& state, CommandInterface& command) override {
    return custom_interface_.initExtensibleCommand(command_name_);
  }

  bool read(OnnxRuntime& runtime, const RobotStateInterface& state,
            const CommandInterface& command) override {
    auto maybe_data = custom_interface_.extensibleCommand(command_name_);
    if (!maybe_data.has_value()) return false;
    auto maybe_buffer = runtime.inputBuffer<float>(key_);
    if (!maybe_buffer.has_value()) return false;
    std::copy(maybe_data->begin(), maybe_data->end(), maybe_buffer->begin());
    return true;
  }

 private:
  std::string key_;
  std::string command_name_;
  CustomInterface& custom_interface_;
};

// Custom matcher that recognizes and handles custom.* inputs
class CustomMatcher : public Matcher {
 public:
  explicit CustomMatcher(CustomInterface* custom_interface)
      : custom_interface_(*custom_interface) {}

  bool matches(const Match& maybe_match) override {
    // Use regex pattern like other matchers to extract name
    std::regex pattern = std::regex("custom\\.([a-zA-Z0-9_]+)");
    std::smatch match;
    if (std::regex_match(maybe_match.name, match, pattern) && match.size() > 1) {
      found_matches_[match[1].str()] = maybe_match;
      return true;
    }
    return false;
  }

  std::vector<std::unique_ptr<Input>> createInputs() const override {
    std::vector<std::unique_ptr<Input>> inputs;
    for (const auto& [command_name, found_match] : found_matches_) {
      // Use the extracted command name directly from regex match
      inputs.push_back(
          std::make_unique<CustomInput>(found_match.name, command_name, &custom_interface_));
    }
    return inputs;
  }

 private:
  CustomInterface& custom_interface_;
};

TEST_F(OnnxControllerTest, ExtensibilityWithCustomMatcher) {
  NiceMock<MockRobotStateInterface> state_mock_;
  NiceMock<MockDataCollectionInterface> data_collection_mock_;
  NiceMock<MockCommandInterface> command_mock_;
  StrictMock<CustomInterface> custom_mock_;

  std::vector<double> test_extensible_data = {1.5, 2.5, 3.5};

  ExpectInitBase(state_mock_);
  ExpectInitOutput(state_mock_);
  ExpectInitSensors(state_mock_);
  ExpectInitBody(state_mock_);
  ExpectInitCommands(command_mock_);

  EXPECT_CALL(custom_mock_, initExtensibleCommand("extensible_data")).WillOnce(Return(true));

  OnnxRLController custom_oc(state_mock_, command_mock_, data_collection_mock_);

  ASSERT_TRUE(custom_oc.load(model_path));

  auto custom_matcher = std::make_unique<CustomMatcher>(&custom_mock_);
  Match test_match{"custom.extensible_data", std::nullopt};
  ASSERT_TRUE(custom_matcher->matches(test_match));

  custom_oc.context().registerMatcher(std::move(custom_matcher));

  ASSERT_TRUE(custom_oc.init(false));

  ExpectReadJointState(state_mock_);
  ExpectReadState(state_mock_);

  ExpectReadCommands(command_mock_);

  // Expect the custom command read method to be called
  EXPECT_CALL(custom_mock_, extensibleCommand("extensible_data"))
      .WillOnce(Return(std::make_optional(test_extensible_data)));

  ExpectSetOutput(state_mock_);

  EXPECT_TRUE(custom_oc.update(0));
}

}  // namespace rai::cs::control::common::onnx
