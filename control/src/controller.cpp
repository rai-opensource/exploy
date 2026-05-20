// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#include "exploy/controller.hpp"
#include "exploy/logging_utils.hpp"

#include <fmt/format.h>

#include <cassert>
#include <chrono>
#include <span>

namespace exploy::control {

OnnxRLController::OnnxRLController(RobotStateInterface& state, CommandInterface& command,
                                   DataCollectionInterface& data_collection)
    : state_(state), command_(command), data_collection_(data_collection) {}

bool OnnxRLController::create(const std::string& onnx_model_path, bool register_default_matchers) {
  if (!default_matchers_registered_ && register_default_matchers) {
    default_matchers_registered_ = true;
    context_.registerMatcher(std::make_unique<StepCountMatcher>());
    context_.registerMatcher(std::make_unique<BasePositionMatcher>());
    context_.registerMatcher(std::make_unique<BaseOrientationMatcher>());
    context_.registerMatcher(std::make_unique<BaseLinearVelocityMatcher>());
    context_.registerMatcher(std::make_unique<BaseAngularVelocityMatcher>());
    context_.registerMatcher(std::make_unique<IMULinearVelocityMatcher>());
    context_.registerMatcher(std::make_unique<IMUAngularVelocityMatcher>());
    context_.registerMatcher(std::make_unique<IMUOrientationMatcher>());
    context_.registerMatcher(std::make_unique<BodyPositionMatcher>());
    context_.registerMatcher(std::make_unique<BodyOrientationMatcher>());
    context_.registerMatcher(std::make_unique<BodyLinearVelocityMatcher>());
    context_.registerMatcher(std::make_unique<BodyAngularVelocityMatcher>());
    context_.registerMatcher(std::make_unique<CommandSE3PoseMatcher>());
    context_.registerMatcher(std::make_unique<CommandSE2VelocityMatcher>());
    context_.registerMatcher(std::make_unique<CommandBooleanMatcher>());
    context_.registerMatcher(std::make_unique<CommandFloatMatcher>());
    context_.registerMatcher(std::make_unique<CommandJointPositionMatcher>());
    context_.registerMatcher(std::make_unique<SE2VelocityMatcher>());
    context_.registerGroupMatcher(std::make_unique<JointMatcher>());
    context_.registerGroupMatcher(std::make_unique<JointTargetMatcher>());
    context_.registerGroupMatcher(std::make_unique<HeightScanMatcher>());
    context_.registerGroupMatcher(std::make_unique<SphericalImageMatcher>());
    context_.registerGroupMatcher(std::make_unique<PinholeImageMatcher>());
    context_.registerGroupMatcher(std::make_unique<MemoryMatcher>());
  }

  if (!onnx_model_.initialize(onnx_model_path)) {
    LOG_STREAM(ERROR, "Error creating OnnxEvaluator from policy: " << onnx_model_path);
    return false;
  }
  if (!context_.createContext(onnx_model_)) {
    LOG_STREAM(ERROR, "Error creating context.");
    return false;
  }
  return true;
}

bool OnnxRLController::init(bool enable_data_collection, WorkerMode mode) {
  if (!onnx_model_.isInitialized()) {
    LOG_STREAM(ERROR, "ONNX model is not initialized.");
    return false;
  }

  for (const auto& input : context_.getInputs()) {
    if (!input->init(state_, command_)) {
      LOG_STREAM(ERROR, "Error initializing input.");
      return false;
    }
  }
  for (const auto& output : context_.getOutputs()) {
    if (!output->init(state_, command_)) {
      LOG_STREAM(ERROR, "Error initializing output.");
      return false;
    }
  }

  auto rate = static_cast<double>(context_.updateRate());
  if (rate <= 0) {
    LOG_STREAM(ERROR, "Invalid update rate: " << rate << " Hz. Must be > 0.");
    return false;
  }
  if (mode == WorkerMode::ASYNC) {
    worker_ = std::make_unique<AsyncWorker>(rate);
  } else {
    worker_ = std::make_unique<SyncWorker>(rate);
  }
  auto success = worker_->setCallbacks(
      [this]() {
        return readInputs();
      },
      [this]() {
        auto start = std::chrono::high_resolution_clock::now();
        bool success = onnx_model_.evaluate();
        auto end = std::chrono::high_resolution_clock::now();
        inference_duration_s_ = std::chrono::duration<double>(end - start).count();
        return success;
      },
      [this]() {
        return writeOutputs();
      });
  if (!success) {
    LOG(ERROR, "Failed to set worker callbacks.");
    return false;
  }

  if (enable_data_collection) {
    for (const auto& name : onnx_model_.inputNames()) {
      auto maybe_buffer = onnx_model_.inputBuffer<float>(name);
      if (!maybe_buffer.has_value()) continue;
      if (!data_collection_.registerDataSource("model/input/" + name, maybe_buffer.value())) {
        LOG_STREAM(ERROR, "Registering data source for model/input/" << name << " failed.");
        return false;
      }
    }
    for (const auto& name : onnx_model_.outputNames()) {
      auto maybe_buffer = onnx_model_.outputBuffer<float>(name);
      if (!maybe_buffer.has_value()) continue;
      if (!data_collection_.registerDataSource("model/output/" + name, maybe_buffer.value())) {
        LOG_STREAM(ERROR, "Registering data source for model/output/" << name << " failed.");
        return false;
      }
    }
    if (!data_collection_.registerDataSource("model/inference/duration_s", inference_duration_s_)) {
      LOG(ERROR, "Registering data source for inference duration failed.");
      return false;
    }
  }

  reset();

  return true;
}

void OnnxRLController::reset() {
  onnx_model_.resetBuffers();
  if (worker_) worker_->reset();
}

bool OnnxRLController::readInputs() {
  for (const auto& input : context_.getInputs()) {
    if (!input->read(onnx_model_, state_, command_)) {
      LOG_STREAM(ERROR, "Failed to read input");
      return false;
    }
  }
  return true;
}

bool OnnxRLController::writeOutputs() {
  for (const auto& output : context_.getOutputs()) {
    if (!output->write(onnx_model_, state_, command_)) {
      LOG_STREAM(ERROR, "Failed to write output");
      return false;
    }
  }
  return true;
}

bool OnnxRLController::update(uint64_t time_us) {
  if (!worker_) return false;
  if (!worker_->update(time_us)) return false;

  if (!data_collection_.collectData(time_us)) {
    LOG(WARN, "Data collection failed.");
  }

  return true;
}

}  // namespace exploy::control
