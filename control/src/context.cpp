// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "exploy/context.hpp"
#include "exploy/logging_utils.hpp"
#include "exploy/metadata.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cmath>
#include <string>
#include <vector>

namespace exploy::control {

namespace {

std::optional<int> parseUpdateRate(OnnxRuntime& onnx_model) {
  const auto maybe_update_rate = onnx_model.getCustomMetadata("update_rate");
  if (!maybe_update_rate.has_value()) {
    LOG(ERROR, "Failed to get update_rate metadata");
    return std::nullopt;
  }
  return static_cast<int>(std::stod(maybe_update_rate.value()));
}

// Parse the `base_names` metadata into a map of {robot_name: base_name} for use in matchers.
std::unordered_map<std::string, std::string> parseBaseNames(const OnnxRuntime& onnx_model) {
  std::unordered_map<std::string, std::string> base_names;
  const auto maybe_base_names = onnx_model.getCustomMetadata("base_names");
  if (!maybe_base_names.has_value()) return base_names;
  try {
    auto json_base_names = json::parse(maybe_base_names.value());
    for (auto it = json_base_names.begin(); it != json_base_names.end(); ++it) {
      base_names[it.key()] = it.value().get<std::string>();
    }
  } catch (const json::exception& e) {
    LOG_STREAM(ERROR, "Failed to parse base_names metadata: " << e.what());
  }
  return base_names;
}

}  // namespace

// Registration methods
void OnnxContext::registerMatcher(std::unique_ptr<Matcher> matcher) {
  matchers_.push_back(std::move(matcher));
}

void OnnxContext::registerGroupMatcher(std::unique_ptr<GroupMatcher> matcher) {
  group_matchers_.push_back(std::move(matcher));
}

bool OnnxContext::createContext(OnnxRuntime& onnx_model, bool strict) {
  // Reset the components and matchers.
  inputs_.clear();
  outputs_.clear();
  for (auto& m : matchers_) m->resetMatcher();
  for (auto& m : group_matchers_) m->resetMatcher();

  // Check if ONNX model is properly loaded before accessing its properties
  if (!onnx_model.isInitialized()) {
    LOG_STREAM(ERROR, "ONNX model not properly loaded, skipping context creation");
    return false;
  }

  // Matchers should now be registered before calling createContext
  if (matchers_.empty() && group_matchers_.empty()) {
    LOG_STREAM(ERROR, "No matchers registered. Please register matchers before creating context.");
    return false;
  }

  if (!metadata::checkExployVersion(onnx_model.getCustomMetadata("exploy_version"))) return false;

  std::optional<int> maybe_update_rate = parseUpdateRate(onnx_model);
  if (!maybe_update_rate.has_value()) return false;
  update_rate_ = maybe_update_rate.value();

  base_names_ = parseBaseNames(onnx_model);

  // Match metadata input names to registered matchers to create components. Each input must match
  // exactly one matcher.
  for (const auto& input_name : onnx_model.inputNames()) {
    Match maybe_match{
        .name = input_name,
        .metadata = onnx_model.getCustomMetadata(input_name),
        .base_names = base_names_,
    };
    std::vector<std::string> matched_by;
    for (auto& group_matcher : group_matchers_) {
      if (group_matcher->matches(maybe_match)) matched_by.push_back(group_matcher->getName());
    }
    for (auto& matcher : matchers_) {
      if (matcher->matches(maybe_match)) matched_by.push_back(matcher->getName());
    }
    if (matched_by.empty()) {
      LOG_STREAM(WARNING, fmt::format("No matcher found for input '{}'", input_name));
      if (strict) {
        return false;
      }
    } else if (matched_by.size() > 1) {
      LOG_STREAM(ERROR, fmt::format("Multiple matchers ({}) found for input '{}': [{}]",
                                    matched_by.size(), input_name, fmt::join(matched_by, ", ")));
      return false;
    }
  }

  // Match metadata output names to registered matchers to create components. Each output must match
  // exactly one matcher.
  for (const auto& output_name : onnx_model.outputNames()) {
    if (output_name == "actions" || output_name == "obs") continue;
    Match maybe_match{
        .name = output_name,
        .metadata = onnx_model.getCustomMetadata(output_name),
        .base_names = base_names_,
    };
    std::vector<std::string> matched_by;
    for (auto& group_matcher : group_matchers_) {
      if (group_matcher->matches(maybe_match)) matched_by.push_back(group_matcher->getName());
    }
    for (auto& matcher : matchers_) {
      if (matcher->matches(maybe_match)) matched_by.push_back(matcher->getName());
    }
    if (matched_by.empty()) {
      LOG_STREAM(WARNING, fmt::format("No matcher found for output '{}'", output_name));
      if (strict) return false;
    } else if (matched_by.size() > 1) {
      LOG_STREAM(ERROR, fmt::format("Multiple matchers ({}) found for output '{}': [{}]",
                                    matched_by.size(), output_name, fmt::join(matched_by, ", ")));
      return false;
    }
  }

  for (auto& group_matcher : group_matchers_) {
    group_matcher->populateGroupMetadata([&onnx_model](const std::string& name) {
      return onnx_model.getCustomMetadata(name);
    });
  }

  auto collect_components = [](auto& components, auto& matchers, auto creator_fn) {
    for (auto& matcher : matchers) {
      auto items = (matcher.get()->*creator_fn)();
      components.insert(components.end(), std::make_move_iterator(items.begin()),
                        std::make_move_iterator(items.end()));
    }
  };

  collect_components(inputs_, matchers_, &Matcher::createInputs);
  collect_components(inputs_, group_matchers_, &GroupMatcher::createInputs);
  collect_components(outputs_, matchers_, &Matcher::createOutputs);
  collect_components(outputs_, group_matchers_, &GroupMatcher::createOutputs);

  return true;
}

}  // namespace exploy::control
