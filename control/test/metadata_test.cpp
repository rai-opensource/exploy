// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "metadata.hpp"

#include <gtest/gtest.h>

namespace exploy::control::metadata {

// ========== Version::toString ==========

TEST(VersionTest, ToString) {
  EXPECT_EQ((Version{1, 2, 3}.toString()), "1.2.3");
  EXPECT_EQ((Version{0, 0, 0}.toString()), "0.0.0");
  EXPECT_EQ((Version{10, 20, 30}.toString()), "10.20.30");
}

// ========== Version::operator<= ==========

TEST(VersionTest, ComparisonEqualVersions) {
  EXPECT_TRUE((Version{1, 2, 3} <= Version{1, 2, 3}));
  EXPECT_TRUE((Version{0, 0, 0} <= Version{0, 0, 0}));
}

TEST(VersionTest, ComparisonMajorDiffers) {
  EXPECT_TRUE((Version{0, 9, 9} <= Version{1, 0, 0}));
  EXPECT_FALSE((Version{2, 0, 0} <= Version{1, 9, 9}));
}

TEST(VersionTest, ComparisonMinorDiffers) {
  EXPECT_TRUE((Version{1, 1, 9} <= Version{1, 2, 0}));
  EXPECT_FALSE((Version{1, 3, 0} <= Version{1, 2, 9}));
}

TEST(VersionTest, ComparisonPatchDiffers) {
  EXPECT_TRUE((Version{1, 2, 2} <= Version{1, 2, 3}));
  EXPECT_FALSE((Version{1, 2, 4} <= Version{1, 2, 3}));
}

// ========== parseVersion ==========

TEST(ParseVersionTest, ValidVersion) {
  auto v = parseVersion("1.2.3");
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->major, 1);
  EXPECT_EQ(v->minor, 2);
  EXPECT_EQ(v->patch, 3);
}

TEST(ParseVersionTest, ZeroVersion) {
  auto v = parseVersion("0.0.0");
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->major, 0);
  EXPECT_EQ(v->minor, 0);
  EXPECT_EQ(v->patch, 0);
}

TEST(ParseVersionTest, LargeNumbers) {
  auto v = parseVersion("10.20.30");
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->major, 10);
  EXPECT_EQ(v->minor, 20);
  EXPECT_EQ(v->patch, 30);
}

TEST(ParseVersionTest, EmptyString) {
  EXPECT_FALSE(parseVersion("").has_value());
}

TEST(ParseVersionTest, MissingPatch) {
  EXPECT_FALSE(parseVersion("1.2").has_value());
}

TEST(ParseVersionTest, ExtraComponent) {
  EXPECT_FALSE(parseVersion("1.2.3.4").has_value());
}

TEST(ParseVersionTest, NonNumericComponents) {
  EXPECT_FALSE(parseVersion("a.b.c").has_value());
}

TEST(ParseVersionTest, NegativeComponent) {
  EXPECT_FALSE(parseVersion("1.2.-3").has_value());
}

TEST(ParseVersionTest, WithVPrefix) {
  EXPECT_FALSE(parseVersion("v1.2.3").has_value());
}

// ========== checkExployVersion ==========

TEST(CheckExployVersionTest, MissingMetadata) {
  EXPECT_FALSE(checkExployVersion(std::nullopt));
}

TEST(CheckExployVersionTest, InvalidJson) {
  EXPECT_FALSE(checkExployVersion("not_valid_json"));
}

TEST(CheckExployVersionTest, JsonNotAString) {
  // JSON-encoded number instead of a string
  EXPECT_FALSE(checkExployVersion("42"));
}

TEST(CheckExployVersionTest, InvalidSemver) {
  // JSON-encoded string that is not a valid semver
  EXPECT_FALSE(checkExployVersion("\"abc\""));
}

TEST(CheckExployVersionTest, VersionBelowMinimum) {
  // kMinSupportedExployVersion is 0.1.0, so 0.0.9 is below the minimum
  EXPECT_FALSE(checkExployVersion("\"0.0.9\""));
}

TEST(CheckExployVersionTest, VersionAboveMaximum) {
  // kMaxSupportedExployVersion is 0.1.0, so 0.2.0 is above the maximum
  EXPECT_FALSE(checkExployVersion("\"0.2.0\""));
}

TEST(CheckExployVersionTest, SupportedVersion) {
  // kMinSupportedExployVersion == kMaxSupportedExployVersion == 0.1.0
  EXPECT_TRUE(checkExployVersion("\"0.1.0\""));
}

}  // namespace exploy::control::metadata
