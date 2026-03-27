// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "exploy/metadata.hpp"

#include <gtest/gtest.h>

namespace exploy::control::metadata {

// ========== Version::toString ==========

TEST(VersionTest, ToString) {
  EXPECT_EQ((Version{1, 2}.toString()), "1.2");
  EXPECT_EQ((Version{0, 0}.toString()), "0.0");
  EXPECT_EQ((Version{10, 20}.toString()), "10.20");
}

// ========== Version::operator<= ==========

TEST(VersionTest, ComparisonEqualVersions) {
  EXPECT_TRUE((Version{1, 2} <= Version{1, 2}));
  EXPECT_TRUE((Version{0, 0} <= Version{0, 0}));
}

TEST(VersionTest, ComparisonMajorDiffers) {
  EXPECT_TRUE((Version{0, 9} <= Version{1, 0}));
  EXPECT_FALSE((Version{2, 0} <= Version{1, 9}));
}

TEST(VersionTest, ComparisonMinorDiffers) {
  EXPECT_TRUE((Version{1, 1} <= Version{1, 2}));
  EXPECT_FALSE((Version{1, 3} <= Version{1, 2}));
}

// ========== parseVersion ==========

TEST(ParseVersionTest, ValidVersion) {
  auto v = parseVersion("1.2.3");
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->major, 1);
  EXPECT_EQ(v->minor, 2);
}

TEST(ParseVersionTest, ZeroVersion) {
  auto v = parseVersion("0.0.0");
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->major, 0);
  EXPECT_EQ(v->minor, 0);
}

TEST(ParseVersionTest, LargeNumbers) {
  auto v = parseVersion("10.20.30");
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->major, 10);
  EXPECT_EQ(v->minor, 20);
}

TEST(ParseVersionTest, EmptyString) {
  EXPECT_FALSE(parseVersion("").has_value());
}

TEST(ParseVersionTest, MissingPatch) {
  auto v = parseVersion("1.2");
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->major, 1);
  EXPECT_EQ(v->minor, 2);
}

TEST(ParseVersionTest, ExtraComponent) {
  auto v = parseVersion("1.2.3.4");
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->major, 1);
  EXPECT_EQ(v->minor, 2);
}

TEST(ParseVersionTest, PythonDevVersion) {
  auto v = parseVersion("0.0.post1.dev96+g1cdbae3db");
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->major, 0);
  EXPECT_EQ(v->minor, 0);
}

TEST(ParseVersionTest, NonNumericComponents) {
  EXPECT_FALSE(parseVersion("a.b.c").has_value());
}

TEST(ParseVersionTest, NegativeComponent) {
  // "1.2.-3" — trailing content after MAJOR.MINOR is ignored
  auto v = parseVersion("1.2.-3");
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->major, 1);
  EXPECT_EQ(v->minor, 2);
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
  constexpr Version min{1, 2};
  EXPECT_FALSE(checkExployVersion("\"1.1.9\"", min));
}

TEST(CheckExployVersionTest, SupportedVersion) {
  constexpr Version min{1, 2};
  EXPECT_TRUE(checkExployVersion("\"1.2.0\"", min));
}

TEST(CheckExployVersionTest, VersionAboveMinimum) {
  constexpr Version min{1, 2};
  EXPECT_TRUE(checkExployVersion("\"1.3.0\"", min));
}

}  // namespace exploy::control::metadata
