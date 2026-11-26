/**
 * @file tests/unit/platform/test_windows_environment_block.cpp
 * @brief Windows process environment serialization contracts.
 */
#include <gtest/gtest.h>

#include "src/platform/windows/misc.h"
#include "src/platform/windows/process_environment.h"

#include <boost/process/v1/environment.hpp>
#include <cstring>
#include <iterator>
#include <string>

namespace {
  namespace bp = boost::process::v1;

  TEST(WindowsEnvironmentBlockTest, EmptyEnvironmentRetainsOneExplicitAndOneImplicitNul) {
    bp::environment env;

    const auto block = platf::create_environment_block(env);

    EXPECT_EQ(block, std::wstring(1, L'\0'));
    EXPECT_EQ(block.data()[block.size()], L'\0');
  }

  TEST(WindowsEnvironmentBlockTest, RetainsOrderEmptyValuesAndEqualsWithoutExtraTerminators) {
    bp::environment env;
    env.emplace("Z_LAST", "one=two");
    env.emplace("A_FIRST", "");
    env.emplace("Middle", " value with spaces\t");
    // The literal's implicit NUL is the explicit final block terminator in the result.
    const wchar_t expected[] = L"Z_LAST=one=two\0A_FIRST=\0Middle= value with spaces\t\0";

    const auto block = platf::create_environment_block(env);

    EXPECT_EQ(block, std::wstring(expected, std::size(expected)));
    EXPECT_EQ(block.data()[block.size()], L'\0');
    EXPECT_EQ(env.at("Z_LAST").to_string(), "one=two");
    EXPECT_EQ(env.size(), 3u);
  }

  TEST(WindowsEnvironmentBlockTest, ConvertsUtf8NamesAndSupplementaryUnicodeValues) {
    bp::environment env;
    env.emplace("na\xc3\xafve", "caf\xc3\xa9=\xf0\x9f\x8c\x8d");
    env.emplace("\xe6\xb8\xa9\xe5\xba\xa6", "\xe9\x9b\xaa");
    const wchar_t expected[] = L"na\u00efve=caf\u00e9=\U0001f30d\0\u6e29\u5ea6=\u96ea\0";

    const auto block = platf::create_environment_block(env);

    EXPECT_EQ(block, std::wstring(expected, std::size(expected)));
  }

  TEST(WindowsEnvironmentBlockTest, MalformedUtf8RetainsTheExistingConversionAuthority) {
    bp::environment env;
    const std::string invalid_utf8 {static_cast<char>(0xc3), '('};
    env.emplace("INVALID", invalid_utf8);
    // Conversion policy, including replacement or empty-on-error behavior, belongs to from_utf8.
    auto expected = L"INVALID=" + platf::from_utf8(invalid_utf8);
    expected.append(2, L'\0');

    EXPECT_EQ(platf::create_environment_block(env), expected);
  }

  TEST(WindowsEnvironmentBlockTest, LargeEnvironmentMatchesTheOriginalUtf8BlockBytes) {
    bp::environment env;
    std::size_t native_block_size = 1;  // Final block NUL, as in the original builder.
    for (unsigned index = 0; index < 32; ++index) {
      const auto name = "LARGE_" + std::to_string(index);
      const std::string value(65'536, static_cast<char>('A' + index % 26));
      env.emplace(name, value);
      native_block_size += name.size() + 1 + value.size() + 1;
    }
    // Boost already owns the ordered UTF-8 NAME=VALUE\0...\0 representation. Converting that
    // whole block is an independent byte baseline for the old per-entry construction.
    const std::string native_block(env.native_handle(), native_block_size);
    const auto expected = platf::from_utf8(native_block);

    const auto block = platf::create_environment_block(env);

    ASSERT_GT(block.size(), 2'000'000u);
    ASSERT_EQ(block.size(), expected.size());
    EXPECT_EQ(std::memcmp(block.data(), expected.data(), block.size() * sizeof(wchar_t)), 0);
    EXPECT_EQ(std::string(env.native_handle(), native_block_size), native_block);
    EXPECT_EQ(block[block.size() - 2], L'\0');
    EXPECT_EQ(block.back(), L'\0');
    EXPECT_EQ(block.data()[block.size()], L'\0');
  }
}  // namespace
