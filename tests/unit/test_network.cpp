/**
 * @file tests/unit/test_network.cpp
 * @brief Test src/network.*
 */
#include "../tests_common.h"

#include <src/network.h>

struct MdnsInstanceNameTest: testing::TestWithParam<std::tuple<std::string, std::string>> {};

TEST_P(MdnsInstanceNameTest, Run) {
  auto [input, expected] = GetParam();
  ASSERT_EQ(net::mdns_instance_name(input), expected);
}

INSTANTIATE_TEST_SUITE_P(
  MdnsInstanceNameTests,
  MdnsInstanceNameTest,
  testing::Values(
    std::make_tuple("shortname-123", "shortname-123"),
    std::make_tuple("space 123", "space-123"),
    std::make_tuple("hostname.domain.test", "hostname"),
    std::make_tuple("&", PROJECT_NAME),
    std::make_tuple("", PROJECT_NAME),
    std::make_tuple("😁", PROJECT_NAME),
    std::make_tuple(std::string(128, 'a'), std::string(63, 'a'))
  )
);

class BindAddressTest: public ::testing::Test {
protected:
  std::string original_bind_address;

  void SetUp() override {
    original_bind_address = config::sunshine.bind_address;
  }

  void TearDown() override {
    config::sunshine.bind_address = std::move(original_bind_address);
  }
};

TEST_F(BindAddressTest, UsesWildcardWhenNotConfigured) {
  config::sunshine.bind_address.clear();
  ASSERT_TRUE(net::get_bind_address(net::IPV4));
  EXPECT_EQ(*net::get_bind_address(net::IPV4), "0.0.0.0");
  ASSERT_TRUE(net::get_bind_address(net::BOTH));
  EXPECT_EQ(*net::get_bind_address(net::BOTH), "::");
}

TEST_F(BindAddressTest, UsesConfiguredAddress) {
  config::sunshine.bind_address = "192.168.1.100";
  ASSERT_TRUE(net::get_bind_address(net::IPV4));
  EXPECT_EQ(*net::get_bind_address(net::IPV4), "192.168.1.100");

  config::sunshine.bind_address = "2001:db8::1";
  ASSERT_TRUE(net::get_bind_address(net::BOTH));
  EXPECT_EQ(*net::get_bind_address(net::BOTH), "2001:db8::1");

  // Invalid explicit restrictions must fail closed instead of resolving to the wildcard.
  config::sunshine.bind_address = "not-an-address";
  EXPECT_FALSE(net::get_bind_address(net::IPV4));

  config::sunshine.bind_address = "127.0.0.1";
  EXPECT_FALSE(net::get_bind_address(net::BOTH));
}

TEST_F(BindAddressTest, ValidatesSyntaxAndAddressFamily) {
  EXPECT_TRUE(net::is_valid_bind_address("", net::IPV4));
  EXPECT_TRUE(net::is_valid_bind_address("127.0.0.1", net::IPV4));
  EXPECT_FALSE(net::is_valid_bind_address("::1", net::IPV4));
  EXPECT_TRUE(net::is_valid_bind_address("::1", net::BOTH));
  EXPECT_FALSE(net::is_valid_bind_address("127.0.0.1", net::BOTH));
  EXPECT_FALSE(net::is_valid_bind_address("not-an-address", net::IPV4));
}
