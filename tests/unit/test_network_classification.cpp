/**
 * @file tests/unit/test_network_classification.cpp
 * @brief Preserve the endpoint classifications used by Web UI authorization.
 */
#include "../tests_common.h"
#include "src/network.h"

#include <array>

namespace {
  namespace ip = boost::asio::ip;

  net::net_e classify_endpoint(const ip::address &address) {
    return net::from_address(address);
  }

  struct address_case_t {
    const char *text;
    net::net_e expected;
  };
}  // namespace

TEST(NetworkClassificationTest, Ipv4BoundariesAndMappedFormsKeepTheirExistingPolicies) {
  // These explicit expectations preserve hosts() membership, including the network and
  // broadcast exclusions. They are intentionally not generated from the production ranges.
  constexpr std::array cases {
    address_case_t {"126.255.255.255", net::WAN},
    address_case_t {"127.0.0.0", net::WAN},
    address_case_t {"127.0.0.1", net::PC},
    address_case_t {"127.255.255.254", net::PC},
    address_case_t {"127.255.255.255", net::WAN},
    address_case_t {"128.0.0.0", net::WAN},

    address_case_t {"192.167.255.255", net::WAN},
    address_case_t {"192.168.0.0", net::WAN},
    address_case_t {"192.168.0.1", net::LAN},
    address_case_t {"192.168.255.254", net::LAN},
    address_case_t {"192.168.255.255", net::WAN},
    address_case_t {"192.169.0.0", net::WAN},

    address_case_t {"172.15.255.255", net::WAN},
    address_case_t {"172.16.0.0", net::WAN},
    address_case_t {"172.16.0.1", net::LAN},
    address_case_t {"172.31.255.254", net::LAN},
    address_case_t {"172.31.255.255", net::WAN},
    address_case_t {"172.32.0.0", net::WAN},

    address_case_t {"9.255.255.255", net::WAN},
    address_case_t {"10.0.0.0", net::WAN},
    address_case_t {"10.0.0.1", net::LAN},
    address_case_t {"10.255.255.254", net::LAN},
    address_case_t {"10.255.255.255", net::WAN},
    address_case_t {"11.0.0.0", net::WAN},

    address_case_t {"100.63.255.255", net::WAN},
    address_case_t {"100.64.0.0", net::WAN},
    address_case_t {"100.64.0.1", net::LAN},
    address_case_t {"100.127.255.254", net::LAN},
    address_case_t {"100.127.255.255", net::WAN},
    address_case_t {"100.128.0.0", net::WAN},

    address_case_t {"169.253.255.255", net::WAN},
    address_case_t {"169.254.0.0", net::WAN},
    address_case_t {"169.254.0.1", net::LAN},
    address_case_t {"169.254.255.254", net::LAN},
    address_case_t {"169.254.255.255", net::WAN},
    address_case_t {"169.255.0.0", net::WAN},

    address_case_t {"0.0.0.0", net::WAN},
    address_case_t {"8.8.8.8", net::WAN},
    address_case_t {"224.0.0.1", net::WAN},
    address_case_t {"255.255.255.255", net::WAN},
  };
  for (const auto &item : cases) {
    SCOPED_TRACE(item.text);
    const auto native = ip::make_address_v4(item.text);
    EXPECT_EQ(classify_endpoint(native), item.expected);
    for (const auto scope : {0u, 7u}) {
      SCOPED_TRACE(scope);
      auto mapped = ip::make_address_v6(ip::v4_mapped, native);
      mapped.scope_id(scope);
      EXPECT_EQ(classify_endpoint(mapped), item.expected);
    }
  }
}

TEST(NetworkClassificationTest, Ipv6BoundariesAndScopesKeepTheirExistingPolicies) {
  constexpr std::array cases {
    address_case_t {"::", net::WAN},
    address_case_t {"::1", net::PC},
    address_case_t {"::2", net::WAN},
    address_case_t {"2001:db8::1", net::WAN},
    address_case_t {"ff02::1", net::WAN},
    address_case_t {"ff0e::1", net::WAN},

    address_case_t {"fbff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", net::WAN},
    address_case_t {"fc00::", net::LAN},
    address_case_t {"fc00::1", net::LAN},
    address_case_t {"fdff:ffff:ffff:ffff:ffff:ffff:ffff:fffe", net::LAN},
    address_case_t {"fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", net::LAN},
    address_case_t {"fe00::", net::WAN},

    address_case_t {"fe7f:ffff:ffff:ffff:ffff:ffff:ffff:ffff", net::WAN},
    address_case_t {"fe80::", net::LAN},
    address_case_t {"fe80::1", net::LAN},
    address_case_t {"fe80::ffff:ffff:ffff:fffe", net::LAN},
    address_case_t {"fe80::ffff:ffff:ffff:ffff", net::LAN},
    address_case_t {"fe80:0:0:1::", net::WAN},
    address_case_t {"fe80:0:0:1::1", net::WAN},
    address_case_t {"fe90::1", net::WAN},

    // IPv4-compatible and NAT64 addresses do not use the IPv4 classification policy.
    address_case_t {"::127.0.0.1", net::WAN},
    address_case_t {"::192.168.1.1", net::WAN},
    address_case_t {"64:ff9b::127.0.0.1", net::WAN},
    address_case_t {"64:ff9b::192.168.1.1", net::WAN},
  };
  for (const auto &item : cases) {
    SCOPED_TRACE(item.text);
    for (const auto scope : {0u, 7u}) {
      SCOPED_TRACE(scope);
      auto address = ip::make_address_v6(item.text);
      address.scope_id(scope);
      EXPECT_EQ(classify_endpoint(address), item.expected);
    }
  }
}

TEST(NetworkNormalizationTest, MappedIpv4UsesNativeAddressAndLogTextWithoutScope) {
  for (const auto *text : {"0.0.0.0", "127.0.0.1", "192.168.1.1", "203.0.113.1", "255.255.255.255"}) {
    SCOPED_TRACE(text);
    const auto native = ip::make_address_v4(text);
    EXPECT_EQ(net::normalize_address(native), ip::address {native});
    EXPECT_EQ(net::addr_to_normalized_string(native), text);
    for (const auto scope : {0u, 7u}) {
      SCOPED_TRACE(scope);
      auto mapped = ip::make_address_v6(ip::v4_mapped, native);
      mapped.scope_id(scope);
      const auto normalized = net::normalize_address(mapped);
      ASSERT_TRUE(normalized.is_v4());
      EXPECT_EQ(normalized.to_v4(), native);
      EXPECT_EQ(net::addr_to_normalized_string(mapped), text);
      EXPECT_EQ(mapped.scope_id(), scope);
    }
  }
}

TEST(NetworkNormalizationTest, NativeIpv6KeepsItsBytesScopeAndLogText) {
  for (const auto *text : {"::1", "fc00::1", "fe80::1", "2001:db8::1"}) {
    SCOPED_TRACE(text);
    auto address = ip::make_address_v6(text);
    EXPECT_EQ(net::addr_to_normalized_string(address), text);
    for (const auto scope : {0u, 7u}) {
      SCOPED_TRACE(scope);
      address.scope_id(scope);
      const auto normalized = net::normalize_address(address);
      ASSERT_TRUE(normalized.is_v6());
      EXPECT_EQ(normalized.to_v6().to_bytes(), address.to_bytes());
      EXPECT_EQ(normalized.to_v6().scope_id(), scope);
    }
  }
  auto scoped_link_local = ip::make_address_v6("fe80::1");
  scoped_link_local.scope_id(7);
  EXPECT_EQ(net::addr_to_normalized_string(scoped_link_local), "fe80::1%7");
}
