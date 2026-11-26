#include "src/platform/windows/audio_endpoint_match.h"

#include <array>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {
  namespace lookup = platf::audio::detail;
  using field_e = lookup::endpoint_match_field_e;

  struct property_t {
    std::wstring text;
    HRESULT status = S_OK;
    VARTYPE type = VT_LPWSTR;
    bool null_string = false;
  };

  struct endpoint_t {
    explicit endpoint_t(std::wstring value):
        id(std::move(value)) {
    }

    property_t &property(field_e field) {
      return properties[static_cast<std::size_t>(field)];
    }

    unsigned reads(field_e field) const {
      return property_reads[static_cast<std::size_t>(field)];
    }

    std::wstring id;
    bool available = true;
    bool property_store_available = true;
    std::array<property_t, 4> properties;
    unsigned id_reads = 0;
    unsigned property_store_opens = 0;
    std::array<unsigned, 4> property_reads {};
  };

  struct reader_t {
    endpoint_t *endpoint;
    bool store_attempted = false;

    std::wstring_view id() const {
      return endpoint->id;
    }

    bool matches(field_e field, std::wstring_view expected) {
      if (!store_attempted) {
        store_attempted = true;
        ++endpoint->property_store_opens;
      }
      if (!endpoint->property_store_available) {
        return false;
      }
      ++endpoint->property_reads[static_cast<std::size_t>(field)];
      const auto &property = endpoint->property(field);
      PROPVARIANT value {};
      value.vt = property.type;
      if (property.type == VT_LPWSTR && !property.null_string) {
        value.pwszVal = const_cast<wchar_t *>(property.text.c_str());
      }
      const auto text = lookup::endpoint_property_string(property.status, value);
      return text && *text == expected;
    }
  };

  std::optional<std::wstring> find(
    std::vector<endpoint_t> &endpoints,
    const lookup::endpoint_match_list_t &matches
  ) {
    return lookup::find_endpoint_id(matches, endpoints.size(), [&endpoints](std::size_t index) -> std::optional<reader_t> {
      auto &endpoint = endpoints[index];
      ++endpoint.id_reads;
      if (!endpoint.available) {
        return std::nullopt;
      }
      return reader_t {&endpoint};
    });
  }
}  // namespace

TEST(WindowsAudioEndpointMatch, ExactIdAvoidsPropertiesAndFurtherEnumeration) {
  std::vector<endpoint_t> endpoints;
  endpoints.emplace_back(L"wanted-id");
  endpoints.emplace_back(L"later-id");

  const auto result = find(endpoints, lookup::match_all_fields(L"wanted-id"));

  EXPECT_EQ(result, L"wanted-id");
  EXPECT_EQ(endpoints[0].id_reads, 1u);
  EXPECT_EQ(endpoints[0].property_store_opens, 0u);
  EXPECT_EQ(endpoints[1].id_reads, 0u);
}

TEST(WindowsAudioEndpointMatch, LaterExactIdOutranksEarlierFriendlyName) {
  std::vector<endpoint_t> endpoints;
  endpoints.emplace_back(L"first-id");
  endpoints[0].property(field_e::device_friendly_name).text = L"wanted";
  endpoints.emplace_back(L"wanted");

  EXPECT_EQ(find(endpoints, lookup::match_all_fields(L"wanted")), L"wanted");
  EXPECT_EQ(endpoints[0].reads(field_e::device_friendly_name), 1u);
  EXPECT_EQ(endpoints[0].reads(field_e::device_description), 0u);
  EXPECT_EQ(endpoints[0].reads(field_e::adapter_friendly_name), 0u);
  EXPECT_EQ(endpoints[1].property_store_opens, 0u);
}

TEST(WindowsAudioEndpointMatch, FieldPriorityOutranksEndpointEnumerationOrder) {
  std::vector<endpoint_t> endpoints;
  endpoints.emplace_back(L"adapter-id");
  endpoints[0].property(field_e::adapter_friendly_name).text = L"wanted";
  endpoints.emplace_back(L"description-id");
  endpoints[1].property(field_e::device_description).text = L"wanted";
  endpoints.emplace_back(L"friendly-id");
  endpoints[2].property(field_e::device_friendly_name).text = L"wanted";

  EXPECT_EQ(find(endpoints, lookup::match_all_fields(L"wanted")), L"friendly-id");
  EXPECT_EQ(endpoints[1].reads(field_e::adapter_friendly_name), 0u);
  EXPECT_EQ(endpoints[2].reads(field_e::device_description), 0u);
  EXPECT_EQ(endpoints[2].reads(field_e::adapter_friendly_name), 0u);
}

TEST(WindowsAudioEndpointMatch, SameFieldKeepsFirstEndpointAndReturnsOwnedId) {
  std::vector<endpoint_t> endpoints;
  endpoints.emplace_back(L"first-id");
  endpoints[0].property(field_e::device_friendly_name).text = L"wanted";
  endpoints.emplace_back(L"second-id");
  endpoints[1].property(field_e::device_friendly_name).text = L"wanted";

  const auto result = find(endpoints, lookup::match_all_fields(L"wanted"));

  EXPECT_EQ(result, L"first-id");
  EXPECT_EQ(endpoints[1].id_reads, 1u);
  EXPECT_EQ(endpoints[1].property_store_opens, 0u);
  endpoints[0].id.assign(L"replaced-id");
  EXPECT_EQ(result, L"first-id");
}

TEST(WindowsAudioEndpointMatch, SteamSearchReadsOnlyAdapterAndStopsAfterFirstMatch) {
  std::vector<endpoint_t> endpoints;
  endpoints.emplace_back(L"ordinary-id");
  endpoints.emplace_back(L"steam-id");
  endpoints[1].property(field_e::adapter_friendly_name).text = L"Steam Streaming Speakers";
  endpoints.emplace_back(L"later-steam-id");
  endpoints[2].property(field_e::adapter_friendly_name).text = L"Steam Streaming Speakers";

  EXPECT_EQ(find(endpoints, lookup::match_steam_speakers()), L"steam-id");
  for (std::size_t index = 0; index < 2; ++index) {
    EXPECT_EQ(endpoints[index].property_store_opens, 1u);
    EXPECT_EQ(endpoints[index].reads(field_e::adapter_friendly_name), 1u);
    EXPECT_EQ(endpoints[index].reads(field_e::device_friendly_name), 0u);
    EXPECT_EQ(endpoints[index].reads(field_e::device_description), 0u);
  }
  EXPECT_EQ(endpoints[2].id_reads, 0u);
}

TEST(WindowsAudioEndpointMatch, UnavailablePropertyStoreCannotBlockLaterExactId) {
  std::vector<endpoint_t> endpoints;
  endpoints.emplace_back(L"unavailable-id");
  endpoints[0].property_store_available = false;
  endpoints.emplace_back(L"wanted");

  EXPECT_EQ(find(endpoints, lookup::match_all_fields(L"wanted")), L"wanted");
  EXPECT_EQ(endpoints[0].property_store_opens, 1u);
  EXPECT_EQ(endpoints[0].reads(field_e::device_friendly_name), 0u);
  EXPECT_EQ(endpoints[1].property_store_opens, 0u);
}

TEST(WindowsAudioEndpointMatch, FailedNonStringAndNullPropertiesCannotAuthorizeAMatch) {
  std::vector<endpoint_t> endpoints;
  endpoints.emplace_back(L"invalid-id");
  auto &failed = endpoints[0].property(field_e::device_friendly_name);
  failed.text = L"wanted";
  failed.status = E_FAIL;
  auto &non_string = endpoints[0].property(field_e::device_description);
  non_string.text = L"wanted";
  non_string.type = VT_UI4;
  auto &null_string = endpoints[0].property(field_e::adapter_friendly_name);
  null_string.text = L"wanted";
  null_string.null_string = true;
  endpoints.emplace_back(L"valid-id");
  endpoints[1].property(field_e::device_friendly_name).text = L"wanted";

  EXPECT_EQ(find(endpoints, lookup::match_all_fields(L"wanted")), L"valid-id");
  EXPECT_EQ(endpoints[0].reads(field_e::device_friendly_name), 1u);
  EXPECT_EQ(endpoints[0].reads(field_e::device_description), 1u);
  EXPECT_EQ(endpoints[0].reads(field_e::adapter_friendly_name), 1u);
}

TEST(WindowsAudioEndpointMatch, EmptyRequestsAndUnavailableIdsDoNotReadProperties) {
  std::vector<endpoint_t> endpoints;
  endpoints.emplace_back(L"unavailable-id");
  endpoints[0].available = false;
  endpoints.emplace_back(L"");
  endpoints[1].property(field_e::device_friendly_name).text = L"wanted";

  EXPECT_FALSE(find(endpoints, {}));
  EXPECT_EQ(endpoints[0].id_reads, 0u);
  EXPECT_EQ(endpoints[1].id_reads, 0u);
  EXPECT_FALSE(find(endpoints, lookup::match_all_fields(L"wanted")));
  EXPECT_EQ(endpoints[0].property_store_opens, 0u);
  EXPECT_EQ(endpoints[1].property_store_opens, 0u);
}

TEST(WindowsAudioEndpointMatch, MissingAndCaseDifferentNamesDoNotMatch) {
  std::vector<endpoint_t> endpoints;
  endpoints.emplace_back(L"WANTED");
  endpoints[0].property(field_e::device_friendly_name).text = L"WANTED";

  EXPECT_FALSE(find(endpoints, lookup::match_all_fields(L"wanted")));
  EXPECT_EQ(endpoints[0].property_store_opens, 1u);
  EXPECT_EQ(endpoints[0].reads(field_e::device_friendly_name), 1u);
  EXPECT_EQ(endpoints[0].reads(field_e::device_description), 1u);
  EXPECT_EQ(endpoints[0].reads(field_e::adapter_friendly_name), 1u);
}

TEST(WindowsAudioEndpointMatch, ConfiguredNamesRetainCStringTermination) {
  std::vector<endpoint_t> endpoints;
  endpoints.emplace_back(L"wanted-id");
  std::wstring requested = L"wanted-id";
  requested.push_back(L'\0');
  requested += L"ignored";

  EXPECT_EQ(find(endpoints, lookup::match_all_fields(requested)), L"wanted-id");
  EXPECT_EQ(endpoints[0].property_store_opens, 0u);
}
