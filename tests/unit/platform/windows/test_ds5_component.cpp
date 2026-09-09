#include "src/platform/windows/ds5/ds5_sidecar_client.h"

#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <gtest/gtest.h>
#include <openssl/evp.h>
#include <windows.h>

std::filesystem::path component_test_root;

namespace {
  class Ds5ComponentTests: public testing::Test {
  protected:
    std::filesystem::path active;
    boost::property_tree::ptree manifest;

    void SetUp() override {
      component_test_root = std::filesystem::temp_directory_path() /
                            ("sunshine-ds5-component-test-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(GetTickCount64()));
      ASSERT_FALSE(std::filesystem::exists(component_test_root));
      active = component_test_root / "tools" / "sunshine-ds5-component" / "active";
      std::filesystem::create_directories(active);
      write("Sunshine.Ds5Sidecar.exe", "test apphost");
      write("Sunshine.Ds5Sidecar.dll", "test managed assembly");
      manifest.put("protocol", 1);
      manifest.put("sidecar_file", "Sunshine.Ds5Sidecar.exe");
      manifest.put("sidecar_sha256", digest("test apphost"));
      boost::property_tree::ptree files;
      for (const auto &[name, content] : std::array {
             std::pair {"Sunshine.Ds5Sidecar.exe", "test apphost"},
             std::pair {"Sunshine.Ds5Sidecar.dll", "test managed assembly"}
           }) {
        boost::property_tree::ptree value;
        value.put_value(digest(content));
        files.push_back({name, value});
      }
      manifest.add_child("files", files);
      save();
    }

    void TearDown() override {
      std::filesystem::remove_all(component_test_root);
    }

    void write(const std::string &name, const std::string &content) {
      std::ofstream(active / name, std::ios::binary) << content;
    }

    void save() {
      boost::property_tree::write_json((active / "component.json").string(), manifest);
    }

    static std::string digest(const std::string &content) {
      std::array<unsigned char, EVP_MAX_MD_SIZE> bytes {};
      unsigned size = 0;
      EXPECT_EQ(EVP_Digest(content.data(), content.size(), bytes.data(), &size, EVP_sha256(), nullptr), 1);
      static constexpr char hex[] = "0123456789abcdef";
      std::string result;
      for (unsigned index = 0; index < size; ++index) {
        result.push_back(hex[bytes[index] >> 4]);
        result.push_back(hex[bytes[index] & 15]);
      }
      return result;
    }
  };

  TEST_F(Ds5ComponentTests, AcceptsCompleteManifestButDoesNotTreatUnrunnablePayloadAsReady) {
    EXPECT_TRUE(platf::ds5::trusted_component_path());
    EXPECT_FALSE(platf::ds5::refresh_component_availability());
    EXPECT_FALSE(platf::ds5::audio_haptics_available());
  }

  TEST_F(Ds5ComponentTests, RejectsTamperedManagedDependency) {
    write("Sunshine.Ds5Sidecar.dll", "modified");
    EXPECT_FALSE(platf::ds5::trusted_component_path());
  }

  TEST_F(Ds5ComponentTests, RejectsMissingManagedDependency) {
    std::filesystem::remove(active / "Sunshine.Ds5Sidecar.dll");
    EXPECT_FALSE(platf::ds5::trusted_component_path());
  }

  TEST_F(Ds5ComponentTests, RejectsUnlistedExecutablePayloadCaseInsensitively) {
    write("extra.DLL", "unlisted");
    EXPECT_FALSE(platf::ds5::trusted_component_path());
  }

  TEST_F(Ds5ComponentTests, RejectsDependencyOutsideFixedComponentDirectory) {
    std::ofstream(active.parent_path() / "outside.dll", std::ios::binary) << "outside";
    boost::property_tree::ptree value;
    value.put_value(digest("outside"));
    manifest.get_child("files").push_back({"../outside.dll", value});
    save();
    EXPECT_FALSE(platf::ds5::trusted_component_path());
  }

  TEST_F(Ds5ComponentTests, RejectsExecutableOnlyLegacyManifest) {
    manifest.erase("files");
    save();
    EXPECT_FALSE(platf::ds5::trusted_component_path());
  }
}  // namespace
