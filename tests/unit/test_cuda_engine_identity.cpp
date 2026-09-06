#include "../tests_common.h"

#ifdef _WIN32
#include <src/cuda_engine_identity.h>
#include <src/model_manager.h>
#include <cstring>

namespace {
  struct fake_device_t {
    std::string name = "NVIDIA GeForce RTX 5080";
    std::size_t memory = 17066033152ull;
    int driver_api = 13040;
    std::array<int, 98> attributes {};
    int fail_query = 0;
    CUdevice_attribute fail_attribute = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
    int queried_device = -1;

    fake_device_t() {
      for (const auto &entry : models::detail::cuda_engine_attributes) {
        attributes[entry.attribute] = 512;
      }
      attributes[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR] = 12;
      attributes[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR] = 0;
      attributes[CU_DEVICE_ATTRIBUTE_INTEGRATED] = 0;
    }
  };
  thread_local fake_device_t fake_device;

  cuda_driver_api fake_api() {
    cuda_driver_api api;
    api.cuDeviceGetName = [](char *name, int size, CUdevice device) -> CUresult {
      fake_device.queried_device = device;
      if (fake_device.fail_query == 1) return CUDA_ERROR_NOT_SUPPORTED;
      std::memset(name, 0, size);
      std::memcpy(name, fake_device.name.data(), std::min<std::size_t>(size, fake_device.name.size()));
      return CUDA_SUCCESS;
    };
    api.cuDriverGetVersion = [](int *value) -> CUresult {
      *value = fake_device.driver_api;
      return fake_device.fail_query == 2 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_SUCCESS;
    };
    api.cuDeviceGetAttribute = [](int *value, CUdevice_attribute attribute, CUdevice) -> CUresult {
      *value = fake_device.attributes[attribute];
      return fake_device.fail_query == 3 && attribute == fake_device.fail_attribute ?
        CUDA_ERROR_NOT_SUPPORTED : CUDA_SUCCESS;
    };
    return api;
  }
}

TEST(CudaEngineIdentityTest, MemoryReservationChangeDoesNotInvalidateEitherCache) {
  fake_device = {};
  const auto api = fake_api();
  std::string error;
  const auto current = models::detail::cuda_engine_device_identity(api, 7, error);
  ASSERT_FALSE(current.empty()) << error;
  EXPECT_EQ(fake_device.queried_device, 7);
  EXPECT_EQ(current, models::detail::cuda_engine_device_identity(api, 7, error));
  fake_device.memory = 17094475776ull;
  // No memory query is supplied or required by the cache identity API. Both observed memory
  // environments remain the same device compatibility key when the checked properties match.
  const auto old = models::detail::cuda_engine_device_identity(api, 7, error);
  ASSERT_FALSE(old.empty()) << error;
  const config::depth_model_info model {
    .name = std::string(models::prod_zipdepth_convex2x::logical_model), .url = {},
  };
  EXPECT_EQ(current, old);
  EXPECT_EQ(models::engine_filename(model, current), models::engine_filename(model, old));
  EXPECT_EQ(models::ocr_engine_filename(current), models::ocr_engine_filename(old));
  EXPECT_EQ(current.find("total-memory"), std::string::npos);
}

TEST(CudaEngineIdentityTest, EveryCheckedPropertyParticipatesWithoutOrdinalOrLocaleDependence) {
  fake_device = {};
  const auto api = fake_api();
  std::string error;
  const auto original = models::detail::cuda_engine_device_identity(api, 0, error);
  ASSERT_FALSE(original.empty()) << error;
  EXPECT_EQ(original, models::detail::cuda_engine_device_identity(api, 3, error));
  for (const auto &entry : models::detail::cuda_engine_attributes) {
    SCOPED_TRACE(entry.name);
    ++fake_device.attributes[entry.attribute];
    const auto changed = models::detail::cuda_engine_device_identity(api, 0, error);
    ASSERT_FALSE(changed.empty()) << error;
    EXPECT_NE(original, changed);
    --fake_device.attributes[entry.attribute];
  }
  ++fake_device.driver_api;
  EXPECT_NE(original, models::detail::cuda_engine_device_identity(api, 0, error));
  --fake_device.driver_api;
  fake_device.name += "|total-memory=1:/:";
  EXPECT_NE(original, models::detail::cuda_engine_device_identity(api, 0, error));
}

TEST(CudaEngineIdentityTest, QueryFailuresNeverProduceFallbackIdentity) {
  const auto api = fake_api();
  std::string error;
  for (int query = 1; query <= 3; ++query) {
    fake_device = {};
    fake_device.fail_query = query;
    for (const auto &entry : models::detail::cuda_engine_attributes) {
      fake_device.fail_attribute = entry.attribute;
      EXPECT_TRUE(models::detail::cuda_engine_device_identity(api, 0, error).empty());
      EXPECT_NE(error.find("failed (CUDA 801)"), std::string::npos);
    }
  }
  fake_device = {};
  auto unavailable = api;
  unavailable.cuDriverGetVersion = nullptr;
  EXPECT_TRUE(models::detail::cuda_engine_device_identity(unavailable, 0, error).empty());
  EXPECT_NE(error.find("unavailable"), std::string::npos);
}

TEST(CudaEngineIdentityTest, InvalidDeviceValuesFailInsteadOfBecomingSharedCacheKeys) {
  const auto api = fake_api();
  std::string error;
  for (const auto &entry : models::detail::cuda_engine_attributes) {
    fake_device = {};
    fake_device.attributes[entry.attribute] = entry.minimum - 1;
    EXPECT_TRUE(models::detail::cuda_engine_device_identity(api, 0, error).empty());
    EXPECT_NE(error.find(entry.name), std::string::npos);
  }
  fake_device = {};
  fake_device.name = std::string(256u, 'x');
  EXPECT_TRUE(models::detail::cuda_engine_device_identity(api, 0, error).empty());
  fake_device.name.clear();
  EXPECT_TRUE(models::detail::cuda_engine_device_identity(api, 0, error).empty());
  fake_device = {};
  fake_device.driver_api = 0;
  EXPECT_TRUE(models::detail::cuda_engine_device_identity(api, 0, error).empty());
}
#endif
