/**
 * @file tests/unit/test_reed_solomon.cpp
 * @brief Validate the nanors runtime-dispatched Reed-Solomon implementation.
 */

#include <rs.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "../tests_common.h"

namespace {
  using rs_ptr = std::unique_ptr<reed_solomon, decltype(&reed_solomon_release)>;

  rs_ptr make_rs(int data_shards, int parity_shards) {
    return {reed_solomon_new(data_shards, parity_shards), reed_solomon_release};
  }
}  // namespace

TEST(ReedSolomonTests, KnownParityVector) {
  reed_solomon_init();
  auto rs = make_rs(3, 2);
  ASSERT_NE(rs, nullptr);

  std::array<std::array<std::uint8_t, 17>, 5> shards {};
  for (std::size_t shard = 0; shard < 3; ++shard) {
    for (std::size_t byte = 0; byte < shards[shard].size(); ++byte) {
      shards[shard][byte] = static_cast<std::uint8_t>(((shard + 1) * 37 + byte * 13 + 5) & 0xff);
    }
  }

  std::array<std::uint8_t *, 5> shard_ptrs {};
  for (std::size_t i = 0; i < shards.size(); ++i) {
    shard_ptrs[i] = shards[i].data();
  }

  ASSERT_EQ(reed_solomon_encode(rs.get(), shard_ptrs.data(), shard_ptrs.size(), shards[0].size()), 0);

  constexpr std::array<std::uint8_t, 17> expected_parity_0 {
    198, 198, 168, 144, 143, 161, 2, 80, 19, 11, 170, 81, 62, 4, 247, 50, 146,
  };
  constexpr std::array<std::uint8_t, 17> expected_parity_1 {
    171, 22, 91, 129, 127, 25, 112, 21, 206, 115, 71, 7, 27, 52, 124, 251, 195,
  };
  EXPECT_EQ(shards[3], expected_parity_0);
  EXPECT_EQ(shards[4], expected_parity_1);
}

TEST(ReedSolomonTests, AcceptsApolloAudioParityMatrix) {
  auto rs = make_rs(4, 2);
  ASSERT_NE(rs, nullptr);

  constexpr std::array<std::uint8_t, 8> audio_parity_matrix {0x77, 0x40, 0x38, 0x0e, 0xc7, 0xa7, 0x0d, 0x6c};
  std::copy(audio_parity_matrix.begin(), audio_parity_matrix.end(), rs->p);

  std::array<std::array<std::uint8_t, 16>, 6> shards {};
  for (std::size_t shard = 0; shard < 4; ++shard) {
    for (std::size_t byte = 0; byte < shards[shard].size(); ++byte) {
      shards[shard][byte] = static_cast<std::uint8_t>(shard * shards[shard].size() + byte);
    }
  }
  std::array<std::uint8_t *, 6> shard_ptrs {};
  for (std::size_t i = 0; i < shards.size(); ++i) {
    shard_ptrs[i] = shards[i].data();
  }

  ASSERT_EQ(reed_solomon_encode(rs.get(), shard_ptrs.data(), shard_ptrs.size(), shards[0].size()), 0);
  EXPECT_EQ(shards[4], (std::array<std::uint8_t, 16> {26, 27, 24, 25, 30, 31, 28, 29, 18, 19, 16, 17, 22, 23, 20, 21}));
  EXPECT_EQ(shards[5], (std::array<std::uint8_t, 16> {144, 145, 146, 147, 148, 149, 150, 151,
                                                     152, 153, 154, 155, 156, 157, 158, 159}));
}

TEST(ReedSolomonTests, RecoversUnalignedSimdTails) {
  constexpr std::array<std::size_t, 14> block_sizes {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1024, 1392, 1400, 1500};

  for (const auto block_size : block_sizes) {
    SCOPED_TRACE(block_size);
    auto rs = make_rs(4, 2);
    ASSERT_NE(rs, nullptr);

    constexpr std::uint8_t canary = 0xa5;
    std::array<std::vector<std::uint8_t>, 6> storage;
    std::array<std::uint8_t *, 6> shard_ptrs {};
    for (std::size_t shard = 0; shard < storage.size(); ++shard) {
      storage[shard].assign(block_size + 2, canary);
      shard_ptrs[shard] = storage[shard].data() + 1;
    }
    for (std::size_t shard = 0; shard < 4; ++shard) {
      for (std::size_t byte = 0; byte < block_size; ++byte) {
        shard_ptrs[shard][byte] = static_cast<std::uint8_t>((shard * 67 + byte * 29 + block_size) & 0xff);
      }
    }

    std::array<std::vector<std::uint8_t>, 4> original;
    for (std::size_t shard = 0; shard < original.size(); ++shard) {
      original[shard].assign(shard_ptrs[shard], shard_ptrs[shard] + block_size);
    }

    ASSERT_EQ(reed_solomon_encode(rs.get(), shard_ptrs.data(), shard_ptrs.size(), block_size), 0);
    for (const auto &buffer : storage) {
      EXPECT_EQ(buffer.front(), canary);
      EXPECT_EQ(buffer.back(), canary);
    }

    constexpr std::array<std::size_t, 2> erased {1, 3};
    std::array<std::uint8_t, 6> marks {};
    for (const auto shard : erased) {
      std::fill_n(shard_ptrs[shard], block_size, 0);
      marks[shard] = 1;
    }

    ASSERT_EQ(reed_solomon_decode(rs.get(), shard_ptrs.data(), marks.data(), shard_ptrs.size(), block_size), 0);
    for (std::size_t shard = 0; shard < 4; ++shard) {
      EXPECT_TRUE(std::equal(original[shard].begin(), original[shard].end(), shard_ptrs[shard]));
    }
    for (const auto &buffer : storage) {
      EXPECT_EQ(buffer.front(), canary);
      EXPECT_EQ(buffer.back(), canary);
    }
  }
}

TEST(ReedSolomonTests, RejectsInvalidShardCountsAndArguments) {
  EXPECT_EQ(reed_solomon_new(0, 1), nullptr);
  EXPECT_EQ(reed_solomon_new(-1, 1), nullptr);
  EXPECT_EQ(reed_solomon_new(1, 0), nullptr);
  EXPECT_EQ(reed_solomon_new(1, -1), nullptr);
  EXPECT_EQ(reed_solomon_new(255, 1), nullptr);
  EXPECT_EQ(reed_solomon_new(1, 255), nullptr);
  EXPECT_EQ(reed_solomon_new(256, 1), nullptr);
  EXPECT_EQ(reed_solomon_new(1, 256), nullptr);
  EXPECT_EQ(reed_solomon_new(128, 128), nullptr);

  auto rs = make_rs(254, 1);
  ASSERT_NE(rs, nullptr);
  EXPECT_EQ(rs->ts, DATA_SHARDS_MAX);

  std::array<std::uint8_t, 1> data {};
  std::array<std::uint8_t, 1> parity {};
  std::array<std::uint8_t *, 2> shards {data.data(), parity.data()};
  auto small_rs = make_rs(1, 1);
  ASSERT_NE(small_rs, nullptr);

  EXPECT_EQ(reed_solomon_encode(nullptr, shards.data(), shards.size(), data.size()), -1);
  EXPECT_EQ(reed_solomon_encode(small_rs.get(), nullptr, shards.size(), data.size()), -1);
  EXPECT_EQ(reed_solomon_encode(small_rs.get(), shards.data(), 1, data.size()), -1);
  EXPECT_EQ(reed_solomon_encode(small_rs.get(), shards.data(), shards.size(), 0), -1);

  std::array<std::uint8_t, 2> marks {};
  EXPECT_EQ(reed_solomon_decode(nullptr, shards.data(), marks.data(), shards.size(), data.size()), -1);
  EXPECT_EQ(reed_solomon_decode(small_rs.get(), nullptr, marks.data(), shards.size(), data.size()), -1);
  EXPECT_EQ(reed_solomon_decode(small_rs.get(), shards.data(), nullptr, shards.size(), data.size()), -1);
  EXPECT_EQ(reed_solomon_decode(small_rs.get(), shards.data(), marks.data(), 1, data.size()), -1);
  EXPECT_EQ(reed_solomon_decode(small_rs.get(), shards.data(), marks.data(), shards.size(), 0), -1);
}
