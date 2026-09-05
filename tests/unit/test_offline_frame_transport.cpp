#include "src/offline_frame_transport.h"

#include <future>
#include <gtest/gtest.h>

using namespace std::chrono_literals;
using namespace offline_sbs;

namespace {
  raw_frame frame(std::uint64_t sequence) {
    raw_frame result;
    result.header = {1, sequence, std::int64_t(sequence * 41), 1, 1000, 2, 1, raw_pixel_format::bgra8};
    result.bgra = {1, 2, 3, 255, 4, 5, 6, 255};
    return result;
  }
}

TEST(OfflineRawFrameTransport, PreservesExactHeaderPixelsAndExclusiveOwnership) {
  raw_frame_channel channel(1, 8);
  auto source = frame(1);
  const auto *allocation = source.bgra.data();
  channel.publish(std::move(source));
  auto received = channel.receive(1);
  EXPECT_EQ(received.bgra.data(), allocation);
  EXPECT_EQ(received.header.pts, 41);
  EXPECT_EQ(received.header.time_base_numerator, 1);
  EXPECT_EQ(received.header.time_base_denominator, 1000);
  EXPECT_EQ(received.bgra, frame(1).bgra);
  channel.acknowledge(1);
  channel.finish();
  EXPECT_THROW(channel.receive(2), std::runtime_error);
  EXPECT_THROW(channel.publish(frame(2)), std::runtime_error);
}

TEST(OfflineRawFrameTransport, RejectsMalformedOrOversizedFramesBeforePublication) {
  raw_frame_channel channel(1, 8);
  auto invalid = frame(1);
  invalid.header.schema = 2;
  EXPECT_THROW(channel.publish(std::move(invalid)), std::runtime_error);
  invalid = frame(1);
  invalid.bgra.push_back(9);
  EXPECT_THROW(channel.publish(std::move(invalid)), std::runtime_error);
  invalid = frame(1);
  invalid.header.format = raw_pixel_format::rgba16f;
  EXPECT_THROW(channel.publish(std::move(invalid)), std::runtime_error);
  invalid = frame(1);
  invalid.header.time_base_denominator = 0;
  EXPECT_THROW(channel.publish(std::move(invalid)), std::runtime_error);
  EXPECT_THROW(channel.publish(frame(2)), std::runtime_error);
  channel.publish(frame(1));
  EXPECT_THROW(channel.acknowledge(1), std::runtime_error);
  EXPECT_THROW(channel.receive(2), std::runtime_error);
  EXPECT_NO_THROW(channel.receive(1));
}

TEST(OfflineRawFrameTransport, FullSlotBackpressureAndSeparateEvidenceAcknowledgement) {
  raw_frame_channel channel(1, 8);
  channel.publish(frame(1));
  std::promise<void> entered;
  auto producer = std::async(std::launch::async, [&] {
    entered.set_value();
    channel.publish(frame(2));
    channel.wait_acknowledged(1);
  });
  entered.get_future().wait();
  EXPECT_EQ(producer.wait_for(20ms), std::future_status::timeout);
  auto first = channel.receive(1);
  EXPECT_EQ(first.header.sequence, 1);
  // Receiving raw pixels must not release the previous adaptive-state snapshot.
  EXPECT_EQ(producer.wait_for(20ms), std::future_status::timeout);
  channel.acknowledge(1);
  EXPECT_EQ(producer.wait_for(1s), std::future_status::ready);
  EXPECT_NO_THROW(producer.get());
  EXPECT_EQ(channel.receive(2).header.sequence, 2);
}

TEST(OfflineRawFrameTransport, CancellationReleasesBlockedProducerConsumerAndEvidenceWaiter) {
  raw_frame_channel full(1, 8), empty(1, 8);
  full.publish(frame(1));
  auto producer = std::async(std::launch::async, [&] { EXPECT_THROW(full.publish(frame(2)), std::runtime_error); });
  auto consumer = std::async(std::launch::async, [&] { EXPECT_THROW(empty.receive(1), std::runtime_error); });
  auto proof = std::async(std::launch::async, [&] { EXPECT_THROW(full.wait_acknowledged(1), std::runtime_error); });
  full.cancel();
  empty.cancel();
  EXPECT_EQ(producer.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(consumer.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(proof.wait_for(1s), std::future_status::ready);
  producer.get(); consumer.get(); proof.get();
}

TEST(OfflineRawFrameTransport, RasterBudgetCoversHdrUploadAndThreeOverlappingOutputs) {
  const auto source = 1280ull * 720 * 12;
  const auto output = 2560ull * 720 * 8;
  const auto serial = raw_raster_byte_bound(source, 2560, 720, true, false);
  const auto overlap = raw_raster_byte_bound(source, 2560, 720, true, true);
  EXPECT_GE(serial, source + source / 12 * 8 + output + 64 * 1024);
  EXPECT_EQ(overlap - serial, source + 2 * output);
  EXPECT_LT(overlap, 128ull * 1024 * 1024);
  EXPECT_THROW(raw_raster_byte_bound(source, 0, 720, true, true), std::runtime_error);
  // A source alone can fit while its HDR upload does not. The analysis-only path must
  // reserve both before allocation even though it has no packed output/encoder.
  constexpr auto fullHdHdrSource = 1920ull * 1080 * 12;
  EXPECT_LT(fullHdHdrSource, 32ull * 1024 * 1024);
  EXPECT_GT(raw_source_byte_bound(fullHdHdrSource, true), 32ull * 1024 * 1024);
  EXPECT_EQ(raw_source_byte_bound(1280ull * 720 * 4, false), 1280ull * 720 * 4 + 1024 * 1024);
}
