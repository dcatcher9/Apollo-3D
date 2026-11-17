/**
 * @file tests/unit/test_offline_completion_transport.cpp
 * @brief Owned native completion evidence, ordering, failure, and cancellation contracts.
 */
#include "src/offline_frame_transport.h"

#include <bit>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

using namespace std::chrono_literals;
using namespace offline_sbs;

namespace {
  completion_header_t header(std::uint64_t count = 1, bool converting = false, raw_pixel_format format = raw_pixel_format::bgra8) {
    completion_header_t result;
    result.adaptive_schema = sbs_adaptive_state::schema_version;
    result.adaptive_contract_sha256 = sbs_adaptive_state::contract_canonical_sha256;
    result.source = sbs_adaptive_state::source;
    result.capture = sbs_adaptive_state::capture;
    for (std::size_t index = 0; index < result.fields.size(); ++index) {
      result.fields[index].name = sbs_adaptive_state::fields[index].name;
      result.fields[index].type = sbs_adaptive_state::fields[index].json_type;
    }
    result.model = "owned-resolved-model";
    result.pop_strength = 1.25;
    result.depth_reuse_interval = 1;
    result.source_format = format;
    result.source_frame_count = count;
    result.sbs_enabled = converting;
    return result;
  }

  completion_frame_t frame(std::uint64_t sequence = 1, bool converting = false, raw_pixel_format format = raw_pixel_format::bgra8) {
    completion_frame_t result;
    result.source = {1, sequence, -41 + static_cast<std::int64_t>(sequence), 3, 90000, 2, 1, format};
    result.adaptive_state = sbs_adaptive_state::initial_words;
    result.adaptive_state[sbs_adaptive_state::index(sbs_adaptive_state::word_e::valid_depth_fraction)] =
      std::bit_cast<std::uint32_t>(0.75f);
    result.depth_updated = true;
    if (converting) {
      result.sbs = completion_sbs_info_t {4, 1, format == raw_pixel_format::gbrpf32le ? raw_pixel_format::rgba16f : raw_pixel_format::bgra8};
    }
    return result;
  }

  completion_terminal_t success(std::uint64_t count = 1, bool converting = false) {
    return {0, count, count, converting ? count : 0, {}};
  }

  raw_frame pixels(std::uint64_t sequence) {
    raw_frame result;
    result.header = frame(sequence).source;
    result.bgra = {1, 2, 3, 255, 4, 5, 6, 255};
    return result;
  }

  struct cancel_channel_t {
    completion_channel_t &channel;

    ~cancel_channel_t() {
      channel.cancel();
    }
  };

  struct cancel_transport_t {
    raw_frame_transport &transport;

    ~cancel_transport_t() {
      transport.cancel();
    }
  };
}  // namespace

TEST(OfflineCompletionTransport, OwnsResolvedHeaderAndFrameEvidenceDespiteProducerMutation) {
  completion_channel_t channel;
  auto produced_header = header();
  channel.publish_header(produced_header);
  produced_header.model.clear();
  produced_header.fields[0].name.clear();
  const auto received_header = channel.receive_header(0ms);
  EXPECT_EQ(received_header.model, "owned-resolved-model");
  EXPECT_EQ(received_header.pop_strength, 1.25);
  EXPECT_EQ(received_header.fields[0].name, sbs_adaptive_state::fields[0].name);

  auto produced_frame = frame();
  const auto expected = produced_frame;
  channel.publish_frame(produced_frame);
  produced_frame.source.pts = 999;
  produced_frame.adaptive_state.fill(0);
  const auto received = channel.receive_frame(1, 0ms);
  EXPECT_EQ(received.source.sequence, 1u);
  EXPECT_EQ(received.source.pts, expected.source.pts);
  EXPECT_EQ(received.source.time_base_numerator, 3);
  EXPECT_EQ(received.source.time_base_denominator, 90000);
  EXPECT_EQ(received.adaptive_state, expected.adaptive_state);
  EXPECT_TRUE(received.depth_updated);
  EXPECT_FALSE(received.sbs);
  channel.finish(success());
}

TEST(OfflineCompletionTransport, AuthenticatesGeneratedHeaderIdentityLayoutAndResolvedConfiguration) {
  using mutation_t = void (*)(completion_header_t &);
  const std::pair<const char *, mutation_t> mutations[] {
    {"protocol schema", [](auto &value) {
       value.schema = 2;
     }},
    {"adaptive schema", [](auto &value) {
       ++value.adaptive_schema;
     }},
    {"canonical fingerprint", [](auto &value) {
       value.adaptive_contract_sha256[0] ^= 1;
     }},
    {"source identity", [](auto &value) {
       value.source += "-other";
     }},
    {"capture identity", [](auto &value) {
       value.capture += "-other";
     }},
    {"field order", [](auto &value) {
       std::swap(value.fields[0], value.fields[1]);
     }},
    {"field name", [](auto &value) {
       value.fields[0].name += "-other";
     }},
    {"field type", [](auto &value) {
       value.fields[0].type = "float32";
     }},
    {"missing model", [](auto &value) {
       value.model.clear();
     }},
    {"nonfinite pop", [](auto &value) {
       value.pop_strength = std::numeric_limits<double>::quiet_NaN();
     }},
    {"small pop", [](auto &value) {
       value.pop_strength = 0.249;
     }},
    {"large pop", [](auto &value) {
       value.pop_strength = 2.001;
     }},
    {"retired cadence", [](auto &value) {
       value.depth_reuse_interval = 2;
     }},
    {"packed format as source", [](auto &value) {
       value.source_format = raw_pixel_format::rgba16f;
     }},
    {"empty timeline", [](auto &value) {
       value.source_frame_count = 0;
     }},
    {"sequence overflow", [](auto &value) {
       value.source_frame_count = std::numeric_limits<std::uint64_t>::max();
     }},
    {"shifted first sequence", [](auto &value) {
       value.first_sequence = 2;
     }},
    {"unbounded owned header", [](auto &value) {
       value.model.assign(1024 * 1024, 'x');
     }},
  };
  for (const auto &[description, mutate] : mutations) {
    SCOPED_TRACE(description);
    completion_channel_t channel;
    auto invalid = header();
    mutate(invalid);
    EXPECT_THROW(channel.publish_header(std::move(invalid)), std::runtime_error);
    EXPECT_NO_THROW(channel.publish_header(header()));
    EXPECT_EQ(channel.receive_header(0ms).source_frame_count, 1u);
  }
  for (const double boundary : {0.25, 2.0}) {
    completion_channel_t channel;
    auto valid = header();
    valid.pop_strength = boundary;
    channel.publish_header(std::move(valid));
    EXPECT_EQ(channel.receive_header(0ms).pop_strength, boundary);
  }
}

TEST(OfflineCompletionTransport, RequiresHeaderAndContiguousBoundedSourceSequences) {
  completion_channel_t channel;
  EXPECT_THROW(channel.publish_frame(frame()), std::runtime_error);
  channel.publish_header(header(2));
  EXPECT_THROW(channel.publish_header(header(2)), std::runtime_error);
  EXPECT_THROW(channel.receive_frame(1, 0ms), std::runtime_error);
  channel.receive_header(0ms);
  EXPECT_THROW(channel.receive_header(0ms), std::runtime_error);
  EXPECT_THROW(channel.publish_header(header(2)), std::runtime_error);
  EXPECT_THROW(channel.publish_frame(frame(2)), std::runtime_error);
  channel.publish_frame(frame(1));
  EXPECT_THROW(channel.receive_frame(2, 0ms), std::runtime_error);
  EXPECT_EQ(channel.receive_frame(1, 0ms).source.sequence, 1u);
  EXPECT_THROW(channel.publish_frame(frame(1)), std::runtime_error);
  channel.publish_frame(frame(2));
  EXPECT_EQ(channel.receive_frame(2, 0ms).source.sequence, 2u);
  EXPECT_THROW(channel.publish_frame(frame(3)), std::runtime_error);
  channel.finish(success(2));
  EXPECT_THROW(channel.receive_frame(3, 0ms), std::runtime_error);
}

TEST(OfflineCompletionTransport, RejectsIncompleteMalformedAndMismatchedFrameEvidence) {
  using mutation_t = void (*)(completion_frame_t &);
  const std::pair<const char *, mutation_t> mutations[] {
    {"source schema", [](auto &value) {
       value.source.schema = 2;
     }},
    {"source sequence", [](auto &value) {
       value.source.sequence = 0;
     }},
    {"source width", [](auto &value) {
       value.source.width = 0;
     }},
    {"source height", [](auto &value) {
       value.source.height = 16385;
     }},
    {"source time base", [](auto &value) {
       value.source.time_base_numerator = 0;
     }},
    {"source denominator", [](auto &value) {
       value.source.time_base_denominator = -1;
     }},
    {"source transfer", [](auto &value) {
       value.source.format = raw_pixel_format::gbrpf32le;
     }},
    {"missing depth update", [](auto &value) {
       value.depth_updated = false;
     }},
    {"NaN", [](auto &value) {
       value.adaptive_state[4] = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::quiet_NaN());
     }},
    {"positive infinity", [](auto &value) {
       value.adaptive_state[5] = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::infinity());
     }},
    {"negative infinity", [](auto &value) {
       value.adaptive_state[6] = std::bit_cast<std::uint32_t>(-std::numeric_limits<float>::infinity());
     }},
    {"reserved state", [](auto &value) {
       value.adaptive_state[3] = 1;
     }},
    {"empty counter overflow", [](auto &value) {
       value.adaptive_state[1] = std::numeric_limits<std::uint32_t>::max();
     }},
    {"collapsed counter overflow", [](auto &value) {
       value.adaptive_state[2] = std::numeric_limits<std::uint32_t>::max();
     }},
    {"missing SBS", [](auto &value) {
       value.sbs.reset();
     }},
    {"empty SBS", [](auto &value) {
       value.sbs->width = 0;
     }},
    {"oversized SBS", [](auto &value) {
       value.sbs->height = 16385;
     }},
    {"planar SBS", [](auto &value) {
       value.sbs->format = raw_pixel_format::gbrpf32le;
     }},
    {"wrong SBS transfer", [](auto &value) {
       value.sbs->format = raw_pixel_format::rgba16f;
     }},
  };
  for (const auto &[description, mutate] : mutations) {
    SCOPED_TRACE(description);
    completion_channel_t channel;
    channel.publish_header(header(1, true));
    channel.receive_header(0ms);
    auto invalid = frame(1, true);
    mutate(invalid);
    EXPECT_THROW(channel.publish_frame(std::move(invalid)), std::runtime_error);
    channel.publish_frame(frame(1, true));
    EXPECT_TRUE(channel.receive_frame(1, 0ms).sbs.has_value());
    channel.finish(success(1, true));
  }
}

TEST(OfflineCompletionTransport, PreservesFiniteFloat32BitsIncludingNativeRepresentationBoundaries) {
  completion_channel_t channel;
  channel.publish_header(header());
  channel.receive_header(0ms);
  auto produced = frame();
  produced.adaptive_state[1] = sbs_adaptive_state::counter_max;
  produced.adaptive_state[2] = sbs_adaptive_state::counter_max;
  produced.adaptive_state[4] = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::max());
  produced.adaptive_state[5] = std::bit_cast<std::uint32_t>(-std::numeric_limits<float>::max());
  produced.adaptive_state[6] = std::bit_cast<std::uint32_t>(-0.0f);
  produced.adaptive_state[7] = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::denorm_min());
  const auto expected = produced.adaptive_state;
  // Native evidence accepts finite float32 values directly. Evaluator JSON retains its old
  // decimal validation, which can reject the rounded decimal spelling of +/-float::max.
  channel.publish_frame(std::move(produced));
  EXPECT_EQ(channel.receive_frame(1, 0ms).adaptive_state, expected);
  channel.finish(success());
}

TEST(OfflineCompletionTransport, SupportsAnalyzeAndConversionWithExactSdrOrHdrDescriptions) {
  for (const auto format : {raw_pixel_format::bgra8, raw_pixel_format::gbrpf32le}) {
    for (const bool converting : {false, true}) {
      SCOPED_TRACE(static_cast<int>(format));
      SCOPED_TRACE(converting);
      completion_channel_t channel;
      channel.publish_header(header(1, converting, format));
      const auto resolved = channel.receive_header(0ms);
      EXPECT_EQ(resolved.source_format, format);
      EXPECT_EQ(resolved.sbs_enabled, converting);
      if (!converting) {
        EXPECT_THROW(channel.publish_frame(frame(1, true, format)), std::runtime_error);
      }
      channel.publish_frame(frame(1, converting, format));
      const auto received = channel.receive_frame(1, 0ms);
      EXPECT_EQ(received.source.format, format);
      ASSERT_EQ(received.sbs.has_value(), converting);
      if (converting) {
        EXPECT_EQ(received.sbs->width, 4u);
        EXPECT_EQ(received.sbs->height, 1u);
        EXPECT_EQ(received.sbs->format, format == raw_pixel_format::gbrpf32le ? raw_pixel_format::rgba16f : raw_pixel_format::bgra8);
      }
      channel.finish(success(1, converting));
    }
  }
}

TEST(OfflineCompletionTransport, OneSlotAppliesBackpressureWithoutLosingThePreviousEvidence) {
  completion_channel_t channel;
  channel.publish_header(header(2));
  channel.receive_header(0ms);
  channel.publish_frame(frame(1));
  std::promise<void> entered;
  auto producer = std::async(std::launch::async, [&] {
    entered.set_value();
    channel.publish_frame(frame(2));
  });
  cancel_channel_t cleanup {channel};
  ASSERT_EQ(entered.get_future().wait_for(1s), std::future_status::ready);
  EXPECT_EQ(producer.wait_for(20ms), std::future_status::timeout);
  EXPECT_EQ(channel.receive_frame(1, 0ms).source.sequence, 1u);
  ASSERT_EQ(producer.wait_for(1s), std::future_status::ready);
  EXPECT_NO_THROW(producer.get());
  EXPECT_EQ(channel.receive_frame(2, 0ms).source.sequence, 2u);
  channel.finish(success(2));
}

TEST(OfflineCompletionTransport, ReceivingEvidenceDoesNotAcknowledgeTheSource) {
  raw_frame_transport transport(1024);
  transport.source.publish(pixels(1));
  const auto received_pixels = transport.source.receive(1);
  transport.completion.publish_header(header());
  transport.completion.receive_header(0ms);
  transport.completion.publish_frame(frame());
  EXPECT_EQ(transport.completion.receive_frame(1, 0ms).source.sequence, received_pixels.header.sequence);
  std::promise<void> entered;
  auto evidence_barrier = std::async(std::launch::async, [&] {
    entered.set_value();
    transport.source.wait_acknowledged(1);
  });
  cancel_transport_t cleanup {transport};
  ASSERT_EQ(entered.get_future().wait_for(1s), std::future_status::ready);
  EXPECT_EQ(evidence_barrier.wait_for(20ms), std::future_status::timeout);
  transport.source.acknowledge(1);
  ASSERT_EQ(evidence_barrier.wait_for(1s), std::future_status::ready);
  EXPECT_NO_THROW(evidence_barrier.get());
  transport.completion.finish(success());
}

TEST(OfflineCompletionTransport, StickySuccessRetainsFinalEvidenceUntilItIsConsumed) {
  completion_channel_t channel;
  channel.publish_header(header());
  channel.receive_header(0ms);
  const auto expected = frame();
  channel.publish_frame(expected);
  channel.finish(success());
  EXPECT_EQ(channel.wait_finished(0ms).exit_code, 0);
  EXPECT_EQ(channel.receive_frame(1, 0ms).adaptive_state, expected.adaptive_state);
  EXPECT_THROW(channel.receive_frame(2, 0ms), std::runtime_error);
  EXPECT_THROW(channel.publish_frame(frame(2)), std::runtime_error);
  channel.finish({10, 0, 0, 0, "later cleanup error"});
  const auto terminal = channel.wait_finished(0ms);
  EXPECT_EQ(terminal.exit_code, 0);
  EXPECT_EQ(terminal.source_frame_count, 1u);
  EXPECT_TRUE(terminal.error.empty());
}

TEST(OfflineCompletionTransport, RejectsSuccessWithoutAllDeclaredEvidenceAndConsistentCounts) {
  completion_channel_t channel;
  EXPECT_THROW(channel.finish(success()), std::runtime_error);
  channel.publish_header(header(2));
  channel.receive_header(0ms);
  channel.publish_frame(frame(1));
  channel.receive_frame(1, 0ms);
  EXPECT_THROW(channel.finish(success(1)), std::runtime_error);
  EXPECT_THROW(channel.finish(success(2)), std::runtime_error);
  channel.publish_frame(frame(2));
  channel.receive_frame(2, 0ms);
  EXPECT_THROW(channel.finish((completion_terminal_t {0, 2, 1, 0, {}})), std::runtime_error);
  EXPECT_THROW(channel.finish((completion_terminal_t {0, 2, 2, 1, {}})), std::runtime_error);
  EXPECT_THROW(channel.finish((completion_terminal_t {0, 2, 2, 0, "error with success"})), std::runtime_error);
  channel.finish(success(2));
  EXPECT_EQ(channel.wait_finished(0ms).adaptive_frame_count, 2u);
}

TEST(OfflineCompletionTransport, FailureBeforeHeaderWakesReceiverAndPreservesBoundedTerminalStatus) {
  completion_channel_t channel;
  std::promise<void> entered;
  auto consumer = std::async(std::launch::async, [&] {
    entered.set_value();
    try {
      channel.receive_header(1s);
      return false;
    } catch (const std::runtime_error &error) {
      return std::string_view {error.what()}.find("native pass failed") != std::string_view::npos;
    }
  });
  cancel_channel_t cleanup {channel};
  ASSERT_EQ(entered.get_future().wait_for(1s), std::future_status::ready);
  channel.finish({10, 0, 0, 0, std::string(2048, 'x')});
  ASSERT_EQ(consumer.wait_for(1s), std::future_status::ready);
  EXPECT_TRUE(consumer.get());
  const auto terminal = channel.wait_finished(0ms);
  EXPECT_EQ(terminal.exit_code, 10);
  EXPECT_EQ(terminal.error, std::string(1024, 'x'));
  channel.finish(success());
  EXPECT_EQ(channel.wait_finished(0ms).exit_code, 10);
  EXPECT_THROW(channel.publish_header(header()), std::runtime_error);
}

TEST(OfflineCompletionTransport, ProducerFailureWakesFrameReceiverInsteadOfWaitingForAnotherFrame) {
  completion_channel_t channel;
  channel.publish_header(header());
  channel.receive_header(0ms);
  std::promise<void> entered;
  auto consumer = std::async(std::launch::async, [&] {
    entered.set_value();
    try {
      channel.receive_frame(1, 1s);
      return false;
    } catch (const std::runtime_error &error) {
      return std::string_view {error.what()}.find("readback failed") != std::string_view::npos;
    }
  });
  cancel_channel_t cleanup {channel};
  ASSERT_EQ(entered.get_future().wait_for(1s), std::future_status::ready);
  auto producer = std::async(std::launch::async, [&] {
    try {
      throw std::runtime_error("readback failed");
    } catch (const std::exception &error) {
      channel.finish({6, 0, 0, 0, error.what()});
    }
  });
  cancel_channel_t producer_cleanup {channel};
  ASSERT_EQ(producer.wait_for(1s), std::future_status::ready);
  EXPECT_NO_THROW(producer.get());
  ASSERT_EQ(consumer.wait_for(1s), std::future_status::ready);
  EXPECT_TRUE(consumer.get());
  EXPECT_EQ(channel.wait_finished(0ms).exit_code, 6);
}

TEST(OfflineCompletionTransport, ConsumerFailureReleasesABlockedEvidencePublisher) {
  completion_channel_t channel;
  channel.publish_header(header(2));
  channel.receive_header(0ms);
  channel.publish_frame(frame(1));
  std::promise<void> entered;
  auto producer = std::async(std::launch::async, [&] {
    entered.set_value();
    try {
      channel.publish_frame(frame(2));
      return false;
    } catch (const std::runtime_error &error) {
      return std::string_view {error.what()}.find("consumer rejected evidence") != std::string_view::npos;
    }
  });
  cancel_channel_t cleanup {channel};
  ASSERT_EQ(entered.get_future().wait_for(1s), std::future_status::ready);
  EXPECT_EQ(producer.wait_for(20ms), std::future_status::timeout);
  channel.finish({10, 0, 0, 0, "consumer rejected evidence"});
  ASSERT_EQ(producer.wait_for(1s), std::future_status::ready);
  EXPECT_TRUE(producer.get());
  EXPECT_THROW(channel.receive_frame(1, 0ms), std::runtime_error);
}

TEST(OfflineCompletionTransport, CancellationReleasesHeaderFrameAndTerminalReceivers) {
  completion_channel_t header_channel;
  completion_channel_t frame_channel;
  completion_channel_t terminal_channel;
  frame_channel.publish_header(header());
  frame_channel.receive_header(0ms);
  std::promise<void> header_entered;
  std::promise<void> frame_entered;
  std::promise<void> terminal_entered;
  auto header_waiter = std::async(std::launch::async, [&] {
    header_entered.set_value();
    try {
      header_channel.receive_header(1s);
      return false;
    } catch (const std::runtime_error &error) {
      return std::string_view {error.what()}.find("cancelled") != std::string_view::npos;
    }
  });
  cancel_channel_t header_cleanup {header_channel};
  auto frame_waiter = std::async(std::launch::async, [&] {
    frame_entered.set_value();
    try {
      frame_channel.receive_frame(1, 1s);
      return false;
    } catch (const std::runtime_error &error) {
      return std::string_view {error.what()}.find("cancelled") != std::string_view::npos;
    }
  });
  cancel_channel_t frame_cleanup {frame_channel};
  auto terminal_waiter = std::async(std::launch::async, [&] {
    terminal_entered.set_value();
    try {
      terminal_channel.wait_finished(1s);
      return false;
    } catch (const std::runtime_error &error) {
      return std::string_view {error.what()}.find("cancelled") != std::string_view::npos;
    }
  });
  cancel_channel_t terminal_cleanup {terminal_channel};
  ASSERT_EQ(header_entered.get_future().wait_for(1s), std::future_status::ready);
  ASSERT_EQ(frame_entered.get_future().wait_for(1s), std::future_status::ready);
  ASSERT_EQ(terminal_entered.get_future().wait_for(1s), std::future_status::ready);
  EXPECT_EQ(header_waiter.wait_for(20ms), std::future_status::timeout);
  EXPECT_EQ(frame_waiter.wait_for(20ms), std::future_status::timeout);
  EXPECT_EQ(terminal_waiter.wait_for(20ms), std::future_status::timeout);
  header_channel.cancel();
  frame_channel.cancel();
  terminal_channel.cancel();
  ASSERT_EQ(header_waiter.wait_for(1s), std::future_status::ready);
  ASSERT_EQ(frame_waiter.wait_for(1s), std::future_status::ready);
  ASSERT_EQ(terminal_waiter.wait_for(1s), std::future_status::ready);
  EXPECT_TRUE(header_waiter.get());
  EXPECT_TRUE(frame_waiter.get());
  EXPECT_TRUE(terminal_waiter.get());
}

TEST(OfflineCompletionTransport, TransportCancellationReleasesRawPixelEvidenceAndAckWaits) {
  raw_frame_transport transport(1024);
  transport.source.publish(pixels(1));
  transport.source.publish(pixels(2));
  std::promise<void> source_entered;
  std::promise<void> sbs_entered;
  std::promise<void> evidence_entered;
  std::promise<void> ack_entered;
  auto source_waiter = std::async(std::launch::async, [&] {
    source_entered.set_value();
    try {
      transport.source.publish(pixels(3));
      return false;
    } catch (const std::runtime_error &error) {
      return std::string_view {error.what()}.find("cancelled") != std::string_view::npos;
    }
  });
  cancel_transport_t source_cleanup {transport};
  auto sbs_waiter = std::async(std::launch::async, [&] {
    sbs_entered.set_value();
    try {
      transport.sbs.receive(1);
      return false;
    } catch (const std::runtime_error &error) {
      return std::string_view {error.what()}.find("cancelled") != std::string_view::npos;
    }
  });
  cancel_transport_t sbs_cleanup {transport};
  auto evidence_waiter = std::async(std::launch::async, [&] {
    evidence_entered.set_value();
    try {
      transport.completion.receive_header(1s);
      return false;
    } catch (const std::runtime_error &error) {
      return std::string_view {error.what()}.find("cancelled") != std::string_view::npos;
    }
  });
  cancel_transport_t evidence_cleanup {transport};
  auto ack_waiter = std::async(std::launch::async, [&] {
    ack_entered.set_value();
    try {
      transport.source.wait_acknowledged(1);
      return false;
    } catch (const std::runtime_error &error) {
      return std::string_view {error.what()}.find("cancelled") != std::string_view::npos;
    }
  });
  cancel_transport_t cleanup {transport};
  ASSERT_EQ(source_entered.get_future().wait_for(1s), std::future_status::ready);
  ASSERT_EQ(sbs_entered.get_future().wait_for(1s), std::future_status::ready);
  ASSERT_EQ(evidence_entered.get_future().wait_for(1s), std::future_status::ready);
  ASSERT_EQ(ack_entered.get_future().wait_for(1s), std::future_status::ready);
  EXPECT_EQ(source_waiter.wait_for(20ms), std::future_status::timeout);
  EXPECT_EQ(sbs_waiter.wait_for(20ms), std::future_status::timeout);
  EXPECT_EQ(evidence_waiter.wait_for(20ms), std::future_status::timeout);
  EXPECT_EQ(ack_waiter.wait_for(20ms), std::future_status::timeout);
  transport.cancel();
  ASSERT_EQ(source_waiter.wait_for(1s), std::future_status::ready);
  ASSERT_EQ(sbs_waiter.wait_for(1s), std::future_status::ready);
  ASSERT_EQ(evidence_waiter.wait_for(1s), std::future_status::ready);
  ASSERT_EQ(ack_waiter.wait_for(1s), std::future_status::ready);
  EXPECT_TRUE(source_waiter.get());
  EXPECT_TRUE(sbs_waiter.get());
  EXPECT_TRUE(evidence_waiter.get());
  EXPECT_TRUE(ack_waiter.get());
}

TEST(OfflineCompletionTransport, CallerDeadlinesDoNotConsumeOrCancelFutureEvidence) {
  completion_channel_t channel;
  EXPECT_THROW(channel.receive_header(-1ms), std::invalid_argument);
  EXPECT_THROW(channel.receive_header(0ms), std::runtime_error);
  channel.publish_header(header());
  EXPECT_EQ(channel.receive_header(0ms).source_frame_count, 1u);
  EXPECT_THROW(channel.receive_frame(1, 0ms), std::runtime_error);
  channel.publish_frame(frame());
  EXPECT_EQ(channel.receive_frame(1, 0ms).source.sequence, 1u);
  EXPECT_THROW(channel.wait_finished(0ms), std::runtime_error);
  channel.finish(success());
  EXPECT_EQ(channel.wait_finished(0ms).exit_code, 0);
}
