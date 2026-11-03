#include <gtest/gtest.h>
#include "src/host_sbs_material_capture_identity.h"
namespace {
  namespace m=models::research_material_capture_v1;
  m::input_t sample(std::uint64_t time,std::uint64_t epoch=7u) {
    m::input_t i;i.capture_epoch=epoch;i.source_stamp=m::clock_t::time_point{std::chrono::microseconds{time}};i.content_stamp=i.source_stamp;
    i.domain.width=1920;i.domain.height=1080;i.domain.analysis_generation=2;i.domain.context_generation=4;i.domain.authority=2;
    i.domain.analysis_ltrb={0,0,1920,1080};i.domain.format=87;i.domain.transfer=0;return i;
  }
}
TEST(MaterialCaptureIdentity, ConversionAndAckRetriesDoNotCreateMaterialObservations) {
  m::tracker_t t;const auto a=t.observe(sample(100));const auto b=t.observe(sample(100));
  ASSERT_TRUE(a.usable());EXPECT_EQ(a.material_serial,b.material_serial);EXPECT_EQ(a.material_capture_clock_us,b.material_capture_clock_us);EXPECT_EQ(b.status,m::status_e::unchanged_material);
}
TEST(MaterialCaptureIdentity, ChangedCaptureAdvancesWithoutClaimingMissingInterval) {
  m::tracker_t t;const auto a=t.observe(sample(100));const auto b=t.observe(sample(200));
  EXPECT_EQ(b.material_serial,a.material_serial+1u);EXPECT_GT(b.material_capture_clock_us,a.material_capture_clock_us);EXPECT_FALSE(b.interval_contiguous);
}
TEST(MaterialCaptureIdentity, ActualCompleteCaptureIntervalCanCertifyContinuity) {
  m::tracker_t t;t.observe(sample(100));auto i=sample(200);i.complete_capture_interval_observed=true;EXPECT_TRUE(t.observe(i).interval_contiguous);
}
TEST(MaterialCaptureIdentity, CleanRoiRetainsMaterialClockAcrossNewDesktopCaptures) {
  m::tracker_t t;const auto a=t.observe(sample(100));auto i=sample(200);i.exact_analysis_pixels_unchanged=true;i.complete_capture_interval_observed=true;
  const auto b=t.observe(i);EXPECT_EQ(b.material_serial,a.material_serial);EXPECT_EQ(b.material_capture_clock_us,a.material_capture_clock_us);EXPECT_EQ(b.latest_capture_stamp,i.source_stamp);
}
TEST(MaterialCaptureIdentity, RegressionCannotRewindThenReplay) {
  m::tracker_t t;const auto a=t.observe(sample(200));EXPECT_FALSE(t.observe(sample(100)).usable());const auto b=t.observe(sample(200));EXPECT_EQ(a.material_serial,b.material_serial);
}
TEST(MaterialCaptureIdentity, SameSourceStampWithDifferentContentIsUnknown) {
  m::tracker_t t;t.observe(sample(100));auto bad=sample(100);bad.content_stamp=m::clock_t::time_point{std::chrono::microseconds{99}};EXPECT_FALSE(t.observe(bad).usable());
}
TEST(MaterialCaptureIdentity, MissingSourceStampNeverGetsSyntheticClock) {
  m::tracker_t t;auto i=sample(100);i.source_stamp.reset();EXPECT_FALSE(t.observe(i).usable());
}
TEST(MaterialCaptureIdentity, DomainAndTransferChangeDoNotInheritIdentity) {
  m::tracker_t t;t.observe(sample(100));auto i=sample(200);i.domain.transfer=2;EXPECT_FALSE(t.observe(i).usable());
}
TEST(MaterialCaptureIdentity, NamespaceMustBeExplicitAndGenuinelyNew) {
  m::tracker_t t;t.observe(sample(100));EXPECT_FALSE(t.begin_source_namespace(7));EXPECT_FALSE(t.observe(sample(200,8)).usable());ASSERT_TRUE(t.begin_source_namespace(8));EXPECT_TRUE(t.observe(sample(200,8)).usable());
}
TEST(MaterialCaptureIdentity, MicrosecondCollisionDoesNotInventClockIncrement) {
  m::tracker_t t;auto a=sample(100);t.observe(a);auto b=a;b.source_stamp=*a.source_stamp+std::chrono::nanoseconds{1};b.content_stamp=b.source_stamp;EXPECT_FALSE(t.observe(b).usable());
}

TEST(MaterialCaptureIdentity, CanonicalFullSourceGenerationZeroIsAllowed) {
  m::tracker_t t;auto in=sample(100);in.domain.authority=0;in.domain.analysis_generation=0;EXPECT_TRUE(t.observe(in).usable());
}
