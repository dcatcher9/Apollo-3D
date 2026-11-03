#include "../tests_common.h"

#include <src/host_sbs_drawn_source_order.h>
using platf::dxgi::detail::host_sbs_drawn_source_order_t;

TEST(HostSbsDrawnSourceOrder, LateAcknowledgedBMayNotReplaceAlreadyDrawnC) {
  host_sbs_drawn_source_order_t p;
  ASSERT_TRUE(p.note_drawn(2));
  ASSERT_TRUE(p.note_drawn(4));
  EXPECT_FALSE(p.permits(3));
  EXPECT_FALSE(p.note_drawn(3));
  EXPECT_EQ(p.last_drawn(), 4u);
}

TEST(HostSbsDrawnSourceOrder, PackedRepeatNeverAdvancesToUnrenderedSource) {
  host_sbs_drawn_source_order_t p;
  ASSERT_TRUE(p.note_drawn(2));
  EXPECT_TRUE(p.permits(2));
  EXPECT_EQ(p.last_drawn(), 2u);
  ASSERT_TRUE(p.note_drawn(3));
  EXPECT_FALSE(p.permits(2));
}

TEST(HostSbsDrawnSourceOrder, CachedCurrentColourUsesNewPixelSourceNotOldDepthOwner) {
  host_sbs_drawn_source_order_t p;
  ASSERT_TRUE(p.note_drawn(4));
  EXPECT_FALSE(p.permits(2));
  EXPECT_TRUE(p.permits(5));
  ASSERT_TRUE(p.note_drawn(5));
  EXPECT_EQ(p.last_drawn(), 5u);
}

TEST(HostSbsDrawnSourceOrder, DuplicateSourceAllowedZeroRejectedExplicitNamespaceReset) {
  host_sbs_drawn_source_order_t p;
  EXPECT_FALSE(p.permits(0));
  ASSERT_TRUE(p.note_drawn(8));
  ASSERT_TRUE(p.note_drawn(8));
  EXPECT_FALSE(p.note_drawn(0));
  EXPECT_FALSE(p.note_drawn(1));
  p.reset();
  ASSERT_TRUE(p.note_drawn(1));
}
