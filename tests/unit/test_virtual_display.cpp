#ifdef _WIN32

  #include "src/platform/windows/virtual_display.h"

  #include <gtest/gtest.h>

namespace {
  SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT identity(uint32_t lowPart, LONG highPart, UINT targetId) {
    return {{lowPart, highPart}, targetId};
  }
}

TEST(VirtualDisplayIdentity, RejectsAnUnrelatedSudoOutput) {
  const auto retiring = identity(42, 7, 3);
  const auto unrelated = identity(42, 7, 4);

  EXPECT_FALSE(VDISPLAY::virtualDisplayIdentityMatchesForTest(
    retiring,
    LR"(\\?\DISPLAY#SMKD1CE#retiring)",
    unrelated,
    LR"(\\?\DISPLAY#SMKD1CE#unrelated)"
  ));
}

TEST(VirtualDisplayIdentity, MatchesTheExactUnpublishedDriverIdentity) {
  const auto retiring = identity(42, 7, 3);

  EXPECT_TRUE(VDISPLAY::virtualDisplayIdentityMatchesForTest(
    retiring,
    {},
    retiring,
    LR"(\\?\DISPLAY#SMKD1CE#candidate)"
  ));
}

TEST(VirtualDisplayIdentity, DoesNotFollowAnExactIdReusedByAPhysicalOutput) {
  const auto retiring = identity(42, 7, 3);

  EXPECT_FALSE(VDISPLAY::virtualDisplayIdentityMatchesForTest(
    retiring,
    {},
    retiring,
    LR"(\\?\DISPLAY#TCL03D4#physical)"
  ));
}

TEST(VirtualDisplayIdentity, FollowsTheLearnedPathAcrossTargetRenumbering) {
  const auto retiring = identity(42, 7, 3);
  const auto renumbered = identity(43, 8, 9);
  constexpr std::wstring_view learnedPath = LR"(\\?\DISPLAY#SMKD1CE#retiring)";

  EXPECT_TRUE(VDISPLAY::virtualDisplayIdentityMatchesForTest(
    retiring,
    learnedPath,
    renumbered,
    learnedPath
  ));
}

#endif
