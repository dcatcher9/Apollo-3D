TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, TrustedInnerAuthoredSoftShadowAndUnoutlinedFill) {
  if (!trusted_inner_shaders_[0u]) GTEST_SKIP() << "Explicit trusted-inner shader root/pin required";
  UINT frame = 0x2900u;
  for (const bool coloured : {false,true}) for (const bool gray_wall : {false,true}) for (const bool shadow : {false,true}) {
    SCOPED_TRACE(frame);expect_authored_trusted_inner(coloured,gray_wall,shadow,false,true,frame++);
  }
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, TrustedInnerAuthoredOpaqueBlackOutlinePreservesExistingProtection) {
  if (!trusted_inner_shaders_[0u]) GTEST_SKIP() << "Explicit trusted-inner shader root/pin required";
  expect_authored_trusted_inner(false,false,false,true,true,0x2920u);
  expect_authored_trusted_inner(true,true,false,true,true,0x2921u);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, TrustedInnerAuthoredNoBlackOrOriginalBodyCanAcquireFill) {
  if (!trusted_inner_shaders_[0u]) GTEST_SKIP() << "Explicit trusted-inner shader root/pin required";
  expect_authored_trusted_inner(false,true,false,false,true,0x2922u,true,true);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, TrustedInnerAuthoredSameSceneWithoutTextCannotAcquireProtection) {
  if (!trusted_inner_shaders_[0u]) GTEST_SKIP() << "Explicit trusted-inner shader root/pin required";
  expect_authored_trusted_inner(false,false,true,false,false,0x2930u);
  expect_authored_trusted_inner(false,true,true,false,false,0x2931u);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, TrustedInnerAuthoredOriginalSupportDoesNotGateCurrentFill) {
  if (!trusted_inner_shaders_[0u]) GTEST_SKIP() << "Explicit trusted-inner shader root/pin required";
  expect_authored_trusted_inner(false,false,true,false,true,0x2940u,false);
  expect_authored_trusted_inner(true,true,false,false,true,0x2941u,false);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, TrustedInnerAuthoredMissingCurrentProbabilityCannotAcquireProtection) {
  if (!trusted_inner_shaders_[0u]) GTEST_SKIP() << "Explicit trusted-inner shader root/pin required";
  expect_authored_trusted_inner(false,false,true,false,true,0x2942u,true,false,0.0f);
  expect_authored_trusted_inner(true,true,false,false,true,0x2943u,true,false,0.0f);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, TrustedInnerAuthoredEqualDominantCurrentFamiliesAbstain) {
  if (!trusted_inner_shaders_[0u]) GTEST_SKIP() << "Explicit trusted-inner shader root/pin required";
  expect_trusted_inner_equal_families();
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, TrustedInnerAuthoredCurrentLifecycleAndOptionalPublisherFallback) {
  if (!trusted_inner_shaders_[0u]) GTEST_SKIP() << "Explicit trusted-inner shader root/pin required";
  expect_trusted_inner_current_lifecycle();
}
