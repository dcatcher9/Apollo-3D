#include <gtest/gtest.h>
#include "src/host_sbs_current_outline_observation.h"

namespace {
  namespace owner = models::current_outline_observation_v1;
  constexpr owner::identity_t current {17, 29000, 8, 91, 1};

  TEST(HostSbsCurrentOutlineObservation, QueuedOwnerDoesNotRequireCompletedPublication) {
    EXPECT_EQ(owner::classify(17,29000,current,current,true,true,3,{},false),owner::role_e::queued);
  }

  TEST(HostSbsCurrentOutlineObservation, PendingRequiresExactIdentityAndTicket) {
    for (unsigned field=0;field<5;++field) {
      auto pending=current;
      if(field==0)++pending.frame;
      if(field==1)++pending.observation_us;
      if(field==2)++pending.generation;
      if(field==3)++pending.token;
      if(field==4)++pending.work;
      EXPECT_EQ(owner::classify(17,29000,current,pending,true,true,3,{},false),owner::role_e::unavailable);
    }
    EXPECT_EQ(owner::classify(17,29000,current,current,true,true,0,{},false),owner::role_e::unavailable);
    EXPECT_EQ(owner::classify(17,29000,current,current,false,true,3,{},false),owner::role_e::unavailable);
    EXPECT_EQ(owner::classify(17,29000,current,current,true,false,3,{},false),owner::role_e::unavailable);
  }

  TEST(HostSbsCurrentOutlineObservation, CompletedOwnerRequiresCurrentPublication) {
    EXPECT_EQ(owner::classify(17,29000,current,{},false,false,0,current,true),owner::role_e::completed);
    EXPECT_EQ(owner::classify(17,29000,current,{},false,false,0,current,false),owner::role_e::unavailable);
    auto stale=current;++stale.token;
    EXPECT_EQ(owner::classify(17,29000,current,{},false,false,0,stale,true),owner::role_e::unavailable);
    // A later admission alone does not revoke an otherwise valid earlier publication.
    EXPECT_EQ(owner::classify(17,29000,current,stale,true,false,0,current,true),owner::role_e::completed);
  }

  TEST(HostSbsCurrentOutlineObservation, WrongRequestedFrameAndInvalidIdentityFailClosed) {
    EXPECT_EQ(owner::classify(18,29000,current,current,true,true,3,current,true),owner::role_e::unavailable);
    auto invalid=current;invalid.token=0;
    EXPECT_EQ(owner::classify(17,29000,invalid,invalid,true,true,3,invalid,true),owner::role_e::unavailable);
    invalid=current;invalid.work=3;
    EXPECT_EQ(owner::classify(17,29000,invalid,invalid,true,true,3,invalid,true),owner::role_e::unavailable);
  }
}
