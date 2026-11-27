/**
 * @file tests/unit/platform/windows/test_user_token.cpp
 * @brief Tests for the interactive-user token helpers.
 */
#include "src/platform/windows/user_token.h"

#include <gtest/gtest.h>

namespace {
  bool thread_token_is_admin() {
    SID_IDENTIFIER_AUTHORITY authority = SECURITY_NT_AUTHORITY;
    PSID administrators = nullptr;
    if (!AllocateAndInitializeSid(&authority, 2, SECURITY_BUILTIN_DOMAIN_RID, DOMAIN_ALIAS_RID_ADMINS, 0, 0, 0, 0, 0, 0, &administrators)) {
      ADD_FAILURE() << "AllocateAndInitializeSid failed: " << GetLastError();
      return false;
    }
    BOOL member = FALSE;
    // A null token makes Windows duplicate the thread's own token for the check.
    const BOOL checked = CheckTokenMembership(nullptr, administrators, &member);
    FreeSid(administrators);
    EXPECT_TRUE(checked) << GetLastError();
    return member != FALSE;
  }
}  // namespace

TEST(UserToken, AdministratorCheckAcceptsTheProcessPrimaryToken) {
  HANDLE process_token = nullptr;
  ASSERT_TRUE(OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY | TOKEN_DUPLICATE, &process_token)) << GetLastError();
  // The callers pass primary tokens, which CheckTokenMembership() itself refuses. The check must
  // still answer, and the answer must match Windows' own check of this thread, whatever account
  // runs the tests.
  const auto admin = platf::token_is_admin(process_token);
  ASSERT_TRUE(admin.has_value());
  EXPECT_EQ(*admin, thread_token_is_admin());
  CloseHandle(process_token);
}
