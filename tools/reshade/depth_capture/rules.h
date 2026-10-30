// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Pure field predicates shared by capture selection. Source identity, liveness,
// ownership, producer-fence queries and queue admission remain at their callers.
#include "../streamline_depth_capture_test_access.h"

namespace sunshine_streamline::depth_capture {
  SUNSHINE_CAPTURE_INTERNAL {
    inline bool completed_pixels(const slot &value) {
      return value.id && value.texture && !value.nomination_only && !value.preservation_only &&
        value.source_nominated && value.finished && value.success && !value.invalid && !value.nomination_invalid &&
        value.producer_submitted;
    }

    inline bool pending_snapshot_core(const slot &value, status result) {
      return !value.nomination_only && !value.invalid && !value.nomination_invalid &&
        (!value.finished || value.success) && (result == status::recorded || result == status::submitted);
    }
  }
}
