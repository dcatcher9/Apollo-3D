// SPDX-License-Identifier: GPL-3.0-only
// The read-only depth-stencil view table (read_only_dsv.h), which decides
// the Generic path's bound-view boundaries (the contract matrix's D11c): a
// view created read-only is read-only until a writable view is created at its
// handle, a handle never created read-only is writable, a slot keeps its
// handle across later views, and once a read-only view found no slot every
// handle the table does not hold may be read-only. depth_addon.cpp feeds it
// from ReShade's init_resource_view (depth_stencil_read for a read-only view)
// and asks it at an unbind or a clear-like draw of the bound view; the
// capture owner's refusal of that copy is the device layer's
// (unbind_read_only_dsv cells).
#include "read_only_dsv.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
  unsigned failures {};

  void check(bool value, const char *what) {
    if (!value) {
      std::fprintf(stderr, "FAIL %s\n", what);
      ++failures;
    }
  }
}  // namespace

int main() {
  {
    sunshine_read_only_dsv::table<64> views;
    check(!views.read_only(0x1000), "A handle never created read-only is writable");
    views.created(0x1000, false);
    check(!views.read_only(0x1000), "A writable view is writable");
    views.created(0x1000, true);
    check(views.read_only(0x1000), "A read-only view is read-only");
    check(!views.read_only(0x1008), "Another handle keeps its own answer");
    views.created(0x1000, false);
    check(!views.read_only(0x1000), "A writable view created over a read-only one is writable");
    views.created(0x1000, true);
    check(views.read_only(0x1000), "A read-only view created again at its handle is read-only again");
    views.created(0, true);
    check(!views.overflowed() && !views.read_only(0x2000), "A null handle is never recorded");
  }
  {
    // Every handle probes the same slots once the table is full: two slots,
    // two probes.
    sunshine_read_only_dsv::table<2, 2> views;
    views.created(0x1000, true);
    views.created(0x2000, true);
    check(views.read_only(0x1000) && views.read_only(0x2000) && !views.overflowed(), "Read-only views fill the table");
    check(!views.read_only(0x3000), "A writable handle is writable while nothing overflowed");
    views.created(0x3000, false);
    check(!views.overflowed() && !views.read_only(0x3000), "A writable view needs no slot");
    views.created(0x2000, false);
    check(!views.read_only(0x2000), "A full table still marks a held handle writable");
    views.created(0x3000, true);
    check(views.overflowed(), "A read-only view with no slot overflows the table");
    check(views.read_only(0x3000) && views.read_only(0x4000), "After an overflow any handle the table does not hold may be read-only");
    check(views.read_only(0x1000) && !views.read_only(0x2000), "A held handle keeps its answer after an overflow");
  }
  if (failures) {
    std::fprintf(stderr, "%u read-only DSV check(s) failed\n", failures);
    return EXIT_FAILURE;
  }
  std::puts("PASS read-only depth-stencil views");
  return EXIT_SUCCESS;
}
