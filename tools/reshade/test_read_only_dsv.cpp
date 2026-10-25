// SPDX-License-Identifier: GPL-3.0-only
// The read-only depth-stencil view table (read_only_dsv.h), which decides
// the Generic path's bound-view boundaries (the contract matrix's D11c): a
// view created read-only is read-only until a writable view is created at its
// handle, a writable view is writable, a slot keeps its handle across later
// views, and a handle the table does not hold (never reported, as a
// descriptor copied into a DSV heap is not, or with no slot left) may be
// read-only. depth_addon.cpp feeds it from ReShade's init_resource_view
// (depth_stencil_read for a read-only view) and asks it at an unbind or a
// clear-like draw of the bound view; the capture owner's refusal of that copy
// is the device layer's (unbind_read_only_dsv cells).
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
    check(views.read_only(0x1000), "A handle whose creation was never reported may be read-only");
    views.created(0x1000, false);
    check(!views.read_only(0x1000), "A writable view is writable");
    views.created(0x1000, true);
    check(views.read_only(0x1000), "A read-only view is read-only");
    check(views.read_only(0x1008), "Another handle never reported may be read-only");
    views.created(0x1008, false);
    check(!views.read_only(0x1008) && views.read_only(0x1000), "Another handle keeps its own answer");
    views.created(0x1000, false);
    check(!views.read_only(0x1000), "A writable view created over a read-only one is writable");
    views.created(0x1000, true);
    check(views.read_only(0x1000), "A read-only view created again at its handle is read-only again");
    views.created(0, false);
    check(views.read_only(0), "A null handle is never recorded");
  }
  {
    // Every handle probes the same slots once the table is full: two slots,
    // two probes.
    sunshine_read_only_dsv::table<2, 2> views;
    views.created(0x1000, false);
    views.created(0x2000, true);
    check(!views.read_only(0x1000) && views.read_only(0x2000), "Writable and read-only views fill the table");
    views.created(0x3000, false);
    check(views.read_only(0x3000), "A writable view with no slot is not held, so it may be read-only");
    views.created(0x2000, false);
    check(!views.read_only(0x2000), "A full table still marks a held handle writable");
    views.created(0x1000, true);
    check(views.read_only(0x1000) && views.read_only(0x4000), "A full table keeps its held answers, and an unheld handle may be read-only");
  }
  if (failures) {
    std::fprintf(stderr, "%u read-only DSV check(s) failed\n", failures);
    return EXIT_FAILURE;
  }
  std::puts("PASS read-only depth-stencil views");
  return EXIT_SUCCESS;
}
