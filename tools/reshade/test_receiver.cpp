// SPDX-License-Identifier: GPL-3.0-only
#include "src/logging.h"
#include "src/platform/windows/reshade_bridge.h"
#include "test_receiver_api.h"

#include <chrono>
#include <cstdio>
#include <exception>

// Keep the production receiver's diagnostics without starting the host or its service stack.
boost::log::sources::severity_logger<int> info(2);
boost::log::sources::severity_logger<int> warning(3);

namespace {
  struct receiver_test_t {
    RECT source;
    platf::foreground_window::observation_t observation;
    platf::reshade_bridge::receiver_t receiver;

    receiver_test_t(ID3D11Device *device, ID3D11DeviceContext *context, HWND window, DWORD pid, RECT rect):
        source(rect),
        receiver(device, context, [this] {
          return observation;
        }) {
      observation.status = platf::foreground_window::status_e::ok;
      observation.window = reinterpret_cast<std::uintptr_t>(window);
      observation.process_id = pid;
      observation.client_screen_rect = {rect.left, rect.top, rect.right, rect.bottom};
    }
  };
}  // namespace

extern "C" __declspec(dllexport) void *SunshineReceiverTestCreate(ID3D11Device *device, ID3D11DeviceContext *context, HWND window, DWORD pid, const RECT *source) {
  if (!device || !context || !window || !pid || !source || source->right <= source->left || source->bottom <= source->top) {
    return nullptr;
  }
  try {
    return new receiver_test_t(device, context, window, pid, *source);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Receiver fixture creation failed: %s\n", error.what());
    return nullptr;
  }
}

extern "C" __declspec(dllexport) void SunshineReceiverTestSetFocus(void *opaque, BOOL focused) {
  if (auto *test = static_cast<receiver_test_t *>(opaque)) {
    test->observation.status = focused ? platf::foreground_window::status_e::ok : platf::foreground_window::status_e::no_foreground;
  }
}

extern "C" __declspec(dllexport) int SunshineReceiverTestPoll(void *opaque, sunshine_receiver_test_frame *output) {
  auto *test = static_cast<receiver_test_t *>(opaque);
  if (!test || !output) {
    return -1;
  }
  *output = {};
  try {
    const auto frame = test->receiver.poll(test->source, 2 * (test->source.right - test->source.left), test->source.bottom - test->source.top);
    if (!frame) {
      return 0;
    }
    output->texture = frame->texture;
    output->linear = frame->linear;
    output->sequence = frame->sequence;
    output->timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(frame->timestamp.time_since_epoch()).count();
    return 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Receiver fixture poll failed: %s\n", error.what());
    return -1;
  }
}

extern "C" __declspec(dllexport) void SunshineReceiverTestDestroy(void *opaque) {
  delete static_cast<receiver_test_t *>(opaque);
}
