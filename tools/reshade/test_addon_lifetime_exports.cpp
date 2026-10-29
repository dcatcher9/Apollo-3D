// SPDX-License-Identifier: GPL-3.0-only
// TEST add-on only: pool work running the add-on's own code, which the add-on
// lifecycle fixture (test_addon_unload_runtime.cpp) holds across ReShade's
// add-on unload, and the passive observations that fixture makes of the work
// the add-on itself leaves in flight there: its asynchronous log writer, its
// shader compiles and its renderer builds, and the order in which the
// exporter's owners go at each unload. The production package excludes
// these exports.
#include "async_log.h"
#include "game3d_renderer.h"
#include "game3d_shader_cache.h"
#include "game3d_test_teardown.h"

#include <cstdio>
#include <d3dcompiler.h>
#include <windows.h>

namespace {
  struct held_work {
    HANDLE proceed {}, returned {};
    volatile LONG *compiled {};
  };

  // Waits for the fixture, then compiles through the add-on's own D3DCompiler
  // import, as the content sampler's and the renderer's pool compiles do. The
  // pool signals `returned` only once this callback has returned into it.
  void CALLBACK held_task(PTP_CALLBACK_INSTANCE instance, void *context) {
    const auto *work = static_cast<const held_work *>(context);
    SetEventWhenCallbackReturns(instance, work->returned);
    LONG result = 0;
    if (WaitForSingleObject(work->proceed, 60000) == WAIT_OBJECT_0) {
      static constexpr char source[] = "RWBuffer<uint> output : register(u0);\n[numthreads(1, 1, 1)] void main() { output[0] = 1; }\n";
      ID3DBlob *code {}, *errors {};
      const bool compiled = SUCCEEDED(D3DCompile(source, sizeof(source) - 1, "held_pool_work", nullptr, nullptr, "main", "cs_5_0", 0, 0, &code, &errors)) && code;
      if (code) {
        code->Release();
      }
      if (errors) {
        errors->Release();
      }
      result = compiled ? 1 : -1;
    }
    InterlockedExchange(work->compiled, result ? result : -2);
    delete work;
  }
}  // namespace

// Submits one pool callback of this add-on's code: it waits for `proceed`,
// compiles a shader and stores 1 in *compiled (-1 for a failed compile, -2
// when `proceed` never came); the pool signals `returned` after it returns.
extern "C" __declspec(dllexport) BOOL SunshineSbsTestHoldPoolWork(HANDLE proceed, HANDLE returned, volatile LONG *compiled) {
  if (!proceed || !returned || !compiled) {
    return FALSE;
  }
  auto *work = new held_work {proceed, returned, compiled};
  if (TrySubmitThreadpoolCallback(held_task, work, nullptr)) {
    return TRUE;
  }
  delete work;
  return FALSE;
}

// Queues count lines "<prefix> <i> of <count>" through the add-on's
// asynchronous log writer, in order, as any add-on line is queued.
extern "C" __declspec(dllexport) BOOL SunshineSbsTestLog(const char *prefix, unsigned count) {
  if (!prefix) {
    return FALSE;
  }
  for (unsigned i = 1; i <= count; ++i) {
    char text[160];
    std::snprintf(text, sizeof(text), "%s %u of %u", prefix, i, count);
    sunshine_log::message(reshade::log::level::info, text);
  }
  return TRUE;
}

// The writer's queue: the lines not yet taken for writing, and whether a
// drain is under way (it may be writing one line it took).
extern "C" __declspec(dllexport) unsigned SunshineSbsTestLogBacklog(BOOL *draining) {
  auto &queue = sunshine_log::detail::queue();
  std::lock_guard<std::mutex> lock(queue.mutex);
  if (draining) {
    *draining = queue.draining ? TRUE : FALSE;
  }
  return unsigned(queue.lines.size());
}

// Shader entry points still loading or compiling on the pool
// (game3d_shader_cache.h, prepare), and the bytecode blobs kept.
extern "C" __declspec(dllexport) unsigned SunshineGame3DTestShaderEntriesPending(unsigned *blobs) {
  auto &store = sunshine_game3d::shader_cache::detail::store();
  std::lock_guard<std::mutex> lock(store.mutex);
  unsigned pending = 0;
  for (const auto &job : store.jobs) {
    pending += job.second->remaining.load(std::memory_order_acquire);
  }
  if (blobs) {
    *blobs = unsigned(store.blobs.size());
  }
  return pending;
}

// The renderer's background builds (game3d_renderer.h): those this process
// started, and those still registered (running, or ready and not adopted).
extern "C" __declspec(dllexport) void SunshineGame3DTestRendererBuilds(unsigned *started, unsigned *in_flight) {
  if (started) {
    *started = sunshine_game3d::renderer::builds_started();
  }
  if (in_flight) {
    *in_flight = sunshine_game3d::renderer::builds_in_flight();
  }
}

// A delay before each step of every later background build, so that the
// fixture destroys a device mid-build; 0 restores the ordinary build.
extern "C" __declspec(dllexport) void SunshineGame3DTestRendererStepDelay(unsigned milliseconds) {
  sunshine_game3d::renderer::build_test_hooks hooks;
  hooks.step_delay_ms = milliseconds;
  sunshine_game3d::renderer::set_build_test_hooks(hooks);
}

// The exporter's teardown order (game3d_test_teardown.h): the events after
// *cursor (a serial; 0 before the first read), at most capacity of them,
// oldest first, into out; *cursor becomes the serial of the last one
// copied. *skipped (optional) counts the events after the old cursor that
// the ring had already overwritten. Returns the number copied.
extern "C" __declspec(dllexport) unsigned SunshineGame3DTestTeardownOrder(std::uint64_t *cursor,
    sunshine_game3d::test::teardown::event *out, unsigned capacity, std::uint64_t *skipped) {
  if (!cursor || (!out && capacity)) {
    return 0;
  }
  std::uint64_t lost = 0;
  const auto copied = sunshine_game3d::test::teardown::read(*cursor, out, capacity, lost);
  if (skipped) {
    *skipped = lost;
  }
  return copied;
}
