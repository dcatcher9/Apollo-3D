// SPDX-License-Identifier: GPL-3.0-only
// TEST add-on only: pool work running the add-on's own code, which the add-on
// lifecycle fixture (test_addon_unload_runtime.cpp) holds across ReShade's
// add-on unload. The production package excludes this export.
#include <windows.h>
#include <d3dcompiler.h>

namespace {
  struct held_work {
    HANDLE proceed{}, returned{};
    volatile LONG *compiled{};
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
      ID3DBlob *code{}, *errors{};
      const bool compiled = SUCCEEDED(D3DCompile(source, sizeof(source) - 1, "held_pool_work", nullptr, nullptr, "main",
        "cs_5_0", 0, 0, &code, &errors)) && code;
      if (code) code->Release();
      if (errors) errors->Release();
      result = compiled ? 1 : -1;
    }
    InterlockedExchange(work->compiled, result ? result : -2);
    delete work;
  }
}

// Submits one pool callback of this add-on's code: it waits for `proceed`,
// compiles a shader and stores 1 in *compiled (-1 for a failed compile, -2
// when `proceed` never came); the pool signals `returned` after it returns.
extern "C" __declspec(dllexport) BOOL SunshineSbsTestHoldPoolWork(HANDLE proceed, HANDLE returned, volatile LONG *compiled) {
  if (!proceed || !returned || !compiled) return FALSE;
  auto *work = new held_work{proceed, returned, compiled};
  if (TrySubmitThreadpoolCallback(held_task, work, nullptr)) return TRUE;
  delete work;
  return FALSE;
}
