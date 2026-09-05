# Joint workflow tests

The local gate runs Sunshine 3D native workflow tests, evaluator contract/provenance tests,
Moonlight 3D native packet/FEC tests and JVM tests in sequence. It checks connection ownership, permissions, mode transitions,
offline job/transport bounds, and capture/presentation timing without opening the host or using a
headset. Run it with both checkouts available and their existing build prerequisites configured:

```powershell
& tools/joint-workflow/Invoke-JointWorkflowTests.ps1 `
  -ClientRoot ../moonlight-android `
  -JavaHome 'C:\path\to\the\client-supported-jdk' `
  -SbsbenchPython 'C:\path\to\the\validated\python.exe'
```

On the development machine, use the interpreter and native runtime described in [AGENTS.md](../AGENTS.md).
The script's default interpreter is that machine's validated runtime; another machine must pass its
validated interpreter explicitly. The client owns its JDK/SDK requirements in its build guide.
The script reads Ninja, compiler and TensorRT locations from the existing host CMake cache, uses
official Windows Node, and installs nothing. It builds `sunshine` and `test_sunshine` before testing;
`-SkipBuild` is for a caller that has already built these sources. A failed stage stops the gate.
The packet test compiles the actual client ANNOUNCE writer, receive queue, AES-GCM, Reed–Solomon and depacketizer
implementations with the host's configured C compiler. It checks first, interior and final data-shard
loss at three packet sizes for HEVC and AV1 through the decoder callback.

Each run writes logs, native XML, both checkout commits, runtime identity and stage exit codes below
`cmake-build-relwithdebinfo/joint-workflow-<time>/`. Client XML remains under the client's standard
`app/build/test-results/testNonRoot_gameDebugUnitTest/` directory. Uncommitted changes are included
in a local run, so commit identities alone are not a source-integrity attestation.

The `Portable workflow boundaries` GitHub Actions workflow compiles the same clock, retained
presentation scheduling, and raw-frame transport tests against the pinned GoogleTest submodule,
and checks generated source contracts using the Python standard library. It can also be run
without the complete host build:

```text
cmake -S tools/joint-workflow -B cmake-build-portable -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-portable
ctest --test-dir cmake-build-portable --output-on-failure
```

The companion Moonlight 3D repository also runs two portable jobs:

- [Transport contracts](https://github.com/dcatcher9/moonlight-android/blob/moonlight-noir/.github/workflows/transport-contracts.yml)
  compiles the production ANNOUNCE/FEC/depacketizer fixture using a C compiler and OpenSSL.
- [Presentation contracts](https://github.com/dcatcher9/moonlight-android/blob/moonlight-noir/.github/workflows/presentation-contracts.yml)
  uses its existing Gradle wrapper and JDK 25 to run the production shared transaction and swap
  proof with the same JUnit tests used by the Android suite. From the client checkout, run
  `.\gradlew.bat -p tools/workflow-tests test`; see the client's
  [standalone project](https://github.com/dcatcher9/moonlight-android/blob/moonlight-noir/tools/workflow-tests/README.md).
  This build does not configure Android plugins or require Android SDK, model assets, or native libraries.

The host portable job needs a C++20 compiler, CMake and the checked-out GoogleTest submodule. It does not
download model assets or claim to exercise Windows capture, TensorRT, NVENC, an Android compositor,
packet loss on a real network, or physical input/display transitions. Use the canonical
[Host SBS evaluation loop](../tools/sbsbench/README.md#required-evaluation-loop) and the explicit
[offline worker smoke](../tests/integration/Invoke-OfflineSbsWorkerSmoke.ps1) when those paths change;
GPU runs remain serial. Device acceptance still follows each repository's physical-test procedure.
