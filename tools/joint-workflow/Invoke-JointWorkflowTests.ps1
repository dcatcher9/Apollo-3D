<#
.SYNOPSIS
Builds and runs the host/client workflow boundary gate without a GPU run or a connected headset.
#>
[CmdletBinding()]
param(
  [string] $HostRoot = '',
  [string] $ClientRoot = '',
  [string] $BuildDirectory = 'cmake-build-relwithdebinfo',
  [string] $SbsbenchPython = 'E:\ApolloDev\modelopt-py312\Scripts\python.exe',
  [string] $JavaHome = $env:JAVA_HOME,
  [switch] $SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $HostRoot) { $HostRoot = Join-Path $PSScriptRoot '../..' }
$HostRoot = (Resolve-Path -LiteralPath $HostRoot).Path
if (-not $ClientRoot) { $ClientRoot = Join-Path $HostRoot '../moonlight-android' }
$ClientRoot = (Resolve-Path -LiteralPath $ClientRoot).Path
if (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
  $BuildDirectory = Join-Path $HostRoot $BuildDirectory
}
$BuildDirectory = (Resolve-Path -LiteralPath $BuildDirectory).Path
$cache = Get-Content -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt')
function Read-CacheValue([string] $Name) {
  $line = $cache | Where-Object { $_ -match ('^' + [regex]::Escape($Name) + ':[^=]+=') } | Select-Object -First 1
  if (-not $line) { throw "The configured host build lacks $Name." }
  return $line.Substring($line.IndexOf('=') + 1)
}
$ninja = Read-CacheValue 'CMAKE_MAKE_PROGRAM'
$compiler = Read-CacheValue 'CMAKE_CXX_COMPILER'
$cCompiler = Read-CacheValue 'CMAKE_C_COMPILER'
$tensorRt = Read-CacheValue 'TENSORRT_DIR'
$testExecutable = Join-Path $BuildDirectory 'tests/test_sunshine.exe'
$gradle = Join-Path $ClientRoot 'gradlew.bat'
foreach ($required in @($ninja, $compiler, $cCompiler, $SbsbenchPython, $gradle)) {
  if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required runtime not found: $required" }
}
$resultsDirectory = Join-Path $BuildDirectory ('joint-workflow-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $resultsDirectory | Out-Null
$previousPath = $env:PATH
$previousJava = $env:JAVA_HOME
$originalLocation = Get-Location
$stages = [Collections.Generic.List[object]]::new()
$hostCommit = $null
$clientCommit = $null

function Invoke-GateStage([string] $Name, [string] $Program, [string[]] $Arguments, [string] $Directory) {
  $log = Join-Path $resultsDirectory ($Name + '.log')
  Write-Host "Running $Name..."
  Set-Location -LiteralPath $Directory
  $started = Get-Date
  # Windows PowerShell wraps native stderr (including successful unittest dots)
  # in ErrorRecords. The native exit code is the stage outcome.
  $stageErrorPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = 'Continue'
    & $Program @Arguments *> $log
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $stageErrorPreference
  }
  $stages.Add([ordered]@{ name = $Name; exit_code = $code; seconds = ((Get-Date) - $started).TotalSeconds; log = $log })
  if ($code -ne 0) {
    Get-Content -LiteralPath $log -Tail 70 | Write-Host
    throw "$Name failed (exit $code). See $log"
  }
}

try {
  # Read the actual configured runtime locations; do not install or silently select toolchains.
  $env:PATH = ((Split-Path -Parent $compiler), (Join-Path $tensorRt 'bin'), (Join-Path $tensorRt 'lib'), 'C:\Program Files\nodejs', $previousPath) -join ';'
  if ($JavaHome) { $env:JAVA_HOME = (Resolve-Path -LiteralPath $JavaHome).Path }
  $hostCommit = & git -C $HostRoot rev-parse HEAD
  if ($LASTEXITCODE -ne 0) { throw 'Could not identify the host checkout.' }
  $clientCommit = & git -C $ClientRoot rev-parse HEAD
  if ($LASTEXITCODE -ne 0) { throw 'Could not identify the client checkout.' }
  $fingerprint = "import sys, numpy, PIL, onnx; print(sys.executable); print(sys.version); print('numpy', numpy.__version__); print('Pillow', PIL.__version__); print('onnx', onnx.__version__)"
  Invoke-GateStage 'runtime' $SbsbenchPython @('-c', $fingerprint) $HostRoot
  if (-not $SkipBuild) {
    Invoke-GateStage 'host-build' $ninja @('-C', $BuildDirectory, 'sunshine', 'test_sunshine') $HostRoot
  }
  if (-not (Test-Path -LiteralPath $testExecutable -PathType Leaf)) { throw "Build the host test binary first: $testExecutable" }
  $nativeFilter = 'Offline*:GpuWorkloadArbiter.*:Rtsp*:Input*:ProcessTest.*:WindowsQpc*:WindowsLocalPresenter*:WebUiDesign.*'
  $nativeXml = Join-Path $resultsDirectory 'host.xml'
  Invoke-GateStage 'host' $testExecutable @("--gtest_filter=$nativeFilter", "--gtest_output=xml:$nativeXml") $BuildDirectory
  [xml] $nativeResult = Get-Content -LiteralPath $nativeXml -Raw
  if ([int] $nativeResult.testsuites.tests -eq 0) { throw 'The native gate selected no tests.' }
  Invoke-GateStage 'evaluator' $SbsbenchPython @(
    '-m', 'unittest', 'test_adaptive_state_contract', 'test_host_sbs_shader_manifest',
    'test_depth_coordinate_v2_contract', 'test_compare_runs', 'test_eval_parallel', 'test_report_parallel'
  ) (Join-Path $HostRoot 'tools/sbsbench')
  $commonC = Join-Path $ClientRoot 'app/src/main/jni/moonlight-core/moonlight-common-c'
  $packetTest = Join-Path $resultsDirectory 'video-packet-size-test.exe'
  Invoke-GateStage 'client-packet-build' $cCompiler @(
    '-std=c11', '-DLC_DEBUG', '-DHAS_SOCKLEN_T',
    '-I', (Join-Path $commonC 'src'), '-I', (Join-Path $commonC 'reedsolomon'),
    '-I', (Join-Path $commonC 'enet/include'),
    (Join-Path $commonC 'tests/VideoPacketSizeTest.c'),
    (Join-Path $commonC 'src/SdpGenerator.c'),
    (Join-Path $commonC 'src/RtpVideoQueue.c'), (Join-Path $commonC 'reedsolomon/rs.c'),
    (Join-Path $commonC 'src/VideoDepacketizer.c'), (Join-Path $commonC 'src/PlatformCrypto.c'),
    (Join-Path $commonC 'src/ByteBuffer.c'), (Join-Path $commonC 'src/LinkedBlockingQueue.c'),
    (Join-Path $commonC 'src/Platform.c'),
    '-lcrypto', '-lwinmm', '-lws2_32', '-o', $packetTest
  ) $HostRoot
  Invoke-GateStage 'client-packet' $packetTest @() $HostRoot
  $clientArguments = @(':app:testNonRoot_gameDebugUnitTest', '--console=plain')
  foreach ($testClass in @(
    'com.limelight.nvstream.http.NvHTTP*Test',
    'com.limelight.nvstream.NvConnection*Test',
    'com.limelight.utils.ClientSbs*Test',
    'com.limelight.utils.Stereo3DRendererSchedulingTest',
    'com.limelight.ui.XrStreamPresenterTransitionTest',
    'com.limelight.ui.StreamContainerSurfaceHandoffContractTest',
    'com.limelight.binding.video.*Test',
    'com.limelight.preferences.XrSessionSettingsControllerTest',
    'com.limelight.preferences.session.SessionSettingsStoreTest',
    'com.limelight.ui.xrcontrols.*Test'
  )) { $clientArguments += @('--tests', $testClass) }
  Invoke-GateStage 'client-jvm' $gradle $clientArguments $ClientRoot
  Write-Host "Joint workflow gate passed. Evidence: $resultsDirectory"
} finally {
  [ordered]@{
    host_root = $HostRoot; client_root = $ClientRoot
    host_commit = $hostCommit; client_commit = $clientCommit
    python = $SbsbenchPython; java_home = $env:JAVA_HOME; skip_build = [bool] $SkipBuild
    stages = @($stages.ToArray())
  } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resultsDirectory 'summary.json') -Encoding UTF8
  $env:PATH = $previousPath
  $env:JAVA_HOME = $previousJava
  Set-Location -LiteralPath $originalLocation
}
