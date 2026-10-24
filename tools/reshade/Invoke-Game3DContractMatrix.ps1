<#
.SYNOPSIS
Builds the standalone Game 3D add-on and runs its API contract matrix, writing one JSON line per case.

.DESCRIPTION
See README.md, "API contract matrix". The Sunshine build forces the add-on's tests off, so this
configures tools/reshade on its own with the test add-on, runs `ctest -L contract` and collects
the results under <BuildDirectory>/contract-matrix-<time>:

- contract-matrix.jsonl: one line per case of the hook layer (L1) and of each runtime family (L3),
  {layer, family, id, seed, dimensions, legal, rule, expected, observed, verdict, rows, evidence}
  as its executable judged it against the oracle and the known-gap table, and one line per contract
  test with its ctest result and PASS summary (the pure and device layers report there);
- summary.json: verdict counts per family, the cases of each known-gap row, and any problem.

The runner then checks those cases against test_game3d_api_contract_known_gaps.h itself: a case
may only meet the oracle, fail as a recorded gap or sit in a pending row, and each expected-failure
row of a family that ran whole must be the only row of some failing case.

The runtime layer is registered only when -ReShadeRuntime and -FrozenShaders are given (or the
build is already configured with them). It and the device layer open GPU work in a hidden
window: run them alone, with no game running, or pass -SkipGpu.

Exit codes: 0 every contract test passed and every case is accounted for; 1 a test failed or a
case or row is not accounted for; 2 setup failed.
#>
[CmdletBinding()]
param(
  [string] $BuildDirectory = 'cmake-build-reshade',
  [string] $ReShadeSdk = '',
  [string] $ReShadeImGui = '',
  [string] $ReShadeRuntime = '',
  [string] $FrozenShaders = '',
  [string] $Toolchain = 'C:\msys64\ucrt64\bin',
  [switch] $SkipBuild,
  [switch] $SkipGpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$sourceDirectory = $PSScriptRoot
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
  $BuildDirectory = Join-Path $repositoryRoot $BuildDirectory
}

function Invoke-Native([string] $Name, [string] $Program, [string[]] $Arguments, [string] $Log) {
  Write-Host "Running $Name..."
  # Windows PowerShell wraps native stderr in ErrorRecords; the exit code is the outcome.
  $preference = $ErrorActionPreference
  try {
    $ErrorActionPreference = 'Continue'
    & $Program @Arguments *> $Log
    return $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $preference
  }
}

# The known-gap rows of one table, in order: {id, status}.
function Read-GapRows([string] $Table) {
  $header = Get-Content -LiteralPath (Join-Path $sourceDirectory 'test_game3d_api_contract_known_gaps.h') -Raw
  $start = $header.IndexOf("$Table[] {")
  if ($start -lt 0) { throw "test_game3d_api_contract_known_gaps.h has no $Table table." }
  $end = $header.IndexOf('};', $start)
  $rows = @()
  foreach ($match in [regex]::Matches($header.Substring($start, $end - $start), '\{"([^"]+)", &rules::\w+, kind::(expected_failure|pending)')) {
    $rows += [pscustomobject]@{ id = $match.Groups[1].Value; status = $match.Groups[2].Value }
  }
  return $rows
}

$previousPath = $env:PATH
$results = $null
try {
  if (-not (Test-Path -LiteralPath (Join-Path $Toolchain 'cmake.exe') -PathType Leaf)) { throw "No CMake in the toolchain directory $Toolchain." }
  $env:PATH = $Toolchain + ';' + $previousPath
  $cmake = Join-Path $Toolchain 'cmake.exe'
  $ctest = Join-Path $Toolchain 'ctest.exe'
  New-Item -ItemType Directory -Force -Path $BuildDirectory | Out-Null
  $results = Join-Path $BuildDirectory ('contract-matrix-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
  New-Item -ItemType Directory -Path $results | Out-Null
  if (-not $SkipBuild) {
    $configure = @('-S', $sourceDirectory, '-B', $BuildDirectory, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=RelWithDebInfo', '-DBUILD_TESTING=ON', '-DSUNSHINE_SBS_RUNTIME_TESTS=ON')
    foreach ($setting in @(@('RESHADE_SDK_DIR', $ReShadeSdk), @('RESHADE_IMGUI_DIR', $ReShadeImGui), @('SUNSHINE_GAME3D_RESHADE_RUNTIME', $ReShadeRuntime), @('SUNSHINE_GAME3D_FROZEN_SHADERS', $FrozenShaders))) {
      if ($setting[1]) { $configure += ('-D' + $setting[0] + '=' + (Resolve-Path -LiteralPath $setting[1]).Path.Replace('\', '/')) }
    }
    if ((Invoke-Native 'configure' $cmake $configure (Join-Path $results 'configure.log')) -ne 0) { throw "Configuring the add-on failed; see $results\configure.log." }
    if ((Invoke-Native 'build' $cmake @('--build', $BuildDirectory) (Join-Path $results 'build.log')) -ne 0) { throw "Building the add-on failed; see $results\build.log." }
  }
} catch {
  $env:PATH = $previousPath
  Write-Error $_ -ErrorAction Continue
  exit 2
}

try {
  $junit = Join-Path $results 'ctest.xml'
  $arguments = @('--test-dir', $BuildDirectory, '-L', 'contract', '--output-on-failure', '--output-junit', $junit, '--test-output-size-passed', '65536', '--test-output-size-failed', '65536')
  if ($SkipGpu) { $arguments += @('-LE', '^(gpu|device)$') }
  # The hook layer writes its case lines where this names; the runtime
  # families write theirs into their own output directories.
  $hookLines = Join-Path $results 'contract-hooks.jsonl'
  $env:SUNSHINE_CONTRACT_MATRIX_JSONL = $hookLines
  try {
    $ctestCode = Invoke-Native 'ctest -L contract' $ctest $arguments (Join-Path $results 'ctest.log')
  } finally {
    Remove-Item Env:SUNSHINE_CONTRACT_MATRIX_JSONL -ErrorAction SilentlyContinue
  }
  if (-not (Test-Path -LiteralPath $junit -PathType Leaf)) { throw "ctest wrote no results; see $results\ctest.log." }

  $lines = [Collections.Generic.List[string]]::new()
  $problems = [Collections.Generic.List[string]]::new()
  $families = [ordered]@{}
  # The runtime layer's Streamline families share sl_runtime_rows.
  $gapTables = @{ 'L1 sl' = (Read-GapRows 'sl_rows'); 'L1 ngx' = (Read-GapRows 'ngx_rows'); 'L3' = (Read-GapRows 'sl_runtime_rows');
    'L3 ngx' = (Read-GapRows 'ngx_runtime_rows'); 'L3 generic' = (Read-GapRows 'generic_runtime_rows'); 'L3 ui' = (Read-GapRows 'ui_runtime_rows') }

  # A layer's case lines: counted per family, each verdict accounted for, and
  # (for a family that ran whole) each expected-failure row the only row of
  # some failing case. A runtime family whose scope is not whole samples its
  # release's cells, so it cannot show a row stale.
  function Add-CaseLines([string] $File, [bool] $Complete) {
    $seen = [ordered]@{}
    foreach ($text in Get-Content -LiteralPath $File) {
      if (-not $text) { continue }
      $lines.Add($text)
      $line = $text | ConvertFrom-Json
      $key = "$($line.layer) $($line.family)"
      $whole = $Complete -and (-not ($line.PSObject.Properties.Name -contains 'scope') -or $line.scope -eq 'whole')
      if (-not $seen.Contains($key)) { $seen[$key] = [ordered]@{ cases = 0; complete = $whole; rows = @{}; alone = @{} } }
      $counts = $seen[$key]
      $counts.cases += 1
      $verdict = [string]$line.verdict
      if ($counts.Contains($verdict)) { $counts[$verdict] += 1 } else { $counts[$verdict] = 1 }
      foreach ($row in @($line.rows)) { $counts.rows[$row] = 1 + [int]$counts.rows[$row] }
      if ($verdict -eq 'known_failure' -and @($line.rows).Count -eq 1) { $counts.alone[[string]@($line.rows)[0]] = $true }
      if (@('meets', 'known_failure', 'pending') -notcontains $verdict) {
        $problems.Add("$key case $($line.id) ($($line.rule)): $verdict")
      }
    }
    foreach ($key in $seen.Keys) {
      $counts = $seen[$key]
      $table = if ($gapTables.ContainsKey($key)) { $gapTables[$key] } else { $gapTables[$key.Split(' ')[0]] }
      foreach ($row in $table) {
        if ($counts.complete -and $row.status -eq 'expected_failure' -and -not $counts.alone.ContainsKey($row.id)) {
          $problems.Add("$key known-gap row $($row.id) is the only row of no failing case")
        }
      }
      $counts.Remove('alone')
      $families[$key] = $counts
    }
  }

  foreach ($case in ([xml](Get-Content -LiteralPath $junit -Raw)).testsuite.testcase) {
    $name = $case.name
    $status = if ($case.SelectSingleNode('failure')) { 'fail' } elseif ($case.SelectSingleNode('skipped')) { 'skipped' } else { 'pass' }
    $output = if ($case.SelectSingleNode('system-out')) { [string]$case.'system-out' } else { '' }
    if ($status -eq 'fail') { $problems.Add("ctest $name failed") }
    $layer = switch -Regex ($name) {
      '^reshade_contract_matrix_cpu$' { 'L0+L1'; break }
      '^reshade_contract_matrix_device$' { 'L2'; break }
      '^reshade_contract_matrix_runtime_output$' { 'L3 setup'; break }
      '^reshade_contract_matrix_runtime_' { 'L3'; break }
      default { 'check' }
    }
    $summary = @($output -split "`r?`n" | Where-Object { $_ -match '^(PASS|FAIL|MEASURE contract-runtime family=.* cases=)' })
    $lines.Add((ConvertTo-Json -Compress -Depth 4 ([ordered]@{ layer = $layer; test = $name; verdict = $status; evidence = [ordered]@{ seconds = [double]$case.time; summary = $summary } })))
    $file = switch ($layer) {
      'L0+L1' { $hookLines; break }
      'L3' { $family = $name.Substring('reshade_contract_matrix_runtime_'.Length); Join-Path $BuildDirectory "contract_runtime/$family/contract-$family.jsonl"; break }
      default { $null }
    }
    if (-not $file) { continue }
    if (Test-Path -LiteralPath $file -PathType Leaf) {
      Add-CaseLines $file ($status -eq 'pass')
    } elseif ($status -ne 'skipped') {
      $problems.Add("$name wrote no case lines ($file)")
    }
  }
  # UTF-8 without a byte-order mark, which would corrupt the first JSON line.
  $utf8 = New-Object Text.UTF8Encoding $false
  [IO.File]::WriteAllLines((Join-Path $results 'contract-matrix.jsonl'), $lines, $utf8)
  $commit = & git -C $repositoryRoot rev-parse HEAD
  $summaryJson = [ordered]@{ commit = $commit; ctest_exit = $ctestCode; skipped_gpu = [bool]$SkipGpu; families = $families; problems = $problems } | ConvertTo-Json -Depth 6
  [IO.File]::WriteAllText((Join-Path $results 'summary.json'), $summaryJson, $utf8)
  if (-not @($families.Keys | Where-Object { $_ -like 'L3 *' }).Count -and -not $SkipGpu) {
    Write-Host 'The runtime layer is not registered: configure with -ReShadeRuntime and -FrozenShaders to run it.'
  }
  foreach ($family in $families.Keys) {
    $counts = $families[$family]
    Write-Host ("{0}: {1} cases, {2} meet the oracle, {3} recorded gaps, {4} pending" -f $family, $counts.cases, [int]$counts['meets'], [int]$counts['known_failure'], [int]$counts['pending'])
  }
  foreach ($problem in $problems) { Write-Host "PROBLEM $problem" }
  Write-Host "Results: $results"
  if ($ctestCode -ne 0 -or $problems.Count) { exit 1 }
  exit 0
} catch {
  Write-Error $_ -ErrorAction Continue
  exit 2
} finally {
  $env:PATH = $previousPath
}
