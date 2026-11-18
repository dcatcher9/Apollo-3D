#requires -Version 7.2
<#
.SYNOPSIS
Qualify offscreen native NVENC inputs with software-decoded authored pixels.
.DESCRIPTION
Runs eight explicitly opted-in native cases serially. Retains all artifacts in a new directory
under this checkout's cmake-build-* tree. Tool paths and their expected hashes are caller supplied.
The native producer alone is not a content verdict: this script probes native encoded bit depth,
checks lossy decoded interiors against authored planes, and optionally compares exact decoded
hashes with a prior qualified run. No capture, host ports, renderer, model, or timing claim.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $TestBinary,
    [Parameter(Mandatory)][string] $FFmpeg,
    [Parameter(Mandatory)][string] $FFprobe,
    [Parameter(Mandatory)][string] $ExpectedTestBinarySha256,
    [Parameter(Mandatory)][string] $ExpectedFFmpegSha256,
    [Parameter(Mandatory)][string] $ExpectedFFprobeSha256,
    [Parameter(Mandatory)][string] $SourcePins,
    [Parameter(Mandatory)][string] $OutputRoot,
    [string] $ControlManifest,
    [ValidateSet(12, 8)][int] $ExpectedDepthTwoCopies = 12,
    [ValidateRange(1, 600)][int] $NativeTimeoutSeconds = 180,
    [ValidateRange(1, 600)][int] $ToolTimeoutSeconds = 120
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$utf8 = [Text.UTF8Encoding]::new($false)

function Assert-That([bool] $Condition, [string] $Message) {
    if (!$Condition) { throw $Message }
}
function Get-Hash([string] $Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}
function Assert-Hash([string] $Path, [string] $Expected) {
    Assert-That ($Expected -match '^[0-9a-fA-F]{64}$') "Invalid SHA-256 for $Path"
    Assert-That ((Get-Hash $Path) -eq $Expected) "SHA-256 mismatch: $Path"
}
function Get-ScopedPath([string] $Root, [string] $Relative) {
    Assert-That (![IO.Path]::IsPathRooted($Relative)) "Expected a relative artifact: $Relative"
    Assert-That (($Relative -split '[/\\]') -notcontains '..') "Parent traversal in artifact: $Relative"
    $resolved = [IO.Path]::GetFullPath((Join-Path $Root $Relative))
    Assert-That ($resolved.StartsWith($Root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) "Artifact escaped its root: $Relative"
    $cursor = $resolved
    while ($cursor.Length -gt $Root.Length) {
        if (Test-Path -LiteralPath $cursor) {
            Assert-That (!((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) "Reparse point in artifact path: $cursor"
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $resolved
}
function Read-Json([string] $Path) {
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -Depth 80
}
function Assert-Pins([string] $Path) {
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $pins = @(Read-Json $Path)
    Assert-That ($pins.Count -gt 0) 'Source manifest is empty'
    foreach ($pin in $pins) {
        Assert-That ($seen.Add([string]$pin.path)) "Duplicate source pin: $($pin.path)"
        Assert-Hash (Get-ScopedPath $repo $pin.path) $pin.hash
    }
}
function Invoke-OwnedChild([string] $Executable, [string[]] $ChildArguments, [string] $Stem, [int] $Seconds, [hashtable] $EnvironmentOverrides = @{}) {
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Executable
    $info.WorkingDirectory = [IO.Path]::GetDirectoryName($Stem)
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($argument in $ChildArguments) { $info.ArgumentList.Add($argument) }
    foreach ($key in $EnvironmentOverrides.Keys) { $info.Environment[$key] = [string]$EnvironmentOverrides[$key] }
    $child = [Diagnostics.Process]::new()
    $child.StartInfo = $info
    $started = $false
    $stdoutTask = $null
    $stderrTask = $null
    $receipt = [ordered]@{
        executable=$Executable;arguments=$ChildArguments;environment_overrides=$EnvironmentOverrides;pid=$null;start_utc=$null;deadline_utc=$null
        timeout_seconds=$Seconds;exit_code=$null;timed_out=$false;finished_utc=$null;completed=$false;error=$null
    }
    $receiptPath = "$Stem.run.json"
    try {
        Assert-That ($child.Start()) "Could not start $Executable"
        $started = $true
        $ownedPid = $child.Id
        $stdoutTask = $child.StandardOutput.ReadToEndAsync()
        $stderrTask = $child.StandardError.ReadToEndAsync()
        $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
        $receipt.pid = $ownedPid
        $receipt.start_utc = $child.StartTime.ToUniversalTime().ToString('o')
        $receipt.deadline_utc = $deadline.ToString('o')
        [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 8), $utf8)
        $remaining = [math]::Max(0, [math]::Ceiling(($deadline - [DateTime]::UtcNow).TotalMilliseconds))
        if (!$child.WaitForExit([int]$remaining)) {
            $receipt.timed_out = $true
            throw "Owned child $ownedPid exceeded ${Seconds}s: $Executable"
        }
        Assert-That ([Threading.Tasks.Task]::WaitAll([Threading.Tasks.Task[]]@($stdoutTask, $stderrTask), 5000)) "Owned child $ownedPid left undrained output pipes"
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        [IO.File]::WriteAllText("$Stem.stdout.log", $stdout, $utf8)
        [IO.File]::WriteAllText("$Stem.stderr.log", $stderr, $utf8)
        Assert-That ($child.ExitCode -eq 0) "Owned child $ownedPid exited $($child.ExitCode); see $Stem.stderr.log"
        Assert-That (($stdout + $stderr) -notmatch '(?i)libgcov|profiling error|profiling:|overwriting an existing profile|checksum mismatch') "Profiling diagnostics invalidate $Stem"
        Assert-That ($stderr.Length -eq 0) "Unexpected stderr invalidates $Stem"
        $receipt.completed = $true
        return [pscustomobject]@{ pid = $ownedPid; exit_code = $child.ExitCode; stdout = $stdout; stderr = $stderr }
    } catch {
        $receipt.error = $_.Exception.Message
        throw
    } finally {
        try {
            try {
                if ($started) {
                    if (!$child.HasExited) {
                        try { $child.Kill($true) } catch { if (!$child.HasExited) { throw } }
                        Assert-That ($child.WaitForExit(5000)) "Could not stop owned child $($child.Id)"
                    }
                    if ($null -ne $stdoutTask -and $null -ne $stderrTask) {
                        $null = [Threading.Tasks.Task]::WaitAll([Threading.Tasks.Task[]]@($stdoutTask, $stderrTask), 5000)
                        if ($stdoutTask.IsCompletedSuccessfully -and !(Test-Path -LiteralPath "$Stem.stdout.log")) { [IO.File]::WriteAllText("$Stem.stdout.log", $stdoutTask.Result, $utf8) }
                        if ($stderrTask.IsCompletedSuccessfully -and !(Test-Path -LiteralPath "$Stem.stderr.log")) { [IO.File]::WriteAllText("$Stem.stderr.log", $stderrTask.Result, $utf8) }
                    }
                }
            } finally {
                if ($started -and $child.HasExited) { $receipt.exit_code = $child.ExitCode }
                $receipt.finished_utc = [DateTime]::UtcNow.ToString('o')
                [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 8), $utf8)
            }
        } finally {
            $child.Dispose()
        }
    }
}
function Get-Sample([byte[]] $Bytes, [int] $Offset, [bool] $TenBit) {
    if ($TenBit) { return [int]$Bytes[$Offset] -bor ([int]$Bytes[$Offset + 1] -shl 8) }
    return [int]$Bytes[$Offset]
}
function Get-CopySum($Counts) {
    Assert-That (@($Counts).Count -eq 2) 'Expected two per-slot copy observations'
    return [long]$Counts[0] + [long]$Counts[1]
}
function Assert-RingLedger($Owner, [int] $Depth, $Accepted, $Refused) {
    $surfaceCount = if ($Depth -eq 2) { 3 } else { 1 }
    Assert-That ($Owner.input_storage -ceq 'registered-surface-ring-v1' -and $Owner.input_surface_count -eq $surfaceCount -and $Owner.copy_path_removed -eq $true) 'Unexpected native input storage'
    $writes = @($Owner.timeline | Where-Object event -eq 'write')
    Assert-That ($writes.Count -eq 7 -and $Owner.conversion_writes -eq $writes.Count) 'Incomplete native input write ledger'
    for ($index = 0; $index -lt $writes.Count; ++$index) {
        $write = $writes[$index]
        Assert-That ($write.authored_revision -eq $index + 1) 'Input producer writes are out of order'
        Assert-That ($write.input_surface -ge 0 -and $write.input_surface -lt $surfaceCount -and $write.input_surface -notin $write.mapped_surfaces_before) 'Producer selected a mapped or unknown input'
        Assert-That ((ConvertTo-Json -InputObject $write.mapped_surfaces_before -Compress) -ceq (ConvertTo-Json -InputObject $write.mapped_surfaces_after -Compress)) 'Producer write changed NVENC mappings'
    }
    for ($index = 0; $index -lt $Accepted.Count; ++$index) {
        $attempt = $Accepted[$index]
        $slot = $index % $Depth
        Assert-That ($attempt.picture_slot -eq $slot -and $attempt.mapped_surfaces_after[$slot] -eq $attempt.input_surface -and !$attempt.input_waits_before) 'Native picture did not map the committed input'
    }
    $expectedFrames = @(if ($Depth -eq 2) { 4,5,5,6,7,10,10,11 })
    $expectedOwnership = @(2,2,1,1,1,2,1,1)
    Assert-That ($Refused.Count -eq $expectedFrames.Count -and $Owner.busy_refusals -eq $expectedFrames.Count) 'Unexpected native busy refusals'
    Assert-That ($Owner.repeat_refusals -eq [math]::Max(0, $expectedFrames.Count - 1)) 'Unexpected repeat-retirement refusals'
    for ($index = 0; $index -lt $Refused.Count; ++$index) {
        $attempt = $Refused[$index]
        Assert-That ($attempt.frame_index -eq $Owner.frame_index_origin + $expectedFrames[$index] -and $attempt.input_waits_before -eq ($index -ne 0)) 'Native refusal did not follow capacity or exact input ownership'
        Assert-That ($attempt.before_in_flight -eq $expectedOwnership[$index] -and $attempt.after_in_flight -eq $attempt.before_in_flight) 'Native refusal changed picture ownership'
        Assert-That ((ConvertTo-Json -InputObject $attempt.mapped_surfaces_before -Compress) -ceq (ConvertTo-Json -InputObject $attempt.mapped_surfaces_after -Compress)) 'Native refusal changed input mappings'
    }
}
function Assert-Ledger($Owner, [int] $Depth, [int] $NativeSchema) {
    $origin = [long]$Owner.frame_index_origin
    Assert-That (@($Owner.returned_indices).Count -eq 12 -and @($Owner.returned_idr).Count -eq 12) 'Incomplete native output ledger'
    for ($index = 0; $index -lt 12; ++$index) {
        Assert-That ([long]$Owner.returned_indices[$index] -eq $origin + $index) 'Native output indices are not exact and ordered'
    }
    Assert-That ($Owner.returned_idr[0] -and $Owner.returned_idr[10]) 'Forced IDR was not returned'
    Assert-That ($Owner.pipeline_depth -eq $Depth -and $Owner.peak_in_flight -eq $Depth) 'Requested native depth was not exercised'
    Assert-That ($Owner.destroy_completed -eq $true) 'Native owner destruction did not finish'
    $submits = @($Owner.timeline | Where-Object event -eq 'submit')
    $accepted = @($submits | Where-Object accepted -eq $true)
    $refused = @($submits | Where-Object accepted -eq $false)
    Assert-That ($accepted.Count -eq 12) 'Native accepted-submission ledger is incomplete'
    for ($index = 0; $index -lt 12; ++$index) {
        $attempt = $accepted[$index]
        Assert-That ([long]$attempt.frame_index -eq $origin + $index) 'Accepted native indices changed order'
        Assert-That ($attempt.after_in_flight -eq $attempt.before_in_flight + 1) 'Accepted submission did not own exactly one slot'
        if ($NativeSchema -eq 2) { continue }
        $copyDelta = (Get-CopySum $attempt.copies_after) - (Get-CopySum $attempt.copies_before)
        $expectedDelta = if ($Depth -eq 1) { 0 } elseif ($ExpectedDepthTwoCopies -eq 12 -or $index -in @(0,1,2,3,4,5,8,9)) { 1 } else { 0 }
        Assert-That ($copyDelta -eq $expectedDelta) "Unexpected native copy delta at accepted frame $index"
        if ($index -in @(2, 3)) {
            Assert-That ($null -eq $attempt.input_revision) 'Unknown direct-call input became a known revision'
            Assert-That ($Depth -eq 1 -or $copyDelta -eq 1) 'Unknown native input did not issue its own copy'
        } else {
            Assert-That ($attempt.input_revision -eq [math]::Floor($index / 4) + 1) 'Authored input revision does not follow actual writes'
        }
    }
    if ($NativeSchema -eq 2) {
        Assert-RingLedger $Owner $Depth $accepted $refused
        return
    }
    $requiredRefusals = if ($Depth -eq 2) { 1 } else { 0 }
    Assert-That ($refused.Count -eq $requiredRefusals -and $Owner.busy_refusals -eq $requiredRefusals) 'Incorrect busy-refusal ledger'
    if ($Depth -eq 2) {
        $attempt = $refused[0]
        Assert-That ($attempt.frame_index -eq $origin + 4 -and $attempt.input_revision -eq 2 -and !$attempt.retained) 'Busy attempt lost the newly authored input identity'
        Assert-That ($attempt.before_in_flight -eq 2 -and $attempt.after_in_flight -eq 2) 'Busy attempt changed picture ownership'
        Assert-That ((ConvertTo-Json -InputObject $attempt.copies_before -Compress) -ceq (ConvertTo-Json -InputObject $attempt.copies_after -Compress)) 'Busy refusal issued an input copy'
        Assert-That ($accepted[4].input_revision -eq 2 -and $accepted[4].retained) 'Busy retry did not retain the same already-written target'
    }
    $copies = Get-CopySum $Owner.copy_counts
    $expectedCopies = if ($Depth -eq 2) { $ExpectedDepthTwoCopies } else { 0 }
    Assert-That ($copies -eq $expectedCopies -and $Owner.copy_counts[0] -eq $Owner.copy_counts[1]) 'Observed copy issues differ from the requested qualification policy'
}
function Test-DecodedContent([string] $DecodedPath, [string] $ExpectedPath, [bool] $TenBit) {
    $actual = [IO.File]::ReadAllBytes($DecodedPath)
    $authored = [IO.File]::ReadAllBytes($ExpectedPath)
    $sampleBytes = if ($TenBit) { 2 } else { 1 }
    [int]$frameBytes = 480 * 256 * 3 / 2 * $sampleBytes
    Assert-That ($actual.Length -eq 12 * $frameBytes -and $authored.Length -eq $actual.Length) 'Decoded/authored raster size or frame count differs'
    $maximum = if ($TenBit) { 16 } else { 4 }
    $meanLimit = if ($TenBit) { 4 } else { 1 }
    $perFrame = [Collections.Generic.List[string]]::new()
    $roiReports = [Collections.Generic.List[object]]::new()
    for ($frame = 0; $frame -lt 12; ++$frame) {
        $slice = [byte[]]::new($frameBytes)
        [Array]::Copy($actual, $frame * $frameBytes, $slice, 0, $frameBytes)
        $perFrame.Add([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($slice)))
        $planeOffset = 0
        for ($plane = 0; $plane -lt 3; ++$plane) {
            $planeWidth = if ($plane -eq 0) { 480 } else { 240 }
            $planeHeight = if ($plane -eq 0) { 256 } else { 128 }
            for ($quadrant = 0; $quadrant -lt 4; ++$quadrant) {
                $centerX = if ($quadrant % 2 -eq 0) { $planeWidth / 4 } else { 3 * $planeWidth / 4 }
                $centerY = if ($quadrant -lt 2) { $planeHeight / 4 } else { 3 * $planeHeight / 4 }
                $sum = 0L
                $largest = 0
                for ($dy = -8; $dy -lt 8; ++$dy) {
                    for ($dx = -8; $dx -lt 8; ++$dx) {
                        $sampleOffset = (($centerY + $dy) * $planeWidth + $centerX + $dx) * $sampleBytes
                        $offset = $frame * $frameBytes + $planeOffset + $sampleOffset
                        $wanted = Get-Sample $authored $offset $TenBit
                        $observed = Get-Sample $actual $offset $TenBit
                        Assert-That (!$TenBit -or ($wanted -le 1023 -and $observed -le 1023)) 'Noncanonical 10-bit decoded sample'
                        $difference = [math]::Abs($wanted - $observed)
                        $sum += $difference
                        $largest = [math]::Max($largest, $difference)
                        # Every authored A/B marker must be distinguishable within this fixed tolerance.
                        $otherFrame = if ($frame -lt 4 -or $frame -ge 8) { 4 } else { 0 }
                        $otherOffset = $otherFrame * $frameBytes + $planeOffset + $sampleOffset
                        Assert-That ([math]::Abs($wanted - (Get-Sample $authored $otherOffset $TenBit)) -gt 2 * $maximum) 'Authored A/B markers are not independently distinguishable'
                    }
                }
                $mean = $sum / 256.0
                Assert-That ($largest -le $maximum -and $mean -le $meanLimit) "Decoded authored-content mismatch at frame=$frame plane=$plane quadrant=$quadrant max=$largest mean=$mean"
                $roiReports.Add([ordered]@{frame=$frame;plane=$plane;quadrant=$quadrant;max_abs=$largest;mean_abs=$mean})
            }
            $planeOffset += $planeWidth * $planeHeight * $sampleBytes
        }
    }
    return [ordered]@{decoded_sha256=(Get-Hash $DecodedPath);authored_sha256=(Get-Hash $ExpectedPath);per_frame_sha256=$perFrame.ToArray();roi_policy=[ordered]@{side=16;max_abs=$maximum;mean_abs=$meanLimit};roi=$roiReports.ToArray()}
}

$TestBinary = (Resolve-Path -LiteralPath $TestBinary).Path
$FFmpeg = (Resolve-Path -LiteralPath $FFmpeg).Path
$FFprobe = (Resolve-Path -LiteralPath $FFprobe).Path
$SourcePins = (Resolve-Path -LiteralPath $SourcePins).Path
Assert-Hash $TestBinary $ExpectedTestBinarySha256
Assert-Hash $FFmpeg $ExpectedFFmpegSha256
Assert-Hash $FFprobe $ExpectedFFprobeSha256
Assert-Pins $SourcePins
Assert-That ([IO.Path]::IsPathRooted($OutputRoot)) 'OutputRoot must be absolute'
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$relativeOutput = [IO.Path]::GetRelativePath($repo, $OutputRoot)
Assert-That (($relativeOutput -split '[/\\]')[0] -like 'cmake-build-*') 'OutputRoot must be inside this checkout cmake-build-* tree'
$null = Get-ScopedPath $repo $relativeOutput
Assert-That (!(Test-Path -LiteralPath $OutputRoot)) 'OutputRoot must be new; existing evidence is never overwritten'
$null = [IO.Directory]::CreateDirectory($OutputRoot)
# Keep instrumented GCC paths short. Strip six components, as in the native CPU qualifier;
# retaining the entire source tree under each long case directory can exceed Windows MAX_PATH.
$buildTree = Get-ScopedPath $repo (($relativeOutput -split '[/\\]')[0])
$coverageRoot = Join-Path $buildTree ('gcov-nvenc-' + [Guid]::NewGuid().ToString('N').Substring(0, 12))
Assert-That (!(Test-Path -LiteralPath $coverageRoot)) 'Coverage root must be new'
$null = [IO.Directory]::CreateDirectory($coverageRoot)
$frozenBinary = Join-Path $OutputRoot 'native-tests.exe'
Copy-Item -LiteralPath $TestBinary -Destination $frozenBinary
Assert-Hash $frozenBinary $ExpectedTestBinarySha256
Copy-Item -LiteralPath $SourcePins -Destination (Join-Path $OutputRoot 'source-pins.json')
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $OutputRoot 'qualifier-source.ps1')
$caseSpecs = @(
    @{name='HevcNv12Depth1';codec='hevc';ten=$false;depth=1}, @{name='HevcNv12Depth2';codec='hevc';ten=$false;depth=2},
    @{name='HevcP010Depth1';codec='hevc';ten=$true;depth=1}, @{name='HevcP010Depth2';codec='hevc';ten=$true;depth=2},
    @{name='Av1Nv12Depth1';codec='av1';ten=$false;depth=1}, @{name='Av1Nv12Depth2';codec='av1';ten=$false;depth=2},
    @{name='Av1P010Depth1';codec='av1';ten=$true;depth=1}, @{name='Av1P010Depth2';codec='av1';ten=$true;depth=2}
)
$cases = [Collections.Generic.List[object]]::new()
$coverageIndex = 0
foreach ($caseSpec in $caseSpecs) {
    $caseRoot = Get-ScopedPath $OutputRoot $caseSpec.name
    $null = [IO.Directory]::CreateDirectory($caseRoot)
    $nativeRoot = Join-Path $caseRoot 'native'
    $xmlPath = Join-Path $caseRoot 'native.xml'
    $coverage = Join-Path $coverageRoot ([string]$coverageIndex)
    $coverageIndex = $coverageIndex + 1
    $null = [IO.Directory]::CreateDirectory($coverage)
    Assert-Hash $frozenBinary $ExpectedTestBinarySha256
    $null = Invoke-OwnedChild $frozenBinary @("--gtest_filter=NativeNvencContentGpuTest.$($caseSpec.name)", "--gtest_output=xml:$xmlPath") (Join-Path $caseRoot 'native-run') $NativeTimeoutSeconds @{
        SUNSHINE_RUN_NATIVE_NVENC_TESTS='1';SUNSHINE_NATIVE_NVENC_OUTPUT_DIR=$nativeRoot;GCOV_PREFIX=$coverage.Replace('\','/');GCOV_PREFIX_STRIP='6';SUNSHINE_TEST_HOST_SBS_SHADER_CACHE=''
    }
    [xml]$xml = Get-Content -LiteralPath $xmlPath -Raw
    $testcases = @($xml.testsuites.testsuite.testcase)
    Assert-That ($testcases.Count -eq 1 -and $xml.testsuites.tests -eq '1' -and $xml.testsuites.failures -eq '0' -and $xml.testsuites.errors -eq '0' -and $xml.testsuites.disabled -eq '0') 'Native child did not pass exactly one enabled test'
    Assert-That ($testcases[0].name -eq $caseSpec.name -and $testcases[0].status -eq 'run' -and $testcases[0].result -eq 'completed') 'Opted-in native child was skipped or selected the wrong case'
    $manifestPath = Get-ScopedPath $nativeRoot 'result.json'
    $native = Read-Json $manifestPath
    $nativeHash = Get-Hash $manifestPath
    $pixelFormat = if ($caseSpec.ten) { 'yuv420p10le' } else { 'yuv420p' }
    $inputFormat = if ($caseSpec.ten) { 'P010' } else { 'NV12' }
    Assert-That ($native.schema -in @(1,2) -and $native.case -eq $caseSpec.name -and $native.codec -eq $caseSpec.codec -and $native.pixel_format -eq $pixelFormat -and $native.input_format -eq $inputFormat) 'Unexpected native artifact schema/profile'
    Assert-That ($native.width -eq 480 -and $native.height -eq 256 -and $native.frames_per_owner -eq 12 -and $native.requested_depth -eq $caseSpec.depth -and $native.all_destroyed -eq $true) 'Native geometry/depth/destruction contract is incomplete'
    Assert-That ($native.adapter.vendor_id -eq 0x10de -and @($native.owners).Count -eq 2) 'Native run did not use NVIDIA and two fresh owners'
    $owners = [Collections.Generic.List[object]]::new()
    for ($ownerIndex = 0; $ownerIndex -lt 2; ++$ownerIndex) {
        $owner = $native.owners[$ownerIndex]
        $name = if ($ownerIndex -eq 0) { 'first' } else { 'rebuilt' }
        Assert-That ($owner.name -eq $name -and $owner.frame_index_origin -eq (1000 + 1000 * $ownerIndex)) 'Incorrect fresh-owner identity'
        Assert-Ledger $owner $caseSpec.depth $native.schema
        $extension = if ($caseSpec.codec -eq 'hevc') { '.hevc' } else { '.obu' }
        Assert-That ($owner.stream_file -ceq "$name$extension" -and $owner.expected_file -ceq "$name.expected.yuv") 'Unexpected native artifact filename'
        $encoded = Get-ScopedPath $nativeRoot $owner.stream_file
        $expected = Get-ScopedPath $nativeRoot $owner.expected_file
        Assert-That ((Get-Item -LiteralPath $encoded).Length -eq $owner.stream_bytes -and $owner.stream_bytes -gt 0) 'Encoded artifact size changed'
        Assert-That ((Get-Item -LiteralPath $expected).Length -eq $owner.expected_bytes) 'Authored artifact size changed'
        $ownerRoot = Join-Path $caseRoot $name
        $null = [IO.Directory]::CreateDirectory($ownerRoot)
        Assert-Hash $FFprobe $ExpectedFFprobeSha256
        $probe = Invoke-OwnedChild $FFprobe @('-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=codec_name,width,height,pix_fmt,nb_read_frames','-of','json',$encoded) (Join-Path $ownerRoot 'probe') $ToolTimeoutSeconds
        $streams = @((ConvertFrom-Json -InputObject $probe.stdout).streams)
        Assert-That ($streams.Count -eq 1) 'Encoded native stream has no unique video stream'
        $videoStream = $streams[0]
        # Probe BEFORE requesting any decoder output conversion: forced 10-bit output can mask an 8-bit encoder.
        Assert-That ($videoStream.codec_name -eq $caseSpec.codec -and $videoStream.width -eq 480 -and $videoStream.height -eq 256 -and $videoStream.pix_fmt -eq $pixelFormat -and $videoStream.nb_read_frames -eq '12') 'Native encoded codec/bit depth/geometry/frame count differs'
        $decoded = Join-Path $ownerRoot 'decoded.yuv'
        Assert-Hash $FFmpeg $ExpectedFFmpegSha256
        $null = Invoke-OwnedChild $FFmpeg @('-hide_banner','-loglevel','error','-nostdin','-hwaccel','none','-threads','1','-i',$encoded,'-map','0:v:0','-an','-sn','-dn','-c:v','rawvideo','-threads','1','-pix_fmt',$pixelFormat,'-fps_mode','passthrough','-f','rawvideo',$decoded) (Join-Path $ownerRoot 'decode') $ToolTimeoutSeconds
        $content = Test-DecodedContent $decoded $expected $caseSpec.ten
        $ownerFacts = [ordered]@{name=$name;content=$content;decoded_file=[IO.Path]::GetRelativePath($OutputRoot,$decoded);expected_file=[IO.Path]::GetRelativePath($OutputRoot,$expected);encoded_file=[IO.Path]::GetRelativePath($OutputRoot,$encoded);encoded_sha256=(Get-Hash $encoded);native_facts=$owner}
        if ($native.schema -eq 1) { $ownerFacts.copy_counts = @($owner.copy_counts) }
        $owners.Add($ownerFacts)
    }
    Assert-Hash $manifestPath $nativeHash
    $cases.Add([ordered]@{name=$caseSpec.name;codec=$caseSpec.codec;pixel_format=$pixelFormat;requested_depth=$caseSpec.depth;coverage_directory=$coverage;adapter=$native.adapter;settings=$native.settings;nvenc_api=$native.nvenc_api;native_manifest=[IO.Path]::GetRelativePath($OutputRoot,$manifestPath);native_manifest_sha256=$nativeHash;owners=$owners.ToArray()})
}
Assert-Pins $SourcePins
Assert-Hash $TestBinary $ExpectedTestBinarySha256
Assert-Hash $frozenBinary $ExpectedTestBinarySha256
Assert-Hash $FFmpeg $ExpectedFFmpegSha256
Assert-Hash $FFprobe $ExpectedFFprobeSha256
$artifacts = [Collections.Generic.List[object]]::new()
foreach ($caseResult in $cases) {
    $paths = [Collections.Generic.List[string]]::new()
    foreach ($name in @('native.xml','native-run.stdout.log','native-run.stderr.log','native-run.run.json')) { $paths.Add((Join-Path $caseResult.name $name)) }
    $paths.Add($caseResult.native_manifest)
    foreach ($ownerResult in $caseResult.owners) {
        $paths.Add($ownerResult.encoded_file)
        $paths.Add($ownerResult.expected_file)
        $paths.Add($ownerResult.decoded_file)
        foreach ($name in @('probe.stdout.log','probe.stderr.log','probe.run.json','decode.stdout.log','decode.stderr.log','decode.run.json')) { $paths.Add((Join-Path (Join-Path $caseResult.name $ownerResult.name) $name)) }
        Assert-Hash (Get-ScopedPath $OutputRoot $ownerResult.encoded_file) $ownerResult.encoded_sha256
        Assert-Hash (Get-ScopedPath $OutputRoot $ownerResult.expected_file) $ownerResult.content.authored_sha256
        Assert-Hash (Get-ScopedPath $OutputRoot $ownerResult.decoded_file) $ownerResult.content.decoded_sha256
    }
    Assert-Hash (Get-ScopedPath $OutputRoot $caseResult.native_manifest) $caseResult.native_manifest_sha256
    foreach ($path in $paths) {
        $file = Get-ScopedPath $OutputRoot $path
        $artifacts.Add([ordered]@{path=$path;bytes=(Get-Item -LiteralPath $file).Length;sha256=(Get-Hash $file)})
    }
}
$result = [ordered]@{
    schema=1;qualified=$true;output_root=$OutputRoot;coverage_root=$coverageRoot;native_binary_sha256=$ExpectedTestBinarySha256.ToUpperInvariant();ffmpeg_sha256=$ExpectedFFmpegSha256.ToUpperInvariant();ffprobe_sha256=$ExpectedFFprobeSha256.ToUpperInvariant();expected_depth_two_copies=$ExpectedDepthTwoCopies
    source_pins_sha256=(Get-Hash (Join-Path $OutputRoot 'source-pins.json'));qualifier_sha256=(Get-Hash (Join-Path $OutputRoot 'qualifier-source.ps1'));cases=$cases.ToArray();artifacts=$artifacts.ToArray()
    boundary='Functional authored native NVENC content only. Logical pictures in flight do not establish GPU overlap; no performance or cancellation claim.'
}
if ($ControlManifest) {
    $ControlManifest = (Resolve-Path -LiteralPath $ControlManifest).Path
    $control = Read-Json $ControlManifest
    Assert-That ($control.schema -eq 1 -and $control.qualified -eq $true -and @($control.cases).Count -eq 8) 'Control is not a complete qualified content matrix'
    $controlRoot = [IO.Path]::GetFullPath($control.output_root)
    $null = Get-ScopedPath $repo ([IO.Path]::GetRelativePath($repo,$controlRoot))
    Assert-That ($ControlManifest -eq (Get-ScopedPath $controlRoot 'qualification.json')) 'Control manifest does not belong to its declared evidence root'
    Assert-Hash (Get-ScopedPath $controlRoot 'source-pins.json') $control.source_pins_sha256
    Assert-Hash (Get-ScopedPath $controlRoot 'qualifier-source.ps1') $control.qualifier_sha256
    Assert-Hash (Get-ScopedPath $controlRoot 'native-tests.exe') $control.native_binary_sha256
    Assert-That (@($control.artifacts).Count -eq $artifacts.Count) 'Control evidence set is incomplete'
    foreach ($artifact in $control.artifacts) {
        $file = Get-ScopedPath $controlRoot $artifact.path
        Assert-That ((Get-Item -LiteralPath $file).Length -eq $artifact.bytes) 'Control artifact size changed'
        Assert-Hash $file $artifact.sha256
    }
    Assert-That ($control.ffmpeg_sha256 -eq $result.ffmpeg_sha256 -and $control.ffprobe_sha256 -eq $result.ffprobe_sha256 -and $control.qualifier_sha256 -eq $result.qualifier_sha256) 'Control/treatment tool or oracle mismatch'
    for ($index = 0; $index -lt 8; ++$index) {
        $before = $control.cases[$index]
        $after = $cases[$index]
        Assert-That ($before.name -eq $after.name -and $before.codec -eq $after.codec -and $before.pixel_format -eq $after.pixel_format -and $before.requested_depth -eq $after.requested_depth) 'Control/treatment case identity changed'
        foreach ($field in @('adapter','settings','nvenc_api')) {
            Assert-That ((ConvertTo-Json -InputObject $before.$field -Depth 30 -Compress) -ceq (ConvertTo-Json -InputObject $after.$field -Depth 30 -Compress)) "Control/treatment $field changed"
        }
        Assert-Hash (Get-ScopedPath $controlRoot $before.native_manifest) $before.native_manifest_sha256
        for ($ownerIndex = 0; $ownerIndex -lt 2; ++$ownerIndex) {
            $oldOwner = $before.owners[$ownerIndex]
            $newOwner = $after.owners[$ownerIndex]
            Assert-Hash (Get-ScopedPath $controlRoot $oldOwner.encoded_file) $oldOwner.encoded_sha256
            Assert-Hash (Get-ScopedPath $controlRoot $oldOwner.expected_file) $oldOwner.content.authored_sha256
            Assert-Hash (Get-ScopedPath $controlRoot $oldOwner.decoded_file) $oldOwner.content.decoded_sha256
            foreach ($field in @('authored_sha256','decoded_sha256','per_frame_sha256','roi_policy')) {
                Assert-That ((ConvertTo-Json -InputObject $oldOwner.content.$field -Depth 10 -Compress) -ceq (ConvertTo-Json -InputObject $newOwner.content.$field -Depth 10 -Compress)) "Exact paired content mismatch: $($after.name)/$($newOwner.name)/$field"
            }
        }
    }
    $result.control_manifest = $ControlManifest
    $result.control_manifest_sha256 = Get-Hash $ControlManifest
    $result.exact_decoded_pair_match = $true
}
$manifest = Join-Path $OutputRoot 'qualification.json'
[IO.File]::WriteAllText($manifest, ($result | ConvertTo-Json -Depth 80), $utf8)
Write-Host "Qualified native content matrix: $manifest"
[pscustomobject]@{manifest=$manifest;sha256=(Get-Hash $manifest);cases=8;owners=16;frames=192;paired=[bool]$ControlManifest}
