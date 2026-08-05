<#
.SYNOPSIS
Runs the production native offline-SBS worker end to end on a deterministic local fixture.

.DESCRIPTION
This is an opt-in GPU integration smoke test. It creates all artifacts below the repository's
.offline-sbs-smoke directory, authenticates the exact worker specification with SHA-256, invokes
the same sunshine.exe --offline-sbs-worker entry point used by the job manager, and verifies the
worker result, full scene audit, compressed Matroska output, stream preservation, timestamps,
attachment bytes, rejected-flash lookahead, an accepted semantic cut, color tags, bit depth, and
failed-job cleanup. The default fixture is SDR. Use -FixtureColor pq or -FixtureColor hlg to opt
into a tagged 10-bit BT.2020 HDR fixture; both HEVC and AV1 remain selectable with -Codec.

No Python runtime or PATH-discovered media tool is used. Stop live streaming before running this
test because the worker and NVENC preflights intentionally use the production D3D11/TensorRT/NVENC
path.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File `
  tests/integration/Invoke-OfflineSbsWorkerSmoke.ps1 `
  -Sunshine .\cmake-build-relwithdebinfo\sunshine.exe `
  -Config E:\ApolloDev\config\sunshine.conf `
  -Ffmpeg E:\ApolloDev\tools\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe `
  -Ffprobe E:\ApolloDev\tools\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffprobe.exe

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File `
  tests/integration/Invoke-OfflineSbsWorkerSmoke.ps1 `
  -Sunshine .\cmake-build-relwithdebinfo\sunshine.exe `
  -Config E:\ApolloDev\config\sunshine.conf `
  -Ffmpeg E:\ApolloDev\tools\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe `
  -Ffprobe E:\ApolloDev\tools\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffprobe.exe `
  -Codec av1_nvenc `
  -FixtureColor hlg

.NOTES
Exit 0 means the complete smoke passed. A prerequisite or contract failure exits nonzero with a
specific phase and retained-artifact hint. Use -KeepArtifacts for detailed inspection.
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string] $Sunshine,

  [Parameter(Mandatory = $true)]
  [string] $Config,

  [Parameter(Mandatory = $true)]
  [string] $Ffmpeg,

  [Parameter(Mandatory = $true)]
  [string] $Ffprobe,

  [ValidateSet('hevc_nvenc', 'av1_nvenc')]
  [string] $Codec = 'hevc_nvenc',

  [ValidateSet('sdr', 'pq', 'hlg')]
  [string] $FixtureColor = 'sdr',

  [ValidateRange(16, 4096)]
  [int] $SceneCacheMiB = 64,

  [switch] $KeepArtifacts
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-Contract {
  param(
    [Parameter(Mandatory = $true)]
    [bool] $Condition,

    [Parameter(Mandatory = $true)]
    [string] $Message
  )

  if (-not $Condition) {
    throw "Contract assertion failed: $Message"
  }
}

function Resolve-RequiredFile {
  param(
    [Parameter(Mandatory = $true)]
    [string] $Path,

    [Parameter(Mandatory = $true)]
    [string] $Description
  )

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "$Description is missing: $Path"
  }
  return (Resolve-Path -LiteralPath $Path).Path
}

function Invoke-Native {
  param(
    [Parameter(Mandatory = $true)]
    [string] $Executable,

    [Parameter(Mandatory = $true)]
    [AllowEmptyCollection()]
    [string[]] $Arguments,

    [Parameter(Mandatory = $true)]
    [string] $WorkingDirectory
  )

  $priorPreference = $ErrorActionPreference
  $output = @()
  $exitCode = -1
  Push-Location -LiteralPath $WorkingDirectory
  try {
    # Windows PowerShell represents redirected native stderr as non-terminating error records.
    # Capture it as text while retaining the real process exit code.
    $ErrorActionPreference = 'Continue'
    $output = @(& $Executable @Arguments 2>&1 | ForEach-Object { $_.ToString() })
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $priorPreference
    Pop-Location
  }

  return [pscustomobject] @{
    ExitCode = $exitCode
    Output = $output
    Text = ($output -join [Environment]::NewLine)
  }
}

function Invoke-RequiredNative {
  param(
    [Parameter(Mandatory = $true)]
    [string] $Executable,

    [Parameter(Mandatory = $true)]
    [AllowEmptyCollection()]
    [string[]] $Arguments,

    [Parameter(Mandatory = $true)]
    [string] $WorkingDirectory,

    [Parameter(Mandatory = $true)]
    [string] $Description
  )

  $run = Invoke-Native `
    -Executable $Executable `
    -Arguments $Arguments `
    -WorkingDirectory $WorkingDirectory
  if ($run.ExitCode -ne 0) {
    throw "$Description failed with exit code $($run.ExitCode).`n$($run.Text)"
  }
  return $run
}

function Write-Utf8NoBom {
  param(
    [Parameter(Mandatory = $true)]
    [string] $Path,

    [Parameter(Mandatory = $true)]
    [AllowEmptyString()]
    [string] $Text
  )

  [IO.File]::WriteAllText(
    $Path,
    $Text,
    [Text.UTF8Encoding]::new($false)
  )
}

function Read-JsonFile {
  param(
    [Parameter(Mandatory = $true)]
    [string] $Path
  )

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "Expected JSON artifact is missing: $Path"
  }
  return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 |
    ConvertFrom-Json
}

function Invoke-JsonProbe {
  param(
    [Parameter(Mandatory = $true)]
    [string] $MediaPath,

    [Parameter(Mandatory = $true)]
    [string] $WorkingDirectory
  )

  $run = Invoke-RequiredNative `
    -Executable $script:FfprobePath `
    -Arguments @(
      '-v', 'error',
      '-show_streams',
      '-show_packets',
      '-show_chapters',
      '-show_format',
      '-of', 'json',
      $MediaPath
    ) `
    -WorkingDirectory $WorkingDirectory `
    -Description "FFprobe inventory for $MediaPath"
  try {
    return $run.Text | ConvertFrom-Json
  } catch {
    throw "FFprobe returned malformed inventory JSON for $MediaPath`: $($_.Exception.Message)"
  }
}

function Invoke-VideoFrameProbe {
  param(
    [Parameter(Mandatory = $true)]
    [string] $MediaPath,

    [Parameter(Mandatory = $true)]
    [string] $WorkingDirectory
  )

  $run = Invoke-RequiredNative `
    -Executable $script:FfprobePath `
    -Arguments @(
      '-v', 'error',
      '-select_streams', 'v:0',
      '-show_frames',
      '-show_entries', 'frame=pts,duration',
      '-of', 'json',
      $MediaPath
    ) `
    -WorkingDirectory $WorkingDirectory `
    -Description "FFprobe video timeline for $MediaPath"
  try {
    return $run.Text | ConvertFrom-Json
  } catch {
    throw "FFprobe returned malformed frame JSON for $MediaPath`: $($_.Exception.Message)"
  }
}

function Get-StreamsOfType {
  param(
    [Parameter(Mandatory = $true)]
    [object] $Probe,

    [Parameter(Mandatory = $true)]
    [string] $Type
  )

  return @($Probe.streams | Where-Object { $_.codec_type -eq $Type })
}

function Get-PacketsForStream {
  param(
    [Parameter(Mandatory = $true)]
    [object] $Probe,

    [Parameter(Mandatory = $true)]
    [int] $StreamIndex
  )

  return @($Probe.packets | Where-Object {
      [int] $_.stream_index -eq $StreamIndex
    })
}

function Get-RationalSeconds {
  param(
    [Parameter(Mandatory = $true)]
    [long] $Ticks,

    [Parameter(Mandatory = $true)]
    [string] $TimeBase
  )

  $parts = $TimeBase.Split('/')
  Assert-Contract ($parts.Count -eq 2) "invalid FFprobe rational '$TimeBase'"
  $numerator = [long] $parts[0]
  $denominator = [long] $parts[1]
  Assert-Contract ($denominator -gt 0) "invalid FFprobe denominator in '$TimeBase'"
  return ([double] $Ticks * [double] $numerator) / [double] $denominator
}

function Assert-NumericAtMost {
  param(
    [Parameter(Mandatory = $true)]
    [object] $Value,

    [Parameter(Mandatory = $true)]
    [double] $Maximum,

    [Parameter(Mandatory = $true)]
    [string] $Description
  )

  $number = [double] $Value
  Assert-Contract (
    -not [double]::IsNaN($number) -and
    -not [double]::IsInfinity($number) -and
    $number -le $Maximum
  ) "$Description is $number, expected <= $Maximum"
}

function Test-TenBitPixelFormat {
  param(
    [Parameter(Mandatory = $true)]
    [string] $PixelFormat
  )

  return $PixelFormat -match '(^|[^0-9])10([^0-9]|$)|p010'
}

function Get-FixtureColorContract {
  param(
    [Parameter(Mandatory = $true)]
    [string] $Mode
  )

  switch ($Mode) {
    'pq' {
      return [pscustomobject] @{
        PixelFormat = 'yuv420p10le'
        Range = 'tv'
        Matrix = 'bt2020nc'
        Transfer = 'smpte2084'
        Primaries = 'bt2020'
        Filter = (
          'format=gbrp,' +
          'zscale=rangein=full:primariesin=bt709:' +
          'transferin=iec61966-2-1:matrixin=gbr:' +
          'range=limited:primaries=bt2020:transfer=smpte2084:' +
          'matrix=bt2020nc,format=yuv420p10le'
        )
      }
    }
    'hlg' {
      return [pscustomobject] @{
        PixelFormat = 'yuv420p10le'
        Range = 'tv'
        Matrix = 'bt2020nc'
        Transfer = 'arib-std-b67'
        Primaries = 'bt2020'
        Filter = (
          'format=gbrp,' +
          'zscale=rangein=full:primariesin=bt709:' +
          'transferin=iec61966-2-1:matrixin=gbr:' +
          'range=limited:primaries=bt2020:transfer=arib-std-b67:' +
          'matrix=bt2020nc,format=yuv420p10le'
        )
      }
    }
    default {
      return [pscustomobject] @{
        PixelFormat = 'yuv420p'
        Range = 'tv'
        Matrix = 'bt709'
        Transfer = 'iec61966-2-1'
        Primaries = 'bt709'
        Filter = (
          'format=gbrp,' +
          'zscale=rangein=full:primariesin=bt709:' +
          'transferin=iec61966-2-1:matrixin=gbr:' +
          'range=limited:primaries=bt709:transfer=iec61966-2-1:' +
          'matrix=bt709,format=yuv420p'
        )
      }
    }
  }
}

function Assert-VideoColorContract {
  param(
    [Parameter(Mandatory = $true)]
    [object] $VideoStream,

    [Parameter(Mandatory = $true)]
    [object] $Expected,

    [Parameter(Mandatory = $true)]
    [string] $Description,

    [Parameter(Mandatory = $true)]
    [bool] $RequireTenBit
  )

  Assert-Contract (
    $VideoStream.color_range -eq $Expected.Range
  ) "$Description color range is not $($Expected.Range)"
  Assert-Contract (
    $VideoStream.color_space -eq $Expected.Matrix
  ) "$Description color matrix is not $($Expected.Matrix)"
  Assert-Contract (
    $VideoStream.color_transfer -eq $Expected.Transfer
  ) "$Description transfer is not $($Expected.Transfer)"
  Assert-Contract (
    $VideoStream.color_primaries -eq $Expected.Primaries
  ) "$Description primaries are not $($Expected.Primaries)"
  if ($RequireTenBit) {
    Assert-Contract (
      Test-TenBitPixelFormat -PixelFormat ([string] $VideoStream.pix_fmt)
    ) "$Description pixel format '$($VideoStream.pix_fmt)' is not 10 bit"
  } else {
    Assert-Contract (
      $VideoStream.pix_fmt -eq $Expected.PixelFormat
    ) "$Description pixel format is not $($Expected.PixelFormat)"
  }
}

function New-WorkerSpec {
  param(
    [Parameter(Mandatory = $true)]
    [string] $JobDirectory,

    [Parameter(Mandatory = $true)]
    [string] $InputPath,

    [Parameter(Mandatory = $true)]
    [string] $StagingOutput,

    [Parameter(Mandatory = $true)]
    [string] $JobId,

    [Parameter(Mandatory = $true)]
    [string] $FfmpegVersion,

    [Parameter(Mandatory = $true)]
    [string] $FfprobeVersion
  )

  $resultDirectory = Join-Path $JobDirectory 'result'
  $specPath = Join-Path $JobDirectory 'worker-spec.json'
  $progressPath = Join-Path $JobDirectory 'progress.json'
  $resultPath = Join-Path $JobDirectory 'result.json'
  New-Item -ItemType Directory -Path $JobDirectory -Force | Out-Null

  $spec = [ordered] @{
    schema = 1
    job_id = $JobId
    operation = 'convert'
    input_path = $InputPath
    job_directory = $JobDirectory
    result_directory = $resultDirectory
    progress_path = $progressPath
    result_path = $resultPath
    staging_output = $StagingOutput
    sunshine = [ordered] @{
      executable = $script:SunshinePath
      config = $script:ConfigPath
    }
    ffmpeg = [ordered] @{
      path = $script:FfmpegPath
      version = $FfmpegVersion
    }
    ffprobe = [ordered] @{
      path = $script:FfprobePath
      version = $FfprobeVersion
    }
    scene_cache = [ordered] @{
      hard_cap_bytes = ([long] $SceneCacheMiB * 1024L * 1024L)
      budget_policy = 'fail'
    }
    codec = $Codec
    planner = [ordered] @{
      implementation = 'native-offline-scene-planner'
      scene_plan_contract = 'scene-plan-v2'
    }
    python_dependency = $false
  }

  $bytes = ($spec | ConvertTo-Json -Depth 16) + "`n"
  Write-Utf8NoBom -Path $specPath -Text $bytes
  $digest = (Get-FileHash -LiteralPath $specPath -Algorithm SHA256).Hash.ToLowerInvariant()
  Assert-Contract ($digest -match '^[0-9a-f]{64}$') 'worker specification SHA-256 is malformed'
  return [pscustomobject] @{
    SpecPath = $specPath
    SpecSha256 = $digest
    ResultPath = $resultPath
    ProgressPath = $progressPath
    ResultDirectory = $resultDirectory
    StagingOutput = $StagingOutput
    JobId = $JobId
  }
}

function Invoke-Worker {
  param(
    [Parameter(Mandatory = $true)]
    [object] $Worker
  )

  return Invoke-Native `
    -Executable $script:SunshinePath `
    -Arguments @(
      $script:ConfigPath,
      '--offline-sbs-worker',
      $Worker.SpecPath,
      $Worker.SpecSha256
    ) `
    -WorkingDirectory (Split-Path -Parent $script:SunshinePath)
}

function Assert-SourceFixture {
  param(
    [Parameter(Mandatory = $true)]
    [object] $Probe,

    [Parameter(Mandatory = $true)]
    [object] $ExpectedColor,

    [Parameter(Mandatory = $true)]
    [bool] $RequireTenBit
  )

  $videos = @(Get-StreamsOfType -Probe $Probe -Type 'video')
  $audio = @(Get-StreamsOfType -Probe $Probe -Type 'audio')
  $subtitles = @(Get-StreamsOfType -Probe $Probe -Type 'subtitle')
  $attachments = @(Get-StreamsOfType -Probe $Probe -Type 'attachment')
  Assert-Contract ($videos.Count -eq 1) 'fixture must have one video stream'
  Assert-Contract ($audio.Count -eq 1) 'fixture must have one audio stream'
  Assert-Contract ($subtitles.Count -eq 1) 'fixture must have one subtitle stream'
  Assert-Contract ($attachments.Count -eq 1) 'fixture must have one attachment stream'
  Assert-Contract (@($Probe.streams).Count -eq 4) 'fixture must have exactly four streams'
  Assert-Contract ($videos[0].codec_name -eq 'ffv1') 'fixture video must be lossless FFV1'
  Assert-Contract (
    $videos[0].sample_aspect_ratio -eq '1:1'
  ) 'fixture video must explicitly signal square pixels'
  Assert-VideoColorContract `
    -VideoStream $videos[0] `
    -Expected $ExpectedColor `
    -Description 'fixture video' `
    -RequireTenBit $RequireTenBit
  Assert-Contract ($audio[0].codec_name -eq 'pcm_s16le') 'fixture audio must be PCM'
  Assert-Contract ($subtitles[0].codec_name -eq 'subrip') 'fixture subtitle must be SubRip'
  Assert-Contract ($attachments[0].codec_name -eq 'ttf') 'fixture attachment must be typed TTF'
  Assert-Contract ($attachments[0].tags.filename -eq 'offline-smoke.ttf') 'fixture attachment filename changed'
  Assert-Contract ($attachments[0].tags.mimetype -eq 'application/x-truetype-font') 'fixture attachment MIME changed'

  $videoPackets = @(Get-PacketsForStream -Probe $Probe -StreamIndex ([int] $videos[0].index))
  $audioPackets = @(Get-PacketsForStream -Probe $Probe -StreamIndex ([int] $audio[0].index))
  $subtitlePackets = @(Get-PacketsForStream -Probe $Probe -StreamIndex ([int] $subtitles[0].index))
  Assert-Contract ($videoPackets.Count -eq 16) 'fixture video must contain 16 packets'
  Assert-Contract ($audioPackets.Count -gt 1) 'fixture audio packet inventory is empty'
  Assert-Contract ($subtitlePackets.Count -eq 1) 'fixture subtitle packet inventory changed'
  Assert-Contract ([long] $videoPackets[0].pts -eq 375) 'fixture video must begin at 375 ms'
  Assert-Contract ([long] $audioPackets[0].pts -eq 625) 'fixture audio must begin at 625 ms'
  Assert-Contract ([long] $subtitlePackets[0].pts -eq 875) 'fixture subtitle must begin at 875 ms'
  Assert-Contract ($videos[0].time_base -eq '1/1000') 'fixture video time base changed'
  Assert-Contract ($audio[0].time_base -eq '1/1000') 'fixture audio time base changed'
  Assert-Contract ($subtitles[0].time_base -eq '1/1000') 'fixture subtitle time base changed'
  Assert-Contract (@($Probe.chapters).Count -eq 1) 'fixture must contain one chapter'
  Assert-Contract ([double] $Probe.chapters[0].start_time -eq 0.5) 'fixture chapter start changed'
  Assert-Contract ([double] $Probe.chapters[0].end_time -eq 1.5) 'fixture chapter end changed'
  Assert-Contract ($Probe.format.tags.title -eq 'Sunshine 3D offline smoke') 'fixture global metadata changed'
}

function Assert-OutputInventory {
  param(
    [Parameter(Mandatory = $true)]
    [object] $SourceProbe,

    [Parameter(Mandatory = $true)]
    [object] $OutputProbe,

    [Parameter(Mandatory = $true)]
    [string] $ExpectedVideoCodec,

    [Parameter(Mandatory = $true)]
    [object] $ExpectedColor,

    [Parameter(Mandatory = $true)]
    [bool] $RequireTenBit
  )

  foreach ($type in @('video', 'audio', 'subtitle', 'attachment')) {
    $sourceStreams = @(Get-StreamsOfType -Probe $SourceProbe -Type $type)
    $outputStreams = @(Get-StreamsOfType -Probe $OutputProbe -Type $type)
    Assert-Contract (
      $sourceStreams.Count -eq $outputStreams.Count
    ) "output $type stream count differs from source"
  }
  Assert-Contract (@($OutputProbe.streams).Count -eq 4) 'output must retain all four streams'

  $sourceVideo = @(Get-StreamsOfType -Probe $SourceProbe -Type 'video')[0]
  $outputVideo = @(Get-StreamsOfType -Probe $OutputProbe -Type 'video')[0]
  Assert-Contract ($outputVideo.codec_name -eq $ExpectedVideoCodec) "output video codec is not $ExpectedVideoCodec"
  Assert-Contract ([int] $outputVideo.width -eq (2 * [int] $sourceVideo.width)) 'output SBS width is not twice the source width'
  Assert-Contract ([int] $outputVideo.height -eq [int] $sourceVideo.height) 'output SBS height differs from source'
  Assert-Contract (
    $outputVideo.sample_aspect_ratio -eq '1:1'
  ) 'output packed SBS sample aspect ratio is not 1:1'
  Assert-VideoColorContract `
    -VideoStream $outputVideo `
    -Expected $ExpectedColor `
    -Description 'output video' `
    -RequireTenBit $RequireTenBit

  foreach ($type in @('audio', 'subtitle', 'attachment')) {
    $sourceStream = @(Get-StreamsOfType -Probe $SourceProbe -Type $type)[0]
    $outputStream = @(Get-StreamsOfType -Probe $OutputProbe -Type $type)[0]
    Assert-Contract ($sourceStream.codec_name -eq $outputStream.codec_name) "output $type codec differs from source"
  }
  $outputAttachment = @(Get-StreamsOfType -Probe $OutputProbe -Type 'attachment')[0]
  Assert-Contract ($outputAttachment.tags.filename -eq 'offline-smoke.ttf') 'output attachment filename changed'
  Assert-Contract ($outputAttachment.tags.mimetype -eq 'application/x-truetype-font') 'output attachment MIME changed'
  Assert-Contract (@($OutputProbe.chapters).Count -eq 1) 'output chapter count differs from source'
  Assert-Contract ($OutputProbe.chapters[0].tags.title -eq 'Smoke chapter') 'output chapter metadata changed'
  Assert-Contract ($OutputProbe.format.tags.title -eq 'Sunshine 3D offline smoke') 'output global metadata changed'

  foreach ($type in @('audio', 'subtitle')) {
    $sourceStream = @(Get-StreamsOfType -Probe $SourceProbe -Type $type)[0]
    $outputStream = @(Get-StreamsOfType -Probe $OutputProbe -Type $type)[0]
    $sourcePackets = @(Get-PacketsForStream -Probe $SourceProbe -StreamIndex ([int] $sourceStream.index))
    $outputPackets = @(Get-PacketsForStream -Probe $OutputProbe -StreamIndex ([int] $outputStream.index))
    Assert-Contract ($sourcePackets.Count -eq $outputPackets.Count) "output $type packet count differs from source"
    for ($packetIndex = 0; $packetIndex -lt $sourcePackets.Count; ++$packetIndex) {
      foreach ($field in @('pts', 'dts', 'duration')) {
        $sourceProperty = $sourcePackets[$packetIndex].PSObject.Properties[$field]
        if ($null -eq $sourceProperty -or $null -eq $sourceProperty.Value) {
          continue
        }
        $outputProperty = $outputPackets[$packetIndex].PSObject.Properties[$field]
        Assert-Contract ($null -ne $outputProperty -and $null -ne $outputProperty.Value) "output $type packet $packetIndex lost $field"
        $sourceSeconds = Get-RationalSeconds `
          -Ticks ([long] $sourceProperty.Value) `
          -TimeBase $sourceStream.time_base
        $outputSeconds = Get-RationalSeconds `
          -Ticks ([long] $outputProperty.Value) `
          -TimeBase $outputStream.time_base
        $outputTickSeconds = Get-RationalSeconds -Ticks 1 -TimeBase $outputStream.time_base
        Assert-Contract (
          [math]::Abs($sourceSeconds - $outputSeconds) -le
            ($outputTickSeconds + 1.0e-12)
        ) "output $type packet $packetIndex $field differs by more than one output tick"
      }
    }
  }
}

function Assert-VideoTimeline {
  param(
    [Parameter(Mandatory = $true)]
    [object] $SourceFrames,

    [Parameter(Mandatory = $true)]
    [object] $OutputFrames,

    [Parameter(Mandatory = $true)]
    [object] $OutputVideoStream
  )

  $source = @($SourceFrames.frames)
  $output = @($OutputFrames.frames)
  Assert-Contract ($source.Count -eq 16) 'source video frame inventory changed'
  Assert-Contract ($source.Count -eq $output.Count) 'output video frame count differs from source'
  $outputTickSeconds = Get-RationalSeconds -Ticks 1 -TimeBase $OutputVideoStream.time_base
  for ($index = 0; $index -lt $source.Count; ++$index) {
    $sourcePtsSeconds = Get-RationalSeconds -Ticks ([long] $source[$index].pts) -TimeBase '1/1000'
    $outputPtsSeconds = Get-RationalSeconds -Ticks ([long] $output[$index].pts) -TimeBase $OutputVideoStream.time_base
    $sourceDurationSeconds = Get-RationalSeconds -Ticks ([long] $source[$index].duration) -TimeBase '1/1000'
    $outputDurationSeconds = Get-RationalSeconds -Ticks ([long] $output[$index].duration) -TimeBase $OutputVideoStream.time_base
    Assert-Contract (
      [math]::Abs($sourcePtsSeconds - $outputPtsSeconds) -le
        ($outputTickSeconds + 1.0e-12)
    ) "output video frame $index PTS differs by more than one output tick"
    Assert-Contract (
      [math]::Abs($sourceDurationSeconds - $outputDurationSeconds) -le
        ($outputTickSeconds + 1.0e-12)
    ) "output video frame $index duration differs by more than one output tick"
  }
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$smokeRoot = Join-Path $repoRoot '.offline-sbs-smoke'
$runId = [guid]::NewGuid().ToString('N')
$runRoot = Join-Path $smokeRoot $runId
$sentinel = Join-Path $runRoot '.sunshine-offline-smoke-owner'
$createdRunRoot = $false
$createdSmokeRoot = $false
$succeeded = $false
$scriptExitCode = 0

try {
  $script:SunshinePath = Resolve-RequiredFile -Path $Sunshine -Description 'sunshine.exe'
  $script:ConfigPath = Resolve-RequiredFile -Path $Config -Description 'Sunshine configuration'
  $script:FfmpegPath = Resolve-RequiredFile -Path $Ffmpeg -Description 'packaged ffmpeg.exe'
  $script:FfprobePath = Resolve-RequiredFile -Path $Ffprobe -Description 'packaged ffprobe.exe'
  Assert-Contract (
    [IO.Path]::GetFileName($script:FfmpegPath).ToLowerInvariant() -eq 'ffmpeg.exe'
  ) 'the FFmpeg path must name ffmpeg.exe'
  Assert-Contract (
    [IO.Path]::GetFileName($script:FfprobePath).ToLowerInvariant() -eq 'ffprobe.exe'
  ) 'the FFprobe path must name ffprobe.exe'

  if (Test-Path -LiteralPath $smokeRoot) {
    $smokeRootItem = Get-Item -LiteralPath $smokeRoot -Force
    Assert-Contract ($smokeRootItem.PSIsContainer) 'smoke root exists but is not a directory'
    Assert-Contract (
      ($smokeRootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0
    ) 'refusing to use a reparse-point smoke root'
  } else {
    New-Item -ItemType Directory -Path $smokeRoot | Out-Null
    $createdSmokeRoot = $true
  }
  Assert-Contract (-not (Test-Path -LiteralPath $runRoot)) 'random smoke run directory already exists'
  New-Item -ItemType Directory -Path $runRoot | Out-Null
  $createdRunRoot = $true
  $runRootItem = Get-Item -LiteralPath $runRoot -Force
  Assert-Contract (
    ($runRootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0
  ) 'refusing to use a reparse-point smoke run directory'
  Write-Utf8NoBom -Path $sentinel -Text $runId
  Write-Host "[offline-sbs-smoke] workspace: $runRoot"

  $ffmpegVersionRun = Invoke-RequiredNative `
    -Executable $script:FfmpegPath `
    -Arguments @('-version') `
    -WorkingDirectory $runRoot `
    -Description 'packaged FFmpeg version probe'
  Assert-Contract ($ffmpegVersionRun.Output.Count -gt 0) 'FFmpeg version output is empty'
  $ffmpegVersion = $ffmpegVersionRun.Output[0]
  $ffprobeVersionRun = Invoke-RequiredNative `
    -Executable $script:FfprobePath `
    -Arguments @('-version') `
    -WorkingDirectory $runRoot `
    -Description 'packaged FFprobe version probe'
  Assert-Contract ($ffprobeVersionRun.Output.Count -gt 0) 'FFprobe version output is empty'
  $ffprobeVersion = $ffprobeVersionRun.Output[0]

  $fixtureColorContract = Get-FixtureColorContract -Mode $FixtureColor
  $fixtureIsHdr = $FixtureColor -ne 'sdr'
  Write-Host (
    "[offline-sbs-smoke] creating deterministic $($FixtureColor.ToUpperInvariant()) " +
    'Matroska fixture'
  )
  $videoPath = Join-Path $runRoot 'fixture-video.mkv'
  $sceneAPath = Join-Path $runRoot 'fixture-scene-a.bmp'
  $flashPath = Join-Path $runRoot 'fixture-flash.bmp'
  $sceneBPath = Join-Path $runRoot 'fixture-scene-b.bmp'
  $audioPath = Join-Path $runRoot 'fixture-audio.wav'
  $subtitlePath = Join-Path $runRoot 'fixture-subtitle.srt'
  $metadataPath = Join-Path $runRoot 'fixture-metadata.ffmeta'
  $attachmentPath = Join-Path $runRoot 'offline-smoke.ttf'
  $sourcePath = Join-Path $runRoot 'source.mkv'
  Write-Utf8NoBom -Path $subtitlePath -Text @'
1
00:00:00,125 --> 00:00:01,375
Sunshine 3D offline worker smoke test

'@
  Write-Utf8NoBom -Path $metadataPath -Text @'
;FFMETADATA1
title=Sunshine 3D offline smoke
comment=deterministic local fixture

[CHAPTER]
TIMEBASE=1/1000
START=500
END=1500
title=Smoke chapter

'@
  Write-Utf8NoBom -Path $attachmentPath -Text 'Sunshine 3D offline SBS deterministic typed attachment.'

  Invoke-RequiredNative `
    -Executable $script:FfmpegPath `
    -Arguments @(
      '-hide_banner', '-loglevel', 'error', '-y',
      '-f', 'lavfi',
      '-i', 'testsrc2=size=320x180:rate=1:duration=1',
      '-frames:v', '1',
      '-pix_fmt', 'bgr24',
      $sceneAPath
    ) `
    -WorkingDirectory $runRoot `
    -Description 'first deterministic scene still generation' | Out-Null
  Invoke-RequiredNative `
    -Executable $script:FfmpegPath `
    -Arguments @(
      '-hide_banner', '-loglevel', 'error', '-y',
      '-i', $sceneAPath,
      '-vf', 'colorlevels=romin=0.75:gomin=0.75:bomin=0.75',
      '-frames:v', '1',
      '-pix_fmt', 'bgr24',
      $flashPath
    ) `
    -WorkingDirectory $runRoot `
    -Description 'one-frame exposure flash still generation' | Out-Null
  Invoke-RequiredNative `
    -Executable $script:FfmpegPath `
    -Arguments @(
      '-hide_banner', '-loglevel', 'error', '-y',
      '-f', 'lavfi',
      '-i', 'smptehdbars=size=320x180:rate=1:duration=1',
      '-frames:v', '1',
      '-pix_fmt', 'bgr24',
      $sceneBPath
    ) `
    -WorkingDirectory $runRoot `
    -Description 'second deterministic scene still generation' | Out-Null
  $fixtureFilter = (
    '[0:v]trim=duration=1.0,setpts=PTS-STARTPTS[a0];' +
    '[1:v]trim=duration=0.125,setpts=PTS-STARTPTS[flash];' +
    '[2:v]trim=duration=0.25,setpts=PTS-STARTPTS[a1];' +
    '[3:v]trim=duration=0.625,setpts=PTS-STARTPTS[b];' +
    '[a0][flash][a1][b]concat=n=4:v=1:a=0,' +
    $fixtureColorContract.Filter + ',setsar=1/1' +
    '[out]'
  )
  Invoke-RequiredNative `
    -Executable $script:FfmpegPath `
    -Arguments @(
      '-hide_banner', '-loglevel', 'error', '-y',
      '-loop', '1', '-framerate', '8', '-i', $sceneAPath,
      '-loop', '1', '-framerate', '8', '-i', $flashPath,
      '-loop', '1', '-framerate', '8', '-i', $sceneAPath,
      '-loop', '1', '-framerate', '8', '-i', $sceneBPath,
      '-filter_complex', $fixtureFilter,
      '-map', '[out]',
      '-frames:v', '16',
      '-c:v', 'ffv1',
      '-level', '3',
      '-g', '1',
      '-color_range', $fixtureColorContract.Range,
      '-colorspace', $fixtureColorContract.Matrix,
      '-color_trc', $fixtureColorContract.Transfer,
      '-color_primaries', $fixtureColorContract.Primaries,
      $videoPath
    ) `
    -WorkingDirectory $runRoot `
    -Description 'flash-and-real-cut video fixture generation' | Out-Null
  Invoke-RequiredNative `
    -Executable $script:FfmpegPath `
    -Arguments @(
      '-hide_banner', '-loglevel', 'error', '-y',
      '-f', 'lavfi',
      '-i', 'sine=frequency=880:sample_rate=48000:duration=1.25',
      '-c:a', 'pcm_s16le',
      $audioPath
    ) `
    -WorkingDirectory $runRoot `
    -Description 'deterministic audio fixture generation' | Out-Null
  Invoke-RequiredNative `
    -Executable $script:FfmpegPath `
    -Arguments @(
      '-hide_banner', '-loglevel', 'error', '-y',
      '-copyts',
      '-itsoffset', '0.375', '-i', $videoPath,
      '-itsoffset', '0.625', '-i', $audioPath,
      '-itsoffset', '0.750', '-i', $subtitlePath,
      '-f', 'ffmetadata', '-i', $metadataPath,
      '-map', '0:v:0',
      '-map', '1:a:0',
      '-map', '2:s:0',
      '-map_metadata', '3',
      '-map_chapters', '3',
      '-c', 'copy',
      '-attach', $attachmentPath,
      '-metadata:s:t:0', 'mimetype=application/x-truetype-font',
      '-metadata:s:t:0', 'filename=offline-smoke.ttf',
      '-avoid_negative_ts', 'disabled',
      '-max_interleave_delta', '0',
      '-f', 'matroska',
      $sourcePath
    ) `
    -WorkingDirectory $runRoot `
    -Description 'deterministic Matroska fixture mux' | Out-Null

  $sourceProbe = Invoke-JsonProbe -MediaPath $sourcePath -WorkingDirectory $runRoot
  $sourceFrames = Invoke-VideoFrameProbe -MediaPath $sourcePath -WorkingDirectory $runRoot
  Assert-SourceFixture `
    -Probe $sourceProbe `
    -ExpectedColor $fixtureColorContract `
    -RequireTenBit $fixtureIsHdr
  Assert-Contract (@($sourceFrames.frames).Count -eq 16) 'fixture frame count differs from packet count'
  $frameMd5Run = Invoke-RequiredNative `
    -Executable $script:FfmpegPath `
    -Arguments @(
      '-hide_banner', '-loglevel', 'error',
      '-i', $sourcePath,
      '-map', '0:v:0',
      '-an', '-sn', '-dn',
      '-f', 'framemd5',
      '-'
    ) `
    -WorkingDirectory $runRoot `
    -Description 'decoded fixture layout hash'
  $frameHashes = @($frameMd5Run.Output | Where-Object {
      $_ -match '^\d+,'
    } | ForEach-Object {
      ($_ -split ',')[-1].Trim()
    })
  Assert-Contract ($frameHashes.Count -eq 16) 'decoded fixture hash count changed'
  $sceneAHash = $frameHashes[0]
  $flashHash = $frameHashes[8]
  $sceneBHash = $frameHashes[11]
  Assert-Contract (
    @($frameHashes[0..7] | Where-Object { $_ -ne $sceneAHash }).Count -eq 0
  ) 'fixture sequences 1-8 are not the same A endpoint'
  Assert-Contract (
    $flashHash -ne $sceneAHash -and $flashHash -ne $sceneBHash
  ) 'fixture sequence 9 is not a distinct one-frame flash'
  Assert-Contract (
    $frameHashes[9] -eq $sceneAHash -and $frameHashes[10] -eq $sceneAHash
  ) 'fixture sequences 10-11 do not return to endpoint A'
  Assert-Contract (
    $sceneBHash -ne $sceneAHash -and
    @($frameHashes[11..15] | Where-Object { $_ -ne $sceneBHash }).Count -eq 0
  ) 'fixture sequences 12-16 are not the same B endpoint'

  Write-Host '[offline-sbs-smoke] checking NVENC prerequisite'
  $nvencPixelFormat = 'yuv420p'
  if ($fixtureIsHdr) {
    $nvencPixelFormat = 'p010le'
  }
  $nvencRun = Invoke-Native `
    -Executable $script:FfmpegPath `
    -Arguments @(
      '-hide_banner', '-loglevel', 'error',
      '-f', 'lavfi',
      '-i', 'color=size=320x180:rate=1:duration=1',
      '-frames:v', '1',
      '-c:v', $Codec,
      '-pix_fmt', $nvencPixelFormat,
      '-f', 'null',
      'NUL'
    ) `
    -WorkingDirectory $runRoot
  if ($nvencRun.ExitCode -ne 0) {
    throw "NVENC prerequisite '$Codec' is unavailable.`n$($nvencRun.Text)"
  }

  Write-Host '[offline-sbs-smoke] verifying failed-job publication before inference'
  $invalidInput = Join-Path $runRoot 'not-media.mkv'
  Write-Utf8NoBom -Path $invalidInput -Text 'This is intentionally not a media container.'
  $failureJobDirectory = Join-Path $runRoot 'failure-job'
  $failureOutput = Join-Path $runRoot 'failure-output.mkv'
  $failureWorker = New-WorkerSpec `
    -JobDirectory $failureJobDirectory `
    -InputPath $invalidInput `
    -StagingOutput $failureOutput `
    -JobId ([guid]::NewGuid().ToString()) `
    -FfmpegVersion $ffmpegVersion `
    -FfprobeVersion $ffprobeVersion
  $failureRun = Invoke-Worker -Worker $failureWorker
  Assert-Contract ($failureRun.ExitCode -eq 2) 'invalid-media worker must exit 2'
  $failureResult = Read-JsonFile -Path $failureWorker.ResultPath
  Assert-Contract ($failureResult.schema -eq 1) 'failed worker result schema changed'
  Assert-Contract ($failureResult.status -eq 'failed') 'failed worker did not publish failed status'
  Assert-Contract ($failureResult.job_id -eq $failureWorker.JobId) 'failed worker result job identity changed'
  Assert-Contract (-not [string]::IsNullOrWhiteSpace($failureResult.error)) 'failed worker result has no diagnostic'
  # A direct worker deliberately leaves native-work for its manager. The real
  # manager owns an identity-pinned job-root handle and removes that child only
  # after the worker process group is reaped; service tests cover that boundary.
  Assert-Contract (
    (Test-Path -LiteralPath $failureOutput -PathType Leaf) -and
    (Get-Item -LiteralPath $failureOutput).Length -eq 0
  ) 'failed worker left a nonempty partial encoded output'

  Write-Host '[offline-sbs-smoke] checking native D3D11/TensorRT/model prerequisite'
  $preflightFrames = Join-Path $runRoot 'preflight-frames'
  $preflightOutput = Join-Path $runRoot 'preflight-output'
  New-Item -ItemType Directory -Path $preflightFrames -Force | Out-Null
  Invoke-RequiredNative `
    -Executable $script:FfmpegPath `
    -Arguments @(
      '-hide_banner', '-loglevel', 'error', '-y',
      '-i', $videoPath,
      '-frames:v', '1',
      '-pix_fmt', 'bgra',
      (Join-Path $preflightFrames 'frame_0000000001.png')
    ) `
    -WorkingDirectory $runRoot `
    -Description 'native model-preflight frame extraction' | Out-Null
  $modelRun = Invoke-Native `
    -Executable $script:SunshinePath `
    -Arguments @(
      $script:ConfigPath,
      '--sbs-bench',
      '--frames', $preflightFrames,
      '--out', $preflightOutput,
      '--limit', '1',
      '--artifacts', 'adaptive'
    ) `
    -WorkingDirectory (Split-Path -Parent $script:SunshinePath)
  if ($modelRun.ExitCode -ne 0) {
    throw (
      "Native D3D11/TensorRT/depth-model prerequisite failed with exit code " +
      "$($modelRun.ExitCode). Verify the configured model, packaged assets, NVIDIA driver, " +
      "and GPU availability before treating this as a worker regression.`n$($modelRun.Text)"
    )
  }

  Write-Host '[offline-sbs-smoke] running authenticated production worker'
  $successJobDirectory = Join-Path $runRoot 'success-job'
  $successOutput = Join-Path $runRoot 'success-output.mkv'
  $successWorker = New-WorkerSpec `
    -JobDirectory $successJobDirectory `
    -InputPath $sourcePath `
    -StagingOutput $successOutput `
    -JobId ([guid]::NewGuid().ToString()) `
    -FfmpegVersion $ffmpegVersion `
    -FfprobeVersion $ffprobeVersion
  $workerRun = Invoke-Worker -Worker $successWorker
  if ($workerRun.ExitCode -ne 0) {
    $failureDetail = ''
    if (Test-Path -LiteralPath $successWorker.ResultPath -PathType Leaf) {
      $failedDocument = Read-JsonFile -Path $successWorker.ResultPath
      $failureDetail = "`nWorker result: $($failedDocument.error)"
    }
    throw (
      "Production offline worker failed with exit code $($workerRun.ExitCode)." +
      "$failureDetail`n$($workerRun.Text)"
    )
  }

  $result = Read-JsonFile -Path $successWorker.ResultPath
  Assert-Contract ($result.schema -eq 1) 'worker result schema changed'
  Assert-Contract ($result.status -eq 'complete') 'worker result is not complete'
  Assert-Contract ($result.job_id -eq $successWorker.JobId) 'worker result job identity changed'
  Assert-Contract ($result.operation -eq 'convert') 'worker result operation changed'
  Assert-Contract ($result.codec -eq $Codec) 'worker result codec changed'
  Assert-Contract ($result.worker_spec_sha256 -eq $successWorker.SpecSha256) 'worker did not attest the exact specification SHA-256'
  Assert-Contract ($result.python_dependency -eq $false) 'worker unexpectedly reports a Python dependency'
  Assert-Contract ([long] $result.source.frame_count -eq 16) 'worker source frame count changed'
  Assert-Contract ([long] $result.scene_count -gt 0) 'worker committed no scenes'
  Assert-Contract (@($result.scenes).Count -eq [long] $result.scene_count) 'worker result scene count is inconsistent'
  Assert-Contract (@($result.replay_contracts).Count -eq [long] $result.scene_count) 'worker replay contract count is inconsistent'
  Assert-Contract ([long] $result.cache.remaining_bytes -eq 0) 'worker retained live scene-cache bytes'
  Assert-Contract (
    [long] $result.cache.peak_cache_plus_raster_bytes -le
    [long] $result.cache.hard_cap_bytes
  ) 'worker exceeded the exact scene-cache-plus-raster hard cap'
  Assert-Contract (
    [long] $result.analysis_contract.tensorrt_enqueue_count -eq 16
  ) 'analysis did not attest exactly one TensorRT enqueue per frame'
  Assert-Contract (
    [long] $result.analysis_contract.scheduled_depth_update_count -eq 16
  ) 'analysis did not schedule one depth update per frame'
  foreach ($replay in @($result.replay_contracts)) {
    Assert-Contract ($replay.depth_inference_enabled -eq $false) "scene $($replay.scene_id) replay enabled depth inference"
    Assert-Contract ([long] $replay.tensorrt_enqueue_count -eq 0) "scene $($replay.scene_id) replay enqueued TensorRT"
    Assert-Contract ([long] $replay.scheduled_depth_update_count -eq 0) "scene $($replay.scene_id) replay scheduled depth inference"
  }
  # The direct-worker smoke has no manager-held root pin. Its enclosing random
  # workspace is removed in this script's guarded finally block.
  Assert-Contract (
    (Test-Path -LiteralPath $successOutput -PathType Leaf) -and
    (Get-Item -LiteralPath $successOutput).Length -gt 0
  ) 'worker produced no compressed Matroska output'

  $auditPath = Join-Path $successWorker.ResultDirectory 'scene-audit.json'
  $audit = Read-JsonFile -Path $auditPath
  Assert-Contract ($audit.schema -eq 2) 'scene audit schema changed'
  Assert-Contract ($audit.version -eq 'whole-clip-scene-audit-v2') 'scene audit version changed'
  Assert-Contract ($audit.status -eq 'complete') 'scene audit is not complete'
  Assert-Contract (@($audit.scenes).Count -eq [long] $result.scene_count) 'full scene audit count differs from result'
  Assert-Contract ($audit.claims.ground_truth -eq $false) 'scene audit incorrectly claims ground truth'
  Assert-Contract ($audit.claims.best_parameters -eq $false) 'scene audit incorrectly claims globally best parameters'
  Assert-Contract (
    [long] $audit.cache.peak_cache_plus_raster_bytes -le
    [long] $audit.cache.hard_cap_bytes
  ) 'scene audit reports a hard-cap violation'
  $boundaryRevisions = @($audit.boundary_revisions)
  $rejectedFlashes = @($boundaryRevisions | Where-Object {
      $_.decision -eq 'rejected_supported_flash_return' -and
      $_.accepted -eq $false -and
      $_.semantic_cut -eq $false
    })
  $appearanceVetoCount = 0L
  foreach ($scene in @($audit.scenes)) {
    $appearanceVetoCount += [long] $scene.evidence.appearance_veto_count
  }
  $falseFlashCuts = @($boundaryRevisions | Where-Object {
      $_.accepted -eq $true -and
      $_.final_sequence -ge 9 -and $_.final_sequence -le 11
    })
  Assert-Contract (
    $falseFlashCuts.Count -eq 0
  ) 'one-frame exposure flash created a semantic scene boundary'
  # The production detector normally vetoes both A->flash and flash->A before a
  # cut pulse reaches the offline planner. The planner's explicit rejected
  # decision remains a valid fallback for a causal pulse from another detector.
  Assert-Contract (
    $rejectedFlashes.Count -ge 1 -or $appearanceVetoCount -ge 2
  ) 'scene audit did not attest the two-sided exposure-flash veto'
  $acceptedSemanticCuts = @($boundaryRevisions | Where-Object {
      $_.accepted -eq $true -and
      $_.semantic_cut -eq $true -and
      $_.final_sequence -eq 12
    })
  Assert-Contract (
    $acceptedSemanticCuts.Count -eq 1
  ) 'scene audit did not accept the later deterministic semantic cut'
  Assert-Contract (
    @($boundaryRevisions | Where-Object {
        $_.accepted -eq $true -and $_.semantic_cut -eq $true
      }).Count -eq 1
  ) 'scene audit accepted an extra semantic boundary'
  Assert-Contract (
    [long] $result.scene_count -eq 2
  ) 'accepted semantic cut did not produce exactly two planned scenes'
  $plannedScenes = @($audit.scenes)
  Assert-Contract (
    [long] $plannedScenes[0].start_sequence -eq 1 -and
    [long] $plannedScenes[0].end_sequence_exclusive -eq 12 -and
    [long] $plannedScenes[0].frame_count -eq 11
  ) 'first planned scene does not cover exact A/flash/A prefix [1,12)'
  Assert-Contract (
    [long] $plannedScenes[1].start_sequence -eq 12 -and
    [long] $plannedScenes[1].end_sequence_exclusive -eq 17 -and
    [long] $plannedScenes[1].frame_count -eq 5
  ) 'second planned scene does not cover exact B suffix [12,17)'

  $adaptiveContract = $result.analysis_contract.adaptive_state
  Assert-Contract (
    $adaptiveContract.transport -eq 'atomic-latest-v1' -and
    $adaptiveContract.retained_history -eq $false -and
    [long] $adaptiveContract.frame_count -eq 16
  ) 'offline analysis did not attest its bounded adaptive-state transport'
  Assert-Contract (
    $result.analysis_contract.PSObject.Properties.Name -notcontains 'cut_state'
  ) 'whole-clip contract still advertises the suppressed cut-state trace'
  $rawAdaptiveHistory = @(Get-ChildItem -LiteralPath (
      Join-Path $successWorker.JobDirectory 'native-work'
    ) -Recurse -File | Where-Object {
      $_.Name -eq 'adaptive_state.jsonl' -or
      $_.Name -eq 'cut_state.json'
    })
  Assert-Contract (
    $rawAdaptiveHistory.Count -eq 0
  ) 'offline worker retained an unbounded per-frame state history'
  $boundedSnapshots = @(Get-ChildItem -LiteralPath (
      Join-Path $successWorker.JobDirectory 'native-work'
    ) -Recurse -File | Where-Object {
      $_.Name -eq 'adaptive_state_header.json' -or
      $_.Name -eq 'adaptive_state_frame.json'
    })
  Assert-Contract (
    @($boundedSnapshots | Where-Object { $_.Length -gt 1MB }).Count -eq 0
  ) 'bounded adaptive-state snapshot exceeded its one-MiB reader contract'

  $timeline = Read-JsonFile -Path (
    Join-Path $successWorker.ResultDirectory 'timeline-contract.json'
  )
  Assert-Contract ($timeline.mode -eq 'bounded-output-timebase') 'Matroska timeline mode changed'
  Assert-Contract ([long] $timeline.max_output_ticks -eq 1) 'Matroska timeline tolerance changed'
  Assert-NumericAtMost $timeline.observed_video.max_pts_error_output_ticks 1.0 'video maximum PTS error'
  Assert-NumericAtMost $timeline.observed_video.max_duration_error_output_ticks 1.0 'video maximum duration error'
  Assert-NumericAtMost $timeline.observed_video.end_to_end_duration_error_output_ticks 1.0 'video end-to-end duration error'
  Assert-NumericAtMost $timeline.observed_auxiliary.max_packet_pts_error_output_ticks 1.0 'auxiliary maximum PTS error'
  Assert-NumericAtMost $timeline.observed_auxiliary.max_packet_dts_error_output_ticks 1.0 'auxiliary maximum DTS error'
  Assert-NumericAtMost $timeline.observed_auxiliary.max_packet_duration_error_output_ticks 1.0 'auxiliary maximum duration error'
  Assert-NumericAtMost $timeline.observed_auxiliary.max_stream_duration_error_output_ticks 1.0 'auxiliary accumulated duration error'
  Assert-NumericAtMost $timeline.observed_auxiliary.max_chapter_error_output_ticks 1.0 'chapter timing error'

  $outputProbe = Invoke-JsonProbe -MediaPath $successOutput -WorkingDirectory $runRoot
  $outputFrames = Invoke-VideoFrameProbe -MediaPath $successOutput -WorkingDirectory $runRoot
  $expectedVideoCodec = 'hevc'
  if ($Codec -eq 'av1_nvenc') {
    $expectedVideoCodec = 'av1'
  }
  Assert-OutputInventory `
    -SourceProbe $sourceProbe `
    -OutputProbe $outputProbe `
    -ExpectedVideoCodec $expectedVideoCodec `
    -ExpectedColor $fixtureColorContract `
    -RequireTenBit $fixtureIsHdr
  $outputVideo = @(Get-StreamsOfType -Probe $outputProbe -Type 'video')[0]
  if ($Codec -eq 'av1_nvenc') {
    Assert-Contract (
      $null -ne $outputVideo.PSObject.Properties['level'] -and
      [int] $outputVideo.level -eq 0
    ) 'AV1 smoke fixture must use the lowest compatible defined level 2.0'
  }
  Assert-VideoTimeline `
    -SourceFrames $sourceFrames `
    -OutputFrames $outputFrames `
    -OutputVideoStream $outputVideo

  $extractedAttachment = Join-Path $runRoot 'output-attachment.ttf'
  Invoke-RequiredNative `
    -Executable $script:FfmpegPath `
    -Arguments @(
      '-hide_banner', '-loglevel', 'error',
      '-dump_attachment:t:0', $extractedAttachment,
      '-i', $successOutput,
      '-f', 'null',
      'NUL'
    ) `
    -WorkingDirectory $runRoot `
    -Description 'output attachment extraction' | Out-Null
  Assert-Contract (
    (Get-FileHash -LiteralPath $attachmentPath -Algorithm SHA256).Hash -eq
    (Get-FileHash -LiteralPath $extractedAttachment -Algorithm SHA256).Hash
  ) 'output attachment bytes differ from source'

  $succeeded = $true
  Write-Host (
    "[offline-sbs-smoke] PASS: $($result.scene_count) scene(s), " +
    "16 frames, $Codec, $($FixtureColor.ToUpperInvariant()), " +
    'flash rejected, semantic cut accepted, exact authenticated worker contract'
  )
} catch {
  $scriptExitCode = 1
  $workspaceDetail = 'No smoke workspace was created.'
  if ($createdRunRoot) {
    $workspaceDetail = "Run workspace: $runRoot"
    if (-not $KeepArtifacts) {
      $workspaceDetail += ' (use -KeepArtifacts to retain it)'
    }
  }
  [Console]::Error.WriteLine(
    "[offline-sbs-smoke] FAIL: $($_.Exception.Message)`n" +
    $workspaceDetail
  )
} finally {
  if ($createdRunRoot -and -not $KeepArtifacts) {
    try {
      $resolvedSmokeRoot = [IO.Path]::GetFullPath($smokeRoot).TrimEnd('\', '/')
      $resolvedRunRoot = [IO.Path]::GetFullPath($runRoot).TrimEnd('\', '/')
      $expectedPrefix = $resolvedSmokeRoot + [IO.Path]::DirectorySeparatorChar
      $smokeRootItem = Get-Item -LiteralPath $resolvedSmokeRoot -Force
      $runRootItem = Get-Item -LiteralPath $resolvedRunRoot -Force
      $owned =
        $resolvedRunRoot.StartsWith(
          $expectedPrefix,
          [StringComparison]::OrdinalIgnoreCase
        ) -and
        ($smokeRootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0 -and
        ($runRootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0 -and
        (Test-Path -LiteralPath $sentinel -PathType Leaf) -and
        ((Get-Content -LiteralPath $sentinel -Raw -Encoding UTF8) -eq $runId)
      if (-not $owned) {
        throw "refusing to clean an unowned smoke directory: $resolvedRunRoot"
      }
      Remove-Item -LiteralPath $resolvedRunRoot -Recurse -Force
      if (Test-Path -LiteralPath $resolvedRunRoot) {
        throw "smoke directory still exists after cleanup: $resolvedRunRoot"
      }
      if (
        (Test-Path -LiteralPath $resolvedSmokeRoot -PathType Container) -and
        @(Get-ChildItem -LiteralPath $resolvedSmokeRoot -Force).Count -eq 0 -and
        $createdSmokeRoot
      ) {
        Remove-Item -LiteralPath $resolvedSmokeRoot -Force
      }
    } catch {
      Write-Warning "Smoke cleanup failed: $($_.Exception.Message)"
      if ($succeeded) {
        $scriptExitCode = 1
      }
    }
  } elseif ($createdRunRoot) {
    Write-Host "[offline-sbs-smoke] retained artifacts: $runRoot"
  }
}

if (-not $succeeded -and $scriptExitCode -eq 0) {
  $scriptExitCode = 1
}
exit $scriptExitCode
