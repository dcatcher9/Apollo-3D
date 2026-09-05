<#
.SYNOPSIS
Runs the production native offline-SBS worker end to end on a deterministic local fixture.

.DESCRIPTION
This is an opt-in GPU integration smoke test. It creates all artifacts below the repository's
.offline-sbs-smoke directory, authenticates the exact worker specification with SHA-256, invokes
the same sunshine.exe --offline-sbs-worker entry point used by the job manager, and verifies the
worker result, full causal scene audit, compressed Matroska output, stream preservation, timestamps,
attachment bytes, exact causal pulse/scene propagation, online flash-veto behavior, color tags, bit depth, and
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
  # Keep SDR and HDR fixtures within the conservative two-source/three-SBS raw-memory
  # overlap reservation by default. Keep the default
  # on the production asynchronous path while allowing callers to exercise the
  # bounded one-source/synchronous-encoder fallback with a smaller explicit
  # value; CPU serialization remains on its one-slot worker thread.
  [int] $TransientRasterMiB = 128,

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

    [AllowNull()]
    [string] $StagingOutput = $null,

    [ValidateSet('evaluate', 'convert')]
    [string] $Operation = 'convert',

    [Parameter(Mandatory = $true)]
    [string] $JobId,

    [Parameter(Mandatory = $true)]
    [string] $FfmpegVersion,

    [Parameter(Mandatory = $true)]
    [string] $FfprobeVersion,

    [int] $RasterMiB = $TransientRasterMiB
  )

  $resultDirectory = Join-Path $JobDirectory 'result'
  $specPath = Join-Path $JobDirectory 'worker-spec.json'
  $progressPath = Join-Path $JobDirectory 'progress.json'
  $resultPath = Join-Path $JobDirectory 'result.json'
  New-Item -ItemType Directory -Path $JobDirectory -Force | Out-Null

  $spec = [ordered] @{
    schema = 2
    job_id = $JobId
    operation = $Operation
    input_path = $InputPath
    job_directory = $JobDirectory
    result_directory = $resultDirectory
    progress_path = $progressPath
    result_path = $resultPath
    staging_output = if ($Operation -eq 'convert') { $StagingOutput } else { $null }
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
    transient_raster = [ordered] @{
      hard_cap_bytes = ([long] $RasterMiB * 1024L * 1024L)
    }
    codec = $Codec
    pipeline = [ordered] @{
      implementation = 'native-causal-single-pass'
      cut_state = 'authenticated-online-pulse-count'
      lookahead = $false
      scene_cache = $false
      replay = $false
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
    JobDirectory = $JobDirectory
    StagingOutput = if ($Operation -eq 'convert') { $StagingOutput } else { $null }
    Operation = $Operation
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

# This fixture has only 16 frames. Assemble its bounded audit in the test after
# authenticating each page; production/UI always read one bounded page at a time.
function Read-VerifiedAudit {
  param([string] $Path, [string] $ExpectedHash)
  Assert-Contract ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -eq $ExpectedHash) 'audit manifest digest differs from worker attestation'
  $manifest = Read-JsonFile -Path $Path
  Assert-Contract ($manifest.schema -eq 4 -and $manifest.version -eq 'whole-clip-scene-audit-v4') 'paged audit contract changed'
  Assert-Contract ([long] $manifest.storage.scene_count -le 16) 'smoke audit exceeds fixture bound'
  $scenes = @()
  $boundaries = @()
  $nextScene = 1L
  $nextSequence = 1L
  $bytes = 0L
  foreach ($descriptor in $manifest.pages) {
    Assert-Contract ([long] $descriptor.index -ge 0 -and [long] $descriptor.index -lt 16) 'audit page index exceeds fixture bound'
    $pagePath = Join-Path (Split-Path -Parent $Path) "scene-audit-page-$($descriptor.index).json"
    $file = Get-Item -LiteralPath $pagePath
    Assert-Contract ($file.Length -eq [long] $descriptor.bytes -and $file.Length -le 2L * 1024 * 1024) 'audit page bytes differ from manifest'
    Assert-Contract ((Get-FileHash -LiteralPath $pagePath -Algorithm SHA256).Hash.ToLowerInvariant() -eq $descriptor.sha256) 'audit page digest differs from manifest'
    $page = Read-JsonFile -Path $pagePath
    Assert-Contract ([long] $descriptor.first_scene_id -eq $nextScene -and [long] $descriptor.start_sequence -eq $nextSequence) 'audit page coverage is not contiguous'
    Assert-Contract (@($page.scenes).Count -eq [long] $descriptor.scene_count) 'audit page scene count differs from manifest'
    $scenes += @($page.scenes)
    $boundaries += @($page.boundary_revisions)
    $nextScene += [long] $descriptor.scene_count
    $nextSequence = [long] $descriptor.end_sequence_exclusive
    $bytes += $file.Length
  }
  Assert-Contract ($nextScene - 1 -eq [long] $manifest.storage.scene_count -and $nextSequence -eq 17 -and $bytes -eq [long] $manifest.storage.total_page_bytes) 'audit manifest totals disagree with authenticated pages'
  $manifest | Add-Member -NotePropertyName scenes -NotePropertyValue $scenes
  $manifest | Add-Member -NotePropertyName boundary_revisions -NotePropertyValue $boundaries
  return $manifest
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
      '-i', 'testsrc2=size=1280x720:rate=1:duration=1',
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
      '-i', 'smptehdbars=size=1280x720:rate=1:duration=1',
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
      '-i', 'color=size=1280x720:rate=1:duration=1',
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

  if ($fixtureIsHdr) {
    Write-Host '[offline-sbs-smoke] checking HDR source/upload budget before allocation'
    foreach ($budgetOperation in @('convert', 'evaluate')) {
      $budgetWorker = New-WorkerSpec `
        -JobDirectory (Join-Path $runRoot "budget-$budgetOperation-job") `
        -InputPath $sourcePath `
        -StagingOutput (Join-Path $runRoot "budget-$budgetOperation-output.mkv") `
        -Operation $budgetOperation `
        -JobId ([guid]::NewGuid().ToString()) `
        -FfmpegVersion $ffmpegVersion `
        -FfprobeVersion $ffprobeVersion `
        -RasterMiB 16
      $budgetRun = Invoke-Worker -Worker $budgetWorker
      Assert-Contract ($budgetRun.ExitCode -eq 2) 'HDR source/upload over-budget worker must fail'
      $budgetResult = Read-JsonFile -Path $budgetWorker.ResultPath
      Assert-Contract (
        $budgetResult.status -eq 'failed' -and
        $budgetResult.error -like '*HDR upload and decoder pipe*before allocation*'
      ) 'HDR source/upload budget was not rejected before allocation'
    }
  }

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
  Assert-Contract ($result.schema -eq 3) 'worker result schema changed'
  Assert-Contract ($result.status -eq 'complete') 'worker result is not complete'
  Assert-Contract ($result.job_id -eq $successWorker.JobId) 'worker result job identity changed'
  Assert-Contract ($result.operation -eq 'convert') 'worker result operation changed'
  Assert-Contract ($result.codec -eq $Codec) 'worker result codec changed'
  Assert-Contract ($result.worker_spec_sha256 -eq $successWorker.SpecSha256) 'worker did not attest the exact specification SHA-256'
  Assert-Contract ($result.python_dependency -eq $false) 'worker unexpectedly reports a Python dependency'
  Assert-Contract ([long] $result.source.frame_count -eq 16) 'worker source frame count changed'
  Assert-Contract ([long] $result.scene_count -gt 0) 'worker committed no scenes'
  Assert-Contract (@($result.scenes).Count -eq [math]::Min(32, [long] $result.scene_count)) 'worker result scene count is inconsistent'
  Assert-Contract (@($result.replay_contracts).Count -eq 0) 'direct conversion unexpectedly published replay contracts'
  Assert-Contract ([long] $result.cache.peak_bytes -eq 0) 'direct conversion unexpectedly wrote a scene cache'
  Assert-Contract ([long] $result.cache.remaining_bytes -eq 0) 'direct conversion retained cached frame bytes'
  Assert-Contract (
    [long] $result.cache.peak_cache_plus_raster_bytes -le
    [long] $result.cache.hard_cap_bytes
  ) 'worker exceeded the exact transient-raster hard cap'

  $analysisContract = $result.analysis_contract
  $runtimeContract = $analysisContract.resolved_runtime
  $sbsContract = $analysisContract.sbs
  Assert-Contract ($analysisContract.artifact_mode -eq 'conversion') 'worker did not run the direct conversion artifact path'
  Assert-Contract ($analysisContract.inference_mode -eq 'single-pass-tensorrt') 'worker did not attest one native conversion pass'
  Assert-Contract ($analysisContract.depth_inference_enabled -eq $true) 'direct conversion disabled online depth inference'
  Assert-Contract (
    [long] $analysisContract.tensorrt_enqueue_count -eq 16
  ) 'conversion did not attest exactly one TensorRT enqueue per source frame'
  Assert-Contract (
    [long] $analysisContract.scheduled_depth_update_count -eq 16
  ) 'conversion did not schedule one online depth update per source frame'
  Assert-Contract ([long] $analysisContract.source_frame_count -eq 16) 'conversion contract source count changed'
  Assert-Contract (
    $runtimeContract.follow_mode -eq $true -and
    [long] $runtimeContract.follow_poll_interval_ms -eq 0 -and
    [long] $runtimeContract.follow_count_bound -eq 16 -and
    [long] $runtimeContract.follow_producer_frame_count -eq 16 -and
    [long] $runtimeContract.follow_first_sequence -eq 1 -and
    $runtimeContract.follow_native_input_deletion -eq $false
  ) 'direct conversion is not using the event-driven unpaced follow path'
  $expectedFollowFormat = 'bgra8'
  $expectedInputFrameFormat = 'sRGB-BGRA8-raw'
  if ($fixtureIsHdr) {
    $expectedFollowFormat = 'gbrpf32le'
    $expectedInputFrameFormat = 'linear-scRGB-f32-planar'
  }
  Assert-Contract (
    $runtimeContract.follow_format -eq $expectedFollowFormat -and
    $null -eq $runtimeContract.follow_frame_pattern -and
    $runtimeContract.frame_transport -eq 'bounded-raw-memory-v1' -and
    $runtimeContract.input_frame_format -eq $expectedInputFrameFormat
  ) 'direct conversion follow raster contract disagrees with the source color mode'
  Assert-Contract (
    $runtimeContract.scene_cache_write -eq $false -and
    $runtimeContract.scene_cache_replay -eq $false -and
    $null -eq $runtimeContract.scene_cache_contract_schema -and
    $null -eq $runtimeContract.scene_plan_schema -and
    $null -eq $runtimeContract.scene_plan_version -and
    $null -eq $runtimeContract.scene_start_sequence -and
    $null -eq $runtimeContract.scene_end_sequence_exclusive -and
    $null -eq $runtimeContract.scene_cache_status_at_replay_start -and
    $null -eq $runtimeContract.scene_cache_processed_count_at_replay_start
  ) 'direct conversion unexpectedly enabled a cache, plan, or replay runtime'
  Assert-Contract (
    $sbsContract.enabled -eq $true -and
    [long] $sbsContract.frame_count -eq 16 -and
    $sbsContract.atomic_publication -eq $true
  ) 'direct conversion SBS count or atomic-publication contract changed'
  Assert-Contract (
    [long] $sbsContract.width -gt 0 -and
    [long] $sbsContract.height -gt 0 -and
    [long] $sbsContract.width -eq [long] $runtimeContract.output_sbs_width -and
    [long] $sbsContract.height -eq [long] $runtimeContract.output_sbs_height -and
    [long] $runtimeContract.output_sbs_width -eq
      (2L * [long] $runtimeContract.output_eye_width) -and
    [long] $runtimeContract.output_sbs_height -eq
      [long] $runtimeContract.output_eye_height
  ) 'direct conversion SBS geometry disagrees with its resolved online renderer geometry'

  # Raw memory bounds include two source leases, the HDR upload conversion,
  # three packed readbacks, bounded OS pipes, and one 64 KiB HDR conversion chunk.
  $sbsPixels = [long] $sbsContract.width * [long] $sbsContract.height
  $sourceBytes = [long] $result.cache.analysis_source_raster_bytes
  $expectedOverlapBytes = 2L * $sourceBytes + 1024L * 1024 + 64L * 1024
  if ($fixtureIsHdr) {
    $expectedOverlapBytes += ($sourceBytes / 12L * 8L) + 3L * 8L * $sbsPixels + 64L * 1024
    $expectedSbsFormat = 'rgba16f'
  } else {
    $expectedOverlapBytes += 3L * 4L * $sbsPixels
    $expectedSbsFormat = 'bgra8'
  }
  $packedBytes = $sbsPixels * $(if ($fixtureIsHdr) { 8L } else { 4L })
  $expectedSerialBytes = $expectedOverlapBytes - $sourceBytes - 2L * $packedBytes
  $expectedRasterBytes = if ($expectedOverlapBytes -le [long] $result.cache.hard_cap_bytes) {
    $expectedOverlapBytes
  } else {
    $expectedSerialBytes
  }
  Assert-Contract (
    $expectedRasterBytes -le [long] $result.cache.hard_cap_bytes
  ) 'smoke hard cap cannot hold the minimum serial raster reservation'
  Assert-Contract (
    [long] $result.cache.peak_live_raster_bytes -eq $expectedRasterBytes
  ) 'worker did not attest the selected serial or overlapping raw-memory reservation'
  Assert-Contract (
    $sbsContract.frame_format -eq $expectedSbsFormat -and
    $null -eq $sbsContract.file_pattern -and
    $sbsContract.row_order -eq 'top-down' -and
    $sbsContract.transfer -eq $runtimeContract.output_transfer -and
    $sbsContract.primaries -eq $runtimeContract.output_primaries -and
    $sbsContract.row_order -eq $runtimeContract.output_row_order -and
    $runtimeContract.follow_atomic_sbs_publication -eq $true
  ) 'direct conversion SBS media contract disagrees with its native runtime'

  $nativeWork = Join-Path $successWorker.JobDirectory 'native-work'
  $rasters = @(Get-ChildItem -LiteralPath (Join-Path $nativeWork 'analysis-input'), (Join-Path $nativeWork 'analysis-output') -File | Where-Object { $_.Extension -in @('.bmp', '.pfm', '.png') })
  Assert-Contract ($rasters.Count -eq 0) 'raw-memory conversion published a disk raster'
  Assert-Contract (-not (Test-Path -LiteralPath (Join-Path $nativeWork 'encoded-video.ffconcat'))) 'raw-memory conversion still uses a concat/HTTP frame bridge'
  foreach ($forbiddenPath in @(
      (Join-Path $nativeWork 'scene-cache'),
      (Join-Path $nativeWork 'render-input'),
      (Join-Path $nativeWork 'render-output'),
      (Join-Path $nativeWork 'logs\render-current-scene.log')
    )) {
    Assert-Contract (-not (Test-Path -LiteralPath $forbiddenPath)) (
      "direct conversion unexpectedly created '$forbiddenPath'"
    )
  }
  # The direct-worker smoke has no manager-held root pin. Its enclosing random
  # workspace is removed in this script's guarded finally block.
  Assert-Contract (
    (Test-Path -LiteralPath $successOutput -PathType Leaf) -and
    (Get-Item -LiteralPath $successOutput).Length -gt 0
  ) 'worker produced no compressed Matroska output'

  $auditPath = Join-Path $successWorker.ResultDirectory 'scene-audit.json'
  $audit = Read-VerifiedAudit -Path $auditPath -ExpectedHash $result.scene_audit_manifest_sha256
  Assert-Contract ($audit.schema -eq 4) 'scene audit schema changed'
  Assert-Contract ($audit.version -eq 'whole-clip-scene-audit-v4') 'scene audit version changed'
  Assert-Contract ($audit.status -eq 'complete') 'scene audit is not complete'
  Assert-Contract (@($audit.scenes).Count -eq [long] $result.scene_count) 'full scene audit count differs from result'
  Assert-Contract ($audit.claims.ground_truth -eq $false) 'scene audit incorrectly claims ground truth'
  Assert-Contract ($audit.claims.best_parameters -eq $false) 'scene audit incorrectly claims globally best parameters'
  Assert-Contract (
    $audit.policy.implementation -eq 'native-causal-scene-tracker' -and
    $audit.policy.version -eq 'causal-scene-epochs-v1' -and
    $audit.policy.lookahead -eq $false -and
    $audit.policy.boundary_only -eq $true -and
    $audit.policy.python_dependency -eq $false
  ) 'scene audit is not the causal online hard-cut epoch policy'
  Assert-Contract ([long] $audit.cache.peak_bytes -eq 0) 'scene audit unexpectedly reports cached frames'
  Assert-Contract (
    [long] $audit.cache.peak_cache_plus_raster_bytes -le
    [long] $audit.cache.hard_cap_bytes
  ) 'scene audit reports a transient-raster hard-cap violation'
  $boundaryRevisions = @($audit.boundary_revisions)
  Assert-Contract (
    $boundaryRevisions.Count -eq ([long] $result.scene_count - 1L)
  ) 'causal scene count does not match the authenticated online pulse count'
  $falseFlashCuts = @($boundaryRevisions | Where-Object {
      $_.accepted -eq $true -and
      $_.final_sequence -ge 9 -and $_.final_sequence -le 11
    })
  Assert-Contract (
    $falseFlashCuts.Count -eq 0
  ) 'online hard-cut state treated the one-frame exposure flash as a scene boundary'
  $causalScenes = @($audit.scenes)
  $appearanceVetoCount = 0L
  foreach ($scene in $causalScenes) {
    $appearanceVetoCount += [long] $scene.evidence.appearance_veto_count
  }
  Assert-Contract (
    $appearanceVetoCount -gt 0
  ) 'deterministic flash fixture did not exercise the online appearance-veto path'
  $expectedSceneStart = 1L
  for ($sceneIndex = 0; $sceneIndex -lt $causalScenes.Count; ++$sceneIndex) {
    $scene = $causalScenes[$sceneIndex]
    $terminalScene = $sceneIndex -eq ($causalScenes.Count - 1)
    Assert-Contract (
      [long] $scene.scene_id -eq ($sceneIndex + 1L) -and
      [long] $scene.start_sequence -eq $expectedSceneStart -and
      [long] $scene.end_sequence_exclusive -eq
        ([long] $scene.start_sequence + [long] $scene.frame_count) -and
      [long] $scene.cache_bytes -eq 0 -and
      $scene.cut_state_semantics -eq 'causal-production-exact'
    ) "causal scene $($sceneIndex + 1) is not a contiguous zero-cache online epoch"
    if ($terminalScene) {
      Assert-Contract (
        [long] $scene.end_sequence_exclusive -eq 17 -and
        $scene.boundary.accepted -eq $true -and
        $scene.boundary.semantic_cut -eq $false -and
        $scene.boundary.decision -eq 'end_of_stream'
      ) 'final causal scene does not close exactly at source EOF'
    } else {
      $pulse = $boundaryRevisions[$sceneIndex]
      Assert-Contract (
        $scene.boundary.accepted -eq $true -and
        $scene.boundary.semantic_cut -eq $true -and
        $scene.boundary.decision -eq 'confirmed' -and
        [long] $scene.boundary.final_sequence -eq
          [long] $scene.end_sequence_exclusive -and
        $pulse.accepted -eq $true -and
        $pulse.semantic_cut -eq $true -and
        $pulse.decision -eq 'confirmed' -and
        [long] $pulse.final_sequence -eq
          [long] $scene.end_sequence_exclusive
      ) "causal scene $($sceneIndex + 1) does not end at its exact online hard-cut pulse"
    }
    $expectedSceneStart = [long] $scene.end_sequence_exclusive
  }

  $adaptiveContract = $analysisContract.adaptive_state
  Assert-Contract (
    $adaptiveContract.transport -eq 'atomic-latest-v1' -and
    $adaptiveContract.retained_history -eq $false -and
    [long] $adaptiveContract.frame_count -eq 16
  ) 'offline analysis did not attest its bounded adaptive-state transport'
  Assert-Contract (
    $analysisContract.PSObject.Properties.Name -notcontains 'cut_state'
  ) 'whole-clip contract still advertises the suppressed cut-state trace'
  $rawAdaptiveHistory = @(Get-ChildItem -LiteralPath (
      $nativeWork
    ) -Recurse -File | Where-Object {
      $_.Name -eq 'adaptive_state.jsonl' -or
      $_.Name -eq 'cut_state.json'
    })
  Assert-Contract (
    $rawAdaptiveHistory.Count -eq 0
  ) 'offline worker retained an unbounded per-frame state history'
  $boundedSnapshots = @(Get-ChildItem -LiteralPath (
      $nativeWork
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
    # The fixed 2560x720 fixture occupies a 2560x768 64x64 coded raster. Its
    # 1,966,080 samples exceed level 3.1 and fit the level 4.0 limit.
    Assert-Contract (
      $null -ne $outputVideo.PSObject.Properties['level'] -and
      [int] $outputVideo.level -eq 8
    ) 'AV1 smoke fixture must use the lowest compatible defined level 4.0'
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

  Write-Host '[offline-sbs-smoke] running authenticated evaluation worker'
  $evaluationJobDirectory = Join-Path $runRoot 'evaluation-job'
  $evaluationWorker = New-WorkerSpec `
    -Operation 'evaluate' `
    -JobDirectory $evaluationJobDirectory `
    -InputPath $sourcePath `
    -JobId ([guid]::NewGuid().ToString()) `
    -FfmpegVersion $ffmpegVersion `
    -FfprobeVersion $ffprobeVersion
  $evaluationRun = Invoke-Worker -Worker $evaluationWorker
  if ($evaluationRun.ExitCode -ne 0) {
    $evaluationFailureDetail = ''
    if (Test-Path -LiteralPath $evaluationWorker.ResultPath -PathType Leaf) {
      $failedEvaluation = Read-JsonFile -Path $evaluationWorker.ResultPath
      $evaluationFailureDetail = "`nWorker result: $($failedEvaluation.error)"
    }
    throw (
      "Production offline evaluation failed with exit code $($evaluationRun.ExitCode)." +
      "$evaluationFailureDetail`n$($evaluationRun.Text)"
    )
  }

  $evaluation = Read-JsonFile -Path $evaluationWorker.ResultPath
  $evaluationAnalysis = $evaluation.analysis_contract
  $evaluationRuntime = $evaluationAnalysis.resolved_runtime
  Assert-Contract (
    $evaluation.schema -eq 3 -and
    $evaluation.status -eq 'complete' -and
    $evaluation.operation -eq 'evaluate' -and
    $evaluation.worker_spec_sha256 -eq $evaluationWorker.SpecSha256 -and
    $null -eq $evaluation.output -and
    $null -eq $evaluation.staging_identity
  ) 'evaluation result identity or output-absence contract changed'
  Assert-Contract (
    $evaluationAnalysis.artifact_mode -eq 'adaptive' -and
    $evaluationAnalysis.inference_mode -eq 'single-pass-tensorrt' -and
    $evaluationAnalysis.depth_inference_enabled -eq $true -and
    [long] $evaluationAnalysis.source_frame_count -eq 16 -and
    [long] $evaluationAnalysis.scheduled_depth_update_count -eq 16 -and
    [long] $evaluationAnalysis.tensorrt_enqueue_count -eq 16 -and
    [long] $evaluationAnalysis.observation_timeline.count -eq 16 -and
    $evaluationAnalysis.observation_timeline.sha256 -eq
      $analysisContract.observation_timeline.sha256
  ) 'evaluation did not run the same frame-driven online estimator pass'
  Assert-Contract (
    $evaluationRuntime.follow_mode -eq $true -and
    $evaluationRuntime.follow_format -eq $expectedFollowFormat -and
    $evaluationRuntime.frame_transport -eq 'bounded-raw-memory-v1' -and
    $null -eq $evaluationRuntime.follow_frame_pattern -and
    $evaluationRuntime.input_frame_format -eq $expectedInputFrameFormat -and
    [long] $evaluationRuntime.follow_poll_interval_ms -eq 0 -and
    $evaluationRuntime.parallax_v2_render -eq $true -and
    $evaluationRuntime.parallax_v2_live -eq $true -and
    $evaluationRuntime.scene_cache_write -eq $false -and
    $evaluationRuntime.scene_cache_replay -eq $false
  ) 'evaluation does not share the direct causal online runtime contract'
  Assert-Contract (
    $evaluationAnalysis.sbs.enabled -eq $false -and
    [long] $evaluationAnalysis.sbs.frame_count -eq 0 -and
    [long] $evaluationAnalysis.sbs.width -eq 0 -and
    [long] $evaluationAnalysis.sbs.height -eq 0 -and
    $evaluationAnalysis.sbs.atomic_publication -eq $false -and
    $evaluationRuntime.follow_atomic_sbs_publication -eq $false -and
    @($evaluation.replay_contracts).Count -eq 0 -and
    [long] $evaluation.cache.peak_bytes -eq 0 -and
    [long] $evaluation.cache.remaining_bytes -eq 0
  ) 'evaluation unexpectedly produced SBS, cache, or replay artifacts'
  Assert-Contract (
    $evaluation.timeline_contract.mode -eq 'evaluation-only' -and
    $null -eq $evaluation.timeline_contract.max_output_ticks -and
    -not (Test-Path -LiteralPath (
      Join-Path $evaluationWorker.ResultDirectory 'timeline-contract.json'
    ))
  ) 'evaluation advertised a conversion timeline or mux artifact'
  Assert-Contract (
    [long] $evaluation.scene_count -eq [long] $result.scene_count
  ) 'evaluation and conversion disagreed on causal online scene count'
  $evaluationScenes = @($evaluation.scenes)
  Assert-Contract (
    $evaluationScenes.Count -eq [math]::Min(32, [long] $evaluation.scene_count)
  ) 'evaluation scene array disagrees with its declared causal scene count'
  for ($sceneIndex = 0; $sceneIndex -lt $causalScenes.Count; ++$sceneIndex) {
    Assert-Contract (
      [long] $evaluationScenes[$sceneIndex].start_sequence -eq
        [long] $causalScenes[$sceneIndex].start_sequence -and
      [long] $evaluationScenes[$sceneIndex].end_sequence_exclusive -eq
        [long] $causalScenes[$sceneIndex].end_sequence_exclusive -and
      $evaluationScenes[$sceneIndex].boundary.decision -eq
        $causalScenes[$sceneIndex].boundary.decision -and
      $evaluationScenes[$sceneIndex].boundary.semantic_cut -eq
        $causalScenes[$sceneIndex].boundary.semantic_cut
    ) "evaluation and conversion disagreed on causal scene $($sceneIndex + 1)"
  }
  $evaluationAudit = Read-VerifiedAudit -Path (
    Join-Path $evaluationWorker.ResultDirectory 'scene-audit.json'
  ) -ExpectedHash $evaluation.scene_audit_manifest_sha256
  Assert-Contract (
    $evaluationAudit.schema -eq 4 -and
    $evaluationAudit.policy.implementation -eq 'native-causal-scene-tracker' -and
    $evaluationAudit.policy.lookahead -eq $false -and
    @($evaluationAudit.scenes).Count -eq [long] $evaluation.scene_count
  ) 'evaluation scene audit is not the same no-lookahead causal contract'

  $succeeded = $true
  Write-Host (
    "[offline-sbs-smoke] PASS: conversion + evaluation, $($result.scene_count) " +
    "causal scene(s), 16 frames each, $Codec, $($FixtureColor.ToUpperInvariant()), " +
    'flash not cut, no lookahead/cache/replay, exact authenticated worker contracts'
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
