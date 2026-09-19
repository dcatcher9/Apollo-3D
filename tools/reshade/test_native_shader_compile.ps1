# SPDX-License-Identifier: GPL-3.0-only
# CPU-only: compile production ReShade FX/Direct3D bytecode and inspect its ABI.
param(
    [Parameter(Mandatory)][string] $ValidatorPath,
    [Parameter(Mandatory)][string] $OutputDirectory
)
$ErrorActionPreference = 'Stop'
$validator = (Get-Item -LiteralPath $ValidatorPath -ErrorAction Stop).FullName
$effect = Join-Path $PSScriptRoot 'Shaders\SunshineDepth3D.fx'
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Use a fresh isolated output directory.' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
function Require([bool] $Condition, [string] $Message) {
    if (-not $Condition) { throw $Message }
}

$variants = @()
foreach ($backend in @('0xb000', '0xc000')) {
    foreach ($source in @(
        @{Name='sdr8'; Color=1; Bits=8},
        @{Name='sdr10'; Color=1; Bits=10},
        @{Name='scrgb'; Color=2; Bits=16},
        @{Name='pq'; Color=3; Bits=10}
    )) {
        foreach ($size in @(@{Width=1280; Height=720}, @{Width=3840; Height=2160})) {
            $variants += @{
                Name=($backend + '-' + $source.Name + '-' + $size.Width)
                Backend=$backend; Color=$source.Color; Bits=$source.Bits
                Width=$size.Width; Height=$size.Height; Performance=0
            }
        }
    }
}
$variants += @{Name='maximum-contract-size'; Backend='0xc000'; Color=2; Bits=16; Width=8192; Height=8192; Performance=0}
# This checks the macro branch only. Actual ReShade performance mode enables
# specialization separately and needs the official-runtime fixture as well.
$variants += @{Name='performance-macro'; Backend='0xb000'; Color=1; Bits=8; Width=1920; Height=1080; Performance=1}

foreach ($variant in $variants) {
    $output = Join-Path $OutputDirectory $variant.Name
    & $validator $effect $output __RESHADE__=60800 "__RENDERER__=$($variant.Backend)" `
        "BUFFER_COLOR_SPACE=$($variant.Color)" "BUFFER_COLOR_BIT_DEPTH=$($variant.Bits)" `
        "BUFFER_WIDTH=$($variant.Width)" "BUFFER_HEIGHT=$($variant.Height)" `
        "__RESHADE_PERFORMANCE_MODE__=$($variant.Performance)"
    if ($LASTEXITCODE -ne 0) { throw "Shader compilation failed: $($variant.Name)" }
    $bindings = [IO.File]::ReadAllText((Join-Path $output 'bindings.txt'))
    $format = if ($variant.Color -eq 1) { 24 } else { 10 }
    $transfer = if ($variant.Color -eq 1) { 'srgb' } else { 'scrgb' }
    $texture = 'V__SunshineDepth3D__DoubleTex'
    foreach ($line in @(
        "resource $texture $format $($variant.Width * 2) $($variant.Height)",
        "annotation $texture sunshine_sbs_export 1",
        "annotation $texture sunshine_sbs_layout `"sbs_lr`"",
        "annotation $texture sunshine_sbs_source_color_space $($variant.Color)",
        "annotation $texture sunshine_sbs_color_space `"$transfer`"",
        "annotation $texture sunshine_sbs_source_width $($variant.Width)",
        "annotation $texture sunshine_sbs_source_height $($variant.Height)",
        "  render_target 0 $texture"
    )) {
        Require ($bindings -match ('(?m)^' + [regex]::Escape($line) + '\r?$')) "Missing export contract in $($variant.Name): $line"
    }
    Require ([regex]::Matches($bindings, '(?m)^pass ').Count -eq 1) 'The native shader must have exactly one offscreen pass.'
    Require ($bindings -match '(?m)^pass SunshineDepth3D/Stereo ') 'The owning stereo technique/pass changed.'
    Require ([regex]::Matches($bindings, '(?m)^  render_target ').Count -eq 1) 'Unexpected render target: native game presentation must remain untouched.'
    Require ([regex]::Matches($bindings, '(?m)^resource_detail .* "DEPTH" 1\r?$').Count -eq 1) 'Native DEPTH semantic is missing.'
    Require ($bindings -notmatch 'HostSBS_|SuperDepth3D|AXAA') 'Native shader depends on an external/custom retired renderer.'

    $defaults = [IO.File]::ReadAllBytes((Join-Path $output 'defaults.bin'))
    foreach ($uniform in @('Strength','ScreenPlane','DepthDirection','DepthView','EdgeAntialias',
        'Sunshine_DepthReady','Sunshine_Calibrated','Sunshine_RawAnchor','Sunshine_RawGain','Sunshine_DepthRect')) {
        $entry = [regex]::Match($bindings, '(?m)^uniform ' + [regex]::Escape($uniform) + ' (\d+) (\d+)\r?$')
        Require $entry.Success "Missing native uniform: $uniform"
        $offset = [int]$entry.Groups[1].Value
        if ($uniform -in @('Sunshine_DepthReady','Sunshine_Calibrated','Sunshine_RawAnchor','Sunshine_RawGain','DepthDirection','DepthView')) {
            Require ([BitConverter]::ToUInt32($defaults, $offset) -eq 0) "Unsafe initial state: $uniform"
        } elseif ($uniform -eq 'Strength') {
            Require ([BitConverter]::ToSingle($defaults, $offset) -eq 1.0) 'Default strength changed.'
        } elseif ($uniform -eq 'Sunshine_DepthRect') {
            for ($component = 0; $component -lt 4; ++$component) {
                Require ([BitConverter]::ToSingle($defaults, $offset + $component * 4) -eq 0.0) 'Uninitialized depth rectangle must remain unavailable until the add-on supplies it.'
            }
        }
    }
}

$rejections = @(
    @('__RESHADE__=60703'), @('__RENDERER__=0xa000'), @('__RENDERER__=0x14300'),
    @('BUFFER_COLOR_SPACE=0'), @('BUFFER_COLOR_SPACE=4'), @('BUFFER_WIDTH=0'),
    @('BUFFER_WIDTH=1921'), @('BUFFER_HEIGHT=1081'), @('BUFFER_WIDTH=8194'), @('BUFFER_HEIGHT=8194')
)
for ($index = 0; $index -lt $rejections.Count; ++$index) {
    $diagnostics = & $validator $effect (Join-Path $OutputDirectory "rejected-$index") `
        __RESHADE__=60800 @($rejections[$index]) 2>&1
    Require ($LASTEXITCODE -ne 0 -and ($diagnostics -join "`n") -match 'Sunshine native-depth stereo') `
        "Incompatible shader configuration was not explicitly rejected: $($rejections[$index] -join ', ')"
}
Write-Output "PASS native shader: $($variants.Count) production compiler/ABI configurations and $($rejections.Count) explicit rejections. No GPU execution."
