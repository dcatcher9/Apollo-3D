# SPDX-License-Identifier: GPL-3.0-only
# Compile the externally supplied, separately licensed Game 3D shader. This does
# not distribute the shader, translate its renderer, or substitute for GPU tests.
param(
    [Parameter(Mandatory)][string] $ValidatorPath,
    [Parameter(Mandatory)][string] $ShaderDirectory,
    [Parameter(Mandatory)][string] $ControlShaderDirectory,
    [Parameter(Mandatory)][string] $OutputDirectory
)
$ErrorActionPreference = 'Stop'
$validator = (Get-Item -LiteralPath $ValidatorPath).FullName
$effect = (Get-Item -LiteralPath (Join-Path $ShaderDirectory 'SunshineGame3D.fx')).FullName
$controlEffect = (Get-Item -LiteralPath (Join-Path $ControlShaderDirectory 'SunshineGame3D.fx')).FullName
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Use a fresh isolated output directory.' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$outputRoot = (Get-Item -LiteralPath $OutputDirectory).FullName

function Require([bool] $Condition, [string] $Message) {
    if (-not $Condition) { throw $Message }
}
function Compile([string] $InputEffect, [string] $Name, [string[]] $Definitions) {
    $directory = Join-Path $outputRoot $Name
    & $validator $InputEffect $directory __RESHADE__=60800 @Definitions *> ($directory + '.txt')
    Require ($LASTEXITCODE -eq 0) "FX/Direct3D compilation failed: $Name (see $directory.txt)."
    return [IO.File]::ReadAllText((Join-Path $directory 'bindings.txt'))
}
function Require-NoEffectDefinitions([string] $Name) {
    $manifestPath = Join-Path (Join-Path $outputRoot $Name) 'preprocessor_definitions.txt'
    Require (Test-Path -LiteralPath $manifestPath -PathType Leaf) 'Rebuild the shader validator to emit actual preprocessor-definition metadata.'
    $manifest = [IO.File]::ReadAllText($manifestPath)
    Require ($manifest -match '(?m)^format 1\r?$') 'Unknown preprocessor-definition manifest format.'
    Require ($manifest -notmatch '(?m)^displayed ') "Game 3D still exposes effect preprocessor settings: $Name"
}
function Require-SameBytecode([string] $First, [string] $Second) {
    $firstDirectory = Join-Path $outputRoot $First
    $secondDirectory = Join-Path $outputRoot $Second
    $entries = @(Get-ChildItem -LiteralPath $firstDirectory -Filter '*.cso' -File)
    Require ($entries.Count -gt 0 -and $entries.Count -eq @(Get-ChildItem -LiteralPath $secondDirectory -Filter '*.cso' -File).Count) "Compiled entry-point count differs: $First / $Second"
    foreach ($entry in $entries) {
        $peer = Join-Path $secondDirectory $entry.Name
        Require ((Test-Path -LiteralPath $peer) -and
            (Get-FileHash -LiteralPath $entry.FullName).Hash -ceq (Get-FileHash -LiteralPath $peer).Hash) "Obsolete setting changed compiled shader: $Second / $($entry.Name)"
    }
}
function UniformDefaults([string] $Bindings, [string] $Directory) {
    $bytes = [IO.File]::ReadAllBytes((Join-Path $Directory 'defaults.bin'))
    $result = @{}
    foreach ($entry in [regex]::Matches($Bindings, '(?m)^uniform (\S+) (\d+) (\d+)\r?$')) {
        $offset = [int] $entry.Groups[2].Value
        $length = [int] $entry.Groups[3].Value
        $result[$entry.Groups[1].Value] = [Convert]::ToBase64String($bytes, $offset, $length)
    }
    return $result
}

# Only these user controls remain. Camera values are dynamic add-on inputs.
$activeControls = @('Depth_Adjustment', 'Depth_Map_View')
$retiredControls = @('Sharpen_Power', 'View_Mode',
    'Warping_Masking', 'Weapon_Near_Halo_Reduction', 'View_Mode_Warping', 'Performance_Level',
    'Compatibility_Power', 'De_Artifacting', 'Reconstruction_Size', 'Extended_Smoothing')
$sourceOwned = @('Sunshine_DepthReady', 'Sunshine_DepthJitter', 'Sunshine_CameraDepthReady',
    'Sunshine_CameraCoordinateBasis', 'Sunshine_CameraProjection', 'Sunshine_CameraRawDepthRange',
    'Sunshine_CameraDepthScale', 'Sunshine_CameraConvergence', 'Sunshine_CameraStrengthBlend',
    'Sunshine_CameraDepthRect')
$sourceText = [IO.File]::ReadAllText($effect) + "`n" +
    [IO.File]::ReadAllText((Join-Path $ShaderDirectory 'SunshineCamera.fxh')) + "`n" +
    [IO.File]::ReadAllText((Join-Path $ShaderDirectory 'SunshineHostWarp.fxh'))
Require ($sourceText -notmatch '\b(Sharpen_Power|SmartSharp|SunshinePostEye|SunshineSharpHDR|SunshinePost\.fxh|SunshineIntegration\.fxh)\b') 'Current shader still depends on retired sharpening/reference postprocessing.'
$includes = @([regex]::Matches($sourceText, '(?m)^\s*#\s*include\s+"([^"]+)"') | ForEach-Object { $_.Groups[1].Value } | Sort-Object)
Require (($includes -join '|') -ceq 'SunshineCamera.fxh|SunshineHostWarp.fxh') 'Current shader must use only its camera and warp includes.'
# The CPU validator does not emit uniform annotations. Require the explicit
# declaration here; the actual-runtime fixture separately reflects this marker.
Require ($sourceText -match '(?s)uniform\s+float\s+Depth_Adjustment\s*<(?:(?!>).)*sunshine_warp_version\s*=\s*2\s*;') 'Missing current warp capability annotation.'
foreach ($name in $sourceOwned) {
    # ReShade can specialize initialized uniforms in Performance Mode even with
    # a source annotation. Their declarations must have no initializer at all.
    $declaration = '(?s)uniform\s+\w+\s+' + [regex]::Escape($name) +
        '\s*<(?:(?!>).)*source\s*=\s*"[^"\r\n]+"(?:(?!>).)*>\s*;'
    Require ($sourceText -match $declaration) "Source-owned input has an initializer or lost its source annotation: $name"
}
function RenderGraph([string] $Bindings) {
    $graph = [Collections.Generic.List[string]]::new()
    $skipPass = $false
    foreach ($line in ($Bindings -split "`r?`n")) {
        if ($line.StartsWith('pass ')) {
            # Upstream's optional text/help technique is outside SBS rendering.
            $skipPass = $line.StartsWith('pass Information_SD/Help ')
            if (-not $skipPass) { $graph.Add($line) }
        } elseif ($line.StartsWith('  pixel_entries ') -and -not $skipPass) {
            $graph.Add($line)
        }
    }
    return $graph -join "`n"
}

$variants = @()
foreach ($backend in @('0xb000', '0xc000')) {
    foreach ($color in @(1, 2, 3)) {
        $variants += @{
            Name="$backend-color$color"; Backend=$backend
            Color=$color; Width=1920; Height=1080; Extra=@()
        }
    }
}
$variants += @{ Name='pq4k'; Backend='0xc000'; Color=3; Width=3840; Height=2160; Extra=@() }
$variants += @{ Name='scrgb4k'; Backend='0xc000'; Color=2; Width=3840; Height=2160; Extra=@() }
$variants += @{ Name='sdr4k'; Backend='0xb000'; Color=1; Width=3840; Height=2160; Extra=@() }
$variants += @{ Name='performance-macro'; Backend='0xc000'; Color=2; Width=1920; Height=1080; Extra=@('__RESHADE_PERFORMANCE_MODE__=1') }

foreach ($variant in $variants) {
    $bits = if ($variant.Color -eq 2) { 16 } elseif ($variant.Color -eq 3) { 10 } else { 8 }
    $definitions = @("__RENDERER__=$($variant.Backend)", "BUFFER_COLOR_SPACE=$($variant.Color)",
        "BUFFER_COLOR_BIT_DEPTH=$bits", "BUFFER_WIDTH=$($variant.Width)", "BUFFER_HEIGHT=$($variant.Height)") + $variant.Extra
    $control = Compile $controlEffect ($variant.Name + '-control') $definitions
    $treatment = Compile $effect ($variant.Name + '-treatment') $definitions
    Require-NoEffectDefinitions ($variant.Name + '-treatment')

    $format = if ($variant.Color -eq 1) { 24 } else { 10 }
    $transfer = if ($variant.Color -eq 1) { 'srgb' } else { 'scrgb' }
    $texture = 'V__SunshineGame3D__DoubleTex'
    foreach ($line in @(
        "resource $texture $format $($variant.Width * 2) $($variant.Height)",
        "annotation $texture sunshine_sbs_export 1",
        "annotation $texture sunshine_sbs_layout `"sbs_lr`"",
        "annotation $texture sunshine_sbs_source_color_space $($variant.Color)",
        "annotation $texture sunshine_sbs_color_space `"$transfer`"",
        "annotation $texture sunshine_sbs_source_width $($variant.Width)",
        "annotation $texture sunshine_sbs_source_height $($variant.Height)"
    )) {
        Require ($treatment -match ('(?m)^' + [regex]::Escape($line) + '\r?$')) "Missing export contract in $($variant.Name): $line"
    }
    foreach ($name in @('SunshineEyeLeft', 'SunshineEyeRight')) {
        $line = "resource V__SunshineGame3D__$name 10 $($variant.Width) $($variant.Height)"
        Require ($treatment -match ('(?m)^' + [regex]::Escape($line) + '\r?$')) "Missing full-resolution eye: $name"
    }
    foreach ($entry in @('SunshineRenderEyesPS', 'SunshinePackEyesPS', 'SunshineMonoOut')) {
        Require ($treatment -match "(?m)^  pixel_entries \S+ \S*$entry\S*\r?$") "Missing export/passthrough entry: $entry"
    }
    Require ($treatment -notmatch '(?m)^pass SunshineGame3D/(Muti_Mode_Reconstruction|InfillMask|USMOut|SDAA|Framed|REST) ') 'Removed physical-display pass returned.'
    Require ($treatment -notmatch '(?m)^resource \S*(texSD_CB_L|texSD_CB_R|texSD_RL) ') 'Removed display-reconstruction texture returned.'
    Require ($treatment -notmatch '(?m)^uniform (Stereoscopic_Mode|Interlace_Optimization|Scaling_Support|Frame_Alternate|Alternate|Frames) ') 'Removed presentation control returned.'
    Require ($treatment -match '(?m)^uniform Sunshine_DepthReady ') 'Depth readiness is no longer dynamic.'
    Require ($treatment -notmatch '(?m)^resource \S*SunshineSharp(Left|Right) ') 'Retired intermediate eye texture returned.'
    Require ($treatment -notmatch '(?m)^pass SunshineGame3D/SunshineSharpen ') 'Retired separate sharpening pass returned.'
    # Cleanup must preserve the rendering graph and resource geometry.
    Require ((RenderGraph $control) -ceq (RenderGraph $treatment)) "Supported render graph changed: $($variant.Name)"
    $passNames = @([regex]::Matches($treatment, '(?m)^pass (\S+) ') | ForEach-Object { $_.Groups[1].Value })
    $expectedPasses = @()
    if ($variant.Color -eq 3) { $expectedPasses += 'SunshineGame3D/SunshinePreparePQ' }
    $expectedPasses += @('SunshineGame3D/SunshineHostCandidate', 'SunshineGame3D/SunshineHostVertical',
        'SunshineGame3D/SunshineHostHorizontal', 'SunshineGame3D/SunshineEyes', 'SunshineGame3D/DoubleOut', 'SunshineGame3D/StereoOut')
    Require (($passNames -join '|') -ceq ($expectedPasses -join '|')) "Sole Host pass graph differs: $($variant.Name)"
    foreach ($name in @('SunshineHostCandidate', 'SunshineHostVerticalMajorant', 'SunshineHostVerticalConditioned', 'SunshineHostFinal')) {
        $line = "resource V__SunshineGame3D__$name 41 $($variant.Width) $($variant.Height)"
        Require ($treatment -match ('(?m)^' + [regex]::Escape($line) + '\r?$')) "Missing full-resolution R32 Host field: $name"
    }
    Require ($treatment -notmatch '(?m)^resource \S*(texDMN|texCN|texMiniReconBuffer|texzBufferN_[PLM]|texzBufferBlurN|texzBufferBlurEx|texReconBuffer|texSmooth|TAABuffer|AccBuffer|Info_Tex|texAvrN) ') 'Retired depth/history resource returned.'
    Require ($treatment -notmatch '(?m)^uniform (Sunshine_WarpMethod|FrameTime|Timer|frametime|timer) ') 'Retired warp/time input returned.'
    foreach ($pass in @(
        "pass SunshineGame3D/SunshineHostCandidate E__SunshineGame3D__SunshineHostCandidateCS_8_8_1 $([int][Math]::Ceiling($variant.Width / 8)) $([int][Math]::Ceiling($variant.Height / 8)) 1",
        "pass SunshineGame3D/SunshineHostVertical E__SunshineGame3D__SunshineHostVerticalCS_32_1_1 $($variant.Width) 1 1",
        "pass SunshineGame3D/SunshineHostHorizontal E__SunshineGame3D__SunshineHostHorizontalCS_32_1_1 $($variant.Height) 1 1",
        "pass SunshineGame3D/SunshineEyes  $($variant.Width) $($variant.Height) 1",
        "pass SunshineGame3D/DoubleOut  $($variant.Width * 2) $($variant.Height) 1",
        'pass SunshineGame3D/StereoOut  0 0 1')) {
        Require ($treatment -match ('(?m)^' + [regex]::Escape($pass) + '\r?$')) "Sole Host pass geometry changed: $pass"
    }
    $before = UniformDefaults $control (Join-Path $outputRoot ($variant.Name + '-control'))
    $after = UniformDefaults $treatment (Join-Path $outputRoot ($variant.Name + '-treatment'))
    foreach ($name in $activeControls) {
        Require ($before.ContainsKey($name) -and $after.ContainsKey($name) -and $after[$name] -ceq $before[$name]) "Shared quality control/default changed: $name / $($variant.Name)"
    }
    foreach ($name in $retiredControls) {
        Require (-not $after.ContainsKey($name)) "Retired quality control returned: $name"
    }
    Require ($after.Count -eq ($activeControls.Count + $sourceOwned.Count)) 'Current shader has unexpected uniforms.'
    foreach ($name in $sourceOwned) {
        Require ($after.ContainsKey($name)) "Dynamic source-owned input missing: $name / $($variant.Name)"
        Require (-not ([Convert]::FromBase64String($after[$name]) | Where-Object { $_ -ne 0 })) "Source-owned default must remain zero: $name"
    }
    foreach ($name in @('Depth_Map', 'Depth_Map_Adjust', 'Depth_Map_Flip', 'Range_Boost',
        'Zero_Parallax_Distance', 'ZPD_OverShoot', 'ZPD_Balance', 'ZPD_Boundary', 'ZPD_Boundary_n_Fade',
        'ZPD_Boundary_n_Cutoff_A', 'ZPD_Screen_Edge_Avoidance', 'Auto_Depth_Adjust', 'WP',
        'WZPD_and_WND', 'Weapon_Adjust', 'Weapon_Depth_Edge', 'Weapon_ZPD_Boundary',
        'Weapon_Reduction_n_Power', 'World_n_Fade_Reduction_Power', 'SunshineGenericWeaponHelp',
        'Starting_Resolution', 'DLSS_FSR_Offset', 'AR_Side_Shrink', 'Auto_Scaler_Adjust',
        'USE_AA', 'Sunshine_DepthJitterBounds')) {
        Require (-not $after.ContainsKey($name)) "Removed Manual/profile/alignment control returned: $name / $($variant.Name)"
    }
    Write-Output "PASS $($variant.Name): unchanged export/render graph, two user controls, dynamic camera/jitter, no sharpening or configurable effect definitions."

}

# Saved global definitions used by other effects must not change this renderer.
# Use adversarial combinations and compare executable entry-point bytecode.
$retiredDefinitions = @('SUNSHINE_GAME3D_AUTOMATIC', 'SUNSHINE_GAME3D_GENERIC_CONFIG',
    'SUNSHINE_SOURCE_OWNED_DEPTH', 'SUNSHINE_GAME3D_COLOR_FIXES', 'SUNSHINE_GAME3D_FUSED_POST',
    'SUNSHINE_GAME3D_NATIVE_WARP', 'EX_DLP_FS_Mode', 'Virtual_Reality_Mode', 'Anaglyph_Mode',
    'Inficolor_3D_Emulator', 'Reconstruction_Mode', 'REST_UI_Mode', 'Super3D_Mode',
    'Use_2D_Plus_Depth', 'Frame_Packed_Mode', 'DoubleBuffer_Mode', 'Anti_Jitter_Mode',
    'SUNSHINE_HOST_WARP_SUPPORTED', 'SUNSHINE_CAMERA_INCLUDED', 'SUNSHINE_HOST_WARP_INCLUDED',
    'SUNSHINE_DEPTH_PROBE', 'HOST_SBS_TEST_FIXED_ELEVEN_REFERENCE',
    'RESHADE_DEPTH_INPUT_IS_REVERSED', 'RESHADE_DEPTH_INPUT_IS_LOGARITHMIC',
    'RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN', 'RESHADE_DEPTH_LINEARIZATION_FAR_PLANE')
foreach ($backend in @('0xb000', '0xc000')) {
    foreach ($color in @(1, 2, 3)) {
        foreach ($value in @(0, 1)) {
            $name = "$backend-color$color-obsolete$value"
            $bits = if ($color -eq 2) { 16 } elseif ($color -eq 3) { 10 } else { 8 }
            $definitions = @("__RENDERER__=$backend", "BUFFER_COLOR_SPACE=$color", "BUFFER_COLOR_BIT_DEPTH=$bits",
                'BUFFER_WIDTH=1920', 'BUFFER_HEIGHT=1080', "SUNSHINE_SBS_EXPORT=$($value * 2)") +
                @($retiredDefinitions | ForEach-Object { "$_=$value" })
            $null = Compile $effect $name $definitions
            Require-NoEffectDefinitions $name
            Require-SameBytecode "$backend-color$color-treatment" $name
        }
    }
}
Write-Output 'PASS obsolete mode/export/debug/global-depth definitions produce identical bytecode across D3D11/12 and SDR/scRGB/PQ, and expose no effect settings.'

$rejections = @(
    @{Name='old-runtime'; Definitions=@('__RESHADE__=60703'); Diagnostic='requires ReShade'},
    @{Name='dx10'; Definitions=@('__RENDERER__=0xa000'); Diagnostic='requires ReShade'},
    @{Name='unknown-color'; Definitions=@('BUFFER_COLOR_SPACE=0'); Diagnostic='requires a known'},
    @{Name='odd-width'; Definitions=@('BUFFER_WIDTH=1921'); Diagnostic='requires even'},
    @{Name='too-wide'; Definitions=@('BUFFER_WIDTH=8194'); Diagnostic='requires even'}
)
foreach ($case in $rejections) {
    $directory = Join-Path $outputRoot ('rejected-' + $case.Name)
    # Windows PowerShell 5 wraps redirected native stderr in error records.
    # Expected compiler diagnostics must not bypass the rejection assertions.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $diagnosticLines = @(& $validator $effect $directory __RESHADE__=60800 @($case.Definitions) 2>&1)
        $rejectionExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    $diagnostics = ($diagnosticLines | ForEach-Object { $_.ToString() }) -join "`n"
    [IO.File]::WriteAllText($directory + '.txt', $diagnostics)
    Require ($rejectionExit -ne 0 -and $diagnostics.Contains($case.Diagnostic)) "Missing explicit rejection: $($case.Name)"
}
Write-Output "PASS current Game3D: $($variants.Count) paired compile/reflection configurations, 12 obsolete-definition bytecode comparisons and $($rejections.Count) invalid-runtime/input rejections. No GPU execution."
