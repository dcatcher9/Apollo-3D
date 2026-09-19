# SPDX-License-Identifier: GPL-3.0-only
# Isolated filesystem checks: never starts a game or loads ReShade/the add-on.
param(
    [Parameter(Mandatory)][string] $PackageDirectory,
    [string] $ShaderDirectory,
    [string] $BaselineShaderDirectory,
    [Parameter(Mandatory)][string] $OutputDirectory
)
$ErrorActionPreference = 'Stop'
$installer = Join-Path $PSScriptRoot 'install.ps1'
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Use a fresh isolated output directory.' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null

function Require([bool] $Condition, [string] $Message) { if (-not $Condition) { throw $Message } }
function File-Hashes([string] $Directory) {
    $hashes = @{}
    foreach ($file in Get-ChildItem -LiteralPath $Directory -File -Recurse) {
        $hashes[$file.FullName] = (Get-FileHash -LiteralPath $file.FullName).Hash
    }
    return $hashes
}
function Require-Unchanged([hashtable] $Before) {
    foreach ($path in $Before.Keys) {
        Require ((Test-Path -LiteralPath $path -PathType Leaf) -and
            (Get-FileHash -LiteralPath $path).Hash -eq $Before[$path]) ('File changed: ' + $path)
    }
}
function New-Fixture([string] $Name, [string] $Configuration, [string] $Preset, [bool] $Legacy = $true) {
    $directory = Join-Path $OutputDirectory $Name
    New-Item -ItemType Directory -Path (Join-Path $directory 'custom addons') -Force | Out-Null
    [IO.File]::WriteAllText((Join-Path $directory 'Test Game.exe'),'UNEXECUTED GAME FIXTURE')
    [IO.File]::WriteAllText((Join-Path $directory 'ReShade.ini'),$Configuration)
    [IO.File]::WriteAllText((Join-Path $directory 'custom preset.ini'),$Preset)
    if ($Legacy) {
        foreach ($name in @('SunshineDepth.addon64','SunshineDepthProbe.addon64','Other.addon64')) {
            [IO.File]::WriteAllText((Join-Path $directory "custom addons\$name"),"OLD $name")
        }
        $oldShaderDirectory = Join-Path $directory 'reshade-shaders\Shaders\Depth3D'
        New-Item -ItemType Directory -Path (Join-Path $oldShaderDirectory 'HostSBS') -Force | Out-Null
        foreach ($name in @('SuperDepth3D.fx','Overwatch.fxh','AXAA.fxh','SunshineIntegration.fxh','SunshinePost.fxh',
            'HostSBS.fxh','HostSBS\Warp.fxh','HostSBS\UserNotes.fxh')) {
            [IO.File]::WriteAllText((Join-Path $oldShaderDirectory $name),"EXISTING USER FILE $name")
        }
    }
    return $directory
}

# A self-contained package fixture exercises installation without a Depth3D
# checkout. An optional source directory lets the same suite validate the real
# independently authored effect's installer interface.
$fixturePackage = Join-Path $OutputDirectory 'package fixture'
$fixtureShaders = Join-Path $fixturePackage 'Shaders'
New-Item -ItemType Directory -Path $fixtureShaders -Force | Out-Null
foreach ($name in @('SunshineSBS.addon64','SunshineSBS-LICENSES.txt')) {
    Copy-Item -LiteralPath (Join-Path $PackageDirectory $name) -Destination (Join-Path $fixturePackage $name)
}
$minimalShader = @'
// Filesystem fixture; not installed into a real game or compiled by this test.
uniform bool Sunshine_DepthReady < source = "bufready_depth"; >;
uniform float Sunshine_RawAnchor;
uniform float Sunshine_RawGain;
uniform bool Sunshine_Calibrated;
texture2D StereoOutput <
    sunshine_sbs_export = 1;
    sunshine_sbs_layout = "sbs_lr";
    sunshine_sbs_color_space = "srgb";
    sunshine_sbs_source_color_space = 1;
    sunshine_sbs_source_width = BUFFER_WIDTH;
    sunshine_sbs_source_height = BUFFER_HEIGHT;
> { Width = BUFFER_WIDTH * 2; Height = BUFFER_HEIGHT; Format = RGBA16F; };
void VS(uint id : SV_VertexID, out float4 position : SV_Position) {
    float2 uv = float2((id << 1) & 2, id & 2);
    position = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float4 PS(float4 position : SV_Position) : SV_Target { return float4(0, 0, 0, 1); }
technique SunshineDepth3D {
    pass { VertexShader = VS; PixelShader = PS; RenderTarget = StereoOutput; }
}
'@
if ($ShaderDirectory) {
    Copy-Item -LiteralPath (Join-Path $ShaderDirectory 'SunshineDepth3D.fx') -Destination (Join-Path $fixtureShaders 'SunshineDepth3D.fx')
} else { [IO.File]::WriteAllText((Join-Path $fixtureShaders 'SunshineDepth3D.fx'), $minimalShader) }

$preset = "Techniques=Curves@Curves.fx,SuperDepth3D@SuperDepth3D.fx,SunshineDepth3D@SunshineDepth3D.fx`nPreprocessorDefinitions=USER_MACRO=7`n`n[SuperDepth3D.fx]`nDepth_Adjustment=37.250000`nDepth_Map=1`nUSE_AA=1`nHostSBS_Algorithm=1`nHostSBS_Baseline=0.064`nStereo_Warp_Algorithm=1`nPreprocessorDefinitions=SUNSHINE_SBS_EXPORT=0,SUNSHINE_DEPTH_PROBE=1,HOST_SBS_TEST_FIXED_ELEVEN_REFERENCE=1,KEEP_ME=5`n[SunshineDepth3D.fx]`nSunshine_Strength=0.375000`nSunshine_ScreenPlane=0.120000`n[Other.fx]`nHostSBS_Algorithm=99`n"
$config = "[ADDON]`nAddonPath=.\custom addons`nDisabledAddons=Other Addon,Sunshine SBS`n[GENERAL]`nPresetPath=.\custom preset.ini`nEffectSearchPaths=.\reshade-shaders\Shaders\**`nPreprocessorDefinitions=SUNSHINE_DEPTH_PROBE=1,KEEP_GLOBAL=9`n[SUNSHINE_DEPTH]`nAutoSelectSceneDepth=0`n"
# The earlier independent renderer remains available only through an explicit
# source directory; these existing compatibility checks keep their full scope.
$game = New-Fixture 'migration with spaces' $config $preset
$oldShaders = File-Hashes (Join-Path $game 'reshade-shaders\Shaders\Depth3D')
$otherBefore = (Get-FileHash -LiteralPath (Join-Path $game 'custom addons\Other.addon64')).Hash
$exeBefore = (Get-FileHash -LiteralPath (Join-Path $game 'Test Game.exe')).Hash
& $installer -GameExecutable (Join-Path $game 'Test Game.exe') -AddonDirectory $fixturePackage -ShaderDirectory $fixtureShaders
$ini = [IO.File]::ReadAllText((Join-Path $game 'ReShade.ini'))
$afterPreset = [IO.File]::ReadAllText((Join-Path $game 'custom preset.ini'))
Require ($ini -match 'AutoSelectSceneDepth=0') 'Installer changed an explicit automatic-selection preference.'
Require ($ini -match 'DisabledAddons=Other Addon,Generic Depth,Sunshine Depth') 'Other disabled add-ons were lost or legacy depth was not disabled by name.'
Require ($afterPreset -match '(?m)^Techniques=Curves@Curves.fx,SunshineDepth3D@SunshineDepth3D.fx$') 'The new technique was not uniquely enabled or the old warp remains active.'
Require ($afterPreset.Replace('Techniques=Curves@Curves.fx,SunshineDepth3D@SunshineDepth3D.fx',
    'Techniques=Curves@Curves.fx,SuperDepth3D@SuperDepth3D.fx,SunshineDepth3D@SunshineDepth3D.fx') -ceq $preset) 'Migration changed shader profiles or unrelated preset entries.'
Require ($ini -match 'PreprocessorDefinitions=SUNSHINE_DEPTH_PROBE=1,KEEP_GLOBAL=9') 'Unrelated global preprocessor settings changed.'
Require-Unchanged $oldShaders
$installedShader = Join-Path $game 'reshade-shaders\Shaders\Sunshine3D\SunshineDepth3D.fx'
Require ((Get-FileHash -LiteralPath $installedShader).Hash -eq (Get-FileHash -LiteralPath (Join-Path $fixtureShaders 'SunshineDepth3D.fx')).Hash) 'The independent shader was not copied intact.'
Require (-not (Test-Path -LiteralPath (Join-Path $game 'custom addons\SunshineDepth.addon64'))) 'Legacy selector still loads.'
Require (-not (Test-Path -LiteralPath (Join-Path $game 'custom addons\SunshineDepthProbe.addon64'))) 'Legacy probe still loads.'
Require ((Get-FileHash -LiteralPath (Join-Path $game 'custom addons\Other.addon64')).Hash -eq $otherBefore) 'Unrelated add-on changed.'
Require ((Get-FileHash -LiteralPath (Join-Path $game 'Test Game.exe')).Hash -eq $exeBefore) 'Game executable changed.'
Require ((Get-FileHash -LiteralPath (Join-Path $game 'custom addons\SunshineSBS.addon64')).Hash -eq (Get-FileHash -LiteralPath (Join-Path $fixturePackage 'SunshineSBS.addon64')).Hash) 'Unified DLL did not reach the configured add-on directory.'
$manifests = @(Get-ChildItem -LiteralPath (Join-Path $game 'Sunshine3D-backups') -Filter manifest.json -Recurse)
Require ($manifests.Count -eq 1) 'Migration backup manifest missing.'
$manifest = Get-Content -LiteralPath $manifests[0].FullName -Raw | ConvertFrom-Json
$oldSelector = $manifest | Where-Object { $_.path -like '*SunshineDepth.addon64' }
Require ([IO.File]::ReadAllText($oldSelector.backup) -eq 'OLD SunshineDepth.addon64') 'Original selector was not backed up.'
Write-Output 'PASS migration enables one independent technique, preserves all old shader files/profiles, and backs up legacy add-ons.'

& $installer -GameExecutable (Join-Path $game 'Test Game.exe') -AddonDirectory $fixturePackage -ShaderDirectory $fixtureShaders
Require ([IO.File]::ReadAllText((Join-Path $game 'ReShade.ini')) -ceq $ini) 'Second installation changed the configured INI.'
Require ([IO.File]::ReadAllText((Join-Path $game 'custom preset.ini')) -ceq $afterPreset) 'Second installation changed the configured preset.'
Write-Output 'PASS repeated installation is configuration-idempotent.'

$fresh = New-Fixture 'new game without Depth3D' "[GENERAL]`n" '' $false
& $installer -GameExecutable (Join-Path $fresh 'Test Game.exe') -AddonDirectory $fixturePackage -ShaderDirectory $fixtureShaders
Require (-not (Test-Path -LiteralPath (Join-Path $fresh 'reshade-shaders\Shaders\Depth3D'))) 'A new installation unexpectedly requires or creates upstream Depth3D.'
$freshIni = [IO.File]::ReadAllText((Join-Path $fresh 'ReShade.ini'))
Require ($freshIni -match '(?m)^EffectSearchPaths=\.\\reshade-shaders\\Shaders\\Sunshine3D$') 'The independent effect search path is missing.'
Require ($freshIni -match 'AutoSelectSceneDepth=1') 'New installations did not enable automatic selection.'
Require ([IO.File]::ReadAllText((Join-Path $fresh 'ReShadePreset.ini')) -match '(?m)^Techniques=SunshineDepth3D@SunshineDepth3D.fx$') 'Fresh installation did not enable its effect.'
Require (Test-Path -LiteralPath (Join-Path $fresh 'SunshineSBS.addon64')) 'Default add-on path was not used.'
Write-Output 'PASS an explicitly selected independent reference installs without a Depth3D checkout or existing shader.'

$override = Join-Path $OutputDirectory 'independent developer shaders'
New-Item -ItemType Directory -Path $override | Out-Null
[IO.File]::WriteAllText((Join-Path $override 'SunshineDepth3D.fx'), $minimalShader + "`n// DEVELOPER OVERRIDE`n")
$spaced = "  [ADDON]  `n  AddonPath = .\custom addons  `n  DisabledAddons = Other Addon  `n  [GENERAL] `n PresetPath = .\custom preset.ini `n [SUNSHINE_DEPTH]`n AutoSelectSceneDepth = 0 `n"
$spacedGame = New-Fixture 'whitespace and shader override' $spaced $preset
& $installer -GameExecutable (Join-Path $spacedGame 'Test Game.exe') -AddonDirectory $fixturePackage -ShaderDirectory $override
$spacedAfter = [IO.File]::ReadAllText((Join-Path $spacedGame 'ReShade.ini'))
Require ($spacedAfter -match 'AutoSelectSceneDepth\s*=\s*0') 'Whitespace caused an explicit preference to be overwritten.'
Require (Test-Path -LiteralPath (Join-Path $spacedGame 'custom addons\SunshineSBS.addon64')) 'Whitespace caused installation in the wrong add-on path.'
Require ((Get-FileHash -LiteralPath (Join-Path $spacedGame 'reshade-shaders\Shaders\Sunshine3D\SunshineDepth3D.fx')).Hash -eq
    (Get-FileHash -LiteralPath (Join-Path $override 'SunshineDepth3D.fx')).Hash) 'The explicit shader override was not used.'
Write-Output 'PASS whitespace, a custom add-on path, and an independent shader source override are respected.'

$commaConfig = "[ADDON]`nAddonPath=.\custom,, addons`nDisabledAddons=Other,, Addon,Sunshine 3D`n[GENERAL]`nPresetPath=.\custom,, preset.ini`nEffectSearchPaths=.\custom,, shaders`n"
$commaPreset = "Techniques=Custom,,Effect@Custom.fx,SuperDepth3D,SunshineDepth3D,SunshineDepth3D@SunshineDepth3D.fx`n[Other.fx]`nValue=3`n"
$commaGame = New-Fixture 'commas' $commaConfig '' $false
[IO.File]::WriteAllText((Join-Path $commaGame 'custom, preset.ini'), $commaPreset)
& $installer -GameExecutable (Join-Path $commaGame 'Test Game.exe') -AddonDirectory $fixturePackage -ShaderDirectory $fixtureShaders
$commaIni = [IO.File]::ReadAllText((Join-Path $commaGame 'ReShade.ini'))
$commaAfter = [IO.File]::ReadAllText((Join-Path $commaGame 'custom, preset.ini'))
Require (Test-Path -LiteralPath (Join-Path $commaGame 'custom, addons\SunshineSBS.addon64')) 'Escaped comma in AddonPath was not decoded.'
Require ($commaIni -match 'DisabledAddons=Other,, Addon,Generic Depth,Sunshine Depth') 'An escaped comma in a disabled add-on name changed.'
Require ($commaIni -match 'EffectSearchPaths=\.\\custom,, shaders,\.\\reshade-shaders\\Shaders\\Sunshine3D') 'Existing escaped effect paths were not preserved.'
Require ($commaAfter -match '(?m)^Techniques=Custom,,Effect@Custom.fx,SunshineDepth3D@SunshineDepth3D.fx$') 'Technique migration mishandled comma escapes or unqualified legacy techniques.'
Write-Output 'PASS ReShade comma escapes and legacy unqualified technique names are migrated correctly.'

foreach ($badCase in @(
    @{Name='duplicate key'; Config=($config.Replace('AddonPath=.\custom addons',"AddonPath=.\custom addons`nAddonPath=other"))},
    @{Name='duplicate section'; Config=($config + "`n[ADDON]`nAddonPath=other`n")},
    @{Name='preset aliases configuration'; Config=($config.Replace('PresetPath=.\custom preset.ini','PresetPath=.\ReShade.ini'))},
    @{Name='preset aliases executable'; Config=($config.Replace('PresetPath=.\custom preset.ini','PresetPath=.\Test Game.exe'))},
    @{Name='shader file is directory'; Config=$config; Directory='reshade-shaders\Shaders\Sunshine3D\SunshineDepth3D.fx'}
)) {
    $badGame = New-Fixture $badCase.Name $badCase.Config $preset
    if ($badCase.Directory) { New-Item -ItemType Directory -Path (Join-Path $badGame $badCase.Directory) -Force | Out-Null }
    $before = File-Hashes $badGame
    $rejected = $false
    try { & $installer -GameExecutable (Join-Path $badGame 'Test Game.exe') -AddonDirectory $fixturePackage -ShaderDirectory $fixtureShaders }
    catch { $rejected = $true }
    Require $rejected ('Unsafe configuration was accepted: ' + $badCase.Name)
    Require-Unchanged $before
    Require (@(Get-ChildItem -LiteralPath $badGame -File -Recurse).Count -eq $before.Count) 'Rejected preflight created files.'
    Write-Output ('PASS preflight rejects ' + $badCase.Name + ' before modifying files.')
}

foreach ($badEffect in @(
    @{Name='missing packaged shader'; Text=$null},
    @{Name='upstream shader named as replacement'; Text='#include "SuperDepth3D.fx"'},
    @{Name='missing calibration input'; Text=$minimalShader.Replace('uniform float Sunshine_RawGain;', '')},
    @{Name='missing export annotation'; Text=$minimalShader.Replace('sunshine_sbs_export = 1;', '')}
)) {
    $source = Join-Path $OutputDirectory ($badEffect.Name + ' source')
    New-Item -ItemType Directory -Path $source | Out-Null
    if ($null -ne $badEffect.Text) { [IO.File]::WriteAllText((Join-Path $source 'SunshineDepth3D.fx'), $badEffect.Text) }
    $badGame = New-Fixture $badEffect.Name $config $preset
    $before = File-Hashes $badGame
    $rejected = $false
    try { & $installer -GameExecutable (Join-Path $badGame 'Test Game.exe') -AddonDirectory $fixturePackage -ShaderDirectory $source }
    catch { $rejected = $true }
    Require $rejected ('Invalid source was accepted: ' + $badEffect.Name)
    Require-Unchanged $before
    Require (@(Get-ChildItem -LiteralPath $badGame -File -Recurse).Count -eq $before.Count) 'Invalid-source preflight created files.'
    Write-Output ('PASS ' + $badEffect.Name + ' is rejected before changing the game installation.')
}

# A sharing violation at preset write occurs after copies and the INI write.
# All prior files must survive byte-for-byte and newly installed files disappear.
$rollbackGame = New-Fixture 'rollback after partial copy' $config $preset
$before = File-Hashes $rollbackGame
$presetLock = [IO.File]::Open((Join-Path $rollbackGame 'custom preset.ini'), [IO.FileMode]::Open,
    [IO.FileAccess]::Read, [IO.FileShare]::Read)
$rejected = $false
try { & $installer -GameExecutable (Join-Path $rollbackGame 'Test Game.exe') -AddonDirectory $fixturePackage -ShaderDirectory $fixtureShaders }
catch { $rejected = $true }
finally { $presetLock.Dispose() }
Require $rejected 'The partial-write failure fixture did not fail.'
Require-Unchanged $before
Require (-not (Test-Path -LiteralPath (Join-Path $rollbackGame 'custom addons\SunshineSBS.addon64'))) 'Rollback left a new add-on installed.'
Require (-not (Test-Path -LiteralPath (Join-Path $rollbackGame 'reshade-shaders\Shaders\Sunshine3D\SunshineDepth3D.fx'))) 'Rollback left the new effect installed.'
Require (@(Get-ChildItem -LiteralPath (Join-Path $rollbackGame 'Sunshine3D-backups') -Filter manifest.json -Recurse).Count -eq 1) 'Rollback evidence is missing.'
Write-Output 'PASS a failure after partial installation restores prior files and removes new files.'

# Native Game 3D requires only the DLL. Migration preserves user settings and
# removes old FX only from the directory owned by previous Game 3D installs.
$nativePreset = $preset + "`n[SunshineGame3D.fx]`nDepth_Adjustment=0.000000`nDepth_Map_View=2`nPreprocessorDefinitions=SUNSHINE_GAME3D_AUTOMATIC=0,USER_GAME3D=3`n"
$nativeGame = New-Fixture 'native migration without shaders' $config $nativePreset
$nativeReference = File-Hashes (Join-Path $nativeGame 'reshade-shaders\Shaders\Depth3D')
$ownedFolder = Join-Path $nativeGame 'reshade-shaders\Shaders\SunshineGame3D'
New-Item -ItemType Directory -Path $ownedFolder -Force | Out-Null
$ownedNames = @('SunshineGame3D.fx','SunshineHostWarp.fxh','SunshineCamera.fxh','SunshineGenericConfig.fxh',
    'SunshineWarpConfig.fxh','Overwatch.fxh','AXAA.fxh','SunshineIntegration.fxh','SunshinePost.fxh')
foreach ($name in ($ownedNames + @('UserNotes.fxh'))) { [IO.File]::WriteAllText((Join-Path $ownedFolder $name), 'OLD OWNED FILE ' + $name) }
$ownedBefore = File-Hashes $ownedFolder
& $installer -GameExecutable (Join-Path $nativeGame 'Test Game.exe') -AddonDirectory $fixturePackage
$nativeIni = [IO.File]::ReadAllText((Join-Path $nativeGame 'ReShade.ini'))
$nativeAfter = [IO.File]::ReadAllText((Join-Path $nativeGame 'custom preset.ini'))
Require ($nativeIni -match '(?m)^Strength=0\.000000\r?$') 'Native migration did not preserve explicit zero strength.'
Require ($nativeIni -match '(?m)^DepthView=2\r?$') 'Native migration did not preserve the selected depth view.'
Require ($nativeIni -match '(?m)^Enabled=1\r?$') 'Native Game 3D was not enabled.'
Require ($nativeIni -match '(?m)^EffectSearchPaths=\.\\reshade-shaders\\Shaders\\\*\*\r?$') 'Native setup changed existing effect search paths.'
Require ($nativeAfter -match '(?m)^Techniques=Curves@Curves.fx\r?$') 'Native migration left competing stereo effects active.'
Require ($nativeAfter -notmatch 'SUNSHINE_GAME3D_AUTOMATIC=') 'Owned retired macro survived.'
Require ($nativeAfter -match 'USER_GAME3D=3') 'Unrelated effect definition was lost.'
Require-Unchanged $nativeReference
Require ((Get-FileHash -LiteralPath (Join-Path $ownedFolder 'UserNotes.fxh')).Hash -eq $ownedBefore[(Join-Path $ownedFolder 'UserNotes.fxh')]) 'Unrelated file in Game3D folder changed.'
$manifestPath = @(Get-ChildItem -LiteralPath (Join-Path $nativeGame 'Sunshine3D-backups') -Filter manifest.json -Recurse)[0].FullName
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($name in $ownedNames) {
    $path = Join-Path $ownedFolder $name
    Require (-not (Test-Path -LiteralPath $path)) ('Retired Game 3D file remains: ' + $name)
    $entry = @($manifest | Where-Object { $_.path -ieq $path })
    Require ($entry.Count -eq 1 -and $entry[0].existed -and (Get-FileHash -LiteralPath $entry[0].backup).Hash -eq $ownedBefore[$path]) ('Removed file lacks a restorable backup: ' + $name)
}
& $installer -GameExecutable (Join-Path $nativeGame 'Test Game.exe') -AddonDirectory $fixturePackage
Require ([IO.File]::ReadAllText((Join-Path $nativeGame 'ReShade.ini')) -ceq $nativeIni) 'Reinstallation changed native preferences.'
Require ([IO.File]::ReadAllText((Join-Path $nativeGame 'custom preset.ini')) -ceq $nativeAfter) 'Reinstallation changed the preset.'
Write-Output 'PASS addon-only migration preserves zero strength/view, removes and backs up owned FX, retains reference/user files, and is idempotent.'

$existingNativeConfig = $config + "`n[SUNSHINE_GAME3D]`nStrength=71.500000`nDepthView=0`nEnabled=0`n"
$existingNative = New-Fixture 'existing native preferences' $existingNativeConfig $nativePreset
& $installer -GameExecutable (Join-Path $existingNative 'Test Game.exe') -AddonDirectory $fixturePackage
$existingAfter = [IO.File]::ReadAllText((Join-Path $existingNative 'ReShade.ini'))
foreach ($line in @('Strength=71.500000','DepthView=0','Enabled=0')) {
    Require ($existingAfter -match ('(?m)^' + [regex]::Escape($line) + '\r?$')) ('Reinstall overwrote saved native preference: ' + $line)
}
Write-Output 'PASS saved native settings take precedence, including explicit disabled and game view.'

foreach ($case in @(
    @{Name='fresh native'; Config="[GENERAL]`n"; Preset=''; Strength=$null; View=$null; File='ReShadePreset.ini'},
    @{Name='missing current strength'; Config=$config; Preset=$preset+"`n[SunshineGame3D.fx]`nDepth_Map_View=1`n"; Strength=$null; View='1'; File='custom preset.ini'},
    @{Name='original settings migrate'; Config=$config; Preset=$preset.Replace('Depth_Adjustment=37.250000','Depth_Adjustment=350.000000').Replace('Depth_Map=1','Depth_Map_View=2'); Strength='350.000000'; View='2'; File='custom preset.ini'},
    @{Name='invalid settings ignored'; Config=$config; Preset=$preset+"`n[SunshineGame3D.fx]`nDepth_Adjustment=NaN`nDepth_Map_View=99`n"; Strength=$null; View=$null; File='custom preset.ini'}
)) {
    $game = New-Fixture $case.Name $case.Config $case.Preset $false
    & $installer -GameExecutable (Join-Path $game 'Test Game.exe') -AddonDirectory $fixturePackage
    $ini = [IO.File]::ReadAllText((Join-Path $game 'ReShade.ini'))
    foreach ($setting in @(@{Key='Strength'; Value=$case.Strength},@{Key='DepthView'; Value=$case.View})) {
        if ($null -eq $setting.Value) {
            Require ($ini -notmatch ('(?m)^' + $setting.Key + '=')) ('Missing/invalid setting should retain native default: ' + $case.Name + '/' + $setting.Key)
        } else {
            Require ($ini -match ('(?m)^' + $setting.Key + '=' + [regex]::Escape($setting.Value) + '\r?$')) ('Migration changed a valid setting: ' + $case.Name + '/' + $setting.Key)
        }
    }
    Require (-not (Test-Path -LiteralPath (Join-Path $game 'reshade-shaders'))) 'Addon-only install created a shader directory.'
    $after = [IO.File]::ReadAllText((Join-Path $game $case.File))
    Require ($after -notmatch '(?m)^Techniques=.*(?:Sunshine|SuperDepth)') 'Native setup enabled an external stereo technique.'
}
Write-Output 'PASS fresh addon-only setup, omitted current defaults, prior reference migration, and invalid setting handling.'

foreach ($badCase in @(
    @{Name='duplicate native key'; Config=$existingNativeConfig.Replace('Strength=71.500000',"Strength=71.500000`nStrength=2")},
    @{Name='duplicate native section'; Config=$existingNativeConfig+"`n[SUNSHINE_GAME3D]`nEnabled=1`n"},
    @{Name='owned shader path directory'; Config=$config; Directory='reshade-shaders\Shaders\SunshineGame3D\SunshineGame3D.fx'}
)) {
    $game = New-Fixture $badCase.Name $badCase.Config $nativePreset
    if ($badCase.Directory) { New-Item -ItemType Directory -Path (Join-Path $game $badCase.Directory) -Force | Out-Null }
    $before = File-Hashes $game
    $rejected = $false
    try { & $installer -GameExecutable (Join-Path $game 'Test Game.exe') -AddonDirectory $fixturePackage } catch { $rejected = $true }
    Require $rejected ('Native preflight accepted ambiguity: ' + $badCase.Name)
    Require-Unchanged $before
    Require (@(Get-ChildItem -LiteralPath $game -File -Recurse).Count -eq $before.Count) 'Native preflight failure wrote files.'
}
Write-Output 'PASS native migration rejects conflicting settings and non-file owned paths before mutation.'

$nativeRollback = New-Fixture 'native rollback after partial write' $config $nativePreset
$rollbackOwned = Join-Path $nativeRollback 'reshade-shaders\Shaders\SunshineGame3D'
New-Item -ItemType Directory -Path $rollbackOwned -Force | Out-Null
foreach ($name in $ownedNames) { [IO.File]::WriteAllText((Join-Path $rollbackOwned $name), 'OLD ' + $name) }
$before = File-Hashes $nativeRollback
$presetLock = [IO.File]::Open((Join-Path $nativeRollback 'custom preset.ini'), [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
$rejected = $false
try { & $installer -GameExecutable (Join-Path $nativeRollback 'Test Game.exe') -AddonDirectory $fixturePackage }
catch { $rejected = $true }
finally { $presetLock.Dispose() }
Require $rejected 'Native partial-write fixture did not fail.'
Require-Unchanged $before
Require (-not (Test-Path -LiteralPath (Join-Path $nativeRollback 'custom addons\SunshineSBS.addon64'))) 'Rollback left a new add-on installed.'
Write-Output 'PASS native partial-write rollback preserves old FX and restores INI/preset/addon files.'

# Force failure after some owned files have already been removed. The existing
# locked final file stays intact; earlier removals must be restored from backup.
$removeRollback = New-Fixture 'native rollback during removal' $config $nativePreset
$removeOwned = Join-Path $removeRollback 'reshade-shaders\Shaders\SunshineGame3D'
New-Item -ItemType Directory -Path $removeOwned -Force | Out-Null
foreach ($name in $ownedNames) { [IO.File]::WriteAllText((Join-Path $removeOwned $name), 'RESTORE ' + $name) }
$before = File-Hashes $removeRollback
$removeLock = [IO.File]::Open((Join-Path $removeOwned 'SunshinePost.fxh'), [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
$rejected = $false
try { & $installer -GameExecutable (Join-Path $removeRollback 'Test Game.exe') -AddonDirectory $fixturePackage }
catch { $rejected = $true }
finally { $removeLock.Dispose() }
Require $rejected 'Locked owned-file removal did not fail.'
Require-Unchanged $before
Require (-not (Test-Path -LiteralPath (Join-Path $removeRollback 'custom addons\SunshineSBS.addon64'))) 'Removal rollback left a new add-on installed.'
Write-Output 'PASS failure midway through owned-file removal restores the old files and configuration.'

$minimalBaseline = @'
#ifndef SUNSHINE_SBS_EXPORT
#define SUNSHINE_SBS_EXPORT 1
#endif
#include "SunshineIntegration.fxh"
#include "SunshinePost.fxh"
texture DoubleTex < sunshine_sbs_export = 1; sunshine_sbs_layout = "sbs_lr"; > {
    Width = BUFFER_WIDTH * 2; Height = BUFFER_HEIGHT; Format = RGBA16F;
};
technique SuperDepth3D {
    pass Export { VertexShader = VS; PixelShader = PS; RenderTarget0 = DoubleTex; }
}
'@
$baselineFiles = @('Overwatch.fxh','AXAA.fxh','SunshineIntegration.fxh','SunshinePost.fxh')
$originalSource = Join-Path $OutputDirectory 'external original shader'
New-Item -ItemType Directory -Path $originalSource | Out-Null
foreach ($name in $baselineFiles) {
    if ($BaselineShaderDirectory) { Copy-Item -LiteralPath (Join-Path $BaselineShaderDirectory $name) -Destination (Join-Path $originalSource $name) }
    else { [IO.File]::WriteAllText((Join-Path $originalSource $name), '// matching external fixture') }
}
if ($BaselineShaderDirectory) { Copy-Item -LiteralPath (Join-Path $BaselineShaderDirectory 'SuperDepth3D.fx') -Destination (Join-Path $originalSource 'SuperDepth3D.fx') }
else { [IO.File]::WriteAllText((Join-Path $originalSource 'SuperDepth3D.fx'), $minimalBaseline) }
$baselinePreset = $preset.Replace('Depth_Adjustment=37.250000','Depth_Adjustment=350.000000')
$originalGame = New-Fixture 'explicit original reference' $existingNativeConfig $baselinePreset
& $installer -GameExecutable (Join-Path $originalGame 'Test Game.exe') -AddonDirectory $fixturePackage -ShaderDirectory $originalSource
$originalAfter = [IO.File]::ReadAllText((Join-Path $originalGame 'custom preset.ini'))
$originalIni = [IO.File]::ReadAllText((Join-Path $originalGame 'ReShade.ini'))
Require ($originalAfter -match '(?m)^Techniques=Curves@Curves.fx,SuperDepth3D@SuperDepth3D.fx\r?$') 'Explicit reference did not uniquely enable SuperDepth3D.'
Require ($originalAfter -match 'Depth_Adjustment=350.000000') 'Reference control was clamped/reset.'
Require ($originalAfter -match 'SUNSHINE_SBS_EXPORT=1,SUNSHINE_DEPTH_PROBE=1,HOST_SBS_TEST_FIXED_ELEVEN_REFERENCE=1,KEEP_ME=5') 'Reference export update changed unrelated macros.'
Require ($originalIni -match '(?m)^Enabled=0\r?$' -and $originalIni -match '(?m)^Strength=71\.500000\r?$') 'Reference setup did not disable native rendering while retaining its strength.'
foreach ($name in ($baselineFiles + @('SuperDepth3D.fx'))) {
    Require ((Get-FileHash -LiteralPath (Join-Path $originalGame ('reshade-shaders\Shaders\Depth3D\'+$name))).Hash -eq
        (Get-FileHash -LiteralPath (Join-Path $originalSource $name)).Hash) ('Reference file was not copied intact: ' + $name)
}
Write-Output 'PASS explicit original reference installs matching files, retains controls, and disables duplicate native rendering.'

$retiredSource = Join-Path $OutputDirectory 'retired Game3D source'
New-Item -ItemType Directory -Path $retiredSource | Out-Null
[IO.File]::WriteAllText((Join-Path $retiredSource 'SunshineGame3D.fx'), '// retired external implementation')
$retiredGame = New-Fixture 'retired external Game3D' $config $nativePreset
$before = File-Hashes $retiredGame
$diagnostic = ''
try { & $installer -GameExecutable (Join-Path $retiredGame 'Test Game.exe') -AddonDirectory $fixturePackage -ShaderDirectory $retiredSource }
catch { $diagnostic = $_.Exception.Message }
Require ($diagnostic.Contains('Omit -ShaderDirectory')) 'Retired external Game3D source lacks migration guidance.'
Require-Unchanged $before
Write-Output 'PASS retired external Game3D source directs users to addon-only setup before mutation.'

foreach ($invalid in @(
    @{Name='wrong owning technique'; Text=$minimalBaseline.Replace('technique SuperDepth3D', 'technique Other3D')},
    @{Name='missing baseline export'; Text=$minimalBaseline.Replace('sunshine_sbs_export = 1;', '')},
    @{Name='unowned export target'; Text=$minimalBaseline.Replace('RenderTarget0 = DoubleTex;', 'RenderTarget0 = OtherTex;')},
    @{Name='commented owning technique'; Text=$minimalBaseline.Replace('technique SuperDepth3D', '// technique SuperDepth3D')}
)) {
    $source = Join-Path $OutputDirectory $invalid.Name
    New-Item -ItemType Directory -Path $source | Out-Null
    foreach ($name in $baselineFiles) { Copy-Item -LiteralPath (Join-Path $originalSource $name) -Destination (Join-Path $source $name) }
    [IO.File]::WriteAllText((Join-Path $source 'SuperDepth3D.fx'), $invalid.Text)
    $badGame = New-Fixture ($invalid.Name + ' game') $config $baselinePreset
    $before = File-Hashes $badGame
    $rejected = $false
    try { & $installer -GameExecutable (Join-Path $badGame 'Test Game.exe') -AddonDirectory $fixturePackage -ShaderDirectory $source }
    catch { $rejected = $true }
    Require $rejected ('Invalid export ownership accepted: ' + $invalid.Name)
    Require-Unchanged $before
}
Write-Output 'PASS external technique ownership and export markers reject incomplete or commented declarations.'
