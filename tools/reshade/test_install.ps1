# SPDX-License-Identifier: GPL-3.0-only
# Isolated filesystem checks: never starts a game or loads ReShade/the add-on.
param(
    [Parameter(Mandatory)][string] $PackageDirectory,
    [Parameter(Mandatory)][string] $OutputDirectory
)
$ErrorActionPreference = 'Stop'
# The checks hash files themselves; as in install.ps1, PowerShell 7's inherited
# PSModulePath must not hide 5.1's Get-FileHash.
if ($PSVersionTable.PSEdition -ne 'Core') {
    Import-Module -Name (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1')
}
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
        [IO.File]::WriteAllText((Join-Path $directory 'custom addons\Other.addon64'),'OLD Other.addon64')
        $oldShaderDirectory = Join-Path $directory 'reshade-shaders\Shaders\Depth3D'
        New-Item -ItemType Directory -Path (Join-Path $oldShaderDirectory 'HostSBS') -Force | Out-Null
        foreach ($name in @('SuperDepth3D.fx','Overwatch.fxh','AXAA.fxh','SunshineIntegration.fxh','SunshinePost.fxh',
            'HostSBS.fxh','HostSBS\Warp.fxh','HostSBS\UserNotes.fxh')) {
            [IO.File]::WriteAllText((Join-Path $oldShaderDirectory $name),"EXISTING USER FILE $name")
        }
    }
    return $directory
}

# A self-contained package fixture exercises installation without a game.
$fixturePackage = Join-Path $OutputDirectory 'package fixture'
New-Item -ItemType Directory -Path $fixturePackage -Force | Out-Null
foreach ($name in @('SunshineSBS.addon64','SunshineSBS-LICENSES.txt')) {
    Copy-Item -LiteralPath (Join-Path $PackageDirectory $name) -Destination (Join-Path $fixturePackage $name)
}

$preset = "Techniques=Curves@Curves.fx,SuperDepth3D@SuperDepth3D.fx,SunshineDepth3D@SunshineDepth3D.fx`nPreprocessorDefinitions=USER_MACRO=7`n`n[SuperDepth3D.fx]`nDepth_Adjustment=37.250000`nDepth_Map=1`nUSE_AA=1`nHostSBS_Algorithm=1`nHostSBS_Baseline=0.064`nStereo_Warp_Algorithm=1`nPreprocessorDefinitions=SUNSHINE_SBS_EXPORT=0,SUNSHINE_DEPTH_PROBE=1,HOST_SBS_TEST_FIXED_ELEVEN_REFERENCE=1,KEEP_ME=5`n[SunshineDepth3D.fx]`nSunshine_Strength=0.375000`nSunshine_ScreenPlane=0.120000`n[Other.fx]`nHostSBS_Algorithm=99`n"
$config = "[ADDON]`nAddonPath=.\custom addons`nDisabledAddons=Other Addon,Sunshine SBS`n[GENERAL]`nPresetPath=.\custom preset.ini`nEffectSearchPaths=.\reshade-shaders\Shaders\**`nPreprocessorDefinitions=SUNSHINE_DEPTH_PROBE=1,KEEP_GLOBAL=9`n[SUNSHINE_DEPTH]`nAutoSelectSceneDepth=0`n"
# A game with effect-based stereo: its stereo techniques leave the preset, the
# reference effect an earlier Sunshine installation copied is removed with a
# backup, and everything else stays as the user left it.
$game = New-Fixture 'migration with spaces' $config $preset
$oldShaders = File-Hashes (Join-Path $game 'reshade-shaders\Shaders\Depth3D')
$oldReference = Join-Path $game 'reshade-shaders\Shaders\Sunshine3D\SunshineDepth3D.fx'
New-Item -ItemType Directory -Path (Split-Path -Parent $oldReference) -Force | Out-Null
[IO.File]::WriteAllText($oldReference, 'OLD REFERENCE EFFECT')
$otherBefore = (Get-FileHash -LiteralPath (Join-Path $game 'custom addons\Other.addon64')).Hash
$exeBefore = (Get-FileHash -LiteralPath (Join-Path $game 'Test Game.exe')).Hash
& $installer -GameExecutable (Join-Path $game 'Test Game.exe') -AddonDirectory $fixturePackage
$ini = [IO.File]::ReadAllText((Join-Path $game 'ReShade.ini'))
$afterPreset = [IO.File]::ReadAllText((Join-Path $game 'custom preset.ini'))
Require ($ini.Contains("[SUNSHINE_DEPTH]`nAutoSelectSceneDepth=0`n")) 'The installer changed the stale [SUNSHINE_DEPTH] line; it must be left byte-for-byte untouched.'
Require ($ini -match '(?m)^DisabledAddons=Other Addon,Generic Depth\r?$') 'Other disabled add-ons were lost or Generic Depth was not disabled by name.'
Require ($afterPreset -match '(?m)^Techniques=Curves@Curves.fx$') 'An old stereo technique remains active.'
Require ($afterPreset.Replace('Techniques=Curves@Curves.fx',
    'Techniques=Curves@Curves.fx,SuperDepth3D@SuperDepth3D.fx,SunshineDepth3D@SunshineDepth3D.fx') -ceq $preset) 'Migration changed shader profiles or unrelated preset entries.'
Require ($ini -match '(?m)^PreprocessorDefinitions=SUNSHINE_DEPTH_PROBE=1,KEEP_GLOBAL=9$') 'Global preprocessor settings changed.'
Require-Unchanged $oldShaders
Require (-not (Test-Path -LiteralPath $oldReference)) 'The old reference effect still loads.'
Require ((Get-FileHash -LiteralPath (Join-Path $game 'custom addons\Other.addon64')).Hash -eq $otherBefore) 'Unrelated add-on changed.'
Require ((Get-FileHash -LiteralPath (Join-Path $game 'Test Game.exe')).Hash -eq $exeBefore) 'Game executable changed.'
Require ((Get-FileHash -LiteralPath (Join-Path $game 'custom addons\SunshineSBS.addon64')).Hash -eq (Get-FileHash -LiteralPath (Join-Path $fixturePackage 'SunshineSBS.addon64')).Hash) 'Unified DLL did not reach the configured add-on directory.'
$manifests = @(Get-ChildItem -LiteralPath (Join-Path $game 'Sunshine3D-backups') -Filter manifest.json -Recurse)
Require ($manifests.Count -eq 1) 'Migration backup manifest missing.'
$manifest = Get-Content -LiteralPath $manifests[0].FullName -Raw | ConvertFrom-Json
$oldEffect = $manifest | Where-Object { $_.path -like '*SunshineDepth3D.fx' }
Require ([IO.File]::ReadAllText($oldEffect.backup) -eq 'OLD REFERENCE EFFECT') 'The old reference effect was not backed up.'
Write-Output 'PASS migration disables the old stereo techniques, preserves all old shader files/profiles, and backs up the old reference effect.'

& $installer -GameExecutable (Join-Path $game 'Test Game.exe') -AddonDirectory $fixturePackage
Require ([IO.File]::ReadAllText((Join-Path $game 'ReShade.ini')) -ceq $ini) 'Second installation changed the configured INI.'
Require ([IO.File]::ReadAllText((Join-Path $game 'custom preset.ini')) -ceq $afterPreset) 'Second installation changed the configured preset.'
Write-Output 'PASS repeated installation is configuration-idempotent.'

$fresh = New-Fixture 'new game without Depth3D' "[GENERAL]`n" '' $false
& $installer -GameExecutable (Join-Path $fresh 'Test Game.exe') -AddonDirectory $fixturePackage
Require (-not (Test-Path -LiteralPath (Join-Path $fresh 'reshade-shaders'))) 'A new installation created a shader directory.'
$freshIni = [IO.File]::ReadAllText((Join-Path $fresh 'ReShade.ini'))
Require ($freshIni -notmatch 'EffectSearchPaths') 'A new installation added an effect search path.'
Require ($freshIni -notmatch 'AutoSelectSceneDepth') 'New installations wrote the removed selection key.'
Require (Test-Path -LiteralPath (Join-Path $fresh 'SunshineSBS.addon64')) 'Default add-on path was not used.'
Write-Output 'PASS a new game installs the add-on alone, at the default add-on path.'

$spaced = "  [ADDON]  `n  AddonPath = .\custom addons  `n  DisabledAddons = Other Addon  `n  [GENERAL] `n PresetPath = .\custom preset.ini `n [SUNSHINE_DEPTH]`n AutoSelectSceneDepth = 0 `n"
$spacedGame = New-Fixture 'whitespace' $spaced $preset
& $installer -GameExecutable (Join-Path $spacedGame 'Test Game.exe') -AddonDirectory $fixturePackage
$spacedAfter = [IO.File]::ReadAllText((Join-Path $spacedGame 'ReShade.ini'))
Require ($spacedAfter.Contains(" [SUNSHINE_DEPTH]`n AutoSelectSceneDepth = 0 `n")) 'The installer changed the stale whitespace-padded [SUNSHINE_DEPTH] line; it must be left byte-for-byte untouched.'
Require (Test-Path -LiteralPath (Join-Path $spacedGame 'custom addons\SunshineSBS.addon64')) 'Whitespace caused installation in the wrong add-on path.'
Write-Output 'PASS whitespace and a custom add-on path are respected.'

$commaConfig = "[ADDON]`nAddonPath=.\custom,, addons`nDisabledAddons=Other,, Addon,Sunshine 3D`n[GENERAL]`nPresetPath=.\custom,, preset.ini`nEffectSearchPaths=.\custom,, shaders`n"
$commaPreset = "Techniques=Custom,,Effect@Custom.fx,SuperDepth3D,SunshineDepth3D,SunshineDepth3D@SunshineDepth3D.fx`n[Other.fx]`nValue=3`n"
$commaGame = New-Fixture 'commas' $commaConfig '' $false
[IO.File]::WriteAllText((Join-Path $commaGame 'custom, preset.ini'), $commaPreset)
& $installer -GameExecutable (Join-Path $commaGame 'Test Game.exe') -AddonDirectory $fixturePackage
$commaIni = [IO.File]::ReadAllText((Join-Path $commaGame 'ReShade.ini'))
$commaAfter = [IO.File]::ReadAllText((Join-Path $commaGame 'custom, preset.ini'))
Require (Test-Path -LiteralPath (Join-Path $commaGame 'custom, addons\SunshineSBS.addon64')) 'Escaped comma in AddonPath was not decoded.'
Require ($commaIni -match '(?m)^DisabledAddons=Other,, Addon,Generic Depth\r?$') 'An escaped comma in a disabled add-on name changed.'
Require ($commaIni -match '(?m)^EffectSearchPaths=\.\\custom,, shaders$') 'Existing escaped effect paths were not preserved.'
Require ($commaAfter -match '(?m)^Techniques=Custom,,Effect@Custom.fx$') 'Technique migration mishandled comma escapes or unqualified legacy techniques.'
Write-Output 'PASS ReShade comma escapes and legacy unqualified technique names are migrated correctly.'

foreach ($badCase in @(
    @{Name='duplicate key'; Config=($config.Replace('AddonPath=.\custom addons',"AddonPath=.\custom addons`nAddonPath=other"))},
    @{Name='duplicate section'; Config=($config + "`n[ADDON]`nAddonPath=other`n")},
    @{Name='preset aliases configuration'; Config=($config.Replace('PresetPath=.\custom preset.ini','PresetPath=.\ReShade.ini'))},
    @{Name='preset aliases executable'; Config=($config.Replace('PresetPath=.\custom preset.ini','PresetPath=.\Test Game.exe'))},
    @{Name='old reference effect is directory'; Config=$config; Directory='reshade-shaders\Shaders\Sunshine3D\SunshineDepth3D.fx'}
)) {
    $badGame = New-Fixture $badCase.Name $badCase.Config $preset
    if ($badCase.Directory) { New-Item -ItemType Directory -Path (Join-Path $badGame $badCase.Directory) -Force | Out-Null }
    $before = File-Hashes $badGame
    $rejected = $false
    try { & $installer -GameExecutable (Join-Path $badGame 'Test Game.exe') -AddonDirectory $fixturePackage }
    catch { $rejected = $true }
    Require $rejected ('Unsafe configuration was accepted: ' + $badCase.Name)
    Require-Unchanged $before
    Require (@(Get-ChildItem -LiteralPath $badGame -File -Recurse).Count -eq $before.Count) 'Rejected preflight created files.'
    Write-Output ('PASS preflight rejects ' + $badCase.Name + ' before modifying files.')
}

# A sharing violation at preset write occurs after copies and the INI write.
# All prior files must survive byte-for-byte and newly installed files disappear.
$rollbackGame = New-Fixture 'rollback after partial copy' $config $preset
$before = File-Hashes $rollbackGame
$presetLock = [IO.File]::Open((Join-Path $rollbackGame 'custom preset.ini'), [IO.FileMode]::Open,
    [IO.FileAccess]::Read, [IO.FileShare]::Read)
$rejected = $false
try { & $installer -GameExecutable (Join-Path $rollbackGame 'Test Game.exe') -AddonDirectory $fixturePackage }
catch { $rejected = $true }
finally { $presetLock.Dispose() }
Require $rejected 'The partial-write failure fixture did not fail.'
Require-Unchanged $before
Require (-not (Test-Path -LiteralPath (Join-Path $rollbackGame 'custom addons\SunshineSBS.addon64'))) 'Rollback left a new add-on installed.'
Require (@(Get-ChildItem -LiteralPath (Join-Path $rollbackGame 'Sunshine3D-backups') -Filter manifest.json -Recurse).Count -eq 1) 'Rollback evidence is missing.'
Write-Output 'PASS a failure after partial installation restores prior files and removes new files.'

# Native Game 3D requires only the DLL. An effect section of an earlier
# installation is not migrated: the add-on's defaults apply, and saved native
# settings take precedence.
$nativePreset = $preset + "`n[SunshineGame3D.fx]`nDepth_Adjustment=0.000000`nDepth_Map_View=2`nPreprocessorDefinitions=SUNSHINE_GAME3D_AUTOMATIC=0,USER_GAME3D=3`n"
$existingNativeConfig = $config + "`n[SUNSHINE_GAME3D]`nStrength=71.500000`nDepthView=0`nEnabled=0`n"
$existingNative = New-Fixture 'existing native preferences' $existingNativeConfig $nativePreset
& $installer -GameExecutable (Join-Path $existingNative 'Test Game.exe') -AddonDirectory $fixturePackage
$existingAfter = [IO.File]::ReadAllText((Join-Path $existingNative 'ReShade.ini'))
foreach ($line in @('Strength=71.500000','DepthView=0','Enabled=0')) {
    Require ($existingAfter -match ('(?m)^' + [regex]::Escape($line) + '\r?$')) ('Reinstall overwrote saved native preference: ' + $line)
}
Write-Output 'PASS saved native settings take precedence, including explicit disabled and game view.'

foreach ($case in @(
    @{Name='fresh native'; Config="[GENERAL]`n"; Preset=''; File='ReShadePreset.ini'},
    @{Name='effect section'; Config=$config; Preset=$nativePreset; File='custom preset.ini'},
    @{Name='original settings'; Config=$config; Preset=$preset.Replace('Depth_Map=1','Depth_Map_View=2'); File='custom preset.ini'}
)) {
    $game = New-Fixture $case.Name $case.Config $case.Preset $false
    & $installer -GameExecutable (Join-Path $game 'Test Game.exe') -AddonDirectory $fixturePackage
    $ini = [IO.File]::ReadAllText((Join-Path $game 'ReShade.ini'))
    foreach ($key in @('Strength','DepthView')) {
        Require ($ini -notmatch ('(?m)^' + $key + '=')) ('An effect setting was migrated instead of the native default: ' + $case.Name + '/' + $key)
    }
    Require ($ini -match '(?m)^Enabled=1\r?$') ('Native Game 3D was not enabled: ' + $case.Name)
    Require (-not (Test-Path -LiteralPath (Join-Path $game 'reshade-shaders'))) 'Addon-only install created a shader directory.'
    $after = [IO.File]::ReadAllText((Join-Path $game $case.File))
    Require ($after -notmatch '(?m)^Techniques=.*(?:Sunshine|SuperDepth)') 'Native setup enabled an external stereo technique.'
    if ($case.Preset) {
        Require (($after -replace '(?m)^Techniques=[^\r\n]*', '') -ceq ($case.Preset -replace '(?m)^Techniques=[^\r\n]*', '')) ('Native setup changed effect settings: ' + $case.Name)
    }
}
Write-Output 'PASS fresh addon-only setup keeps the native defaults and leaves effect sections as they were.'

foreach ($badCase in @(
    @{Name='duplicate native key'; Config=$existingNativeConfig.Replace('Strength=71.500000',"Strength=71.500000`nStrength=2")},
    @{Name='duplicate native section'; Config=$existingNativeConfig+"`n[SUNSHINE_GAME3D]`nEnabled=1`n"}
)) {
    $game = New-Fixture $badCase.Name $badCase.Config $nativePreset
    $before = File-Hashes $game
    $rejected = $false
    try { & $installer -GameExecutable (Join-Path $game 'Test Game.exe') -AddonDirectory $fixturePackage } catch { $rejected = $true }
    Require $rejected ('Native preflight accepted ambiguity: ' + $badCase.Name)
    Require-Unchanged $before
    Require (@(Get-ChildItem -LiteralPath $game -File -Recurse).Count -eq $before.Count) 'Native preflight failure wrote files.'
}
Write-Output 'PASS native setup rejects conflicting settings before mutation.'

$nativeRollback = New-Fixture 'native rollback after partial write' $config $nativePreset
$before = File-Hashes $nativeRollback
$presetLock = [IO.File]::Open((Join-Path $nativeRollback 'custom preset.ini'), [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
$rejected = $false
try { & $installer -GameExecutable (Join-Path $nativeRollback 'Test Game.exe') -AddonDirectory $fixturePackage }
catch { $rejected = $true }
finally { $presetLock.Dispose() }
Require $rejected 'Native partial-write fixture did not fail.'
Require-Unchanged $before
Require (-not (Test-Path -LiteralPath (Join-Path $nativeRollback 'custom addons\SunshineSBS.addon64'))) 'Rollback left a new add-on installed.'
Write-Output 'PASS native partial-write rollback restores INI/preset/addon files.'

# Force failure at the removal of the old reference effect, after the add-on,
# INI and preset were written. The locked file stays intact; every earlier
# write must be restored from backup.
$removeRollback = New-Fixture 'native rollback during removal' $config $nativePreset
$removeReference = Join-Path $removeRollback 'reshade-shaders\Shaders\Sunshine3D\SunshineDepth3D.fx'
New-Item -ItemType Directory -Path (Split-Path -Parent $removeReference) -Force | Out-Null
[IO.File]::WriteAllText($removeReference, 'RESTORE REFERENCE EFFECT')
$before = File-Hashes $removeRollback
$removeLock = [IO.File]::Open($removeReference, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
$rejected = $false
try { & $installer -GameExecutable (Join-Path $removeRollback 'Test Game.exe') -AddonDirectory $fixturePackage }
catch { $rejected = $true }
finally { $removeLock.Dispose() }
Require $rejected 'Locked reference-effect removal did not fail.'
Require-Unchanged $before
Require (-not (Test-Path -LiteralPath (Join-Path $removeRollback 'custom addons\SunshineSBS.addon64'))) 'Removal rollback left a new add-on installed.'
Write-Output 'PASS failure at the reference-effect removal restores the old files and configuration.'

$guardGame = New-Fixture 'running fresh guard' $config $nativePreset $false
[IO.File]::WriteAllText((Join-Path $guardGame 'custom addons\SunshineSBS.addon64'), 'EXISTING ADDON')
$before = File-Hashes $guardGame
$rejected = $false
try { & $installer -GameExecutable (Join-Path $guardGame 'Test Game.exe') -AddonDirectory $fixturePackage -AllowRunningGame }
catch { $rejected = $_.Exception.Message.Contains('must be updated while the game is closed') }
Require $rejected 'AllowRunningGame replaced an existing add-on.'
Require-Unchanged $before
$freshGame = New-Fixture 'running fresh allowed' $config $nativePreset $false
& $installer -GameExecutable (Join-Path $freshGame 'Test Game.exe') -AddonDirectory $fixturePackage -AllowRunningGame
Require (Test-Path -LiteralPath (Join-Path $freshGame 'custom addons\SunshineSBS.addon64') -PathType Leaf) 'Fresh installation with AllowRunningGame did not install the add-on.'
Write-Output 'PASS AllowRunningGame only permits a missing native add-on and preserves an existing one.'

# The automatic host path runs as another account (SYSTEM for the service), where the game
# user's %LOCALAPPDATA% or %USERPROFILE% would name a different profile.
$environmentSettings = @(
    "[ADDON]`nAddonPath=%LOCALAPPDATA%\ReShade\Addons`n",
    "[GENERAL]`nPresetPath=%USERPROFILE%\ReShadePreset.ini`n"
)
for ($index = 0; $index -lt $environmentSettings.Count; ++$index) {
    $environmentGame = New-Fixture ('running environment path ' + $index) $environmentSettings[$index] $nativePreset $false
    $before = File-Hashes $environmentGame
    $rejected = $false
    try { & $installer -GameExecutable (Join-Path $environmentGame 'Test Game.exe') -AddonDirectory $fixturePackage -AllowRunningGame }
    catch { $rejected = $_.Exception.Message.Contains('needs manual setup') }
    Require $rejected ('AllowRunningGame resolved an environment-variable path in the host account: ' + $environmentSettings[$index])
    Require-Unchanged $before
    Require (@(Get-ChildItem -LiteralPath $environmentGame -Recurse -File).Count -eq $before.Count) 'A refused environment-variable path still created files.'
}
Write-Output 'PASS AllowRunningGame refuses environment-variable ReShade paths before writing anything.'

# Windows PowerShell 5.1 initializes $PSScriptRoot after binding -File parameters.
# Exercise the actual packaged CLI without -AddonDirectory, as documented for users.
$standalonePackage = Join-Path $OutputDirectory 'Windows PowerShell File package with spaces'
New-Item -ItemType Directory -Path $standalonePackage | Out-Null
$standaloneInstaller = Join-Path $standalonePackage 'install.ps1'
Copy-Item -LiteralPath $installer -Destination $standaloneInstaller
foreach ($name in @('SunshineSBS.addon64','SunshineSBS-LICENSES.txt')) {
    [IO.File]::WriteAllText((Join-Path $standalonePackage $name), 'UNEXECUTED CLI FIXTURE ' + $name)
}
$standaloneGame = New-Fixture 'Windows PowerShell File default addon directory' "[GENERAL]`n" '' $false
$standaloneExecutable = Join-Path $standaloneGame 'Test Game.exe'
$standaloneGameHash = (Get-FileHash -LiteralPath $standaloneExecutable).Hash
$windowsPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$previousErrorAction = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    $standaloneOutput = & $windowsPowerShell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $standaloneInstaller -GameExecutable $standaloneExecutable -AllowRunningGame 2>&1
    $standaloneExit = $LASTEXITCODE
} finally { $ErrorActionPreference = $previousErrorAction }
Require ($standaloneExit -eq 0) ('Windows PowerShell -File setup without AddonDirectory failed: ' + ($standaloneOutput -join [Environment]::NewLine))
foreach ($name in @('SunshineSBS.addon64','SunshineSBS-LICENSES.txt')) {
    Require ((Get-FileHash -LiteralPath (Join-Path $standaloneGame $name)).Hash -eq
        (Get-FileHash -LiteralPath (Join-Path $standalonePackage $name)).Hash) ('Packaged CLI did not use its co-located file: ' + $name)
}
Require ([IO.File]::ReadAllText((Join-Path $standaloneGame 'ReShade.ini')) -match '(?m)^Enabled=1\r?$') 'Packaged CLI did not enable native Game 3D.'
Require ((Get-FileHash -LiteralPath $standaloneExecutable).Hash -eq $standaloneGameHash) 'Packaged CLI changed the game executable.'
Write-Output 'PASS Windows PowerShell 5.1 -File uses co-located addon files when AddonDirectory is omitted.'

# Windows PowerShell started from a PowerShell 7 window inherits PowerShell 7's
# PSModulePath, whose Microsoft.PowerShell.Utility comes first. Its manifest
# lists the Utility cmdlets, Get-FileHash among them (a cmdlet there), over a
# nested Microsoft.PowerShell.Commands.Utility.dll that 5.1 resolves to its own
# assembly. 5.1 loads it for the first Utility cmdlet a script uses; then
# Get-FileHash, a function of 5.1's own module, is not recognized. A stand-in
# of that manifest (5.1's cmdlet list plus Get-FileHash) reproduces it. Only the
# child installer sees it.
$leakedModules = Join-Path $OutputDirectory 'PowerShell 7 modules'
$leakedUtility = Join-Path $leakedModules 'Microsoft.PowerShell.Utility'
New-Item -ItemType Directory -Path $leakedUtility | Out-Null
$leakedCmdlets = @((Get-Module Microsoft.PowerShell.Utility).ExportedCmdlets.Keys) + 'Get-FileHash' |
    ForEach-Object { "'" + $_ + "'" }
[IO.File]::WriteAllText((Join-Path $leakedUtility 'Microsoft.PowerShell.Utility.psd1'), ((
    "@{",
    "    GUID = '1da87e53-152b-403e-98dc-74d7b4d63d59'",
    "    ModuleVersion = '7.0.0.0'",
    "    CompatiblePSEditions = @('Core')",
    "    PowerShellVersion = '3.0'",
    "    NestedModules = @('Microsoft.PowerShell.Commands.Utility.dll')",
    ("    CmdletsToExport = @(" + ($leakedCmdlets -join ', ') + ")"),
    "    FunctionsToExport = @()",
    "}") -join "`r`n"))
$leakedGame = New-Fixture 'Windows PowerShell File with PowerShell 7 module path' "[GENERAL]`n" '' $false
$leakedExecutable = Join-Path $leakedGame 'Test Game.exe'
$leakedGameHash = (Get-FileHash -LiteralPath $leakedExecutable).Hash
$previousModulePath = $env:PSModulePath
$previousErrorAction = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    $env:PSModulePath = $leakedModules + ';' + $previousModulePath
    $leakedOutput = & $windowsPowerShell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $standaloneInstaller -GameExecutable $leakedExecutable -AllowRunningGame 2>&1
    $leakedExit = $LASTEXITCODE
} finally {
    $env:PSModulePath = $previousModulePath
    $ErrorActionPreference = $previousErrorAction
}
Require ($leakedExit -eq 0) ('Windows PowerShell -File setup under PowerShell 7''s module path failed: ' + ($leakedOutput -join [Environment]::NewLine))
foreach ($name in @('SunshineSBS.addon64','SunshineSBS-LICENSES.txt')) {
    Require ((Get-FileHash -LiteralPath (Join-Path $leakedGame $name)).Hash -eq
        (Get-FileHash -LiteralPath (Join-Path $standalonePackage $name)).Hash) ('Setup under PowerShell 7''s module path did not install: ' + $name)
}
Require ((Get-FileHash -LiteralPath $leakedExecutable).Hash -eq $leakedGameHash) 'Setup under PowerShell 7''s module path changed the game executable.'
Write-Output 'PASS Windows PowerShell 5.1 -File installs when PowerShell 7''s module path was inherited.'
