# SPDX-License-Identifier: GPL-3.0-only
# Run after installing ReShade 6.8 with full add-on support for the game.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $GameExecutable,
    [string] $AddonDirectory,
    [switch] $AllowRunningGame
)
$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 started from a PowerShell 7 window inherits PowerShell
# 7's PSModulePath, whose Core-only Microsoft.PowerShell.Utility comes first and
# hides 5.1's own Get-FileHash ("not recognized"). Load this edition's module
# from $PSHOME; the environment and every other command stay as they are.
if ($PSVersionTable.PSEdition -ne 'Core') {
    Import-Module -Name (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1')
}
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
# Windows PowerShell 5.1 does not populate $PSScriptRoot while binding -File
# parameter defaults. Resolve the co-located package only after binding.
if (-not $AddonDirectory) { $AddonDirectory = $PSScriptRoot }

function Assert-UnambiguousIni([string] $Text, [string] $Path) {
    $sections = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $keys = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $section = ''
    foreach ($line in ($Text -split '\r?\n')) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed[0] -in @(';','#','/')) { continue }
        if ($trimmed.StartsWith('[')) {
            $closing = $trimmed.IndexOf(']')
            if ($closing -lt 0) { throw "Malformed section header in $Path. Fix it before installing." }
            $section = $trimmed.Substring(1,$closing-1).Trim()
            if ($section -cin @('ADDON','GENERAL','SUNSHINE_GAME3D') -and -not $sections.Add($section)) {
                throw "Duplicate [$section] sections in $Path. Merge them before installing."
            }
            continue
        }
        $equals = $trimmed.IndexOf('=')
        if ($equals -lt 0) { continue }
        $key = $trimmed.Substring(0,$equals).Trim()
        $relevant = $key -ceq 'PreprocessorDefinitions' -or
            ($section -ceq '' -and $key -ceq 'Techniques') -or
            ($section -ceq 'ADDON' -and $key -cin @('AddonPath','DisabledAddons')) -or
            ($section -ceq 'GENERAL' -and $key -cin @('PresetPath','EffectSearchPaths')) -or
            ($section -ceq 'SUNSHINE_GAME3D' -and $key -cin @('Strength','DepthView','Enabled'))
        if ($relevant -and -not $keys.Add($section + "`0" + $key)) {
            throw "Duplicate $key entries in [$section] in $Path. Merge them before installing."
        }
    }
}

function Find-IniSection([string] $Text, [string] $Section) {
    $sectionPattern = if ($Section -ceq '') { '(?ms)\A(?<body>.*?)(?=^[\t ]*\[|\z)' } else {
        '(?ms)^[\t ]*\[[\t ]*' + [regex]::Escape($Section) + '[\t ]*\][^\r\n]*(?:\r?\n|\z)(?<body>.*?)(?=^[\t ]*\[|\z)'
    }
    $sectionMatches = [regex]::Matches($Text, $sectionPattern)
    if ($sectionMatches.Count -gt 1) { throw "Duplicate [$Section] sections must be merged before installing." }
    if ($sectionMatches.Count -eq 1) { return $sectionMatches[0] }
    return $null
}

function Get-IniValue([string] $Text, [string] $Section, [string] $Key) {
    $sectionMatch = Find-IniSection $Text $Section
    if (-not $sectionMatch) { return '' }
    $valueMatches = [regex]::Matches($sectionMatch.Groups['body'].Value, '(?m)^[\t ]*' + [regex]::Escape($Key) + '[\t ]*=[\t ]*(?<value>[^\r\n]*)')
    if ($valueMatches.Count -gt 1) { throw "Duplicate $Key entries in [$Section] must be merged before installing." }
    if ($valueMatches.Count -eq 1) { return $valueMatches[0].Groups['value'].Value.Trim() }
    return ''
}

# ReShade uses doubled commas inside an array element; a single comma separates
# elements. Preserve decoded entries when adding/removing our own settings.
function Split-ReShadeList([string] $Value) {
    if ($Value -ceq '') { return }
    $element = [Text.StringBuilder]::new()
    for ($index = 0; $index -lt $Value.Length; ++$index) {
        if ($Value[$index] -eq ',') {
            if ($index + 1 -lt $Value.Length -and $Value[$index+1] -eq ',') {
                [void]$element.Append(',')
                ++$index
            } else {
                $element.ToString()
                [void]$element.Clear()
            }
        } else { [void]$element.Append($Value[$index]) }
    }
    $element.ToString()
}

function Join-ReShadeList([string[]] $Values) {
    return (@($Values | Where-Object { $_ -cne '' } | ForEach-Object { $_.Replace(',',',,') }) -join ',')
}

function Get-IniScalarValue([string] $Text, [string] $Section, [string] $Key) {
    $values = @(Split-ReShadeList (Get-IniValue $Text $Section $Key))
    if ($values.Count -gt 1) { throw "$Key in [$Section] must have one value; escape a literal comma as two commas." }
    if ($values.Count -eq 1) { return $values[0] }
    return ''
}

function Set-IniValue([string] $Text, [string] $Section, [string] $Key, [string] $Value) {
    $newline = if ($Text.Contains("`r`n")) { "`r`n" } else { "`n" }
    $sectionMatch = Find-IniSection $Text $Section
    if (-not $sectionMatch) {
        return $Text.TrimEnd("`r", "`n") + $newline + $newline + "[$Section]" + $newline + "$Key=$Value" + $newline
    }
    $body = $sectionMatch.Groups['body'].Value
    $keyPattern = '(?m)^[\t ]*' + [regex]::Escape($Key) + '[\t ]*=[^\r\n]*'
    $keyMatches = [regex]::Matches($body, $keyPattern)
    if ($keyMatches.Count -gt 1) { throw "Duplicate $Key entries in [$Section] must be merged before installing." }
    if ($keyMatches.Count -eq 1) {
        $keyMatch = $keyMatches[0]
        $body = $body.Remove($keyMatch.Index, $keyMatch.Length).Insert($keyMatch.Index, "$Key=$Value")
    } else { $body = "$Key=$Value" + $newline + $body }
    $bodyGroup = $sectionMatch.Groups['body']
    return $Text.Remove($bodyGroup.Index, $bodyGroup.Length).Insert($bodyGroup.Index, $body)
}

function Resolve-GamePath([string] $Path) {
    # The game expands environment variables for its own user. The automatic path runs as the
    # host's account (SYSTEM for the service), where %LOCALAPPDATA% names another profile.
    if ($AllowRunningGame -and $Path.Contains('%')) {
        throw "ReShade.ini path '$Path' uses environment variables and needs manual setup: close the game and run this installer without -AllowRunningGame from your own account."
    }
    $expanded = [Environment]::ExpandEnvironmentVariables($Path)
    if ([IO.Path]::IsPathRooted($expanded)) { return [IO.Path]::GetFullPath($expanded) }
    return [IO.Path]::GetFullPath((Join-Path $gameDirectory $expanded))
}

$gameExe = (Get-Item -LiteralPath $GameExecutable -ErrorAction Stop).FullName
if ([IO.Path]::GetExtension($gameExe) -ine '.exe') { throw 'GameExecutable must name the actual game EXE beside ReShade.ini.' }
$gameDirectory = Split-Path -Parent $gameExe
$runningGame = Get-CimInstance Win32_Process | Where-Object {
    $_.ExecutablePath -ieq $gameExe -or (-not $_.ExecutablePath -and $_.Name -ieq [IO.Path]::GetFileName($gameExe))
}
if ($runningGame -and -not $AllowRunningGame) { throw 'Close the game before installing Sunshine 3D, then run this command again.' }
$iniPath = Join-Path $gameDirectory 'ReShade.ini'
if (-not (Test-Path -LiteralPath $iniPath -PathType Leaf)) { throw 'Install ReShade 6.8 with full add-on support for this game first.' }
$iniBefore = [IO.File]::ReadAllText($iniPath)
Assert-UnambiguousIni $iniBefore $iniPath
$addonSearch = Get-IniScalarValue $iniBefore 'ADDON' 'AddonPath'
if (-not $addonSearch) { $addonSearch = '.' }
$addonDestination = Resolve-GamePath $addonSearch
if ($AllowRunningGame -and (Test-Path -LiteralPath (Join-Path $addonDestination 'SunshineSBS.addon64'))) {
    throw 'An existing Sunshine add-on must be updated while the game is closed.'
}
$addonSource = (Get-Item -LiteralPath (Join-Path $AddonDirectory 'SunshineSBS.addon64')).FullName
$licenseSource = (Get-Item -LiteralPath (Join-Path $AddonDirectory 'SunshineSBS-LICENSES.txt')).FullName
if ([IO.Path]::GetFullPath((Join-Path $addonDestination 'SunshineSBS.addon64')) -ieq $addonSource) {
    throw 'Use the build/package directory as AddonDirectory, separate from the game installation.'
}
$presetSetting = Get-IniScalarValue $iniBefore 'GENERAL' 'PresetPath'
if (-not $presetSetting) { $presetSetting = '.\ReShadePreset.ini' }
$presetPath = Resolve-GamePath $presetSetting
if ([IO.Path]::GetExtension($presetPath) -ine '.ini' -or $presetPath -ieq $iniPath) {
    throw 'PresetPath must name a separate preset .ini file, not ReShade.ini or another installation file.'
}
$presetBefore = if (Test-Path -LiteralPath $presetPath -PathType Leaf) { [IO.File]::ReadAllText($presetPath) } else { '' }
Assert-UnambiguousIni $presetBefore $presetPath

$copies = [Collections.Generic.List[object]]::new()
$copies.Add(@{Source=$addonSource; Destination=(Join-Path $addonDestination 'SunshineSBS.addon64')})
$copies.Add(@{Source=$licenseSource; Destination=(Join-Path $addonDestination 'SunshineSBS-LICENSES.txt')})
$iniAfter = $iniBefore
$disabled = @(Split-ReShadeList (Get-IniValue $iniAfter 'ADDON' 'DisabledAddons') | Where-Object {
    $_ -and $_ -notmatch '^Sunshine (3D|SBS)(@SunshineSBS\.addon64)?$'
})
# ReShade's built-in Generic Depth add-on would otherwise compete with the
# integrated depth selector on the first launch after install.
if ($disabled -cnotcontains 'Generic Depth') { $disabled += 'Generic Depth' }
$iniAfter = Set-IniValue $iniAfter 'ADDON' 'DisabledAddons' (Join-ReShadeList $disabled)
$iniAfter = Set-IniValue $iniAfter 'GENERAL' 'PresetPath' (Join-ReShadeList @($presetSetting))
$presetAfter = $presetBefore
# Saved Game 3D settings take precedence on every reinstall; a new installation
# enables Game 3D and leaves strength and view at the add-on's defaults.
if ((Get-IniValue $iniBefore 'SUNSHINE_GAME3D' 'Enabled') -ceq '') {
    $iniAfter = Set-IniValue $iniAfter 'SUNSHINE_GAME3D' 'Enabled' '1'
}
# Techniques is a preset-wide key before any named shader section. The stereo
# techniques of earlier effect-based installations are disabled: Game 3D renders
# in the add-on.
$techniques = @(Split-ReShadeList (Get-IniValue $presetAfter '' 'Techniques') | Where-Object {
    $_ -and $_ -cnotin @('SuperDepth3D','SuperDepth3D@SuperDepth3D.fx','SunshineDepth3D','SunshineDepth3D@SunshineDepth3D.fx',
        'SunshineGame3D','SunshineGame3D@SunshineGame3D.fx')
})
$presetAfter = Set-IniValue $presetAfter '' 'Techniques' (Join-ReShadeList $techniques)

# The reference effect an explicit -ShaderDirectory installation copied is
# removed from the directory those installations owned. A user's own
# SuperDepth3D files and settings stay in their separate path.
$legacyPaths = @(Join-Path (Resolve-GamePath '.\reshade-shaders\Shaders\Sunshine3D') 'SunshineDepth3D.fx')
$destinationSet = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($destination in (@($copies | ForEach-Object { $_.Destination }) + @($iniPath,$presetPath) + @($legacyPaths))) {
    if (Test-Path -LiteralPath $destination -PathType Container) {
        throw "An installation file path is a directory: $destination. Resolve it before installing."
    }
    if (-not $destinationSet.Add([IO.Path]::GetFullPath($destination))) {
        throw 'Installation paths overlap. Use separate add-on, shader, configuration, and preset files.'
    }
}
$backupDirectory = Join-Path $gameDirectory ('Sunshine3D-backups\' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $backupDirectory -Force | Out-Null
$destinations = @($copies | ForEach-Object { $_.Destination }) + @($iniPath,$presetPath) + @($legacyPaths)
$manifest = @()
foreach ($destination in ($destinations | Select-Object -Unique)) {
    $entry = [ordered]@{path=$destination; existed=(Test-Path -LiteralPath $destination -PathType Leaf); backup=$null}
    if ($entry.existed) {
        $backupFile = Join-Path $backupDirectory (('{0:D3}-' -f $manifest.Count) + [IO.Path]::GetFileName($destination))
        Copy-Item -LiteralPath $destination -Destination $backupFile
        $entry.backup = $backupFile
    }
    $manifest += $entry
}
$manifest | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $backupDirectory 'manifest.json') -Encoding utf8
try {
    foreach ($copy in $copies) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $copy.Destination) -Force | Out-Null
        Copy-Item -LiteralPath $copy.Source -Destination $copy.Destination -Force
        if ((Get-FileHash -LiteralPath $copy.Source).Hash -ne (Get-FileHash -LiteralPath $copy.Destination).Hash) { throw 'Copy verification failed.' }
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $presetPath) -Force | Out-Null
    [IO.File]::WriteAllText($iniPath,$iniAfter,[Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText($presetPath,$presetAfter,[Text.UTF8Encoding]::new($false))
    foreach ($legacyPath in $legacyPaths) {
        if (Test-Path -LiteralPath $legacyPath -PathType Leaf) { Remove-Item -LiteralPath $legacyPath -Force }
    }
} catch {
    $installationFailure = $_
    $restoreFailures = [Collections.Generic.List[string]]::new()
    foreach ($entry in $manifest) {
        try {
            if ($entry.existed) { Copy-Item -LiteralPath $entry.backup -Destination $entry.path -Force }
            elseif (Test-Path -LiteralPath $entry.path -PathType Leaf) { Remove-Item -LiteralPath $entry.path -Force }
        } catch { $restoreFailures.Add($entry.path + ': ' + $_.Exception.Message) }
    }
    if ($restoreFailures.Count -gt 0) {
        throw "Installation failed: $($installationFailure.Exception.Message). Some files could not be restored; backups are in $backupDirectory. $($restoreFailures -join '; ')"
    }
    throw $installationFailure
}
Write-Output "Installed Sunshine 3D for $([IO.Path]::GetFileName($gameExe)) using native Game 3D."
if ($runningGame) { Write-Output 'Installation is ready for the next game launch; the current game has not loaded the new add-on.' }
Write-Output 'On the next game launch, open ReShade > Add-ons > Sunshine 3D, then choose Game 3D in Moonlight.'
Write-Output "Backup: $backupDirectory"
