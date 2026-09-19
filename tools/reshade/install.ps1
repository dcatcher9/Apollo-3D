# SPDX-License-Identifier: GPL-3.0-only
# Run after installing ReShade 6.8 with full add-on support for the game.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $GameExecutable,
    [string] $AddonDirectory = $PSScriptRoot,
    [string] $ShaderDirectory
)
$ErrorActionPreference = 'Stop'

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
            if ($section -cin @('ADDON','GENERAL','SUNSHINE_DEPTH','SUNSHINE_GAME3D') -and -not $sections.Add($section)) {
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
            ($section -ceq 'SUNSHINE_DEPTH' -and $key -ceq 'AutoSelectSceneDepth') -or
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
    $expanded = [Environment]::ExpandEnvironmentVariables($Path)
    if ([IO.Path]::IsPathRooted($expanded)) { return [IO.Path]::GetFullPath($expanded) }
    return [IO.Path]::GetFullPath((Join-Path $gameDirectory $expanded))
}

function Enable-SunshineExport([string] $Text, [string] $Section, [bool] $Create) {
    $definitions = @(Split-ReShadeList (Get-IniValue $Text $Section 'PreprocessorDefinitions'))
    $found = $false
    $updated = @(foreach ($definition in $definitions) {
        if ($definition -cmatch '^\s*SUNSHINE_SBS_EXPORT(?:\s*=.*)?\s*$') {
            if (-not $found) { 'SUNSHINE_SBS_EXPORT=1' }
            $found = $true
        } else { $definition }
    })
    if (-not $found -and -not $Create) { return $Text }
    if (-not $found) { $updated += 'SUNSHINE_SBS_EXPORT=1' }
    return Set-IniValue $Text $Section 'PreprocessorDefinitions' (Join-ReShadeList $updated)
}

function Remove-Game3DDefinitions([string] $Text, [string] $Section) {
    $value = Get-IniValue $Text $Section 'PreprocessorDefinitions'
    if (-not $value) { return $Text }
    $definitions = @(Split-ReShadeList $value | Where-Object {
        $_ -cnotmatch '^\s*(?:SUNSHINE_GAME3D_(?:AUTOMATIC|CAMERA_DEPTH|GENERIC_CONFIG|COLOR_FIXES|FUSED_POST|NATIVE_WARP)|SUNSHINE_SOURCE_OWNED_DEPTH|SUNSHINE_HOST_WARP_SUPPORTED|SUNSHINE_DEPTH_PROBE|HOST_SBS_TEST_FIXED_ELEVEN_REFERENCE)(?:\s*=.*)?\s*$' -and
        # Shared global definitions may still belong to the original reference.
        # Only the owned effect's obsolete display/export definitions are removed.
        ($Section -cne 'SunshineGame3D.fx' -or
         $_ -cnotmatch '^\s*(?:SUNSHINE_SBS_EXPORT|DoubleBuffer_Mode|EX_DLP_FS_Mode|Virtual_Reality_Mode|Anaglyph_Mode|Inficolor_3D_Emulator|Reconstruction_Mode|REST_UI_Mode|Super3D_Mode|Use_2D_Plus_Depth|Frame_Packed_Mode)(?:\s*=.*)?\s*$')
    })
    return Set-IniValue $Text $Section 'PreprocessorDefinitions' (Join-ReShadeList $definitions)
}

function Get-EffectTechniqueBody([string] $Code, [string] $Name) {
    $declaration = [regex]::Match($Code, '\btechnique(?:11)?\s+' + [regex]::Escape($Name) + '\b[^{}]*\{')
    if (-not $declaration.Success) { return '' }
    $start = $declaration.Index + $declaration.Length
    $depth = 1
    for ($index = $start; $index -lt $Code.Length; ++$index) {
        if ($Code[$index] -ceq '{') { ++$depth }
        elseif ($Code[$index] -ceq '}') {
            --$depth
            if ($depth -eq 0) { return $Code.Substring($start, $index - $start) }
        }
    }
    return ''
}

$gameExe = (Get-Item -LiteralPath $GameExecutable -ErrorAction Stop).FullName
if ([IO.Path]::GetExtension($gameExe) -ine '.exe') { throw 'GameExecutable must name the actual game EXE beside ReShade.ini.' }
$gameDirectory = Split-Path -Parent $gameExe
$runningGame = Get-CimInstance Win32_Process | Where-Object {
    $_.ExecutablePath -ieq $gameExe -or (-not $_.ExecutablePath -and $_.Name -ieq [IO.Path]::GetFileName($gameExe))
}
if ($runningGame) { throw 'Close the game before installing Sunshine 3D, then run this command again.' }
$iniPath = Join-Path $gameDirectory 'ReShade.ini'
if (-not (Test-Path -LiteralPath $iniPath -PathType Leaf)) { throw 'Install ReShade 6.8 with full add-on support for this game first.' }
$iniBefore = [IO.File]::ReadAllText($iniPath)
Assert-UnambiguousIni $iniBefore $iniPath
$addonSearch = Get-IniScalarValue $iniBefore 'ADDON' 'AddonPath'
if (-not $addonSearch) { $addonSearch = '.' }
$addonDestination = Resolve-GamePath $addonSearch
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
$nativeGame3D = [string]::IsNullOrWhiteSpace($ShaderDirectory)
$original = $false
$effectName = 'native Game 3D'
# ShaderDirectory is an explicit reference-renderer opt-in. Shipping Game 3D
# owns its GPU passes inside the DLL and never discovers/copies external FX.
if (-not $nativeGame3D) {
    $original = -not (Test-Path -LiteralPath (Join-Path $ShaderDirectory 'SunshineDepth3D.fx') -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $ShaderDirectory 'SuperDepth3D.fx') -PathType Leaf)
    $effectName = if ($original) { 'SuperDepth3D' } else { 'SunshineDepth3D' }
    $effectPath = Join-Path $ShaderDirectory ($effectName + '.fx')
    if (-not (Test-Path -LiteralPath $effectPath -PathType Leaf)) {
        throw 'Game 3D now runs in the add-on. Omit -ShaderDirectory for normal installation; supply an original SuperDepth3D or independent SunshineDepth3D directory only for reference testing.'
    }
    $shaderRelativePath = if ($original) { '.\reshade-shaders\Shaders\Depth3D' } else { '.\reshade-shaders\Shaders\Sunshine3D' }
    $shaderDestination = Resolve-GamePath $shaderRelativePath
    $effectCode = [regex]::Replace([IO.File]::ReadAllText($effectPath), '(?m)//[^\r\n]*|(?s)/\*.*?\*/', '')
    $shaderFiles = if ($original) { @('SuperDepth3D.fx','Overwatch.fxh','AXAA.fxh','SunshineIntegration.fxh','SunshinePost.fxh') } else { @('SunshineDepth3D.fx') }
    foreach ($name in $shaderFiles) {
        $source = Join-Path $ShaderDirectory $name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "$name is missing. Use the complete matching reference shader source and all its integration/support includes."
        }
        $source = (Get-Item -LiteralPath $source).FullName
        $destination = Join-Path $shaderDestination $name
        if ($source -ieq $destination) { throw 'Use a shader source separate from the game installation.' }
        $copies.Add(@{Source=$source; Destination=$destination})
    }
    $techniqueBody = Get-EffectTechniqueBody $effectCode $effectName
    if (-not $techniqueBody -or
        $effectCode -notmatch '\bsunshine_sbs_export\s*=\s*1\s*;' -or
        $effectCode -notmatch '\bsunshine_sbs_layout\s*=\s*"sbs_lr"\s*;') {
        throw "$effectName.fx does not expose the expected Sunshine stereo technique and export interface."
    }
    if ($original) {
        if ($effectCode -notmatch '\btexture(?:2D)?\s+DoubleTex\s*<[^>]*\bsunshine_sbs_export\s*=\s*1\s*;' -or
            $techniqueBody -notmatch '\bRenderTarget(?:0)?\s*=\s*DoubleTex\s*;') {
            throw 'SuperDepth3D.fx must render its annotated export texture from its own technique.'
        }
        if ($effectCode -notmatch '#\s*include\s+"SunshineIntegration\.fxh"' -or
            $effectCode -notmatch '#\s*include\s+"SunshinePost\.fxh"') {
            throw 'SuperDepth3D.fx must include the matching Sunshine integration/post files.'
        }
    } else {
        foreach ($uniform in @('Sunshine_DepthReady','Sunshine_RawAnchor','Sunshine_RawGain','Sunshine_Calibrated')) {
            if ($effectCode -notmatch ('\buniform\s+(?:bool|float)\s+' + $uniform + '\b')) {
                throw "SunshineDepth3D.fx is missing the $uniform calibration input. Use the complete matching package."
            }
        }
    }
}
$iniAfter = $iniBefore
$disabled = @(Split-ReShadeList (Get-IniValue $iniAfter 'ADDON' 'DisabledAddons') | Where-Object {
    $_ -and $_ -notmatch '^Sunshine (3D|SBS)(@SunshineSBS\.addon64)?$'
})
# ReShade checks the exported name, so this also prevents renamed legacy copies
# from registering before the integrated add-on on the first launch after install.
foreach ($name in @('Generic Depth','Sunshine Depth')) {
    if ($disabled -cnotcontains $name) { $disabled += $name }
}
$iniAfter = Set-IniValue $iniAfter 'ADDON' 'DisabledAddons' (Join-ReShadeList $disabled)
if (-not (Get-IniScalarValue $iniAfter 'SUNSHINE_DEPTH' 'AutoSelectSceneDepth')) {
    $iniAfter = Set-IniValue $iniAfter 'SUNSHINE_DEPTH' 'AutoSelectSceneDepth' '1'
}
$iniAfter = Set-IniValue $iniAfter 'GENERAL' 'PresetPath' (Join-ReShadeList @($presetSetting))
if (-not $nativeGame3D) {
    $searchPaths = @(Split-ReShadeList (Get-IniValue $iniAfter 'GENERAL' 'EffectSearchPaths'))
    $shaderDiscovered = $false
    foreach ($searchPath in $searchPaths) {
        try {
            $recursive = $searchPath -match '[\\/]\*\*$'
            $searchRoot = Resolve-GamePath ($searchPath -replace '[\\/]\*\*$', '')
            if ($searchRoot -ieq $shaderDestination -or
                ($recursive -and $shaderDestination.StartsWith($searchRoot.TrimEnd('\','/') + '\', [StringComparison]::OrdinalIgnoreCase))) {
                $shaderDiscovered = $true
            }
        } catch { } # Preserve search entries we do not understand and add our path.
    }
    if (-not $shaderDiscovered) {
        $searchPaths += $shaderRelativePath
        $iniAfter = Set-IniValue $iniAfter 'GENERAL' 'EffectSearchPaths' (Join-ReShadeList $searchPaths)
    }
}
$presetAfter = $presetBefore
if ($nativeGame3D) {
    # Native settings take precedence on every reinstall. A current FX section
    # with omitted values uses the same defaults as native Game 3D; only a
    # wholly absent section may inherit the prior original reference settings.
    $migrationSection = if (Find-IniSection $presetBefore 'SunshineGame3D.fx') { 'SunshineGame3D.fx' } else { 'SuperDepth3D.fx' }
    if ((Get-IniValue $iniBefore 'SUNSHINE_GAME3D' 'Strength') -ceq '') {
        $value = Get-IniValue $presetBefore $migrationSection 'Depth_Adjustment'
        $numeric = [single]0
        if ([single]::TryParse($value,[Globalization.NumberStyles]::Float,[Globalization.CultureInfo]::InvariantCulture,[ref]$numeric) -and
            -not [single]::IsNaN($numeric) -and -not [single]::IsInfinity($numeric)) {
            $iniAfter = Set-IniValue $iniAfter 'SUNSHINE_GAME3D' 'Strength' $value
        }
    }
    if ((Get-IniValue $iniBefore 'SUNSHINE_GAME3D' 'DepthView') -ceq '') {
        $value = Get-IniValue $presetBefore $migrationSection 'Depth_Map_View'
        if ($value -cmatch '^[012]$') { $iniAfter = Set-IniValue $iniAfter 'SUNSHINE_GAME3D' 'DepthView' $value }
    }
    if ((Get-IniValue $iniBefore 'SUNSHINE_GAME3D' 'Enabled') -ceq '') {
        $iniAfter = Set-IniValue $iniAfter 'SUNSHINE_GAME3D' 'Enabled' '1'
    }
} else {
    # Explicit reference testing owns stereo output; preserve native preferences
    # for returning to Game 3D, while disabling its renderer to avoid two writers.
    $iniAfter = Set-IniValue $iniAfter 'SUNSHINE_GAME3D' 'Enabled' '0'
}
# Techniques is a preset-wide key before any named shader section.
$techniques = @(Split-ReShadeList (Get-IniValue $presetAfter '' 'Techniques') | Where-Object {
    $_ -and $_ -cnotin @('SuperDepth3D','SuperDepth3D@SuperDepth3D.fx','SunshineDepth3D','SunshineDepth3D@SunshineDepth3D.fx',
        'SunshineGame3D','SunshineGame3D@SunshineGame3D.fx')
})
if (-not $nativeGame3D) { $techniques += ($effectName + '@' + $effectName + '.fx') }
$presetAfter = Set-IniValue $presetAfter '' 'Techniques' (Join-ReShadeList $techniques)
if ($original) {
    # Global and preset-wide definitions have different precedence in ReShade.
    # Set only our export flag; retain every unrelated macro and shader uniform.
    $iniAfter = Enable-SunshineExport $iniAfter 'GENERAL' $true
    $iniAfter = Enable-SunshineExport $iniAfter ($effectName + '.fx') $false
    $presetAfter = Enable-SunshineExport $presetAfter '' $true
    $presetAfter = Enable-SunshineExport $presetAfter ($effectName + '.fx') $false
}
if ($nativeGame3D) {
    foreach ($scope in @('GENERAL','SunshineGame3D.fx')) {
        $iniAfter = Remove-Game3DDefinitions $iniAfter $scope
    }
    foreach ($scope in @('','SunshineGame3D.fx')) {
        $presetAfter = Remove-Game3DDefinitions $presetAfter $scope
    }
}

$legacyPaths = @('SunshineDepth.addon64','SunshineDepthProbe.addon64') | ForEach-Object { Join-Path $addonDestination $_ }
# Retired support files are removed only from the owned Game 3D directory. The
# original reference's files and settings remain available in its separate path.
if ($nativeGame3D) {
    $shaderDestination = Resolve-GamePath '.\reshade-shaders\Shaders\SunshineGame3D'
    $legacyPaths += @('SunshineGame3D.fx','SunshineHostWarp.fxh','SunshineCamera.fxh','SunshineGenericConfig.fxh','SunshineWarpConfig.fxh','Overwatch.fxh','AXAA.fxh','SunshineIntegration.fxh','SunshinePost.fxh') | ForEach-Object { Join-Path $shaderDestination $_ }
}
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
Write-Output "Installed Sunshine 3D for $([IO.Path]::GetFileName($gameExe)) using $effectName."
Write-Output 'Start the game, open ReShade > Add-ons > Sunshine 3D, then choose Game 3D in Moonlight.'
Write-Output "Backup: $backupDirectory"
