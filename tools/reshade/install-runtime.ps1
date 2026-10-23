# SPDX-License-Identifier: GPL-3.0-only
# Fresh ReShade setup only. The running game loads these files on its next launch.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $GameExecutable,
    [string] $SetupArchive,
    [string] $CacheDirectory = (Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Sunshine3D\cache\ReShade')
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$setupUri = 'https://reshade.me/downloads/ReShade_Setup_6.8.0_Addon.exe'
$setupHash = 'AFE4C8F13048306307983B8B3D41D5BF00A86820440B0E57DEA10950E1176445'
$proxyNames = @('dxgi.dll','d3d9.dll','d3d10.dll','d3d10_1.dll','d3d11.dll','d3d12.dll',
    'opengl32.dll','d3d8.dll','dinput8.dll','version.dll','winmm.dll','dsound.dll')

function Assert-Native64Pe([string] $Path, [bool] $RequireDll) {
    $stream = [IO.File]::OpenRead($Path)
    $reader = [IO.BinaryReader]::new($stream)
    try {
        if ($stream.Length -lt 64 -or $reader.ReadUInt16() -ne 0x5a4d) { throw "Not a Windows PE image: $Path" }
        $stream.Position = 0x3c
        $header = $reader.ReadInt32()
        if ($header -lt 64 -or $header -gt $stream.Length - 26) { throw "Invalid PE header: $Path" }
        $stream.Position = $header
        if ($reader.ReadUInt32() -ne 0x4550 -or $reader.ReadUInt16() -ne 0x8664) {
            throw "Game 3D requires a native x64 Windows image: $Path"
        }
        $stream.Position = $header + 22
        $characteristics = $reader.ReadUInt16()
        if ($RequireDll -and ($characteristics -band 0x2000) -eq 0) { throw "ReShade payload is not a DLL: $Path" }
        if ($reader.ReadUInt16() -ne 0x20b) { throw "Game 3D requires a PE32+ image: $Path" }
    } finally {
        $reader.Dispose()
        $stream.Dispose()
    }
}

function Assert-FreshTarget([string] $Directory) {
    foreach ($name in (@('ReShade.ini') + $proxyNames)) {
        $existing = Join-Path $Directory $name
        if (Test-Path -LiteralPath $existing) {
            throw "Existing ReShade configuration or proxy prevents automatic runtime installation: $existing. Existing files are never replaced; use ReShade setup manually for this game."
        }
    }
    # A preset and shader directory may remain after an old runtime was removed.
    # They do not load ReShade and must not block fresh runtime setup. Leave the
    # preset untouched here; the add-on installer backs it up before migration.
    $preset = Get-Item -LiteralPath (Join-Path $Directory 'ReShadePreset.ini') -Force -ErrorAction SilentlyContinue
    if ($preset -and ($preset.PSIsContainer -or ($preset.Attributes -band [IO.FileAttributes]::ReparsePoint))) {
        throw "Existing ReShade preset must be a regular file for automatic runtime installation: $($preset.FullName). Existing files are never replaced; resolve the preset path before installing."
    }
}

function Assert-PinnedArchive([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ine $setupHash) {
        throw 'ReShade setup SHA-256 verification failed. Only the pinned official 6.8.0 full-add-on setup is accepted.'
    }
}

function Assert-CompatibleGame([string] $Name) {
    # The official installer consults this same list. It may only restrict our actual detected
    # executable; remote InstallTarget entries never redirect writes into another directory.
    $cache = [IO.Path]::GetFullPath($CacheDirectory)
    $compatibility = Join-Path $cache 'Compatibility.ini'
    $temporary = Join-Path $cache ('reshade-compatibility-' + [guid]::NewGuid().ToString('N') + '.tmp')
    $cached = Get-Item -LiteralPath $compatibility -ErrorAction SilentlyContinue
    if (-not $cached -or $cached.LastWriteTimeUtc -lt [DateTime]::UtcNow.AddDays(-1)) {
        $previousTls = [Net.ServicePointManager]::SecurityProtocol
        try {
            [void][IO.Directory]::CreateDirectory($cache)
            [Net.ServicePointManager]::SecurityProtocol = $previousTls -bor [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/crosire/reshade-shaders/list/Compatibility.ini' -OutFile $temporary -TimeoutSec 10
            if ((Get-Item -LiteralPath $temporary).Length -gt 1MB) { throw 'ReShade compatibility list is unexpectedly large.' }
            if (Test-Path -LiteralPath $compatibility -PathType Leaf) { [IO.File]::Replace($temporary, $compatibility, [NullString]::Value) }
            else { [IO.File]::Move($temporary, $compatibility) }
        } catch {
            # Match official setup's offline fallback. An existing cached restriction remains
            # authoritative even when refreshing it failed.
            Write-Warning 'The official ReShade compatibility list could not be refreshed; using cached information when available.'
        } finally {
            [Net.ServicePointManager]::SecurityProtocol = $previousTls
            if (Test-Path -LiteralPath $temporary -PathType Leaf) { Remove-Item -LiteralPath $temporary -Force }
        }
    }
    if (-not (Test-Path -LiteralPath $compatibility -PathType Leaf)) { return }
    if ((Get-Item -LiteralPath $compatibility).Length -gt 1MB) { throw 'ReShade compatibility list is unexpectedly large.' }
    $matching = $false
    $forcedApi = $false
    $renderApi = ''
    foreach ($line in [IO.File]::ReadAllLines($compatibility)) {
        $text = $line.Trim().TrimStart([char]0xfeff)
        if (-not $text -or $text[0] -in @('#',';')) { continue }
        if ($text -match '^\[([^\]]+)\]') { $matching = $Matches[1] -ieq $Name; continue }
        if (-not $matching -or $text -notmatch '^([^=]+)=(.*)$') { continue }
        $key = $Matches[1].Trim()
        $value = $Matches[2].Trim()
        if ($key -ieq 'Banned' -and $value -eq '1') {
            throw "Official ReShade compatibility information blocks installation for $Name (Banned=1)."
        }
        if ($key -ieq 'ForceInstallApi' -and $value -eq '1') { $forcedApi = $true }
        if ($key -ieq 'RenderApi') { $renderApi = $value }
        if ($key -ieq 'InstallTarget' -and $value -and $value -notin @('.','.\','./')) {
            throw "ReShade requires a different installation layout for $Name. Automatic setup uses only the detected executable directory; use official ReShade setup manually."
        }
        if ($key -ieq 'RenderApi' -and $value -and $value -notin @('DXGI','D3D11','D3D12')) {
            throw "Official ReShade compatibility information names unsupported $value rendering for $Name. Game 3D requires Direct3D 11 or 12."
        }
    }
    if ($forcedApi -and $renderApi -ine 'DXGI') {
        throw "ReShade requires a specific graphics API proxy for $Name. Use official ReShade setup manually for this game."
    }
}

function Get-PinnedArchive {
    if ($SetupArchive) {
        $archive = (Get-Item -LiteralPath $SetupArchive -ErrorAction Stop).FullName
        Assert-PinnedArchive $archive
        return $archive
    }
    $cache = [IO.Path]::GetFullPath($CacheDirectory)
    $archive = Join-Path $cache 'ReShade_Setup_6.8.0_Addon.exe'
    if ((Test-Path -LiteralPath $archive -PathType Leaf) -and
        (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ieq $setupHash) {
        return $archive
    }
    [void][IO.Directory]::CreateDirectory($cache)
    $temporary = Join-Path $cache ('reshade-download-' + [guid]::NewGuid().ToString('N') + '.tmp')
    $previousTls = [Net.ServicePointManager]::SecurityProtocol
    try {
        [Net.ServicePointManager]::SecurityProtocol = $previousTls -bor [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri $setupUri -OutFile $temporary -TimeoutSec 60
        Assert-PinnedArchive $temporary
        try {
            if (Test-Path -LiteralPath $archive -PathType Leaf) {
                if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ine $setupHash) {
                    [IO.File]::Replace($temporary, $archive, [NullString]::Value)
                }
            } else { [IO.File]::Move($temporary, $archive) }
        }
        catch {
            # A concurrent host may have populated the same cache. Never replace its file;
            # accept it only after independently verifying the same pinned digest.
            if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) { throw }
            Assert-PinnedArchive $archive
        }
        Assert-PinnedArchive $archive
        return $archive
    } finally {
        [Net.ServicePointManager]::SecurityProtocol = $previousTls
        if (Test-Path -LiteralPath $temporary -PathType Leaf) { Remove-Item -LiteralPath $temporary -Force }
    }
}

function Write-RuntimePayload([string] $Archive, [string] $Destination) {
    Add-Type -AssemblyName System.IO.Compression
    $bytes = [IO.File]::ReadAllBytes($Archive)
    $offset = -1
    # Match the official setup's appended ZIP discovery: its payload starts at a 512-byte
    # aligned local-file header. Do not execute the installer or load the extracted DLL.
    for ($index = 0; $index -le $bytes.Length - 4; $index += 512) {
        if ($bytes[$index] -eq 0x50 -and $bytes[$index+1] -eq 0x4b -and
            $bytes[$index+2] -eq 3 -and $bytes[$index+3] -eq 4) { $offset = $index; break }
    }
    if ($offset -lt 0) { throw 'The verified ReShade setup has no appended ZIP payload.' }
    $memory = [IO.MemoryStream]::new($bytes, $offset, $bytes.Length - $offset, $false)
    $zip = $null
    try {
        $zip = [IO.Compression.ZipArchive]::new($memory, [IO.Compression.ZipArchiveMode]::Read)
        $entries = @($zip.Entries | Where-Object { $_.FullName -ceq 'ReShade64.dll' })
        if ($entries.Count -ne 1) { throw 'ReShade setup must contain exactly one ReShade64.dll.' }
        $input = $entries[0].Open()
        $output = $null
        try {
            $output = [IO.File]::Open($Destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
            $input.CopyTo($output)
            $output.Flush($true)
        } finally {
            if ($output) { $output.Dispose() }
            $input.Dispose()
        }
    } finally {
        if ($zip) { $zip.Dispose() }
        $memory.Dispose()
    }
    Assert-Native64Pe $Destination $true
    $version = [Diagnostics.FileVersionInfo]::GetVersionInfo($Destination)
    if ($version.ProductName -cne 'ReShade' -or $version.OriginalFilename -cne 'ReShade64.dll' -or
        $version.FileMajorPart -ne 6 -or $version.FileMinorPart -ne 8 -or $version.FileBuildPart -ne 0) {
        throw 'Extracted payload is not the expected ReShade 6.8.0 x64 runtime.'
    }
}

$game = Get-Item -LiteralPath $GameExecutable -ErrorAction Stop
if ($game.PSIsContainer -or $game.Extension -ine '.exe') { throw 'GameExecutable must name the actual game EXE.' }
Assert-Native64Pe $game.FullName $false
$gameDirectory = $game.DirectoryName
Assert-FreshTarget $gameDirectory
Assert-CompatibleGame $game.Name
$archive = Get-PinnedArchive
$runtimeDestination = Join-Path $gameDirectory 'dxgi.dll'
$configurationDestination = Join-Path $gameDirectory 'ReShade.ini'
$runtimeTemporary = Join-Path $gameDirectory ('Sunshine3D-runtime-' + [guid]::NewGuid().ToString('N') + '.tmp')
$configurationTemporary = Join-Path $gameDirectory ('Sunshine3D-config-' + [guid]::NewGuid().ToString('N') + '.tmp')
$created = [Collections.Generic.List[object]]::new()
try {
    Write-RuntimePayload $archive $runtimeTemporary
    $configuration = "[ADDON]`r`nAddonPath=.`r`n`r`n[SUNSHINE_GAME3D]`r`nEnabled=1`r`n"
    $configurationBytes = [Text.UTF8Encoding]::new($false).GetBytes($configuration)
    $configurationFile = [IO.File]::Open($configurationTemporary, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $configurationFile.Write($configurationBytes, 0, $configurationBytes.Length); $configurationFile.Flush($true) }
    finally { $configurationFile.Dispose() }
    $runtimeHash = (Get-FileHash -LiteralPath $runtimeTemporary -Algorithm SHA256).Hash
    $configurationHash = (Get-FileHash -LiteralPath $configurationTemporary -Algorithm SHA256).Hash
    Assert-FreshTarget $gameDirectory
    # Publish the config before the proxy so a next launch cannot load our runtime with a
    # half-written configuration. File.Move is create-new on Windows PowerShell/.NET Framework.
    [IO.File]::Move($configurationTemporary, $configurationDestination)
    $created.Add(@{Path=$configurationDestination; Hash=$configurationHash})
    [IO.File]::Move($runtimeTemporary, $runtimeDestination)
    $created.Add(@{Path=$runtimeDestination; Hash=$runtimeHash})
    if ((Get-FileHash -LiteralPath $runtimeDestination -Algorithm SHA256).Hash -cne $runtimeHash -or
        (Get-FileHash -LiteralPath $configurationDestination -Algorithm SHA256).Hash -cne $configurationHash) {
        throw 'ReShade publication verification failed.'
    }
} catch {
    $failure = $_
    $restoreFailures = [Collections.Generic.List[string]]::new()
    foreach ($entry in $created) {
        try {
            if (Test-Path -LiteralPath $entry.Path -PathType Leaf) {
                if ((Get-FileHash -LiteralPath $entry.Path -Algorithm SHA256).Hash -cne $entry.Hash) {
                    throw 'The newly created file was changed externally; preserving it.'
                }
                Remove-Item -LiteralPath $entry.Path -Force
            }
        } catch { $restoreFailures.Add($entry.Path + ': ' + $_.Exception.Message) }
    }
    if ($restoreFailures.Count) { throw "ReShade setup failed: $($failure.Exception.Message). Rollback: $($restoreFailures -join '; ')" }
    throw $failure
} finally {
    foreach ($temporary in @($runtimeTemporary, $configurationTemporary)) {
        if (Test-Path -LiteralPath $temporary -PathType Leaf) { Remove-Item -LiteralPath $temporary -Force }
    }
}
Write-Output "Installed official ReShade 6.8.0 with full add-on support for $($game.Name)."
Write-Output 'ReShade will load on the next game launch. Install the Sunshine 3D add-on before restarting the game.'
