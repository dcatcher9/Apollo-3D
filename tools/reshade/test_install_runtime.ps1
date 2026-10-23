# SPDX-License-Identifier: GPL-3.0-only
# Pester 3/4 filesystem fixtures; never executes a game, vendor installer or runtime DLL.
param(
    [string] $SetupArchive = (Join-Path $PSScriptRoot '..\..\cmake-build-relwithdebinfo\ReShade_Setup_6.8.0_Addon.exe')
)
$ErrorActionPreference = 'Stop'
$script:runtimeInstaller = Join-Path $PSScriptRoot 'install-runtime.ps1'
$script:officialArchive = (Get-Item -LiteralPath $SetupArchive -ErrorAction Stop).FullName
$script:pinnedHash = 'AFE4C8F13048306307983B8B3D41D5BF00A86820440B0E57DEA10950E1176445'
if ((Get-FileHash -LiteralPath $script:officialArchive -Algorithm SHA256).Hash -cne $script:pinnedHash) {
    throw 'Runtime tests require the unmodified pinned official full-add-on archive.'
}

Describe 'Fresh ReShade runtime installation' {
    function New-RuntimeFixture([string] $Name = 'Fixture Game.exe') {
        $root = Join-Path $TestDrive ([guid]::NewGuid().ToString('N') + ' game with spaces')
        [void][IO.Directory]::CreateDirectory($root)
        $game = Join-Path $root $Name
        Copy-Item -LiteralPath (Join-Path $env:SystemRoot 'System32\where.exe') -Destination $game
        $cache = Join-Path $root 'download cache'
        [void][IO.Directory]::CreateDirectory($cache)
        [IO.File]::WriteAllText((Join-Path $cache 'Compatibility.ini'), '[Other Game.exe]')
        return @{Root=$root; Game=$game; Cache=$cache}
    }

    It 'extracts the pinned official native x64 runtime without changing the game' {
        $fixture = New-RuntimeFixture
        $before = (Get-FileHash -LiteralPath $fixture.Game).Hash
        & $script:runtimeInstaller -GameExecutable $fixture.Game -SetupArchive $script:officialArchive -CacheDirectory $fixture.Cache
        (Get-FileHash -LiteralPath $fixture.Game).Hash | Should Be $before
        $runtime = Join-Path $fixture.Root 'dxgi.dll'
        (Test-Path -LiteralPath $runtime -PathType Leaf) | Should Be $true
        ([Diagnostics.FileVersionInfo]::GetVersionInfo($runtime)).ProductName | Should Be 'ReShade'
        ([Diagnostics.FileVersionInfo]::GetVersionInfo($runtime)).OriginalFilename | Should Be 'ReShade64.dll'
        [IO.File]::ReadAllText((Join-Path $fixture.Root 'ReShade.ini')) | Should Match 'Enabled=1'
        @(Get-ChildItem -LiteralPath $fixture.Root -Filter '*.tmp').Count | Should Be 0
    }

    It 'downloads once and reuses the verified persistent cache for another game' {
        $fixture = New-RuntimeFixture
        $second = New-RuntimeFixture
        $archiveForMock = $script:officialArchive
        Mock Invoke-WebRequest ({ param($Uri, $OutFile, $TimeoutSec) Copy-Item -LiteralPath $archiveForMock -Destination $OutFile }.GetNewClosure())
        & $script:runtimeInstaller -GameExecutable $fixture.Game -CacheDirectory $fixture.Cache
        & $script:runtimeInstaller -GameExecutable $second.Game -CacheDirectory $fixture.Cache
        Assert-MockCalled Invoke-WebRequest -Times 1 -Exactly -Scope It -ParameterFilter {
            $Uri -eq 'https://reshade.me/downloads/ReShade_Setup_6.8.0_Addon.exe' -and $TimeoutSec -eq 60
        }
        (Get-FileHash -LiteralPath (Join-Path $fixture.Cache 'ReShade_Setup_6.8.0_Addon.exe')).Hash | Should Be $script:pinnedHash
    }

    It 'redownloads a corrupt cached setup and verifies its replacement' {
        $fixture = New-RuntimeFixture
        [IO.File]::WriteAllText((Join-Path $fixture.Cache 'ReShade_Setup_6.8.0_Addon.exe'), 'CORRUPT CACHE')
        $archiveForMock = $script:officialArchive
        Mock Invoke-WebRequest ({ param($Uri, $OutFile, $TimeoutSec) Copy-Item -LiteralPath $archiveForMock -Destination $OutFile }.GetNewClosure())
        & $script:runtimeInstaller -GameExecutable $fixture.Game -CacheDirectory $fixture.Cache
        Assert-MockCalled Invoke-WebRequest -Times 1 -Exactly -Scope It
        (Get-FileHash -LiteralPath (Join-Path $fixture.Cache 'ReShade_Setup_6.8.0_Addon.exe')).Hash | Should Be $script:pinnedHash
    }

    It 'rejects a substituted explicit setup before creating game files' {
        $fixture = New-RuntimeFixture
        $archive = Join-Path $fixture.Cache 'wrong.exe'
        [IO.File]::WriteAllText($archive, 'NOT THE PINNED OFFICIAL ARCHIVE')
        { & $script:runtimeInstaller -GameExecutable $fixture.Game -SetupArchive $archive -CacheDirectory $fixture.Cache } | Should Throw 'SHA-256'
        (Test-Path -LiteralPath (Join-Path $fixture.Root 'dxgi.dll')) | Should Be $false
        (Test-Path -LiteralPath (Join-Path $fixture.Root 'ReShade.ini')) | Should Be $false
    }

    It 'rejects a substituted download and removes its temporary cache file' {
        $fixture = New-RuntimeFixture
        Mock Invoke-WebRequest { [IO.File]::WriteAllText($OutFile, 'SUBSTITUTED DOWNLOAD') }
        { & $script:runtimeInstaller -GameExecutable $fixture.Game -CacheDirectory $fixture.Cache } | Should Throw 'SHA-256'
        @(Get-ChildItem -LiteralPath $fixture.Cache -Filter '*.tmp').Count | Should Be 0
        (Test-Path -LiteralPath (Join-Path $fixture.Cache 'ReShade_Setup_6.8.0_Addon.exe')) | Should Be $false
        (Test-Path -LiteralPath (Join-Path $fixture.Root 'dxgi.dll')) | Should Be $false
    }

    It 'preserves every existing proxy and configuration without downloading' {
        Mock Invoke-WebRequest { throw 'No download is allowed for an existing installation.' }
        foreach ($name in @('dxgi.dll','d3d11.dll','d3d12.dll','d3d10.dll','d3d10_1.dll','d3d9.dll','opengl32.dll','dinput8.dll','version.dll','ReShade.ini')) {
            $fixture = New-RuntimeFixture
            $existing = Join-Path $fixture.Root $name
            [IO.File]::WriteAllText($existing, 'EXISTING USER CONTENT')
            { & $script:runtimeInstaller -GameExecutable $fixture.Game -CacheDirectory $fixture.Cache } | Should Throw 'Existing'
            [IO.File]::ReadAllText($existing) | Should Be 'EXISTING USER CONTENT'
            @(Get-ChildItem -LiteralPath $fixture.Root -File).Count | Should Be 2
        }
        Assert-MockCalled Invoke-WebRequest -Times 0 -Exactly -Scope It
    }

    It 'installs beside an orphan preset without changing it or existing shaders' {
        $fixture = New-RuntimeFixture 'bf6.exe'
        $preset = Join-Path $fixture.Root 'ReShadePreset.ini'
        $shaderDirectory = Join-Path $fixture.Root 'reshade-shaders\Shaders'
        [void][IO.Directory]::CreateDirectory($shaderDirectory)
        $shader = Join-Path $shaderDirectory 'Curves.fx'
        [IO.File]::WriteAllBytes($preset, [byte[]](0xef,0xbb,0xbf,0x3b,0x20,0x75,0x73,0x65,0x72,0x0d,0x0a))
        [IO.File]::WriteAllText($shader, 'EXISTING USER SHADER')
        $presetHash = (Get-FileHash -LiteralPath $preset).Hash
        $shaderHash = (Get-FileHash -LiteralPath $shader).Hash
        & $script:runtimeInstaller -GameExecutable $fixture.Game -SetupArchive $script:officialArchive -CacheDirectory $fixture.Cache
        (Test-Path -LiteralPath (Join-Path $fixture.Root 'dxgi.dll') -PathType Leaf) | Should Be $true
        (Get-FileHash -LiteralPath $preset).Hash | Should Be $presetHash
        (Get-FileHash -LiteralPath $shader).Hash | Should Be $shaderHash
    }

    It 'rejects directory and reparse preset paths before downloading' {
        Mock Invoke-WebRequest { throw 'No download is allowed for an unsafe preset path.' }
        $directoryFixture = New-RuntimeFixture
        $directoryPreset = Join-Path $directoryFixture.Root 'ReShadePreset.ini'
        [void][IO.Directory]::CreateDirectory($directoryPreset)
        { & $script:runtimeInstaller -GameExecutable $directoryFixture.Game -CacheDirectory $directoryFixture.Cache } | Should Throw 'regular file'
        (Test-Path -LiteralPath (Join-Path $directoryFixture.Root 'dxgi.dll')) | Should Be $false
        $reparseFixture = New-RuntimeFixture
        $reparsePreset = Join-Path $reparseFixture.Root 'ReShadePreset.ini'
        $target = Join-Path $TestDrive ([guid]::NewGuid().ToString('N') + ' preset target')
        [void][IO.Directory]::CreateDirectory($target)
        $sentinel = Join-Path $target 'user.ini'
        [IO.File]::WriteAllText($sentinel, 'EXISTING LINK TARGET')
        New-Item -ItemType Junction -Path $reparsePreset -Target $target | Out-Null
        try {
            ([bool]((Get-Item -LiteralPath $reparsePreset -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) | Should Be $true
            { & $script:runtimeInstaller -GameExecutable $reparseFixture.Game -CacheDirectory $reparseFixture.Cache } | Should Throw 'regular file'
            (Test-Path -LiteralPath (Join-Path $reparseFixture.Root 'dxgi.dll')) | Should Be $false
            [IO.File]::ReadAllText($sentinel) | Should Be 'EXISTING LINK TARGET'
        } finally {
            # Remove the junction itself without traversing its target.
            [IO.Directory]::Delete($reparsePreset)
        }
        Assert-MockCalled Invoke-WebRequest -Times 0 -Exactly -Scope It
    }

    It 'backs up and migrates an orphan preset through the native add-on installer' {
        $fixture = New-RuntimeFixture 'bf6.exe'
        $preset = Join-Path $fixture.Root 'ReShadePreset.ini'
        $presetBefore = "Techniques=Curves@Curves.fx,SuperDepth3D@SuperDepth3D.fx`r`nPreprocessorDefinitions=USER_MACRO=7`r`n`r`n[SuperDepth3D.fx]`r`nDepth_Adjustment=37.25`r`nDepth_Map_View=2`r`n`r`n[Curves.fx]`r`nContrast=0.42`r`n"
        [IO.File]::WriteAllText($preset, $presetBefore, [Text.UTF8Encoding]::new($false))
        $presetHash = (Get-FileHash -LiteralPath $preset).Hash
        $gameHash = (Get-FileHash -LiteralPath $fixture.Game).Hash
        $package = Join-Path $TestDrive ([guid]::NewGuid().ToString('N') + ' addon package')
        [void][IO.Directory]::CreateDirectory($package)
        [IO.File]::WriteAllText((Join-Path $package 'SunshineSBS.addon64'), 'UNEXECUTED ADDON FIXTURE')
        [IO.File]::WriteAllText((Join-Path $package 'SunshineSBS-LICENSES.txt'), 'ADDON LICENSE FIXTURE')
        & $script:runtimeInstaller -GameExecutable $fixture.Game -SetupArchive $script:officialArchive -CacheDirectory $fixture.Cache
        (Get-FileHash -LiteralPath $preset).Hash | Should Be $presetHash
        & (Join-Path $PSScriptRoot 'install.ps1') -GameExecutable $fixture.Game -AddonDirectory $package
        $presetAfter = [IO.File]::ReadAllText($preset)
        $presetAfter.Replace('Techniques=Curves@Curves.fx', 'Techniques=Curves@Curves.fx,SuperDepth3D@SuperDepth3D.fx') | Should Be $presetBefore
        $configuration = [IO.File]::ReadAllText((Join-Path $fixture.Root 'ReShade.ini'))
        $configuration | Should Match 'Strength=37\.25'
        $configuration | Should Match 'DepthView=2'
        (Get-FileHash -LiteralPath $fixture.Game).Hash | Should Be $gameHash
        (Get-FileHash -LiteralPath (Join-Path $fixture.Root 'SunshineSBS.addon64')).Hash | Should Be (Get-FileHash -LiteralPath (Join-Path $package 'SunshineSBS.addon64')).Hash
        $manifests = @(Get-ChildItem -LiteralPath (Join-Path $fixture.Root 'Sunshine3D-backups') -Filter manifest.json -Recurse)
        $manifests.Count | Should Be 1
        $manifest = [IO.File]::ReadAllText($manifests[0].FullName) | ConvertFrom-Json
        $presetBackup = @($manifest | Where-Object { $_.path -eq $preset })
        $presetBackup.Count | Should Be 1
        $presetBackup[0].existed | Should Be $true
        (Get-FileHash -LiteralPath $presetBackup[0].backup).Hash | Should Be $presetHash
    }

    It 'honors official banned-game restrictions even in cached compatibility data' {
        $fixture = New-RuntimeFixture
        [IO.File]::WriteAllText((Join-Path $fixture.Cache 'Compatibility.ini'), "[fixture game.EXE]`nBanned=1`n")
        { & $script:runtimeInstaller -GameExecutable $fixture.Game -SetupArchive $script:officialArchive -CacheDirectory $fixture.Cache } | Should Throw 'Banned=1'
        (Test-Path -LiteralPath (Join-Path $fixture.Root 'dxgi.dll')) | Should Be $false
    }

    It 'does not redirect installation using remote compatibility paths' {
        $fixture = New-RuntimeFixture
        [IO.File]::WriteAllText((Join-Path $fixture.Cache 'Compatibility.ini'), "[Fixture Game.exe]`nInstallTarget=..\Elsewhere`n")
        { & $script:runtimeInstaller -GameExecutable $fixture.Game -SetupArchive $script:officialArchive -CacheDirectory $fixture.Cache } | Should Throw 'different installation layout'
        (Test-Path -LiteralPath (Join-Path $fixture.Root 'dxgi.dll')) | Should Be $false
    }

    It 'rejects incompatible official render APIs' {
        $fixture = New-RuntimeFixture
        [IO.File]::WriteAllText((Join-Path $fixture.Cache 'Compatibility.ini'), "[Fixture Game.exe]`nRenderApi=Vulkan`n")
        { & $script:runtimeInstaller -GameExecutable $fixture.Game -SetupArchive $script:officialArchive -CacheDirectory $fixture.Cache } | Should Throw 'unsupported'
    }

    It 'preserves games whose official compatibility requires a specific graphics proxy' {
        $fixture = New-RuntimeFixture
        $before = (Get-FileHash -LiteralPath $fixture.Game).Hash
        [IO.File]::WriteAllText((Join-Path $fixture.Cache 'Compatibility.ini'), "[Fixture Game.exe]`nForceInstallApi=1`nRenderApi=D3D12`n")
        { & $script:runtimeInstaller -GameExecutable $fixture.Game -SetupArchive $script:officialArchive -CacheDirectory $fixture.Cache } | Should Throw 'specific graphics API proxy'
        (Get-FileHash -LiteralPath $fixture.Game).Hash | Should Be $before
        @(Get-ChildItem -LiteralPath $fixture.Root -File).Count | Should Be 1
    }

    It 'retains cached banned restrictions when compatibility refresh fails' {
        $fixture = New-RuntimeFixture
        $compatibility = Join-Path $fixture.Cache 'Compatibility.ini'
        [IO.File]::WriteAllText($compatibility, "[Fixture Game.exe]`nBanned=1`n")
        (Get-Item -LiteralPath $compatibility).LastWriteTimeUtc = [DateTime]::UtcNow.AddDays(-2)
        Mock Invoke-WebRequest { throw 'OFFLINE FIXTURE' }
        { & $script:runtimeInstaller -GameExecutable $fixture.Game -SetupArchive $script:officialArchive -CacheDirectory $fixture.Cache -WarningAction SilentlyContinue } | Should Throw 'Banned=1'
        Assert-MockCalled Invoke-WebRequest -Times 1 -Exactly -Scope It -ParameterFilter { $TimeoutSec -eq 10 }
    }

    It 'rejects non-native executable fixtures before setup' {
        $fixture = New-RuntimeFixture
        [IO.File]::WriteAllText($fixture.Game, 'NOT A WINDOWS PE FILE')
        { & $script:runtimeInstaller -GameExecutable $fixture.Game -SetupArchive $script:officialArchive -CacheDirectory $fixture.Cache } | Should Throw 'PE image'
        (Test-Path -LiteralPath (Join-Path $fixture.Root 'dxgi.dll')) | Should Be $false
    }

    It 'rolls back its own published files after publication verification fails' {
        $fixture = New-RuntimeFixture
        $preset = Join-Path $fixture.Root 'ReShadePreset.ini'
        [IO.File]::WriteAllText($preset, 'EXISTING USER PRESET')
        $presetHash = (Get-FileHash -LiteralPath $preset).Hash
        $originalHash = Get-Command Get-FileHash
        $failureState = @{Injected=$false}
        Mock Get-FileHash ({
            param($LiteralPath, $Algorithm)
            if ([IO.Path]::GetFileName($LiteralPath) -eq 'dxgi.dll' -and -not $failureState.Injected) {
                $failureState.Injected = $true
                throw 'SIMULATED PUBLICATION READ FAILURE'
            }
            & $originalHash -LiteralPath $LiteralPath -Algorithm SHA256
        }.GetNewClosure())
        { & $script:runtimeInstaller -GameExecutable $fixture.Game -SetupArchive $script:officialArchive -CacheDirectory $fixture.Cache } | Should Throw 'SIMULATED PUBLICATION READ FAILURE'
        $failureState.Injected | Should Be $true
        (Test-Path -LiteralPath (Join-Path $fixture.Root 'dxgi.dll')) | Should Be $false
        (Test-Path -LiteralPath (Join-Path $fixture.Root 'ReShade.ini')) | Should Be $false
        (Get-FileHash -LiteralPath $preset).Hash | Should Be $presetHash
        @(Get-ChildItem -LiteralPath $fixture.Root -Filter '*.tmp').Count | Should Be 0
        @(Get-ChildItem -LiteralPath $fixture.Root -File).Count | Should Be 2
    }
}
