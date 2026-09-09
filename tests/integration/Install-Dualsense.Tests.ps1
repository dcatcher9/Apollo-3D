# Run with Windows PowerShell 5.1 and its bundled Pester 3:
# Invoke-Pester tests/integration/Install-Dualsense.Tests.ps1 -EnableExit
# Every install/download action below is mocked. The only executed child is a
# freshly compiled harmless environment probe; no vendor drivers are installed.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repo 'src_assets\windows\misc\client-drivers\install-dualsense.ps1')

function New-Ds5TestArchive([string]$Path, [string]$Core = 'fixture core') {
    Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
    $stream = [IO.File]::Create($Path)
    $archive = New-Object IO.Compression.ZipArchive($stream, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($item in @(@('HIDMaestro.Core.dll', $Core), @('../../escaped.txt', 'must not extract'), @('sdk/signtool.exe', 'must not extract'))) {
            $entry = $archive.CreateEntry($item[0])
            $writer = New-Object IO.StreamWriter($entry.Open())
            try { $writer.Write($item[1]) } finally { $writer.Dispose() }
        }
    } finally { $archive.Dispose(); $stream.Dispose() }
}

function Save-Ds5TestManifest {
    $script:FixtureManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $script:Payload 'component.json') -Encoding UTF8
}

Describe 'Optional DualSense bootstrap transaction' {
    BeforeEach {
        $script:Installation = Join-Path $TestDrive ([Guid]::NewGuid().ToString('N'))
        $script:Component = Join-Path $script:Installation 'tools\sunshine-ds5-component'
        $script:Payload = Join-Path $script:Component 'payload'
        $script:Logs = Join-Path $script:Installation 'config\driver-setup'
        New-Item -ItemType Directory -Path $script:Payload -Force | Out-Null
        [IO.File]::WriteAllText((Join-Path $script:Payload 'Sunshine.Ds5Sidecar.exe'), 'fixture apphost')
        [IO.File]::WriteAllText((Join-Path $script:Payload 'Sunshine.Ds5Sidecar.dll'), 'fixture managed assembly')
        $files = [ordered]@{}
        foreach ($file in Get-ChildItem -LiteralPath $script:Payload -File) {
            $files[$file.Name] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        $files['HIDMaestro.Core.dll'] = $script:Ds5CoreSha256
        $script:FixtureManifest = [ordered]@{
            protocol = 1; installer_driver_setup = 1; sidecar_file = 'Sunshine.Ds5Sidecar.exe'
            sidecar_sha256 = $files['Sunshine.Ds5Sidecar.exe']; hidmaestro_version = '1.6.2.0'
            hidmaestro_archive_url = $script:Ds5ArchiveUrl; hidmaestro_archive_sha256 = $script:Ds5ArchiveSha256
            files = $files
        }
        Save-Ds5TestManifest
        $script:Archive = Join-Path $script:Installation 'fixture.zip'
        New-Ds5TestArchive $script:Archive
        $script:FixtureArchiveHash = (Get-FileHash -LiteralPath $script:Archive -Algorithm SHA256).Hash.ToLowerInvariant()
        $coreBytes = [Text.Encoding]::UTF8.GetBytes('fixture core')
        $sha = [Security.Cryptography.SHA256]::Create()
        try { $script:FixtureCoreHash = ([BitConverter]::ToString($sha.ComputeHash($coreBytes))).Replace('-', '').ToLowerInvariant() } finally { $sha.Dispose() }
        $script:HelperCalls = New-Object 'Collections.Generic.List[object]'
        Mock Assert-Ds5Administrator {}
        Mock Assert-Ds5ProtectedPath {}
        Mock New-Ds5PrivateDirectory {
            param($Path)
            $full = Get-Ds5Path $Path
            if (Test-Path -LiteralPath $full) { throw 'Test staging directory already exists.' }
            New-Item -ItemType Directory -Path $full | Out-Null
            return $full
        }
        # Cryptographic fixtures replace the two externally pinned payloads only.
        # Every helper/manifest file and every altered fixture uses its real hash.
        Mock Get-Ds5Sha256 {
            param($Path)
            $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -eq $script:FixtureArchiveHash) { return $script:Ds5ArchiveSha256 }
            if ($actual -eq $script:FixtureCoreHash) { return $script:Ds5CoreSha256 }
            return $actual
        }
        Mock Invoke-Ds5Download { param($Destination) [IO.File]::Copy($script:Archive, $Destination, $false) }
        Mock Move-Ds5Directory { param($Source, $Destination, $Root) [IO.Directory]::Move($Source, $Destination) }
        Mock Invoke-Ds5Helper {
            param($Executable, $Mode, $PrivateTemp, $LogFile)
            $script:HelperCalls.Add(@{ Executable = $Executable; Mode = $Mode; Temp = $PrivateTemp })
            if ($Mode -eq '--probe-host') { return 0x43 }
            return 0
        }
    }

    It 'downloads the pinned archive, extracts only Core, and activates after install plus readiness' {
        $result = Install-Ds5Component $script:Payload $script:Component $script:Logs
        $result.ExitCode | Should Be 0
        $active = Join-Path $script:Component 'active'
        @(Get-ChildItem -LiteralPath $active).Count | Should Be 4
        [IO.File]::ReadAllText((Join-Path $active 'HIDMaestro.Core.dll')) | Should Be 'fixture core'
        (Test-Path -LiteralPath (Join-Path $script:Component 'escaped.txt')) | Should Be $false
        $script:HelperCalls.Count | Should Be 2
        $script:HelperCalls[0].Mode | Should Be '--install-drivers'
        $script:HelperCalls[1].Mode | Should Be '--probe-host'
        ([IO.Path]::GetDirectoryName($script:HelperCalls[0].Temp)) | Should Be ([IO.Path]::GetDirectoryName([IO.Path]::GetDirectoryName($script:HelperCalls[0].Executable)))
        ([IO.Path]::GetFileName($script:HelperCalls[0].Temp)) | Should Be 'temp'
        @(Get-ChildItem -LiteralPath $script:Component -Filter 'install-*').Count | Should Be 0
        Assert-MockCalled Invoke-Ds5Download -Times 1 -Exactly -Scope It
    }

    It 'accepts a verified offline archive without downloading' {
        $result = Install-Ds5Component $script:Payload $script:Component $script:Logs $script:Archive
        $result.ExitCode | Should Be 0
        Assert-MockCalled Invoke-Ds5Download -Times 0 -Exactly -Scope It
    }

    It 'rejects a streaming-only helper before downloading or executing anything' {
        $script:FixtureManifest.Remove('installer_driver_setup')
        Save-Ds5TestManifest
        (Install-Ds5Component $script:Payload $script:Component $script:Logs).ExitCode | Should Be 1
        Assert-MockCalled Invoke-Ds5Download -Times 0 -Exactly -Scope It
        Assert-MockCalled Invoke-Ds5Helper -Times 0 -Exactly -Scope It
    }

    It 'rejects changed release pins instead of trusting new hashes supplied in the manifest' {
        $script:FixtureManifest.hidmaestro_archive_sha256 = '0' * 64
        Save-Ds5TestManifest
        (Install-Ds5Component $script:Payload $script:Component $script:Logs).ExitCode | Should Be 1
        Assert-MockCalled Invoke-Ds5Download -Times 0 -Exactly -Scope It
        Assert-MockCalled Invoke-Ds5Helper -Times 0 -Exactly -Scope It
    }

    It 'rejects tampered managed dependencies and additional payload files before installation' {
        [IO.File]::AppendAllText((Join-Path $script:Payload 'Sunshine.Ds5Sidecar.dll'), 'tamper')
        (Install-Ds5Component $script:Payload $script:Component $script:Logs).ExitCode | Should Be 1
        Assert-MockCalled Invoke-Ds5Helper -Times 0 -Exactly -Scope It
        [IO.File]::WriteAllText((Join-Path $script:Payload 'Sunshine.Ds5Sidecar.dll'), 'fixture managed assembly')
        [IO.File]::WriteAllText((Join-Path $script:Payload 'unlisted.dll'), 'extra')
        (Install-Ds5Component $script:Payload $script:Component $script:Logs).ExitCode | Should Be 1
        Assert-MockCalled Invoke-Ds5Helper -Times 0 -Exactly -Scope It
    }

    It 'refuses to bundle Core in the installer payload' {
        [IO.File]::WriteAllText((Join-Path $script:Payload 'HIDMaestro.Core.dll'), 'fixture core')
        (Install-Ds5Component $script:Payload $script:Component $script:Logs).ExitCode | Should Be 1
        Assert-MockCalled Invoke-Ds5Helper -Times 0 -Exactly -Scope It
    }

    It 'rejects archive and Core corruption before invoking the helper' {
        [IO.File]::AppendAllText($script:Archive, 'tamper')
        (Install-Ds5Component $script:Payload $script:Component $script:Logs $script:Archive).ExitCode | Should Be 1
        Assert-MockCalled Invoke-Ds5Helper -Times 0 -Exactly -Scope It
        Remove-Item -LiteralPath $script:Archive
        New-Ds5TestArchive $script:Archive 'wrong core'
        $script:FixtureArchiveHash = (Get-FileHash -LiteralPath $script:Archive -Algorithm SHA256).Hash.ToLowerInvariant()
        (Install-Ds5Component $script:Payload $script:Component $script:Logs $script:Archive).ExitCode | Should Be 1
        Assert-MockCalled Invoke-Ds5Helper -Times 0 -Exactly -Scope It
    }

    It 'rejects traversal and duplicate keys in the manifest' {
        $script:FixtureManifest.files['../outside.dll'] = '0' * 64
        Save-Ds5TestManifest
        { Read-Ds5Manifest $script:Payload } | Should Throw
        $script:FixtureManifest.files.Remove('../outside.dll')
        Save-Ds5TestManifest
        $path = Join-Path $script:Payload 'component.json'
        $json = [IO.File]::ReadAllText($path).Replace('"protocol":', '"protocol": 1, "protocol":')
        [IO.File]::WriteAllText($path, $json)
        { Read-Ds5Manifest $script:Payload } | Should Throw
    }

    It 'preserves the previous active component on driver setup failure and reports partial system changes' {
        $active = Join-Path $script:Component 'active'
        New-Item -ItemType Directory -Path $active | Out-Null
        [IO.File]::WriteAllText((Join-Path $active 'old.txt'), 'previous component')
        Mock Invoke-Ds5Helper { return 1 }
        $result = Install-Ds5Component $script:Payload $script:Component $script:Logs
        $result.ExitCode | Should Be 1
        $result.Message | Should Match 'may already be partial'
        [IO.File]::ReadAllText((Join-Path $active 'old.txt')) | Should Be 'previous component'
        @(Get-ChildItem -LiteralPath $script:Component -Filter 'previous-*').Count | Should Be 0
    }

    It 'rejects ordinary failures, untagged readiness, and incomplete tagged readiness' {
        Mock Invoke-Ds5Helper { param($Executable, $Mode) if ($Mode -eq '--probe-host') { return $script:ProbeExit }; return 0 }
        foreach ($script:ProbeExit in @(1, 3, 0x41, 0x42)) {
            $result = Install-Ds5Component $script:Payload $script:Component $script:Logs
            $result.ExitCode | Should Be 1
            (Test-Path -LiteralPath (Join-Path $script:Component 'active')) | Should Be $false
            $result.Message | Should Match 'readiness'
        }
    }

    It 'restores the previous directory when activation rename fails' {
        $active = Join-Path $script:Component 'active'
        New-Item -ItemType Directory -Path $active | Out-Null
        [IO.File]::WriteAllText((Join-Path $active 'old.txt'), 'previous component')
        Mock Move-Ds5Directory {
            param($Source, $Destination, $Root)
            if ([IO.Path]::GetFileName($Source) -eq 'component') { throw 'injected activation failure' }
            [IO.Directory]::Move($Source, $Destination)
        }
        $result = Install-Ds5Component $script:Payload $script:Component $script:Logs
        $result.ExitCode | Should Be 1
        $result.Message | Should Match 'injected activation failure'
        [IO.File]::ReadAllText((Join-Path $active 'old.txt')) | Should Be 'previous component'
        Assert-MockCalled Move-Ds5Directory -Times 3 -Exactly -Scope It
    }

    It 'skips installation and download for an identical healthy active component' {
        (Install-Ds5Component $script:Payload $script:Component $script:Logs).ExitCode | Should Be 0
        $script:HelperCalls.Clear()
        $result = Install-Ds5Component $script:Payload $script:Component $script:Logs
        $result.ExitCode | Should Be 0
        $result.Message | Should Match 'already ready'
        $script:HelperCalls.Count | Should Be 1
        $script:HelperCalls[0].Mode | Should Be '--probe-host'
        Assert-MockCalled Invoke-Ds5Download -Times 1 -Exactly -Scope It
    }
}

Describe 'DualSense setup path and process isolation' {
    It 'rejects traversal, alternate streams, and junction paths' {
        { Get-Ds5Path 'C:\Program Files\Sunshine\..\Windows' } | Should Throw
        { Get-Ds5Path 'C:\Program Files\Sunshine\file:stream' } | Should Throw
        $target = Join-Path $TestDrive 'target'
        New-Item -ItemType Directory -Path $target | Out-Null
        $link = Join-Path $TestDrive 'junction'
        New-Item -ItemType Junction -Path $link -Target $target | Out-Null
        try { { Get-Ds5Path (Join-Path $link 'child') } | Should Throw }
        finally { [IO.Directory]::Delete($link) }
        (Test-Path -LiteralPath $target) | Should Be $true
    }

    It 'rejects installation files writable by ordinary users' {
        $script:UntrustedAcl = New-Object Security.AccessControl.DirectorySecurity
        $script:UntrustedAcl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))
        $script:UntrustedAcl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
            (New-Object Security.Principal.SecurityIdentifier('S-1-5-32-545')), 'Modify', 'Allow')))
        Mock Get-Acl { return $script:UntrustedAcl }
        $failure = ''
        try { Assert-Ds5ProtectedPath $TestDrive } catch { $failure = $_.Exception.Message }
        $failure | Should Match 'unelevated identity'
    }

    It 'rejects parent ACLs that allow replacing an otherwise protected subtree' {
        $script:ProtectedAcl = New-Object Security.AccessControl.DirectorySecurity
        $script:ProtectedAcl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))
        $script:ReplaceableAcl = New-Object Security.AccessControl.DirectorySecurity
        $script:ReplaceableAcl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))
        $script:ReplaceableAcl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
            (New-Object Security.Principal.SecurityIdentifier('S-1-5-32-545')), 'DeleteSubdirectoriesAndFiles', 'Allow')))
        $script:ReplaceableParent = $TestDrive
        $leaf = Join-Path $TestDrive 'protected-leaf'
        New-Item -ItemType Directory -Path $leaf | Out-Null
        Mock Get-Acl { param($LiteralPath) if ($LiteralPath -eq $script:ReplaceableParent) { return $script:ReplaceableAcl }; return $script:ProtectedAcl }
        $failure = ''
        try { Assert-Ds5ProtectedPath $leaf } catch { $failure = $_.Exception.Message }
        $failure | Should Match 'replaceable by an unelevated identity'
    }

    It 'executes a harmless fixture with private TEMP and sanitized .NET settings without changing its parent' {
        $directory = Join-Path $TestDrive 'probe'
        $temp = Join-Path $TestDrive 'private-temp'
        New-Item -ItemType Directory -Path $directory, $temp | Out-Null
        $executable = Join-Path $directory 'fixture.exe'
        Add-Type -OutputAssembly $executable -OutputType ConsoleApplication -TypeDefinition @'
using System;
class EnvironmentProbe { public static int Main(string[] args) {
    if (args.Length != 1 || args[0] != "--probe-host") return 99;
    Console.WriteLine("TEMP=" + Environment.GetEnvironmentVariable("TEMP"));
    Console.WriteLine("TMP=" + Environment.GetEnvironmentVariable("TMP"));
    Console.WriteLine("HOOK=" + Environment.GetEnvironmentVariable("DOTNET_STARTUP_HOOKS"));
    Console.WriteLine("PROFILER=" + Environment.GetEnvironmentVariable("COR_ENABLE_PROFILING"));
    Console.WriteLine("MANAGER=" + Environment.GetEnvironmentVariable("APPDOMAIN_MANAGER_ASM"));
    return 0x43;
} }
'@
        $oldTemp = $env:TEMP
        $oldTmp = $env:TMP
        $log = Join-Path $TestDrive 'probe.log'
        $saved = @{}
        foreach ($name in @('DOTNET_STARTUP_HOOKS', 'COR_ENABLE_PROFILING', 'APPDOMAIN_MANAGER_ASM')) {
            $saved[$name] = [Environment]::GetEnvironmentVariable($name)
            [Environment]::SetEnvironmentVariable($name, 'untrusted-user-override')
        }
        try {
            (Invoke-Ds5Helper $executable '--probe-host' $temp $log) | Should Be 0x43
            $output = [IO.File]::ReadAllText($log)
            $output | Should Match ([regex]::Escape('TEMP=' + $temp))
            $output | Should Match ([regex]::Escape('TMP=' + $temp))
            $output | Should Not Match 'untrusted-user-override'
            $env:DOTNET_STARTUP_HOOKS | Should Be 'untrusted-user-override'
        } finally {
            foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
        }
        $env:TEMP | Should Be $oldTemp
        $env:TMP | Should Be $oldTmp
    }
}
