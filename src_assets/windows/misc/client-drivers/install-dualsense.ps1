[CmdletBinding()]
param(
    [string]$PayloadDirectory,
    [string]$ComponentRoot,
    [string]$LogDirectory,
    [string]$HIDMaestroArchive = ''
)

# Windows PowerShell 5.1. Dot-sourcing exposes the functions for isolated tests;
# normal installer invocation always executes the entry point below.
$ErrorActionPreference = 'Stop'
$script:Ds5ArchiveUrl = 'https://github.com/hifihedgehog/HIDMaestro/releases/download/v1.6.2/HIDMaestro-v1.6.2.zip'
$script:Ds5ArchiveSha256 = '6ae8df0cf317baf7e65777e2929f618916a67831b5ff1162205310f2c08b80ff'
$script:Ds5CoreSha256 = 'df59cd5b6031311f03031c32b794901fc1f45503410f4f6ad55eaa854cee8746'
$script:Ds5ArchiveBytes = 118881819
# --probe-host tags readiness with 0x40 so ordinary exit failures cannot be
# mistaken for HID (1) or audio (2) support. Both ready is 0x43, not bare 3.
$script:Ds5ReadyExitCode = 0x43

function Get-Ds5Path([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path) -or $Path -notmatch '^[A-Za-z]:[\\/]' -or
        $Path.Substring(2).Contains(':') -or $Path -match '(^|[\\/])\.{1,2}([\\/]|$)') {
        throw "A local absolute path without traversal or alternate streams is required: $Path"
    }
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    if ($full.Length -lt 4) { throw 'A drive root is not an installation directory.' }
    $cursor = $full
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            if ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse points are not permitted in driver setup paths: $cursor"
            }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $full
}

function Assert-Ds5ProtectedPath([string]$Path) {
    $full = Get-Ds5Path $Path
    $trusted = @('S-1-5-18', 'S-1-5-32-544', 'S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464')
    $replacementRights = [Security.AccessControl.FileSystemRights]::Delete -bor
        [Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles -bor
        [Security.AccessControl.FileSystemRights]::ChangePermissions -bor
        [Security.AccessControl.FileSystemRights]::TakeOwnership
    $cursor = $full
    while ($cursor) {
        $acl = Get-Acl -LiteralPath $cursor
        if ($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -notin $trusted) {
            throw "Driver setup requires administrator-owned paths: $cursor"
        }
        # Drive roots normally allow creating unrelated children. Ancestors
        # must still prevent replacement of this protected installation subtree.
        $unsafeRights = $replacementRights
        if ($cursor -eq $full) { $unsafeRights = $unsafeRights -bor [Security.AccessControl.FileSystemRights]::Write }
        foreach ($rule in $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
            if ($rule.AccessControlType -eq [Security.AccessControl.AccessControlType]::Allow -and
                -not ($rule.PropagationFlags -band [Security.AccessControl.PropagationFlags]::InheritOnly) -and
                ($rule.FileSystemRights -band $unsafeRights) -and $rule.IdentityReference.Value -notin $trusted) {
                throw "Driver setup refuses a path writable or replaceable by an unelevated identity: $cursor"
            }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
}

function Assert-Ds5Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'DualSense driver setup must be launched by the elevated Sunshine installer.'
    }
}

function New-Ds5PrivateDirectory([string]$Path) {
    $full = Get-Ds5Path $Path
    if (Test-Path -LiteralPath $full) { throw "Staging directory already exists: $full" }
    Assert-Ds5ProtectedPath ([IO.Path]::GetDirectoryName($full))
    $acl = New-Object Security.AccessControl.DirectorySecurity
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))
    foreach ($sid in @('S-1-5-18', 'S-1-5-32-544')) {
        $identity = New-Object Security.Principal.SecurityIdentifier($sid)
        $rule = New-Object Security.AccessControl.FileSystemAccessRule($identity, 'FullControl', 'ContainerInherit, ObjectInherit', 'None', 'Allow')
        $acl.AddAccessRule($rule)
    }
    # Apply the protected DACL and owner atomically at creation, including when
    # the parent has an inherited child-only write rule.
    [IO.Directory]::CreateDirectory($full, $acl) | Out-Null
    Assert-Ds5ProtectedPath $full
    return $full
}

function Get-Ds5Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Read-Ds5Manifest([string]$Directory) {
    $path = Join-Path (Get-Ds5Path $Directory) 'component.json'
    Assert-Ds5ProtectedPath $path
    if ((Get-Item -LiteralPath $path).Length -gt 262144) { throw 'Component manifest is too large.' }
    $json = [IO.File]::ReadAllText($path)
    $manifest = $json | ConvertFrom-Json
    $keys = @('protocol', 'installer_driver_setup', 'sidecar_file', 'sidecar_sha256', 'hidmaestro_version', 'hidmaestro_archive_url', 'hidmaestro_archive_sha256', 'files')
    $actualKeys = @($manifest.PSObject.Properties.Name)
    if ($actualKeys.Count -ne $keys.Count -or @($actualKeys | Where-Object { $_ -notin $keys }).Count -ne 0 -or
        $manifest.protocol -ne 1 -or $manifest.installer_driver_setup -ne 1 -or
        $manifest.sidecar_file -cne 'Sunshine.Ds5Sidecar.exe' -or $manifest.hidmaestro_version -cne '1.6.2.0' -or
        $manifest.hidmaestro_archive_url -cne $script:Ds5ArchiveUrl -or
        $manifest.hidmaestro_archive_sha256 -cne $script:Ds5ArchiveSha256) {
        throw 'Component manifest does not identify the supported installer-capable helper and pinned HIDMaestro release.'
    }
    $files = @{}
    foreach ($property in $manifest.files.PSObject.Properties) {
        $name = $property.Name
        if ($name -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$' -or $name.EndsWith('.') -or
            $name -match '^(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(\.|$)' -or
            $name -ieq 'component.json' -or $files.ContainsKey($name) -or
            $property.Value -isnot [string] -or $property.Value -cnotmatch '^[0-9a-f]{64}$') {
            throw "Invalid or duplicate flat component manifest entry: $name"
        }
        $files.Add($name, $property.Value)
    }
    if ($files.Count -lt 3 -or $files.Count -gt 512 -or
        $files['HIDMaestro.Core.dll'] -cne $script:Ds5CoreSha256 -or
        $files['Sunshine.Ds5Sidecar.exe'] -cne $manifest.sidecar_sha256 -or
        -not $files.ContainsKey('Sunshine.Ds5Sidecar.dll')) {
        throw 'The full component manifest must include the pinned Core and helper assemblies.'
    }
    # All allowed values above are simple version/hash/filename strings. Reject
    # duplicate JSON property spellings too, before different readers disagree.
    $seen = @{}
    foreach ($match in [regex]::Matches($json, '"((?:\\.|[^"\\])*)"\s*:')) {
        $name = ('"' + $match.Groups[1].Value + '"') | ConvertFrom-Json
        if ($seen.ContainsKey($name)) { throw "Duplicate JSON property in component manifest: $name" }
        $seen[$name] = $true
    }
    return @{ Files = $files; Hash = (Get-Ds5Sha256 $path) }
}

function Assert-Ds5Component([string]$Directory, $Manifest, [switch]$Payload) {
    $directory = Get-Ds5Path $Directory
    Assert-Ds5ProtectedPath $directory
    $expected = @{} + $Manifest.Files
    if ($Payload) { $expected.Remove('HIDMaestro.Core.dll') }
    $expected['component.json'] = $Manifest.Hash
    $entries = @(Get-ChildItem -LiteralPath $directory -Force)
    if ($entries.Count -ne $expected.Count) { throw 'Component has missing or additional files.' }
    foreach ($entry in $entries) {
        $null = Get-Ds5Path $entry.FullName
        if ($entry.PSIsContainer -or -not $expected.ContainsKey($entry.Name)) { throw "Unexpected component entry: $($entry.Name)" }
        Assert-Ds5ProtectedPath $entry.FullName
        if ((Get-Ds5Sha256 $entry.FullName) -cne $expected[$entry.Name]) { throw "Component hash mismatch: $($entry.Name)" }
    }
}

function Invoke-Ds5Download([string]$Destination) {
    Add-Type -AssemblyName System.Net.Http
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $client = New-Object Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromSeconds(180)
    $client.MaxResponseContentBufferSize = 134217728
    try {
        $bytes = $client.GetByteArrayAsync($script:Ds5ArchiveUrl).GetAwaiter().GetResult()
        if ($bytes.Length -ne $script:Ds5ArchiveBytes) { throw 'Official HIDMaestro download has an unexpected size.' }
        $file = [IO.File]::Open($Destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try { $file.Write($bytes, 0, $bytes.Length) } finally { $file.Dispose() }
    } finally { $client.Dispose() }
}

function Expand-Ds5PinnedCore([string]$Archive, [string]$Directory) {
    if ((Get-Ds5Sha256 $Archive) -cne $script:Ds5ArchiveSha256) { throw 'HIDMaestro archive SHA-256 does not match the pinned official release.' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        $entries = @($zip.Entries | Where-Object { $_.FullName -ieq 'HIDMaestro.Core.dll' })
        if ($entries.Count -ne 1 -or $entries[0].FullName -cne 'HIDMaestro.Core.dll' -or $entries[0].Length -gt 134217728) {
            throw 'Pinned archive must contain exactly one root HIDMaestro.Core.dll.'
        }
        # Never expand the full ZIP: drivers/signing tools are owned by the SDK,
        # and arbitrary archive paths must not reach the installation filesystem.
        $core = Join-Path $Directory 'HIDMaestro.Core.dll'
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entries[0], $core)
    } finally { $zip.Dispose() }
    if ((Get-Ds5Sha256 $core) -cne $script:Ds5CoreSha256) { throw 'Extracted HIDMaestro Core SHA-256 does not match the pinned assembly.' }
}

function Initialize-Ds5ProcessJob {
    if ('SunshineDriverSetup.ProcessJob' -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
namespace SunshineDriverSetup {
    public sealed class ProcessJob : IDisposable {
        [StructLayout(LayoutKind.Sequential)] struct BasicLimits {
            public long ProcessTime, JobTime; public uint Flags;
            public UIntPtr MinimumWorkingSet, MaximumWorkingSet; public uint ActiveProcesses;
            public UIntPtr Affinity; public uint Priority, Scheduling;
        }
        [StructLayout(LayoutKind.Sequential)] struct IoCounters { public ulong A, B, C, D, E, F; }
        [StructLayout(LayoutKind.Sequential)] struct ExtendedLimits {
            public BasicLimits Basic; public IoCounters Io;
            public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
        }
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetInformationJobObject(IntPtr job, int info, IntPtr value, uint size);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateJobObject(IntPtr job, uint code);
        [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
        IntPtr handle;
        public ProcessJob() {
            handle = CreateJobObject(IntPtr.Zero, null);
            if (handle == IntPtr.Zero) throw new Win32Exception();
            var limits = new ExtendedLimits(); limits.Basic.Flags = 0x2000;
            int size = Marshal.SizeOf(limits); IntPtr memory = Marshal.AllocHGlobal(size);
            try {
                Marshal.StructureToPtr(limits, memory, false);
                if (!SetInformationJobObject(handle, 9, memory, (uint)size)) throw new Win32Exception();
            } catch { Dispose(); throw; } finally { Marshal.FreeHGlobal(memory); }
        }
        public void Attach(IntPtr process) { if (!AssignProcessToJobObject(handle, process)) throw new Win32Exception(); }
        public void Stop() { if (handle != IntPtr.Zero && !TerminateJobObject(handle, 1)) throw new Win32Exception(); }
        public void Dispose() { if (handle != IntPtr.Zero) { CloseHandle(handle); handle = IntPtr.Zero; } }
    }
}
'@
}

function Invoke-Ds5Helper([string]$Executable, [ValidateSet('--install-drivers', '--probe-host')][string]$Mode,
    [string]$PrivateTemp, [string]$LogFile) {
    Initialize-Ds5ProcessJob
    $job = New-Object SunshineDriverSetup.ProcessJob
    $process = New-Object Diagnostics.Process
    $process.StartInfo = New-Object Diagnostics.ProcessStartInfo
    $process.StartInfo.FileName = $Executable
    $process.StartInfo.Arguments = $Mode
    $process.StartInfo.WorkingDirectory = [IO.Path]::GetDirectoryName($Executable)
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    # A verified .NET payload must not load a user-selected startup hook,
    # profiler, additional dependency store, or executable from the user's PATH.
    foreach ($name in @($process.StartInfo.EnvironmentVariables.Keys)) {
        if ($name -match '^(DOTNET_|COMPlus_|CORECLR_|COR_)' -or
            $name -in @('APPDOMAIN_MANAGER_ASM', 'APPDOMAIN_MANAGER_TYPE', 'DEVPATH', '__COMPAT_LAYER')) {
            $process.StartInfo.EnvironmentVariables.Remove($name)
        }
    }
    $system = [Environment]::GetFolderPath([Environment+SpecialFolder]::System)
    $windows = [Environment]::GetFolderPath([Environment+SpecialFolder]::Windows)
    $powershell = Join-Path $system 'WindowsPowerShell\v1.0'
    $process.StartInfo.EnvironmentVariables['PATH'] = "$system;$windows;$powershell"
    $process.StartInfo.EnvironmentVariables['PSModulePath'] = Join-Path $powershell 'Modules'
    $process.StartInfo.EnvironmentVariables['COMSPEC'] = Join-Path $system 'cmd.exe'
    $process.StartInfo.EnvironmentVariables['SystemRoot'] = $windows
    $process.StartInfo.EnvironmentVariables['windir'] = $windows
    # HIDMaestro's prewarm cache trusts an existing TEMP directory. Never allow
    # an elevated installer to consume a cache planted in the user's TEMP.
    $process.StartInfo.EnvironmentVariables['TEMP'] = $PrivateTemp
    $process.StartInfo.EnvironmentVariables['TMP'] = $PrivateTemp
    $started = $false
    try {
        $started = $process.Start()
        if (-not $started) { throw 'Unable to start the verified DualSense helper.' }
        $job.Attach($process.Handle)
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $timeout = if ($Mode -eq '--install-drivers') { 300000 } else { 30000 }
        if (-not $process.WaitForExit($timeout)) {
            $job.Stop()
            $null = $process.WaitForExit(5000)
            throw "DualSense helper exceeded its bounded $Mode timeout. Driver changes may be partial."
        }
        if (-not $stdout.Wait(5000) -or -not $stderr.Wait(5000)) { throw 'DualSense helper output did not close after exit.' }
        [IO.File]::AppendAllText($LogFile, "$Mode exit=$($process.ExitCode)`r`n$($stdout.Result)`r`n$($stderr.Result)`r`n")
        return $process.ExitCode
    } finally {
        $job.Dispose()
        if ($started -and -not $process.HasExited) {
            $process.Kill()
            $null = $process.WaitForExit(5000)
        }
        $process.Dispose()
    }
}

function Remove-Ds5WorkDirectory([string]$Path, [string]$Root) {
    $full = Get-Ds5Path $Path
    $root = Get-Ds5Path $Root
    if ([IO.Path]::GetDirectoryName($full) -ine $root -or [IO.Path]::GetFileName($full) -notmatch '^install-[0-9a-f]{32}$') {
        throw 'Refusing to clean an unexpected driver staging path.'
    }
    $pending = New-Object 'Collections.Generic.Queue[string]'
    $pending.Enqueue($full)
    $count = 0
    while ($pending.Count) {
        foreach ($entry in Get-ChildItem -LiteralPath $pending.Dequeue() -Force) {
            if (++$count -gt 10000) { throw 'Driver staging cleanup exceeded its bounded entry count.' }
            $null = Get-Ds5Path $entry.FullName
            if ($entry.PSIsContainer) { $pending.Enqueue($entry.FullName) }
        }
    }
    Remove-Item -LiteralPath $full -Recurse -Force
}

function Move-Ds5Directory([string]$Source, [string]$Destination, [string]$Root) {
    $root = (Get-Ds5Path $Root) + '\'
    $source = Get-Ds5Path $Source
    $destination = Get-Ds5Path $Destination
    if (-not $source.StartsWith($root, [StringComparison]::OrdinalIgnoreCase) -or
        -not $destination.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Component activation must remain within its protected installation root.'
    }
    [IO.Directory]::Move($source, $destination)
}

function Install-Ds5Component([string]$PayloadDirectory, [string]$ComponentRoot, [string]$LogDirectory, [string]$HIDMaestroArchive = '') {
    $work = $null
    $backup = $null
    $logFile = $null
    $driverSetupAttempted = $false
    try {
        Assert-Ds5Administrator
        $root = Get-Ds5Path $ComponentRoot
        $payload = Get-Ds5Path $PayloadDirectory
        $logs = Get-Ds5Path $LogDirectory
        $tools = [IO.Path]::GetDirectoryName($root)
        $installation = [IO.Path]::GetDirectoryName($tools)
        if ([IO.Path]::GetFileName($root) -ine 'sunshine-ds5-component' -or [IO.Path]::GetFileName($tools) -ine 'tools' -or
            $payload -ine (Join-Path $root 'payload') -or $logs -ine (Join-Path $installation 'config\driver-setup')) {
            throw 'Driver setup paths must use the fixed Sunshine installation layout.'
        }
        foreach ($path in @($installation, $tools, $root, $payload)) { Assert-Ds5ProtectedPath $path }
        $config = Join-Path $installation 'config'
        if (-not (Test-Path -LiteralPath $config)) { $null = New-Ds5PrivateDirectory $config }
        Assert-Ds5ProtectedPath $config
        if (-not (Test-Path -LiteralPath $logs)) { $null = New-Ds5PrivateDirectory $logs }
        Assert-Ds5ProtectedPath $logs
        $logFile = Join-Path $logs ("dualsense-" + [Guid]::NewGuid().ToString('N') + '.log')
        [IO.File]::WriteAllText($logFile, "Optional DualSense driver setup`r`n")
        $manifest = Read-Ds5Manifest $payload
        Assert-Ds5Component $payload $manifest -Payload
        $active = Join-Path $root 'active'
        $null = Get-Ds5Path $active
        $work = New-Ds5PrivateDirectory (Join-Path $root ('install-' + [Guid]::NewGuid().ToString('N')))
        $temp = New-Ds5PrivateDirectory (Join-Path $work 'temp')
        if (Test-Path -LiteralPath $active) {
            # A previous component may have an older manifest, but it must never
            # be a link or contain linked/nested entries when moved for rollback.
            foreach ($entry in Get-ChildItem -LiteralPath $active -Force) {
                $null = Get-Ds5Path $entry.FullName
                if ($entry.PSIsContainer) { throw 'Existing active component is not flat.' }
            }
            try {
                Assert-Ds5Component $active $manifest
                if ((Invoke-Ds5Helper (Join-Path $active 'Sunshine.Ds5Sidecar.exe') '--probe-host' $temp $logFile) -eq $script:Ds5ReadyExitCode) {
                    return @{ ExitCode = 0; Message = 'The identical DualSense component and both drivers are already ready.'; LogFile = $logFile }
                }
            } catch { [IO.File]::AppendAllText($logFile, "Existing component needs replacement or repair: $($_.Exception.Message)`r`n") }
        }
        $stage = New-Ds5PrivateDirectory (Join-Path $work 'component')
        foreach ($entry in Get-ChildItem -LiteralPath $payload -File -Force) {
            [IO.File]::Copy($entry.FullName, (Join-Path $stage $entry.Name), $false)
        }
        $archive = Join-Path $work 'HIDMaestro-v1.6.2.zip'
        if ([string]::IsNullOrWhiteSpace($HIDMaestroArchive)) {
            Invoke-Ds5Download $archive
        } else {
            $sourceArchive = Get-Ds5Path $HIDMaestroArchive
            [IO.File]::Copy($sourceArchive, $archive, $false)
        }
        Expand-Ds5PinnedCore $archive $stage
        Assert-Ds5Component $stage $manifest
        $driverSetupAttempted = $true
        if ((Invoke-Ds5Helper (Join-Path $stage 'Sunshine.Ds5Sidecar.exe') '--install-drivers' $temp $logFile) -ne 0) {
            throw 'The verified helper reported driver setup failure.'
        }
        if ((Invoke-Ds5Helper (Join-Path $stage 'Sunshine.Ds5Sidecar.exe') '--probe-host' $temp $logFile) -ne $script:Ds5ReadyExitCode) {
            throw 'DualSense UMDF2 and USB/IP readiness did not both pass after driver setup.'
        }
        Assert-Ds5Component $stage $manifest
        if (Test-Path -LiteralPath $active) {
            $backup = Join-Path $root ('previous-' + [Guid]::NewGuid().ToString('N'))
            Move-Ds5Directory $active $backup $root
        }
        try { Move-Ds5Directory $stage $active $root }
        catch {
            if ($backup -and -not (Test-Path -LiteralPath $active)) { Move-Ds5Directory $backup $active $root; $backup = $null }
            throw
        }
        [IO.File]::AppendAllText($logFile, "Activated verified component: $active`r`nPrevious component: $backup`r`n")
        return @{ ExitCode = 0; Message = 'DualSense driver setup and readiness checks passed.'; LogFile = $logFile; PreviousComponent = $backup }
    } catch {
        $message = $_.Exception.Message
        if ($driverSetupAttempted) { $message += ' Driver or certificate changes may already be partial; preserving the previous component does not undo Windows driver changes.' }
        if ($backup) { $message += " Previous component remains at $backup." }
        if ($logFile) { [IO.File]::AppendAllText($logFile, "FAILED: $message`r`n") }
        return @{ ExitCode = 1; Message = $message; LogFile = $logFile }
    } finally {
        if ($work -and (Test-Path -LiteralPath $work)) {
            try { Remove-Ds5WorkDirectory $work $root }
            catch { Write-Warning "Driver staging was retained for inspection: $work. $($_.Exception.Message)" }
        }
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    $result = Install-Ds5Component $PayloadDirectory $ComponentRoot $LogDirectory $HIDMaestroArchive
    $result | ConvertTo-Json -Compress | Write-Output
    exit $result.ExitCode
}
