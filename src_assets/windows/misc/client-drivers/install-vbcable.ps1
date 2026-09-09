<#
.SYNOPSIS
Installs the optional, unmodified VB-CABLE Pack45 with its vendor setup.
.DESCRIPTION
Run only after the user selects VB-CABLE in the Sunshine installer. The Pack45
readme requires Administrator rights and a restart. -Silent uses the install-only
-i and hidden -h switches verified in this exact pinned vendor binary; see
VB-CABLE-SETUP-CONTRACT.md. No remove/uninstall switch is passed.
Exit 0: an existing device is ready. Exit 3010: installation succeeded or an
existing device needs a restart. Exit 1: failed, canceled, or not verified.
No uninstall, certificate-store, default-audio, or Sunshine configuration changes
are made. The vendor package and its readme are retained beside the setup log.
#>
[CmdletBinding()]
param(
    [string]$PackageDirectory,
    [string]$LogDirectory,
    [switch]$Silent
)

function Test-VBCableAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-VBCableLocalPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or $Path -notmatch '^[A-Za-z]:[\\/]' -or
        $Path.Substring(2).Contains(':') -or $Path -match '(^|[\\/])\.{1,2}([\\/]|$)') {
        throw "A local absolute path without traversal or alternate streams is required: $Path"
    }
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    if ($full.Length -lt 4) { throw 'A drive root is not a driver setup directory.' }
    $cursor = $full
    while ($cursor) {
        if ((Test-Path -LiteralPath $cursor) -and
            ((Get-Item -LiteralPath $cursor -Force -ErrorAction Stop).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Reparse points are not permitted in driver setup paths: $cursor"
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $full
}

function Assert-VBCableProtectedPath {
    param([string]$Path)
    $full = Get-VBCableLocalPath $Path
    $trusted = @('S-1-5-18', 'S-1-5-32-544',
        'S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464')
    $replacementRights = [Security.AccessControl.FileSystemRights]::Delete -bor
        [Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles -bor
        [Security.AccessControl.FileSystemRights]::ChangePermissions -bor
        [Security.AccessControl.FileSystemRights]::TakeOwnership
    $cursor = $full
    while ($cursor) {
        $acl = Get-Acl -LiteralPath $cursor -ErrorAction Stop
        if ($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -notin $trusted) {
            throw "Driver setup requires administrator-owned paths: $cursor"
        }
        # An ancestor may allow creating unrelated children (the system drive
        # normally does), but must not allow replacing this protected subtree.
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

function New-VBCablePrivateDirectory {
    param([string]$Path)
    $full = Get-VBCableLocalPath $Path
    if (Test-Path -LiteralPath $full) { throw "Driver staging directory already exists: $full" }
    Assert-VBCableProtectedPath ([IO.Path]::GetDirectoryName($full))
    $acl = New-Object Security.AccessControl.DirectorySecurity
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))
    foreach ($sid in @('S-1-5-18', 'S-1-5-32-544')) {
        $identity = New-Object Security.Principal.SecurityIdentifier($sid)
        $rule = New-Object Security.AccessControl.FileSystemAccessRule($identity, 'FullControl',
            'ContainerInherit, ObjectInherit', 'None', 'Allow')
        $acl.AddAccessRule($rule)
    }
    # Supply the protected DACL at creation, avoiding a create-then-SetAcl window
    # in which an inherited child-only permission could permit file replacement.
    [IO.Directory]::CreateDirectory($full, $acl) | Out-Null
    Assert-VBCableProtectedPath $full
    return $full
}

function Initialize-VBCableLogDirectory {
    param([string]$Path)
    $full = Get-VBCableLocalPath $Path
    $missing = New-Object 'System.Collections.Generic.List[string]'
    $cursor = $full
    while (-not (Test-Path -LiteralPath $cursor)) {
        $missing.Add($cursor)
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    Assert-VBCableProtectedPath $cursor
    for ($index = $missing.Count - 1; $index -ge 0; --$index) {
        $null = New-VBCablePrivateDirectory $missing[$index]
    }
    Assert-VBCableProtectedPath $full
    return $full
}

function Get-VBCableDriverState {
    # The signed Pack45 INF declares this hardware ID. A friendly name such as
    # "CABLE Input" is insufficient: it can name an unrelated audio endpoint.
    $devices = @(Get-CimInstance -ClassName Win32_PnPEntity `
        -Filter "ClassGuid='{4d36e96c-e325-11ce-bfc1-08002be10318}'" -ErrorAction Stop |
        Where-Object {
            @($_.HardwareID) -contains 'VBAudioVACWDM' -or
            @($_.HardwareID) -contains 'ROOT\VBAudioVACWDM'
        })
    return [pscustomobject]@{
        Present = $devices.Count -gt 0
        Ready = @($devices | Where-Object { $_.Present -eq $true -and $_.ConfigManagerErrorCode -eq 0 }).Count -gt 0
        RebootRequired = @($devices | Where-Object { $_.Present -eq $true -and $_.ConfigManagerErrorCode -eq 14 }).Count -gt 0
    }
}

function Write-VBCableSetupLog {
    param([string]$LogPath, [string]$Message)
    $entry = '{0:o} {1}' -f [DateTimeOffset]::Now, $Message
    Add-Content -LiteralPath $LogPath -Value $entry -Encoding UTF8 -ErrorAction Stop
    Write-Host $Message
}

function Expand-VBCablePackage {
    param([string]$PackageDirectory, [string]$LogDirectory)
    $PackageDirectory = Get-VBCableLocalPath $PackageDirectory
    $LogDirectory = Get-VBCableLocalPath $LogDirectory
    Assert-VBCableProtectedPath $PackageDirectory
    Assert-VBCableProtectedPath $LogDirectory
    $archivePath = Join-Path $PackageDirectory 'VBCABLE_Driver_Pack45.zip'
    Assert-VBCableProtectedPath $archivePath
    $expectedArchiveHash = 'B950E39F01AF1D04EA623C8F6D8EB9B6EA5C477C637295FABF20631C85116BFB'
    if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256 -ErrorAction Stop).Hash -ne $expectedArchiveHash) {
        throw 'VB-CABLE Pack45 archive SHA-256 verification failed.'
    }

    # Always extract into a fresh directory; never reuse an old setup's files.
    $stage = Join-Path $LogDirectory ('VB-CABLE-Pack45-' + [Guid]::NewGuid().ToString('N'))
    $null = New-VBCablePrivateDirectory $stage
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        foreach ($entry in $archive.Entries) {
            if ($entry.FullName -notmatch '^[A-Za-z0-9_.-]+$' -or $entry.FullName -in @('.', '..')) {
                throw 'Unexpected path in the VB-CABLE archive.'
            }
        }
    }
    finally {
        $archive.Dispose()
    }
    [IO.Compression.ZipFile]::ExtractToDirectory($archivePath, $stage)

    foreach ($required in @('readme.txt', 'vbMmeCable64_win10.inf', 'vbaudio_cable64_win10.cat',
            'vbaudio_cable64_win10.sys', 'VBCABLE_Setup_x64.exe')) {
        if (-not (Test-Path -LiteralPath (Join-Path $stage $required) -PathType Leaf)) {
            throw "Incomplete VB-CABLE package: $required is missing."
        }
    }
    $setup = Join-Path $stage 'VBCABLE_Setup_x64.exe'
    Assert-VBCableProtectedPath $setup
    $expectedSetupHash = '734C35DFA6D98F48782A451633CEB471166EC70D60482FD89A1123D0EE3C4F41'
    if ((Get-FileHash -LiteralPath $setup -Algorithm SHA256 -ErrorAction Stop).Hash -ne $expectedSetupHash) {
        throw 'VB-CABLE setup SHA-256 verification failed.'
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $setup -ErrorAction Stop
    if ($signature.Status -ne 'Valid' -or -not $signature.SignerCertificate -or
        $signature.SignerCertificate.Thumbprint -ne 'A77952D93229D0EC36E2543081EEA7D125732B9C' -or
        $signature.SignerCertificate.Subject -notmatch '(^|,\s*)CN=BUREL VINCENT Entrepreneur individuel(,|$)') {
        throw 'VB-CABLE setup does not have its pinned, valid VB-Audio publisher signature.'
    }
    return $setup
}

function Invoke-VBCableSetup {
    param([string]$SetupPath, [switch]$Silent)
    $SetupPath = Get-VBCableLocalPath $SetupPath
    Assert-VBCableProtectedPath $SetupPath
    $startArguments = @{
        FilePath = $SetupPath
        WorkingDirectory = Split-Path -Parent $SetupPath
        WindowStyle = 'Normal'
        PassThru = $true
        ErrorAction = 'Stop'
    }
    if ($Silent) {
        $startArguments.WindowStyle = 'Hidden'
        $startArguments.ArgumentList = @('-i', '-h')
    }
    $process = Start-Process @startArguments
    try {
        if (-not $process.WaitForExit(600000)) {
            # Do not terminate a driver operation part-way through installation.
            return [pscustomobject]@{ Exited = $false; ExitCode = $null }
        }
        return [pscustomobject]@{ Exited = $true; ExitCode = $process.ExitCode }
    }
    finally {
        $process.Dispose()
    }
}

function Invoke-VBCableInstall {
    param(
        [Parameter(Mandatory = $true)][string]$PackageDirectory,
        [Parameter(Mandatory = $true)][string]$LogDirectory,
        [switch]$Silent
    )
    $logPath = $null
    try {
        if (-not (Test-VBCableAdministrator)) {
            throw 'Administrator rights are required. Run the Sunshine installer as Administrator.'
        }
        $PackageDirectory = Get-VBCableLocalPath $PackageDirectory
        $LogDirectory = Initialize-VBCableLogDirectory $LogDirectory
        $candidateLog = Get-VBCableLocalPath (Join-Path $LogDirectory 'vbcable-install.log')
        if (Test-Path -LiteralPath $candidateLog) { Assert-VBCableProtectedPath $candidateLog }
        $logPath = $candidateLog
        Write-VBCableSetupLog $logPath 'VB-CABLE is donationware by VB-Audio: https://www.vb-cable.com/ . Donations and license payments are welcome.'
        if (-not [Environment]::Is64BitOperatingSystem) {
            throw 'This Sunshine package requires 64-bit Windows.'
        }

        $before = Get-VBCableDriverState
        if ($before.Ready) {
            Write-VBCableSetupLog $logPath 'An existing VB-CABLE device is ready; no installation was performed.'
            return 0
        }
        if ($before.RebootRequired) {
            Write-VBCableSetupLog $logPath 'The existing VB-CABLE device requires a Windows restart; no installation was performed.'
            return 3010
        }
        if ($before.Present) {
            throw 'VB-CABLE is already installed but is not ready. Repair or enable it with the vendor tools; automatic reinstallation is disabled.'
        }

        $setup = Expand-VBCablePackage $PackageDirectory $LogDirectory
        if ($Silent) {
            Write-VBCableSetupLog $logPath 'Running the verified VB-CABLE vendor setup with its install-only, hidden switches (-i -h).'
        }
        else {
            Write-VBCableSetupLog $logPath "Opening the verified VB-CABLE vendor setup: $setup . Select Install Driver and finish its prompts."
        }
        $result = Invoke-VBCableSetup -SetupPath $setup -Silent:$Silent
        if (-not $result.Exited) {
            throw 'The vendor setup is still running after ten minutes. Check the setup process before retrying; it was not terminated during a possible driver operation.'
        }
        Write-VBCableSetupLog $logPath "Vendor setup exit code: $($result.ExitCode)"
        if ($result.ExitCode -ne 0) {
            throw "VB-CABLE setup returned error $($result.ExitCode)."
        }
        for ($attempt = 0; $attempt -lt 10; ++$attempt) {
            $after = Get-VBCableDriverState
            if ($after.Ready -or $after.RebootRequired) {
                Write-VBCableSetupLog $logPath 'VB-CABLE installation verified. Restart Windows to finalize installation, then select its cable endpoints in Sunshine and the receiving application.'
                return 3010
            }
            if ($attempt -lt 9) {
                Start-Sleep -Seconds 1
            }
        }
        throw 'No working or restart-pending VB-CABLE device was registered. The setup may have been canceled; installation is not verified.'
    }
    catch {
        $message = 'VB-CABLE setup failed: ' + $_.Exception.Message
        if ($logPath) {
            try { Write-VBCableSetupLog $logPath $message } catch { Write-Host $message }
        }
        else {
            Write-Host $message
        }
        return 1
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    if ([string]::IsNullOrWhiteSpace($PackageDirectory) -or [string]::IsNullOrWhiteSpace($LogDirectory)) {
        Write-Host 'Both -PackageDirectory and -LogDirectory are required.'
        exit 1
    }
    exit (Invoke-VBCableInstall -PackageDirectory $PackageDirectory -LogDirectory $LogDirectory -Silent:$Silent)
}
