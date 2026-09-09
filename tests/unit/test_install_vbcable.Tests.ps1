# Run with Windows PowerShell/Pester. All device, privilege, and process calls are
# mocked. No executable from a vendor package is started and no driver is installed.
$repository = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $repository 'src_assets/windows/misc/client-drivers/install-vbcable.ps1')

Describe 'VB-CABLE device identity' {
    It 'ignores friendly names and other VB-Audio products' {
        Mock Get-CimInstance {
            [pscustomobject]@{ Name = 'CABLE Input'; HardwareID = @('OTHER_AUDIO'); ConfigManagerErrorCode = 0 }
            [pscustomobject]@{ Name = 'VB-Audio Voicemeeter'; HardwareID = @('VBAudioVMVAIO'); ConfigManagerErrorCode = 0 }
        }
        $state = Get-VBCableDriverState
        $state.Present | Should Be $false
        $state.Ready | Should Be $false
        Assert-MockCalled Get-CimInstance -Exactly 1 -Scope It -ParameterFilter {
            $ClassName -eq 'Win32_PnPEntity' -and $Filter -eq "ClassGuid='{4d36e96c-e325-11ce-bfc1-08002be10318}'"
        }
    }

    It 'identifies the INF hardware ID independently of localized names' {
        Mock Get-CimInstance {
            [pscustomobject]@{ Name = 'Localized name'; HardwareID = @('VBAudioVACWDM'); Present = $true; ConfigManagerErrorCode = 0 }
        }
        (Get-VBCableDriverState).Ready | Should Be $true
    }

    It 'retains an actual disabled cable without declaring it ready' {
        Mock Get-CimInstance {
            [pscustomobject]@{ HardwareID = @('ROOT\VBAudioVACWDM'); Present = $true; ConfigManagerErrorCode = 22 }
        }
        $state = Get-VBCableDriverState
        $state.Present | Should Be $true
        $state.Ready | Should Be $false
        $state.RebootRequired | Should Be $false
    }

    It 'does not report a stale nonpresent device as ready' {
        Mock Get-CimInstance {
            [pscustomobject]@{ HardwareID = @('VBAudioVACWDM'); Present = $false; ConfigManagerErrorCode = 0 }
        }
        $state = Get-VBCableDriverState
        $state.Present | Should Be $true
        $state.Ready | Should Be $false
        $state.RebootRequired | Should Be $false
    }
}

Describe 'VB-CABLE install outcome and preservation' {
    BeforeEach {
        Mock Test-VBCableAdministrator { $true }
        Mock Assert-VBCableProtectedPath {}
        Mock Initialize-VBCableLogDirectory { param($Path) return $Path }
        Mock Write-VBCableSetupLog {}
        Mock Get-VBCableDriverState { [pscustomobject]@{ Present = $false; Ready = $false; RebootRequired = $false } }
        Mock Expand-VBCablePackage { 'C:\mock\VBCABLE_Setup_x64.exe' }
        Mock Invoke-VBCableSetup { [pscustomobject]@{ Exited = $true; ExitCode = 0 } }
        Mock Start-Sleep {}
        Mock Start-Process { throw 'A unit test must never launch a driver installer.' }
    }

    It 'rejects a nonadministrator before extracting or invoking setup' {
        Mock Test-VBCableAdministrator { $false }
        (Invoke-VBCableInstall $TestDrive "$TestDrive\logs" -Silent) | Should Be 1
        Assert-MockCalled Expand-VBCablePackage -Exactly 0 -Scope It
        Assert-MockCalled Invoke-VBCableSetup -Exactly 0 -Scope It
    }

    It 'preserves an existing healthy cable with a zero exit' {
        Mock Get-VBCableDriverState { [pscustomobject]@{ Present = $true; Ready = $true; RebootRequired = $false } }
        (Invoke-VBCableInstall $TestDrive "$TestDrive\logs" -Silent) | Should Be 0
        Assert-MockCalled Expand-VBCablePackage -Exactly 0 -Scope It
        Assert-MockCalled Invoke-VBCableSetup -Exactly 0 -Scope It
    }

    It 'preserves an existing restart-pending cable and propagates reboot' {
        Mock Get-VBCableDriverState { [pscustomobject]@{ Present = $true; Ready = $false; RebootRequired = $true } }
        (Invoke-VBCableInstall $TestDrive "$TestDrive\logs" -Silent) | Should Be 3010
        Assert-MockCalled Invoke-VBCableSetup -Exactly 0 -Scope It
    }

    It 'does not reinstall or toggle removal for an existing problem device' {
        Mock Get-VBCableDriverState { [pscustomobject]@{ Present = $true; Ready = $false; RebootRequired = $false } }
        (Invoke-VBCableInstall $TestDrive "$TestDrive\logs" -Silent) | Should Be 1
        Assert-MockCalled Invoke-VBCableSetup -Exactly 0 -Scope It
    }

    It 'verifies a new healthy device and requires the vendor-mandated restart' {
        $script:deviceChecks = 0
        Mock Get-VBCableDriverState {
            ++$script:deviceChecks
            [pscustomobject]@{ Present = $script:deviceChecks -gt 1; Ready = $script:deviceChecks -gt 1; RebootRequired = $false }
        }
        (Invoke-VBCableInstall $TestDrive "$TestDrive\logs" -Silent) | Should Be 3010
        Assert-MockCalled Invoke-VBCableSetup -Exactly 1 -Scope It -ParameterFilter { $Silent }
        Assert-MockCalled Get-VBCableDriverState -Exactly 2 -Scope It
    }

    It 'accepts a newly registered restart-pending device' {
        $script:deviceChecks = 0
        Mock Get-VBCableDriverState {
            ++$script:deviceChecks
            [pscustomobject]@{ Present = $script:deviceChecks -gt 1; Ready = $false; RebootRequired = $script:deviceChecks -gt 1 }
        }
        (Invoke-VBCableInstall $TestDrive "$TestDrive\logs" -Silent) | Should Be 3010
    }

    It 'does not call cancellation or a zero-exit no-op successful' {
        (Invoke-VBCableInstall $TestDrive "$TestDrive\logs" -Silent) | Should Be 1
        Assert-MockCalled Invoke-VBCableSetup -Exactly 1 -Scope It
        Assert-MockCalled Get-VBCableDriverState -Exactly 11 -Scope It
        Assert-MockCalled Start-Sleep -Exactly 9 -Scope It
    }

    It 'reports a nonzero vendor failure without retrying installation' {
        Mock Invoke-VBCableSetup { [pscustomobject]@{ Exited = $true; ExitCode = -100 } }
        (Invoke-VBCableInstall $TestDrive "$TestDrive\logs" -Silent) | Should Be 1
        Assert-MockCalled Invoke-VBCableSetup -Exactly 1 -Scope It
        Assert-MockCalled Get-VBCableDriverState -Exactly 1 -Scope It
    }

    It 'reports timeout without retrying or claiming device readiness' {
        Mock Invoke-VBCableSetup { [pscustomobject]@{ Exited = $false; ExitCode = $null } }
        (Invoke-VBCableInstall $TestDrive "$TestDrive\logs" -Silent) | Should Be 1
        Assert-MockCalled Invoke-VBCableSetup -Exactly 1 -Scope It
        Assert-MockCalled Get-VBCableDriverState -Exactly 1 -Scope It
    }

    It 'does not launch setup after package verification fails' {
        Mock Expand-VBCablePackage { throw 'Invalid signature' }
        (Invoke-VBCableInstall $TestDrive "$TestDrive\logs" -Silent) | Should Be 1
        Assert-MockCalled Invoke-VBCableSetup -Exactly 0 -Scope It
    }
}

Describe 'VB-CABLE process invocation' {
    BeforeEach {
        $script:fakeProcess = [pscustomobject]@{ ExitCode = 0; WaitTimeout = 0; Disposed = $false; Exited = $true }
        $script:fakeProcess | Add-Member ScriptMethod WaitForExit { param($milliseconds) $this.WaitTimeout = $milliseconds; return $this.Exited }
        $script:fakeProcess | Add-Member ScriptMethod Dispose { $this.Disposed = $true }
        Mock Start-Process { $script:fakeProcess }
        Mock Stop-Process { throw 'Do not kill a driver operation.' }
        Mock Assert-VBCableProtectedPath {}
    }

    It 'passes only the verified install and hide switches for silent setup' {
        (Invoke-VBCableSetup -SetupPath 'C:\mock\VBCABLE_Setup_x64.exe' -Silent).Exited | Should Be $true
        Assert-MockCalled Start-Process -Exactly 1 -Scope It -ParameterFilter {
            $FilePath -eq 'C:\mock\VBCABLE_Setup_x64.exe' -and $WorkingDirectory -eq 'C:\mock' -and
            $WindowStyle -eq 'Hidden' -and $ArgumentList.Count -eq 2 -and
            $ArgumentList[0] -eq '-i' -and $ArgumentList[1] -eq '-h'
        }
        $script:fakeProcess.WaitTimeout | Should Be 600000
        $script:fakeProcess.Disposed | Should Be $true
    }

    It 'keeps explicitly interactive setup visible without hidden switches' {
        (Invoke-VBCableSetup 'C:\mock\VBCABLE_Setup_x64.exe').ExitCode | Should Be 0
        Assert-MockCalled Start-Process -Exactly 1 -Scope It -ParameterFilter {
            $WindowStyle -eq 'Normal' -and -not $ArgumentList
        }
    }

    It 'releases the process handle but never kills an installation on timeout' {
        $script:fakeProcess.Exited = $false
        $result = Invoke-VBCableSetup 'C:\mock\VBCABLE_Setup_x64.exe' -Silent
        $result.Exited | Should Be $false
        $script:fakeProcess.Disposed | Should Be $true
        Assert-MockCalled Stop-Process -Exactly 0 -Scope It
    }
}

Describe 'VB-CABLE package trust boundary' {
    BeforeEach {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $script:packageRoot = Join-Path $TestDrive ([Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $script:packageRoot | Out-Null
        $source = Join-Path $script:packageRoot 'source'
        New-Item -ItemType Directory -Path $source | Out-Null
        foreach ($name in @('readme.txt', 'vbMmeCable64_win10.inf', 'vbaudio_cable64_win10.cat',
                'vbaudio_cable64_win10.sys', 'VBCABLE_Setup_x64.exe')) {
            Set-Content -LiteralPath (Join-Path $source $name) -Value 'inert unit-test fixture'
        }
        [IO.Compression.ZipFile]::CreateFromDirectory($source, (Join-Path $script:packageRoot 'VBCABLE_Driver_Pack45.zip'))
        Mock Get-FileHash {
            param($LiteralPath)
            if ($LiteralPath.EndsWith('.zip')) {
                return [pscustomobject]@{ Hash = 'B950E39F01AF1D04EA623C8F6D8EB9B6EA5C477C637295FABF20631C85116BFB' }
            }
            return [pscustomobject]@{ Hash = '734C35DFA6D98F48782A451633CEB471166EC70D60482FD89A1123D0EE3C4F41' }
        }
        Mock Get-AuthenticodeSignature {
            [pscustomobject]@{ Status = 'Valid'; SignerCertificate = [pscustomobject]@{
                Thumbprint = 'A77952D93229D0EC36E2543081EEA7D125732B9C'
                Subject = 'CN=BUREL VINCENT Entrepreneur individuel, O=BUREL VINCENT Entrepreneur individuel'
            } }
        }
        Mock Start-Process { throw 'Fixture executables must never run.' }
        Mock Assert-VBCableProtectedPath {}
        Mock New-VBCablePrivateDirectory { param($Path) New-Item -ItemType Directory -Path $Path | Out-Null; return $Path }
    }

    It 'extracts a complete trusted package into a fresh directory' {
        $setup = Expand-VBCablePackage $script:packageRoot $script:packageRoot
        (Test-Path -LiteralPath $setup -PathType Leaf) | Should Be $true
        (Split-Path -Leaf (Split-Path -Parent $setup)) | Should Match '^VB-CABLE-Pack45-'
        Assert-MockCalled Get-AuthenticodeSignature -Exactly 1 -Scope It
    }

    It 'rejects archive substitution before inspecting a publisher' {
        Mock Get-FileHash { [pscustomobject]@{ Hash = 'wrong' } }
        { Expand-VBCablePackage $script:packageRoot $script:packageRoot } | Should Throw 'archive SHA-256'
        Assert-MockCalled Get-AuthenticodeSignature -Exactly 0 -Scope It
    }

    It 'rejects a signed executable from a different publisher' {
        Mock Get-AuthenticodeSignature {
            [pscustomobject]@{ Status = 'Valid'; SignerCertificate = [pscustomobject]@{ Thumbprint = 'OTHER'; Subject = 'CN=Other Publisher' } }
        }
        { Expand-VBCablePackage $script:packageRoot $script:packageRoot } | Should Throw 'pinned, valid'
        Assert-MockCalled Start-Process -Exactly 0 -Scope It
    }

    It 'rejects an invalid signature even if the certificate matches' {
        Mock Get-AuthenticodeSignature { [pscustomobject]@{ Status = 'HashMismatch'; SignerCertificate = $null } }
        { Expand-VBCablePackage $script:packageRoot $script:packageRoot } | Should Throw 'pinned, valid'
    }
}

Describe 'VB-CABLE protected execution paths' {
    BeforeEach {
        $script:protectedAcl = New-Object Security.AccessControl.DirectorySecurity
        $script:protectedAcl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))
        $script:protectedAcl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
            (New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')), 'FullControl', 'Allow')))
        Mock Get-Acl { $script:protectedAcl }
        Mock Start-Process { throw 'A path test must never launch an installer.' }
    }

    It 'rejects relative, network, traversal, drive root, and alternate stream paths' {
        foreach ($candidate in @('relative\path', '\\server\share\setup', 'C:\setup\..\other',
                'C:\setup\file:stream', 'C:\')) {
            $rejected = $false
            try { $null = Get-VBCableLocalPath $candidate } catch { $rejected = $true }
            $rejected | Should Be $true
        }
        Assert-MockCalled Start-Process -Exactly 0 -Scope It
    }

    It 'rejects a reparse point in any existing ancestor' {
        Mock Test-Path { $true }
        Mock Get-Item {
            param($LiteralPath)
            $attributes = [IO.FileAttributes]::Directory
            if ($LiteralPath -eq 'C:\redirect') { $attributes = $attributes -bor [IO.FileAttributes]::ReparsePoint }
            [pscustomobject]@{ Attributes = $attributes }
        }
        { Get-VBCableLocalPath 'C:\redirect\drivers\setup.exe' } | Should Throw 'Reparse points'
    }

    It 'rejects an execution path owned by an unelevated identity' {
        $script:protectedAcl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-21-111-222-333-1001')))
        { Assert-VBCableProtectedPath 'C:\Program Files\Sunshine\drivers' } | Should Throw 'administrator-owned'
    }

    It 'rejects a writable stage even when its owner is Administrators' {
        $script:protectedAcl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
            (New-Object Security.Principal.SecurityIdentifier('S-1-5-32-545')), 'Write', 'Allow')))
        { Assert-VBCableProtectedPath 'C:\Program Files\Sunshine\drivers' } | Should Throw 'unelevated identity'
    }

    It 'rejects an ancestor that lets users replace the protected subtree' {
        $script:unsafeParentAcl = New-Object Security.AccessControl.DirectorySecurity
        $script:unsafeParentAcl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))
        $script:unsafeParentAcl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
            (New-Object Security.Principal.SecurityIdentifier('S-1-5-32-545')), 'DeleteSubdirectoriesAndFiles', 'Allow')))
        Mock Get-Acl {
            param($LiteralPath)
            if ($LiteralPath -eq 'C:\Program Files\Sunshine') { return $script:unsafeParentAcl }
            return $script:protectedAcl
        }
        { Assert-VBCableProtectedPath 'C:\Program Files\Sunshine\drivers' } | Should Throw 'unelevated identity'
    }

    It 'accepts a protected subtree beneath a drive allowing unrelated child creation' {
        $script:driveAcl = New-Object Security.AccessControl.DirectorySecurity
        $script:driveAcl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))
        $script:driveAcl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
            (New-Object Security.Principal.SecurityIdentifier('S-1-5-32-545')), 'CreateDirectories', 'Allow')))
        Mock Get-Acl {
            param($LiteralPath)
            if ($LiteralPath -eq 'C:\') { return $script:driveAcl }
            return $script:protectedAcl
        }
        { Assert-VBCableProtectedPath 'C:\Program Files\Sunshine\drivers' } | Should Not Throw
    }

    It 'blocks execution before process creation when the setup path is writable' {
        $script:protectedAcl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
            (New-Object Security.Principal.SecurityIdentifier('S-1-5-32-545')), 'Modify', 'Allow')))
        { Invoke-VBCableSetup 'C:\Program Files\Sunshine\drivers\setup.exe' -Silent } | Should Throw 'unelevated identity'
        Assert-MockCalled Start-Process -Exactly 0 -Scope It
    }
}
