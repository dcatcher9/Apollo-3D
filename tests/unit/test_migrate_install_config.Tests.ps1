# Isolated fixture copies only. Privileged ACL application is mocked; no installed
# profile, service, process, driver, or development configuration is changed.
$repository = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $repository 'src_assets/windows/misc/migration/migrate-install-config.ps1')

Describe 'Installed configuration directory migration' {
    BeforeEach {
        $script:oldInstall = Join-Path $TestDrive ('Apollo-' + [Guid]::NewGuid().ToString('N'))
        $script:newInstall = Join-Path $TestDrive ('Sunshine3D-' + [Guid]::NewGuid().ToString('N'))
        $script:oldConfig = Join-Path $oldInstall 'config'
        $script:newConfig = Join-Path $newInstall 'config'
        New-Item -ItemType Directory -Path $oldConfig, $newInstall -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $oldConfig 'sunshine.conf') -Value 'sunshine_name = Original host'
        Mock Assert-InstallMigrationDestination {}
        Mock New-InstallMigrationStage { param($Path) [IO.Directory]::CreateDirectory($Path) | Out-Null }
        Mock Set-Acl {}
    }

    It 'copies persistent files, covers and pairing credentials without changing the source' {
        New-Item -ItemType Directory -Path "$oldConfig\credentials", "$oldConfig\covers" | Out-Null
        [IO.File]::WriteAllBytes("$oldConfig\credentials\cakey.pem", [byte[]](1,2,3,4))
        [IO.File]::WriteAllBytes("$oldConfig\covers\game.png", [byte[]](5,6,7))
        Set-Content "$oldConfig\sunshine_state.json" '{"uniqueid":"existing-pairing"}'
        Set-Content "$oldConfig\apps.json" '{"apps":[{"name":"Desktop"}]}'
        (Invoke-InstallConfigMigration $oldInstall $newInstall) | Should Be 0
        (Get-Content "$newConfig\sunshine_state.json" -Raw) | Should Match 'existing-pairing'
        [Convert]::ToBase64String([IO.File]::ReadAllBytes("$newConfig\credentials\cakey.pem")) | Should Be 'AQIDBA=='
        (Test-Path "$newConfig\covers\game.png") | Should Be $true
        (Test-Path "$oldConfig\credentials\cakey.pem") | Should Be $true
        Assert-MockCalled Set-Acl -Exactly 1 -Scope It -ParameterFilter {
            $LiteralPath.EndsWith('\credentials') -and $AclObject.AreAccessRulesProtected
        }
    }

    It 'rewrites only known config paths and app cover paths rooted in the old install' {
        $oldForward = $oldInstall.Replace('\', '/')
        @"
pkey = $oldInstall\config\credentials\cakey.pem
file_apps = $oldForward/config/apps.json
log_path = E:\OtherProfile\sunshine.log
file_state = sunshine_state.json
sunshine_name = $oldInstall
"@ | Set-Content "$oldConfig\sunshine.conf"
        @{ apps = @(
            @{ name = 'Internal'; 'image-path' = "$oldInstall\config\covers\one.png"; cmd = "$oldInstall\run-game.exe" },
            @{ name = 'External'; 'image-path' = 'E:\Games\cover.png' },
            @{ name = 'Relative'; 'image-path' = './config/covers/two.png' }
        ) } | ConvertTo-Json -Depth 8 | Set-Content "$oldConfig\apps.json"
        (Invoke-InstallConfigMigration $oldInstall $newInstall) | Should Be 0
        $conf = Get-Content "$newConfig\sunshine.conf" -Raw
        $conf.Contains("pkey = $newInstall\config\credentials\cakey.pem") | Should Be $true
        $conf.Contains("file_apps = $newInstall\config\apps.json") | Should Be $true
        $conf.Contains('log_path = E:\OtherProfile\sunshine.log') | Should Be $true
        $conf.Contains('file_state = sunshine_state.json') | Should Be $true
        $conf.Contains("sunshine_name = $oldInstall") | Should Be $true
        $apps = Get-Content "$newConfig\apps.json" -Raw | ConvertFrom-Json
        $apps.apps[0].'image-path' | Should Be "$newInstall\config\covers\one.png"
        $apps.apps[0].cmd | Should Be "$oldInstall\run-game.exe"
        $apps.apps[1].'image-path' | Should Be 'E:\Games\cover.png'
        $apps.apps[2].'image-path' | Should Be './config/covers/two.png'
        (Get-Content "$oldConfig\sunshine.conf" -Raw).Contains("pkey = $oldInstall") | Should Be $true
    }

    It 'preserves an existing destination profile without merging credentials' {
        New-Item -ItemType Directory -Path $newConfig | Out-Null
        Set-Content "$newConfig\sunshine.conf" 'sunshine_name = New profile'
        (Invoke-InstallConfigMigration $oldInstall $newInstall) | Should Be 0
        (Get-Content "$newConfig\sunshine.conf" -Raw) | Should Match 'New profile'
        Assert-MockCalled New-InstallMigrationStage -Exactly 0 -Scope It
    }

    It 'preserves destination settings in the legacy root layout' {
        Set-Content "$newInstall\sunshine_state.json" '{"uniqueid":"destination"}'
        (Invoke-InstallConfigMigration $oldInstall $newInstall) | Should Be 0
        (Test-Path $newConfig) | Should Be $false
        Assert-MockCalled New-InstallMigrationStage -Exactly 0 -Scope It
    }

    It 'can replace an empty destination config directory' {
        New-Item -ItemType Directory -Path $newConfig | Out-Null
        (Invoke-InstallConfigMigration $oldInstall $newInstall) | Should Be 0
        (Test-Path "$newConfig\sunshine.conf") | Should Be $true
    }

    It 'treats a same-directory upgrade as a no-op' {
        (Invoke-InstallConfigMigration $oldInstall $oldInstall) | Should Be 0
        Assert-MockCalled New-InstallMigrationStage -Exactly 0 -Scope It
    }

    It 'finds retained Apollo config after its uninstaller removed registration' {
        $saved = $env:ProgramW6432
        try {
            $env:ProgramW6432 = Join-Path $TestDrive ('ProgramFiles-' + [Guid]::NewGuid().ToString('N'))
            New-Item -ItemType Directory -Path "$env:ProgramW6432\Apollo\config" -Force | Out-Null
            Set-Content "$env:ProgramW6432\Apollo\config\sunshine.conf" 'sunshine_name = Retained Apollo'
            (Invoke-InstallConfigMigration '' $newInstall) | Should Be 0
            (Get-Content "$newConfig\sunshine.conf" -Raw) | Should Match 'Retained Apollo'
        }
        finally { $env:ProgramW6432 = $saved }
    }

    It 'treats a fresh install with no source as a no-op' {
        $saved = $env:ProgramW6432
        try {
            $env:ProgramW6432 = Join-Path $TestDrive ('NoOldInstall-' + [Guid]::NewGuid().ToString('N'))
            (Invoke-InstallConfigMigration '' $newInstall) | Should Be 0
            (Test-Path $newConfig) | Should Be $false
        }
        finally { $env:ProgramW6432 = $saved }
    }

    It 'rejects source path traversal even if the supplied source does not exist' {
        (Invoke-InstallConfigMigration "$oldInstall\missing\..\config" $newInstall) | Should Be 1
        Assert-MockCalled New-InstallMigrationStage -Exactly 0 -Scope It
    }

    It 'rejects a reparse point in the source tree and removes the partial stage' {
        Set-Content "$oldConfig\linked-file" 'fixture'
        Mock Get-Item { [pscustomobject]@{ Attributes = [IO.FileAttributes]::ReparsePoint } } -ParameterFilter {
            $LiteralPath -eq "$oldConfig\linked-file"
        }
        (Invoke-InstallConfigMigration $oldInstall $newInstall) | Should Be 1
        (Test-Path $newConfig) | Should Be $false
        @(Get-ChildItem $newInstall -Force).Count | Should Be 0
        (Test-Path "$oldConfig\sunshine.conf") | Should Be $true
    }

    It 'rejects an unsafe destination before copying anything' {
        Mock Assert-InstallMigrationDestination { throw 'unsafe destination ancestor' }
        (Invoke-InstallConfigMigration $oldInstall $newInstall) | Should Be 1
        Assert-MockCalled New-InstallMigrationStage -Exactly 0 -Scope It
    }

    It 'leaves a destination created during staging untouched' {
        Mock Update-InstallMigrationPaths {
            New-Item -ItemType Directory -Path $newConfig | Out-Null
            Set-Content "$newConfig\sunshine.conf" 'sunshine_name = Concurrent profile'
        }
        (Invoke-InstallConfigMigration $oldInstall $newInstall) | Should Be 1
        (Get-Content "$newConfig\sunshine.conf" -Raw) | Should Match 'Concurrent profile'
        @(Get-ChildItem $newInstall -Force).Count | Should Be 1
    }

    It 'does not publish partial configuration when a copy fails' {
        Mock Copy-InstallMigrationTree {
            param($Source, $Destination)
            Set-Content (Join-Path $Destination 'partial-file') 'partial'
            throw 'fixture copy failure'
        }
        (Invoke-InstallConfigMigration $oldInstall $newInstall) | Should Be 1
        (Test-Path $newConfig) | Should Be $false
        @(Get-ChildItem $newInstall -Force).Count | Should Be 0
        (Test-Path "$oldConfig\sunshine.conf") | Should Be $true
    }
}

Describe 'Configuration migration path and ACL boundaries' {
    It 'does not rewrite prefix collisions, relative paths or traversal values' {
        (Convert-InstallMigrationValue 'C:\ApolloOther\config\apps.json' 'C:\Apollo' 'C:\Sunshine3D') | Should Be 'C:\ApolloOther\config\apps.json'
        (Convert-InstallMigrationValue 'config/apps.json' 'C:\Apollo' 'C:\Sunshine3D') | Should Be 'config/apps.json'
        (Convert-InstallMigrationValue 'C:\Apollo\..\External\apps.json' 'C:\Apollo' 'C:\Sunshine3D') | Should Be 'C:\Apollo\..\External\apps.json'
    }

    It 'restricts the new credential ACL to SYSTEM and Administrators' {
        $acl = New-InstallMigrationPrivateAcl
        $acl.AreAccessRulesProtected | Should Be $true
        $rules = @($acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]))
        $rules.Count | Should Be 2
        @($rules | Where-Object { $_.IdentityReference.Value -notin @('S-1-5-18', 'S-1-5-32-544') }).Count | Should Be 0
        $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value | Should Be 'S-1-5-32-544'
    }

    It 'rejects alternate streams and UNC destination paths' {
        { Get-InstallMigrationPath 'C:\Apollo:stream' } | Should Throw
        { Get-InstallMigrationPath '\\server\share\Sunshine3D' } | Should Throw
    }

    It 'rejects a writable destination ACL' {
        $acl = New-InstallMigrationPrivateAcl
        $identity = New-Object Security.Principal.SecurityIdentifier('S-1-5-32-545')
        $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($identity, 'Write', 'Allow')))
        Mock Get-Acl { $acl }
        { Assert-InstallMigrationDestination $TestDrive } | Should Throw
    }
}
