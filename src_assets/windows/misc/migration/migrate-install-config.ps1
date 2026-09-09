# Preserve the old installed host's configuration when its installation directory
# changes. Run before migrate-config.bat and before starting the new host. This
# copies settings only; the old installation and development profiles are untouched.
[CmdletBinding()]
param([string]$SourceDirectory, [string]$DestinationDirectory)

function Get-InstallMigrationPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or $Path -notmatch '^[A-Za-z]:[\\/]' -or
        $Path.Substring(2).Contains(':') -or $Path -match '(^|[\\/])\.{1,2}([\\/]|$)') {
        throw 'Configuration migration requires local absolute paths without traversal or alternate streams.'
    }
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    if ($full.Length -lt 4) { throw 'A drive root is not an installation directory.' }
    $cursor = $full
    while ($cursor) {
        if ((Test-Path -LiteralPath $cursor) -and
            ((Get-Item -LiteralPath $cursor -Force -ErrorAction Stop).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Configuration migration refuses reparse points: $cursor"
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $full
}

function Assert-InstallMigrationDestination {
    param([string]$Path)
    $trusted = @('S-1-5-18', 'S-1-5-32-544',
        'S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464')
    $cursor = Get-InstallMigrationPath $Path
    if (-not (Test-Path -LiteralPath $cursor -PathType Container)) {
        throw 'The destination installation directory must already exist.'
    }
    while ($cursor) {
        $acl = Get-Acl -LiteralPath $cursor -ErrorAction Stop
        if ($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -notin $trusted) {
            throw "Configuration migration requires administrator-owned destination ancestors: $cursor"
        }
        $rights = [Security.AccessControl.FileSystemRights]::Delete -bor
            [Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles -bor
            [Security.AccessControl.FileSystemRights]::ChangePermissions -bor
            [Security.AccessControl.FileSystemRights]::TakeOwnership
        if ($cursor -eq $Path) { $rights = $rights -bor [Security.AccessControl.FileSystemRights]::Write }
        foreach ($rule in $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
            if ($rule.AccessControlType -eq 'Allow' -and
                -not ($rule.PropagationFlags -band [Security.AccessControl.PropagationFlags]::InheritOnly) -and
                ($rule.FileSystemRights -band $rights) -and $rule.IdentityReference.Value -notin $trusted) {
                throw "Configuration migration refuses a replaceable destination ancestor: $cursor"
            }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
}

function New-InstallMigrationPrivateAcl {
    $acl = New-Object Security.AccessControl.DirectorySecurity
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))
    foreach ($sid in @('S-1-5-18', 'S-1-5-32-544')) {
        $identity = New-Object Security.Principal.SecurityIdentifier($sid)
        $rule = New-Object Security.AccessControl.FileSystemAccessRule($identity, 'FullControl',
            'ContainerInherit, ObjectInherit', 'None', 'Allow')
        $acl.AddAccessRule($rule)
    }
    return $acl
}

function New-InstallMigrationStage {
    param([string]$Path)
    [IO.Directory]::CreateDirectory($Path, (New-InstallMigrationPrivateAcl)) | Out-Null
}

function Copy-InstallMigrationTree {
    param([string]$Source, [string]$Destination)
    foreach ($item in Get-ChildItem -LiteralPath $Source -Force -ErrorAction Stop) {
        $null = Get-InstallMigrationPath $item.FullName
        $target = Join-Path $Destination $item.Name
        if ($item.PSIsContainer) {
            [IO.Directory]::CreateDirectory($target) | Out-Null
            Copy-InstallMigrationTree $item.FullName $target
        }
        else {
            # File.Copy creates a new file with the destination's inherited ACL;
            # do not copy source security descriptors with the credentials.
            [IO.File]::Copy($item.FullName, $target, $false)
        }
    }
}

function Convert-InstallMigrationValue {
    param([string]$Value, [string]$Source, [string]$Destination)
    $normalized = $Value.Replace('/', '\')
    if ($normalized.StartsWith($Source + '\', [StringComparison]::OrdinalIgnoreCase) -and
        $normalized -notmatch '(^|\\)\.{1,2}(\\|$)') {
        return $Destination + $normalized.Substring($Source.Length)
    }
    return $Value
}

function Update-InstallMigrationPaths {
    param([string]$Config, [string]$Source, [string]$Destination)
    $utf8 = New-Object Text.UTF8Encoding($false)
    $conf = Join-Path $Config 'sunshine.conf'
    if (Test-Path -LiteralPath $conf -PathType Leaf) {
        $content = [IO.File]::ReadAllText($conf)
        $keys = 'pkey|cert|log_path|file_state|credentials_file|file_apps'
        $content = [regex]::Replace($content, "(?m)^(\s*(?:$keys)\s*=\s*)([^\r\n]*?)(\s*)$", {
            param($match)
            $match.Groups[1].Value + (Convert-InstallMigrationValue $match.Groups[2].Value $Source $Destination) + $match.Groups[3].Value
        })
        [IO.File]::WriteAllText($conf, $content, $utf8)
    }
    $appsFile = Join-Path $Config 'apps.json'
    if (Test-Path -LiteralPath $appsFile -PathType Leaf) {
        $apps = [IO.File]::ReadAllText($appsFile) | ConvertFrom-Json -ErrorAction Stop
        $changed = $false
        foreach ($app in $apps.apps) {
            $property = $app.PSObject.Properties['image-path']
            if ($null -ne $property -and $property.Value -is [string]) {
                $updated = Convert-InstallMigrationValue $property.Value $Source $Destination
                if ($updated -cne $property.Value) { $property.Value = $updated; $changed = $true }
            }
        }
        if ($changed) { [IO.File]::WriteAllText($appsFile, ($apps | ConvertTo-Json -Depth 100), $utf8) }
    }
}

function Invoke-InstallConfigMigration {
    param([string]$SourceDirectory, [string]$DestinationDirectory)
    $ErrorActionPreference = 'Stop'
    $stage = $null
    try {
        $destination = Get-InstallMigrationPath $DestinationDirectory
        if (-not [string]::IsNullOrWhiteSpace($SourceDirectory)) {
            $SourceDirectory = Get-InstallMigrationPath $SourceDirectory
        }
        if ([string]::IsNullOrWhiteSpace($SourceDirectory) -or -not (Test-Path -LiteralPath $SourceDirectory)) {
            $programFiles = $env:ProgramW6432
            if ([string]::IsNullOrWhiteSpace($programFiles)) { $programFiles = $env:ProgramFiles }
            $SourceDirectory = Join-Path $programFiles 'Apollo'
        }
        $source = Get-InstallMigrationPath $SourceDirectory
        if ($source -ieq $destination -or -not (Test-Path -LiteralPath $source -PathType Container)) { return 0 }
        $oldConfig = Get-InstallMigrationPath (Join-Path $source 'config')
        if (-not (Test-Path -LiteralPath $oldConfig -PathType Container)) { return 0 }
        $newConfig = Get-InstallMigrationPath (Join-Path $destination 'config')
        # Preserve any existing destination profile, including the older layout
        # that migrate-config.bat will move into config immediately after this.
        if ((Test-Path -LiteralPath $newConfig) -and
            (-not (Test-Path -LiteralPath $newConfig -PathType Container) -or
             @(Get-ChildItem -LiteralPath $newConfig -Force).Count -gt 0)) { return 0 }
        foreach ($name in @('sunshine.conf', 'sunshine_state.json', 'apps.json', 'credentials', 'covers')) {
            if (Test-Path -LiteralPath (Join-Path $destination $name)) { return 0 }
        }
        if (@(Get-ChildItem -LiteralPath $oldConfig -Force).Count -eq 0) { return 0 }
        if ($destination.StartsWith($source + '\', [StringComparison]::OrdinalIgnoreCase) -or
            $source.StartsWith($destination + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Source and destination installations must not be nested.'
        }
        Assert-InstallMigrationDestination $destination
        $stage = Join-Path $destination ('.sunshine-config-migration-' + [Guid]::NewGuid().ToString('N'))
        New-InstallMigrationStage $stage
        Copy-InstallMigrationTree $oldConfig $stage
        Update-InstallMigrationPaths $stage $source $destination
        $credentials = Join-Path $stage 'credentials'
        if (Test-Path -LiteralPath $credentials -PathType Container) {
            # Explicit protection survives migrate-config.bat resetting config's
            # ACL, while children inherit SYSTEM/Administrators access only.
            Set-Acl -LiteralPath $credentials -AclObject (New-InstallMigrationPrivateAcl)
        }
        # An empty directory is safe to remove nonrecursively. Directory.Move
        # refuses a destination created meanwhile instead of merging profiles.
        if (Test-Path -LiteralPath $newConfig) { [IO.Directory]::Delete($newConfig, $false) }
        [IO.Directory]::Move($stage, $newConfig)
        $stage = $null
        Write-Host 'Preserved configuration from the previous installation.'
        return 0
    }
    catch {
        Write-Host "Configuration migration failed: $($_.Exception.Message)"
        return 1
    }
    finally {
        if ($stage -and (Test-Path -LiteralPath $stage)) {
            # The only removable tree is this invocation's private staging tree.
            $checked = Get-InstallMigrationPath $stage
            if ([IO.Path]::GetDirectoryName($checked) -ine $destination -or
                [IO.Path]::GetFileName($checked) -notmatch '^\.sunshine-config-migration-[0-9a-f]{32}$') {
                throw 'Refusing cleanup outside the migration staging directory.'
            }
            Remove-Item -LiteralPath $checked -Recurse -Force -ErrorAction Stop
        }
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    exit (Invoke-InstallConfigMigration -SourceDirectory $SourceDirectory -DestinationDirectory $DestinationDirectory)
}
