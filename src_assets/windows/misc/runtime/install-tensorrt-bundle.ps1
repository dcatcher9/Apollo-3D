# Expand the offline TensorRT payload. No network, SDK, registry, or system PATH
# changes are made. Run before starting Sunshine. Exit 0 succeeds; exit 1 fails.
[CmdletBinding()]
param([string]$ArchivePath, [string]$ManifestPath, [string]$SevenZipPath, [string]$DestinationDirectory)

function Get-TensorRTBundlePath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or $Path -notmatch '^[A-Za-z]:[\\/]' -or
        $Path.Substring(2).Contains(':') -or $Path -match '[<>"|?*]' -or
        $Path -match '(^|[\\/])\.{1,2}([\\/]|$)') { throw 'A local absolute path without traversal or alternate streams is required.' }
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    if ($full.Length -lt 4) { throw 'A drive root is not a TensorRT setup directory.' }
    $cursor = $full
    while ($cursor) {
        if ((Test-Path -LiteralPath $cursor) -and
            ((Get-Item -LiteralPath $cursor -Force -ErrorAction Stop).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "TensorRT setup refuses reparse points: $cursor"
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $full
}

function Assert-TensorRTBundleProtectedPath {
    param([string]$Path)
    $full = Get-TensorRTBundlePath $Path
    $trusted = @('S-1-5-18', 'S-1-5-32-544',
        'S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464')
    $cursor = $full
    while ($cursor) {
        $acl = Get-Acl -LiteralPath $cursor -ErrorAction Stop
        if ($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -notin $trusted) { throw "Untrusted TensorRT setup path owner: $cursor" }
        $rights = [Security.AccessControl.FileSystemRights]::Delete -bor
            [Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles -bor
            [Security.AccessControl.FileSystemRights]::ChangePermissions -bor
            [Security.AccessControl.FileSystemRights]::TakeOwnership
        if ($cursor -eq $full) { $rights = $rights -bor [Security.AccessControl.FileSystemRights]::Write }
        foreach ($rule in $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
            if ($rule.AccessControlType -eq 'Allow' -and
                -not ($rule.PropagationFlags -band [Security.AccessControl.PropagationFlags]::InheritOnly) -and
                ($rule.FileSystemRights -band $rights) -and $rule.IdentityReference.Value -notin $trusted) {
                throw "TensorRT setup path is writable or replaceable by an unelevated identity: $cursor"
            }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
}

function New-TensorRTBundleAcl {
    param([switch]$File)
    if ($File) { $acl = New-Object Security.AccessControl.FileSecurity }
    else { $acl = New-Object Security.AccessControl.DirectorySecurity }
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))
    foreach ($sid in @('S-1-5-18', 'S-1-5-32-544')) {
        $identity = New-Object Security.Principal.SecurityIdentifier($sid)
        if ($File) { $rule = New-Object Security.AccessControl.FileSystemAccessRule($identity, 'FullControl', 'Allow') }
        else { $rule = New-Object Security.AccessControl.FileSystemAccessRule($identity, 'FullControl', 'ContainerInherit, ObjectInherit', 'None', 'Allow') }
        $acl.AddAccessRule($rule)
    }
    if ($File) {
        # The host may run as an ordinary desktop user; runtime DLLs must remain
        # readable/executable after being moved out of private staging.
        $users = New-Object Security.Principal.SecurityIdentifier('S-1-5-32-545')
        $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($users, 'ReadAndExecute', 'Allow')))
    }
    return $acl
}

function New-TensorRTBundleDirectory {
    param([string]$Path)
    [IO.Directory]::CreateDirectory($Path, (New-TensorRTBundleAcl)) | Out-Null
}

function Read-TensorRTBundleManifest {
    param([string]$Path)
    $manifest = [IO.File]::ReadAllText($Path) | ConvertFrom-Json -ErrorAction Stop
    $expected = @('nvinfer_11.dll', 'nvinfer_plugin_11.dll', 'nvonnxparser_11.dll',
        'nvinfer_dispatch_11.dll', 'nvinfer_lean_11.dll', 'nvinfer_vc_plugin_11.dll',
        'nvinfer_builder_resource_ptx_11.dll', 'nvinfer_builder_resource_sm75_11.dll',
        'nvinfer_builder_resource_sm80_11.dll', 'nvinfer_builder_resource_sm86_11.dll',
        'nvinfer_builder_resource_sm89_11.dll', 'nvinfer_builder_resource_sm90_11.dll',
        'nvinfer_builder_resource_sm100_11.dll', 'nvinfer_builder_resource_sm120_11.dll')
    if ($manifest.archiveSha256 -notmatch '^[a-fA-F0-9]{64}$' -or @($manifest.files).Count -ne $expected.Count) {
        throw 'Invalid TensorRT bundle manifest.'
    }
    $seen = @{}
    foreach ($file in $manifest.files) {
        if ($file.name -cnotin $expected -or $seen.ContainsKey($file.name) -or
            $file.sha256 -notmatch '^[a-fA-F0-9]{64}$' -or
            [long]$file.size -le 0 -or [long]$file.size -gt 1073741824) { throw 'Invalid TensorRT runtime file manifest.' }
        $seen[$file.name] = $true
    }
    return $manifest
}

function Test-TensorRTBundleFiles {
    param([string]$Directory, $Files, [switch]$Exact)
    if ($Exact) {
        $actual = @(Get-ChildItem -LiteralPath $Directory -Force -ErrorAction Stop)
        if ($actual.Count -ne @($Files).Count -or @($actual | Where-Object { $_.PSIsContainer }).Count) { return $false }
    }
    foreach ($file in $Files) {
        $path = Get-TensorRTBundlePath (Join-Path $Directory $file.name)
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
        if ((Get-Item -LiteralPath $path -Force).Length -ne [long]$file.size -or
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ine $file.sha256) { return $false }
    }
    return $true
}

function Expand-TensorRTBundle {
    param([string]$Archive, [string]$Executable, [string]$Stage)
    $process = Start-Process -FilePath $Executable -ArgumentList @('x', '-y', '-bd', ('-o"' + $Stage + '"'), '--', ('"' + $Archive + '"')) `
        -WorkingDirectory ([IO.Path]::GetDirectoryName($Executable)) -WindowStyle Hidden -PassThru -ErrorAction Stop
    try {
        if (-not $process.WaitForExit(900000)) {
            # Stop only our archive extractor before touching its private stage.
            $process.Kill()
            if (-not $process.WaitForExit(30000)) { throw 'TensorRT extractor did not stop after its timeout.' }
            throw 'TensorRT extraction timed out.'
        }
        if ($process.ExitCode -ne 0) { throw "TensorRT extraction failed with exit code $($process.ExitCode)." }
    }
    finally { $process.Dispose() }
}

function Move-TensorRTBundleFile {
    param([string]$Source, [string]$Destination)
    [IO.File]::Move($Source, $Destination)
}

function Invoke-TensorRTBundleInstall {
    param([string]$ArchivePath, [string]$ManifestPath, [string]$SevenZipPath, [string]$DestinationDirectory)
    $ErrorActionPreference = 'Stop'
    $work = $null
    $retainWork = $false
    $published = $false
    $changes = New-Object 'System.Collections.Generic.List[object]'
    try {
        $destination = Get-TensorRTBundlePath $DestinationDirectory
        $archive = Get-TensorRTBundlePath $ArchivePath
        $manifestFile = Get-TensorRTBundlePath $ManifestPath
        $extractor = Get-TensorRTBundlePath $SevenZipPath
        if ($archive -ine (Join-Path $destination 'tools\tensorrt\runtime.7z') -or
            $manifestFile -ine (Join-Path $destination 'scripts\tensorrt-runtime.json') -or
            $extractor -ine (Join-Path $destination 'tools\tensorrt\7zip\7z.exe')) { throw 'Unexpected TensorRT bundle installation layout.' }
        Assert-TensorRTBundleProtectedPath $destination
        Assert-TensorRTBundleProtectedPath $manifestFile
        $manifest = Read-TensorRTBundleManifest $manifestFile
        if (Get-Process -Name sunshine,Sunshine.Ds5Sidecar -ErrorAction SilentlyContinue) { throw 'Close all Sunshine hosts and DualSense helpers before installing the TensorRT runtime.' }
        foreach ($file in $manifest.files) {
            $path = Get-TensorRTBundlePath (Join-Path $destination $file.name)
            if (Test-Path -LiteralPath $path) { Assert-TensorRTBundleProtectedPath $path }
        }
        if (Test-TensorRTBundleFiles $destination $manifest.files) {
            if (Test-Path -LiteralPath $archive) {
                Assert-TensorRTBundleProtectedPath $archive
                Remove-Item -LiteralPath $archive -Force
            }
            Write-Host 'The installed TensorRT runtime already matches this package.'
            return 0
        }
        foreach ($path in @($archive, $extractor, (Join-Path ([IO.Path]::GetDirectoryName($extractor)) '7z.dll'))) {
            Assert-TensorRTBundleProtectedPath $path
        }
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ine $manifest.archiveSha256) { throw 'TensorRT bundle SHA-256 verification failed.' }
        $work = Join-Path $destination ('.tensorrt-setup-' + [Guid]::NewGuid().ToString('N'))
        New-TensorRTBundleDirectory $work
        $stage = Join-Path $work 'new'
        $backup = Join-Path $work 'previous'
        New-TensorRTBundleDirectory $stage
        New-TensorRTBundleDirectory $backup
        Write-Host 'Extracting the bundled TensorRT runtime.'
        Expand-TensorRTBundle $archive $extractor $stage
        if (-not (Test-TensorRTBundleFiles $stage $manifest.files -Exact)) { throw 'Extracted TensorRT files do not match the package manifest.' }
        if (Get-Process -Name sunshine,Sunshine.Ds5Sidecar -ErrorAction SilentlyContinue) { throw 'A Sunshine host or DualSense helper started during setup. Close it before retrying.' }
        foreach ($file in $manifest.files) {
            $new = Join-Path $stage $file.name
            Set-Acl -LiteralPath $new -AclObject (New-TensorRTBundleAcl -File)
            $target = Join-Path $destination $file.name
            $previous = Join-Path $backup $file.name
            $change = [pscustomobject]@{ Target = $target; Backup = $previous; Installed = $false }
            $changes.Add($change)
            if (Test-Path -LiteralPath $target) { Move-TensorRTBundleFile $target $previous }
            Move-TensorRTBundleFile $new $target
            $change.Installed = $true
        }
        $published = $true
        Remove-Item -LiteralPath $archive -Force
        Write-Host 'The bundled TensorRT runtime is ready.'
        return 0
    }
    catch {
        Write-Host "TensorRT setup failed: $($_.Exception.Message)"
        if (-not $published) {
            for ($index = $changes.Count - 1; $index -ge 0; --$index) {
                $change = $changes[$index]
                try {
                    if ($change.Installed) { [IO.File]::Delete($change.Target) }
                    if (Test-Path -LiteralPath $change.Backup) { Move-TensorRTBundleFile $change.Backup $change.Target }
                }
                catch { $retainWork = $true; Write-Host "Runtime rollback needs manual recovery from: $work" }
            }
        }
        return 1
    }
    finally {
        if ($work -and -not $retainWork -and (Test-Path -LiteralPath $work)) {
            $checked = Get-TensorRTBundlePath $work
            if ([IO.Path]::GetDirectoryName($checked) -ine $destination -or
                [IO.Path]::GetFileName($checked) -notmatch '^\.tensorrt-setup-[0-9a-f]{32}$') { throw 'Refusing cleanup outside the TensorRT staging directory.' }
            Remove-Item -LiteralPath $checked -Recurse -Force -ErrorAction Stop
        }
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    exit (Invoke-TensorRTBundleInstall -ArchivePath $ArchivePath -ManifestPath $ManifestPath -SevenZipPath $SevenZipPath -DestinationDirectory $DestinationDirectory)
}
