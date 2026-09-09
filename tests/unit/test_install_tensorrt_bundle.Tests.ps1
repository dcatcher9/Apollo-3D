# Only temporary text fixtures are published. Privileged ACL changes and archive
# extraction are mocked; no installed runtime, SDK, service, or driver is touched.
$repository = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $repository 'src_assets/windows/misc/runtime/install-tensorrt-bundle.ps1')

Describe 'Offline TensorRT bundle installation' {
    BeforeEach {
        $script:bundleRoot = Join-Path $TestDrive ([Guid]::NewGuid().ToString('N'))
        $script:bundleArchive = Join-Path $bundleRoot 'tools\tensorrt\runtime.7z'
        $script:bundleManifestPath = Join-Path $bundleRoot 'scripts\tensorrt-runtime.json'
        $script:bundleExtractor = Join-Path $bundleRoot 'tools\tensorrt\7zip\7z.exe'
        $script:bundleFixtures = Join-Path $TestDrive ('fixtures-' + [Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path "$bundleRoot\scripts", "$bundleRoot\tools\tensorrt\7zip", $bundleFixtures -Force | Out-Null
        [IO.File]::WriteAllText($bundleArchive, 'fixture archive')
        [IO.File]::WriteAllText($bundleExtractor, 'never executed')
        [IO.File]::WriteAllText("$bundleRoot\tools\tensorrt\7zip\7z.dll", 'never loaded')
        $names = @('nvinfer_11.dll', 'nvinfer_plugin_11.dll', 'nvonnxparser_11.dll',
            'nvinfer_dispatch_11.dll', 'nvinfer_lean_11.dll', 'nvinfer_vc_plugin_11.dll',
            'nvinfer_builder_resource_ptx_11.dll', 'nvinfer_builder_resource_sm75_11.dll',
            'nvinfer_builder_resource_sm80_11.dll', 'nvinfer_builder_resource_sm86_11.dll',
            'nvinfer_builder_resource_sm89_11.dll', 'nvinfer_builder_resource_sm90_11.dll',
            'nvinfer_builder_resource_sm100_11.dll', 'nvinfer_builder_resource_sm120_11.dll')
        $files = foreach ($name in $names) {
            $path = Join-Path $bundleFixtures $name
            [IO.File]::WriteAllText($path, ('fixture for ' + $name))
            [pscustomobject]@{ name = $name; sha256 = (Get-FileHash $path -Algorithm SHA256).Hash; size = (Get-Item $path).Length }
        }
        $script:bundleManifest = [pscustomobject]@{ archiveSha256 = (Get-FileHash $bundleArchive -Algorithm SHA256).Hash; files = @($files) }
        $bundleManifest | ConvertTo-Json -Depth 5 | Set-Content $bundleManifestPath
        Mock Assert-TensorRTBundleProtectedPath {}
        Mock New-TensorRTBundleDirectory { param($Path) [IO.Directory]::CreateDirectory($Path) | Out-Null }
        Mock Get-Process {}
        Mock Set-Acl {}
        Mock Expand-TensorRTBundle {
            param($Archive, $Executable, $Stage)
            foreach ($file in Get-ChildItem -LiteralPath $bundleFixtures -File) {
                [IO.File]::Copy($file.FullName, (Join-Path $Stage $file.Name), $false)
            }
        }
        Mock Move-TensorRTBundleFile { param($Source, $Destination) [IO.File]::Move($Source, $Destination) }
    }

    It 'installs the complete verified runtime and removes the bundled archive' {
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 0
        (Test-TensorRTBundleFiles $bundleRoot $bundleManifest.files) | Should Be $true
        (Test-Path $bundleArchive) | Should Be $false
        @(Get-ChildItem $bundleRoot -Directory -Force | Where-Object { $_.Name -like '.tensorrt-setup-*' }).Count | Should Be 0
        Assert-MockCalled Expand-TensorRTBundle -Exactly 1 -Scope It
        Assert-MockCalled Set-Acl -Exactly 14 -Scope It
    }

    It 'skips extraction when every installed DLL matches' {
        foreach ($file in $bundleManifest.files) { [IO.File]::Copy((Join-Path $bundleFixtures $file.name), (Join-Path $bundleRoot $file.name)) }
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 0
        (Test-Path $bundleArchive) | Should Be $false
        Assert-MockCalled Expand-TensorRTBundle -Exactly 0 -Scope It
    }

    It 'accepts an already complete runtime when its archive was previously cleaned up' {
        foreach ($file in $bundleManifest.files) { [IO.File]::Copy((Join-Path $bundleFixtures $file.name), (Join-Path $bundleRoot $file.name)) }
        Remove-Item -LiteralPath $bundleArchive
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 0
        Assert-MockCalled Expand-TensorRTBundle -Exactly 0 -Scope It
    }

    It 'replaces an incomplete old runtime as a complete verified set' {
        [IO.File]::WriteAllText("$bundleRoot\nvinfer_11.dll", 'old runtime')
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 0
        (Test-TensorRTBundleFiles $bundleRoot $bundleManifest.files) | Should Be $true
        Assert-MockCalled Expand-TensorRTBundle -Exactly 1 -Scope It
    }

    It 'rejects a corrupt archive before extraction or replacement' {
        [IO.File]::WriteAllText($bundleArchive, 'corrupt')
        [IO.File]::WriteAllText("$bundleRoot\nvinfer_11.dll", 'old runtime')
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 1
        [IO.File]::ReadAllText("$bundleRoot\nvinfer_11.dll") | Should Be 'old runtime'
        (Test-Path $bundleArchive) | Should Be $true
        Assert-MockCalled Expand-TensorRTBundle -Exactly 0 -Scope It
    }

    It 'requires all hosts and helpers to be closed without killing a process' {
        Mock Get-Process { [pscustomobject]@{ Name = 'sunshine' } }
        Mock Stop-Process { throw 'Never stop the user host from this script.' }
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 1
        (Test-Path $bundleArchive) | Should Be $true
        Assert-MockCalled New-TensorRTBundleDirectory -Exactly 0 -Scope It
        Assert-MockCalled Stop-Process -Exactly 0 -Scope It
    }

    It 'does not publish files when extracted content has the wrong hash' {
        [IO.File]::WriteAllText("$bundleFixtures\nvinfer_11.dll", 'corrupt fixture')
        [IO.File]::WriteAllText("$bundleRoot\nvinfer_11.dll", 'old runtime')
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 1
        [IO.File]::ReadAllText("$bundleRoot\nvinfer_11.dll") | Should Be 'old runtime'
        Assert-MockCalled Move-TensorRTBundleFile -Exactly 0 -Scope It
        (Test-Path $bundleArchive) | Should Be $true
    }

    It 'rechecks running hosts after extraction and before replacing DLLs' {
        $script:bundleProcessChecks = 0
        Mock Get-Process {
            ++$script:bundleProcessChecks
            if ($script:bundleProcessChecks -gt 1) { [pscustomobject]@{ Name = 'sunshine' } }
        }
        [IO.File]::WriteAllText("$bundleRoot\nvinfer_11.dll", 'old runtime')
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 1
        [IO.File]::ReadAllText("$bundleRoot\nvinfer_11.dll") | Should Be 'old runtime'
        Assert-MockCalled Expand-TensorRTBundle -Exactly 1 -Scope It
        Assert-MockCalled Move-TensorRTBundleFile -Exactly 0 -Scope It
    }

    It 'rejects extra archive files before publishing' {
        [IO.File]::WriteAllText("$bundleFixtures\unexpected.dll", 'unexpected')
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 1
        Assert-MockCalled Move-TensorRTBundleFile -Exactly 0 -Scope It
    }

    It 'retains old files and cleans staging after extractor failure' {
        Mock Expand-TensorRTBundle { throw 'fixture extractor failure' }
        [IO.File]::WriteAllText("$bundleRoot\nvinfer_11.dll", 'old runtime')
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 1
        [IO.File]::ReadAllText("$bundleRoot\nvinfer_11.dll") | Should Be 'old runtime'
        @(Get-ChildItem $bundleRoot -Directory -Force | Where-Object { $_.Name -like '.tensorrt-setup-*' }).Count | Should Be 0
        Assert-MockCalled Move-TensorRTBundleFile -Exactly 0 -Scope It
    }

    It 'restores all original DLLs after a partial replacement fails' {
        foreach ($file in $bundleManifest.files) { [IO.File]::WriteAllText((Join-Path $bundleRoot $file.name), ('old ' + $file.name)) }
        Mock Move-TensorRTBundleFile {
            param($Source, $Destination)
            if ($Source -like '*\new\nvinfer_plugin_11.dll') { throw 'fixture replacement failure' }
            [IO.File]::Move($Source, $Destination)
        }
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 1
        foreach ($file in $bundleManifest.files) { [IO.File]::ReadAllText((Join-Path $bundleRoot $file.name)) | Should Be ('old ' + $file.name) }
        (Test-Path $bundleArchive) | Should Be $true
        @(Get-ChildItem $bundleRoot -Directory -Force | Where-Object { $_.Name -like '.tensorrt-setup-*' }).Count | Should Be 0
    }

    It 'preserves backups for recovery if rollback itself is blocked' {
        [IO.File]::WriteAllText("$bundleRoot\nvinfer_11.dll", 'old runtime')
        Mock Move-TensorRTBundleFile {
            param($Source, $Destination)
            if ($Source -like '*\new\nvinfer_plugin_11.dll' -or $Source -like '*\previous\nvinfer_11.dll') { throw 'fixture blocked move' }
            [IO.File]::Move($Source, $Destination)
        }
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 1
        $retained = @(Get-ChildItem $bundleRoot -Directory -Force | Where-Object { $_.Name -like '.tensorrt-setup-*' })
        $retained.Count | Should Be 1
        [IO.File]::ReadAllText((Join-Path $retained[0].FullName 'previous\nvinfer_11.dll')) | Should Be 'old runtime'
    }

    It 'rejects traversal or duplicate names in the manifest' {
        $bundleManifest.files[0].name = '..\nvinfer_11.dll'
        $bundleManifest | ConvertTo-Json -Depth 5 | Set-Content $bundleManifestPath
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 1
        $bundleManifest.files[0].name = $bundleManifest.files[1].name
        $bundleManifest | ConvertTo-Json -Depth 5 | Set-Content $bundleManifestPath
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 1
        Assert-MockCalled Expand-TensorRTBundle -Exactly 0 -Scope It
    }

    It 'rejects a package outside the expected installation layout' {
        (Invoke-TensorRTBundleInstall "$TestDrive\runtime.7z" $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 1
        Assert-MockCalled Expand-TensorRTBundle -Exactly 0 -Scope It
    }

    It 'refuses an untrusted destination before creating staging' {
        Mock Assert-TensorRTBundleProtectedPath { throw 'unsafe destination' }
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 1
        Assert-MockCalled New-TensorRTBundleDirectory -Exactly 0 -Scope It
    }

    It 'rejects a reparse point at the manifest path' {
        Mock Get-Item { [pscustomobject]@{ Attributes = [IO.FileAttributes]::ReparsePoint } } -ParameterFilter { $LiteralPath -eq $bundleManifestPath }
        (Invoke-TensorRTBundleInstall $bundleArchive $bundleManifestPath $bundleExtractor $bundleRoot) | Should Be 1
        Assert-MockCalled Expand-TensorRTBundle -Exactly 0 -Scope It
    }
}

Describe 'Offline TensorRT extractor and access controls' {
    It 'rejects a failed extractor process' {
        $script:extractorProcess = [pscustomobject]@{ ExitCode = 2 }
        $extractorProcess | Add-Member ScriptMethod WaitForExit { param($Timeout) return $true }
        $extractorProcess | Add-Member ScriptMethod Dispose {}
        Mock Start-Process { $extractorProcess }
        { Expand-TensorRTBundle 'C:\fixture\runtime.7z' 'C:\fixture\7z.exe' 'C:\fixture\new' } | Should Throw
        Assert-MockCalled Start-Process -Exactly 1 -Scope It -ParameterFilter { $WindowStyle -eq 'Hidden' -and $PassThru }
    }

    It 'bounds extraction and stops only its own timed-out extractor' {
        $script:extractorKilled = $false
        $script:extractorProcess = [pscustomobject]@{ ExitCode = -1 }
        $extractorProcess | Add-Member ScriptMethod WaitForExit { param($Timeout) return $script:extractorKilled }
        $extractorProcess | Add-Member ScriptMethod Kill { $script:extractorKilled = $true }
        $extractorProcess | Add-Member ScriptMethod Dispose {}
        Mock Start-Process { $extractorProcess }
        { Expand-TensorRTBundle 'C:\fixture\runtime.7z' 'C:\fixture\7z.exe' 'C:\fixture\new' } | Should Throw
        $extractorKilled | Should Be $true
    }

    It 'allows ordinary users to load published DLLs without granting write access' {
        $acl = New-TensorRTBundleAcl -File
        $acl.AreAccessRulesProtected | Should Be $true
        $rules = @($acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]))
        $users = @($rules | Where-Object { $_.IdentityReference.Value -eq 'S-1-5-32-545' })
        $users.Count | Should Be 1
        ($users[0].FileSystemRights -band [Security.AccessControl.FileSystemRights]::ReadAndExecute) | Should Be ([Security.AccessControl.FileSystemRights]::ReadAndExecute)
        ($users[0].FileSystemRights -band [Security.AccessControl.FileSystemRights]::Write) | Should Be 0
    }

    It 'keeps staging private and rejects UNC, traversal and alternate-stream paths' {
        $acl = New-TensorRTBundleAcl
        @($acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]) | Where-Object { $_.IdentityReference.Value -notin @('S-1-5-18', 'S-1-5-32-544') }).Count | Should Be 0
        { Get-TensorRTBundlePath '\\server\share\runtime.7z' } | Should Throw
        { Get-TensorRTBundlePath 'C:\runtime\..\elsewhere' } | Should Throw
        { Get-TensorRTBundlePath 'C:\runtime.7z:stream' } | Should Throw
    }
}
