# Windows PowerShell 5.1 runner for the isolated optional-driver installer tests.
# The suites mock vendor setup, driver, and download operations. They use only
# temporary fixture files and a harmless locally compiled environment probe.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
try {
    if ($PSVersionTable.PSEdition -ne 'Desktop' -or
        $PSVersionTable.PSVersion.Major -ne 5 -or $PSVersionTable.PSVersion.Minor -ne 1) {
        throw 'Run these installer tests with Windows PowerShell 5.1 (powershell.exe).'
    }

    # CTest/MSYS and PowerShell 7 can inherit a PSModulePath containing incompatible
    # Core modules ahead of the OS modules. Establish the 5.1 module environment
    # in this test process before Pester discovers or mocks any system command.
    $osModules = [IO.Path]::Combine($PSHOME, 'Modules')
    $sharedModules = [IO.Path]::Combine([Environment]::GetFolderPath('ProgramFiles'), 'WindowsPowerShell', 'Modules')
    $env:PSModulePath = $osModules + ';' + $sharedModules
    foreach ($moduleName in @('Microsoft.PowerShell.Management', 'Microsoft.PowerShell.Utility',
            'Microsoft.PowerShell.Security', 'CimCmdlets')) {
        $manifest = [IO.Path]::Combine($osModules, $moduleName, $moduleName + '.psd1')
        Import-Module -Name $manifest -Global -Force -ErrorAction Stop
    }

    $pester = Get-Module -ListAvailable -Name Pester |
        Where-Object { $_.Version.Major -in @(3, 4) } |
        Sort-Object Version -Descending |
        Select-Object -First 1
    if (-not $pester) {
        throw 'Pester 3 or 4 must already be available to Windows PowerShell. This runner does not install modules.'
    }
    Import-Module -Name $pester.Path -Force -ErrorAction Stop

    $testRoot = Split-Path -Parent $PSScriptRoot
    $suites = @(
        (Join-Path $testRoot 'unit\test_install_vbcable.Tests.ps1'),
        (Join-Path $testRoot 'integration\Install-Dualsense.Tests.ps1'),
        (Join-Path $testRoot 'unit\test_migrate_install_config.Tests.ps1'),
        (Join-Path $testRoot 'unit\test_install_tensorrt_bundle.Tests.ps1')
    )
    foreach ($suite in $suites) {
        if (-not (Test-Path -LiteralPath $suite -PathType Leaf)) {
            throw "Driver installer test suite is missing: $suite"
        }
    }

    Write-Host "Running isolated driver installer tests with Windows PowerShell $($PSVersionTable.PSVersion) and Pester $($pester.Version)."
    $total = 0
    foreach ($suite in $suites) {
        $result = Invoke-Pester -Script $suite -PassThru
        if ($null -eq $result -or $result.TotalCount -lt 1) {
            throw "No tests ran in driver installer suite: $suite"
        }
        if ($result.FailedCount -gt 0 -or $result.SkippedCount -gt 0 -or
            $result.PendingCount -gt 0 -or $result.InconclusiveCount -gt 0) {
            throw "Driver installer suite did not fully pass: $suite (failed=$($result.FailedCount), skipped=$($result.SkippedCount), pending=$($result.PendingCount), inconclusive=$($result.InconclusiveCount))."
        }
        $total += $result.TotalCount
    }
    Write-Host "All $total isolated driver installer tests passed."
    exit 0
}
catch {
    Write-Output "Driver installer tests failed: $($_.Exception.Message)"
    exit 1
}
