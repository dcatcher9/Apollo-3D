param(
    [string]$Dotnet = '',
    [string]$HIDMaestroArchive = '',
    [string]$OutputDirectory = 'build/ds5-sidecar-component'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$version = 'v1.6.2'
$archiveSha256 = '6ae8df0cf317baf7e65777e2929f618916a67831b5ff1162205310f2c08b80ff'
$coreSha256 = 'df59cd5b6031311f03031c32b794901fc1f45503410f4f6ad55eaa854cee8746'
$archiveUrl = "https://github.com/hifihedgehog/HIDMaestro/releases/download/$version/HIDMaestro-$version.zip"
$output = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $root $OutputDirectory))
}

# Never clean a caller-supplied directory. Publishing a fresh component avoids
# mixing runtimes and ensures the manifest describes this build exactly.
if (Test-Path -LiteralPath $output) {
    if (@(Get-ChildItem -LiteralPath $output -Force).Count -ne 0) {
        throw "OutputDirectory must be empty or absent: $output"
    }
}

if ([string]::IsNullOrWhiteSpace($Dotnet)) {
    $command = Get-Command dotnet -CommandType Application -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw 'Supply -Dotnet with a .NET 10 SDK dotnet.exe path; a portable SDK is supported.'
    }
    $Dotnet = $command.Source
}
$sdks = @(& $Dotnet --list-sdks 2>&1)
if ($LASTEXITCODE -ne 0 -or -not ($sdks | Where-Object { [string]$_ -match '^\s*10\.' })) {
    throw 'The helper and HIDMaestro 1.6.2 require the .NET 10 SDK. An installed runtime alone is insufficient.'
}

$buildRoot = Join-Path $root 'build'
$cache = Join-Path $buildRoot 'ds5-sidecar-cache'
New-Item -ItemType Directory -Force -Path $cache | Out-Null
if ([string]::IsNullOrWhiteSpace($HIDMaestroArchive)) {
    $HIDMaestroArchive = Join-Path $cache "HIDMaestro-$version.zip"
    if (-not (Test-Path -LiteralPath $HIDMaestroArchive)) {
        Invoke-WebRequest -Uri $archiveUrl -OutFile $HIDMaestroArchive -UseBasicParsing
    }
}
if ((Get-FileHash -LiteralPath $HIDMaestroArchive -Algorithm SHA256).Hash -ne $archiveSha256) {
    throw 'HIDMaestro archive does not match the pinned official v1.6.2 SHA-256.'
}

$intermediate = Join-Path $cache ([System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $intermediate | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($HIDMaestroArchive)
try {
    foreach ($name in @('HIDMaestro.Core.dll', 'LICENSE', 'README.md', 'THIRD-PARTY-NOTICES.txt')) {
        $entry = $archive.GetEntry($name)
        if ($null -eq $entry) { throw "Pinned HIDMaestro archive is missing $name" }
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $intermediate $name))
    }
}
finally { $archive.Dispose() }
$core = Join-Path $intermediate 'HIDMaestro.Core.dll'
if ((Get-FileHash -LiteralPath $core -Algorithm SHA256).Hash -ne $coreSha256) {
    throw 'HIDMaestro Core does not match the pinned official assembly SHA-256.'
}

$project = Join-Path $root 'tools/sunshine-ds5-sidecar/Sunshine.Ds5Sidecar.csproj'
$obj = (Join-Path $intermediate 'obj') + [System.IO.Path]::DirectorySeparatorChar
$bin = (Join-Path $intermediate 'bin') + [System.IO.Path]::DirectorySeparatorChar
& $Dotnet publish $project --configuration Release --runtime win-x64 --self-contained true `
    --source https://api.nuget.org/v3/index.json --output $output `
    "-p:HIDMaestroCorePath=$core" "-p:BaseIntermediateOutputPath=$obj" "-p:BaseOutputPath=$bin" `
    -p:DebugType=None -p:DebugSymbols=false
if ($LASTEXITCODE -ne 0) { throw "DualSense helper publish failed with exit code $LASTEXITCODE" }

Copy-Item -LiteralPath (Join-Path $root 'LICENSE') -Destination (Join-Path $output 'LICENSE-Sunshine3D')
Copy-Item -LiteralPath (Join-Path $intermediate 'LICENSE') -Destination (Join-Path $output 'LICENSE-HIDMaestro')
Copy-Item -LiteralPath (Join-Path $intermediate 'README.md') -Destination (Join-Path $output 'README-HIDMaestro.md')
Copy-Item -LiteralPath (Join-Path $intermediate 'THIRD-PARTY-NOTICES.txt') -Destination (Join-Path $output 'THIRD-PARTY-NOTICES-HIDMaestro.txt')

$sidecar = Join-Path $output 'Sunshine.Ds5Sidecar.exe'
& $sidecar --self-check
if ($LASTEXITCODE -ne 0) { throw "DualSense helper self-check failed with exit code $LASTEXITCODE" }

$files = [ordered]@{}
foreach ($file in Get-ChildItem -LiteralPath $output -File | Sort-Object Name) {
    $files[$file.Name] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
if (@(Get-ChildItem -LiteralPath $output -Directory).Count -ne 0) {
    throw 'Component publish unexpectedly contains nested files; the native manifest contract requires a flat directory.'
}
$manifest = [ordered]@{
    protocol = 1
    installer_driver_setup = 1
    sidecar_file = 'Sunshine.Ds5Sidecar.exe'
    sidecar_sha256 = $files['Sunshine.Ds5Sidecar.exe']
    hidmaestro_version = '1.6.2.0'
    hidmaestro_archive_url = $archiveUrl
    hidmaestro_archive_sha256 = $archiveSha256
    files = $files
} | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText((Join-Path $output 'component.json'), $manifest + [Environment]::NewLine,
    (New-Object System.Text.UTF8Encoding($false)))

Write-Host "Built and verified local DualSense component: $output"
Write-Host 'No drivers or certificates were installed. See tools/sunshine-ds5-sidecar/README.md for deployment and qualification.'
