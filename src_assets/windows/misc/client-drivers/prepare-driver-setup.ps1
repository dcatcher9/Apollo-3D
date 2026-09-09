# Called only when an optional driver component was selected. Stop the installed
# service, but leave manually launched hosts to the user rather than killing them.
$ErrorActionPreference = 'Stop'
try {
    foreach ($name in @('ApolloService', 'sunshinesvc')) {
        $service = Get-Service -Name $name -ErrorAction SilentlyContinue
        if ($null -ne $service -and $service.Status -ne 'Stopped') {
            Stop-Service -InputObject $service -ErrorAction Stop
            $service.WaitForStatus('Stopped', [TimeSpan]::FromSeconds(30))
        }
    }
    if (Get-Process -Name sunshine,Sunshine.Ds5Sidecar -ErrorAction SilentlyContinue) {
        throw 'A Sunshine host or DualSense helper is still running. Close it before installing optional drivers.'
    }
    exit 0
}
catch {
    Write-Output $_.Exception.Message
    exit 1
}
