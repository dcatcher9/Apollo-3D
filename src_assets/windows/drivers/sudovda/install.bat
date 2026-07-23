@echo off
pushd %~dp0

set "CERTUTIL=certutil"
where certutil >nul 2>&1 || set "CERTUTIL=%SystemRoot%\System32\certutil.exe"

echo ================
echo Installing cert for the SudoVDA driver...

%CERTUTIL% -addstore -f root "sudovda.cer"
%CERTUTIL% -addstore -f TrustedPublisher "sudovda.cer"

echo ================
echo Configuring the SudoVDA watchdog...

REM SudoVDA's built-in three-second default is too short for GPU initialization and
REM debugger activity. Keep crash cleanup enabled, but give Apollo enough time to
REM service the heartbeat. /reg:64 is required when this script is launched by the
REM 32-bit NSIS installer because the 64-bit UMDF driver reads the 64-bit registry view.
%SystemRoot%\System32\reg.exe add "HKLM\SOFTWARE\SudoMaker\SudoVDA" /v watchdog /t REG_DWORD /d 30 /f /reg:64
if errorlevel 1 (
    echo Failed to configure the SudoVDA watchdog.
    popd
    exit /b 1
)

echo ================
echo Removing the old driver... It's OK to show an error if you're installing the driver for the first time.

nefconc.exe --remove-device-node --hardware-id root\sudomaker\sudovda --class-guid "4D36E968-E325-11CE-BFC1-08002BE10318"

echo ================
echo Installing the new driver...

nefconc.exe --create-device-node --class-name Display --class-guid "4D36E968-E325-11CE-BFC1-08002BE10318" --hardware-id root\sudomaker\sudovda
nefconc.exe --install-driver --inf-path "SudoVDA.inf"

echo ================
echo Done!

popd
