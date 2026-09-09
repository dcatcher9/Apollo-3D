; ViGEm setup is synchronous and suppresses both EXE and MSI UI.
SectionGetFlags ${gamepad} $R3
IntOp $R3 $R3 & ${SF_SELECTED}
StrCmp $R3 0 client_drivers_gamepad_done
nsExec::ExecToLog '"$ClientDriverPowerShell" -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\scripts\install-gamepad.ps1"'
Pop $R2
StrCmp $R2 0 client_drivers_gamepad_done
StrCmp $R2 3010 client_drivers_gamepad_reboot
DetailPrint "Virtual Gamepad setup failed."
SetErrorLevel 1
StrCpy $ClientDriverExitCode 1
Goto client_drivers_gamepad_done
client_drivers_gamepad_reboot:
SetRebootFlag true
StrCpy $ClientDriverExitCode 3010
client_drivers_gamepad_done:

; Component selection, not stale files from an earlier installation, controls setup.
SectionGetFlags ${vbcable} $R0
IntOp $R0 $R0 & ${SF_SELECTED}
SectionGetFlags ${dualsense} $R1
IntOp $R1 $R1 & ${SF_SELECTED}
IntOp $R2 $R0 | $R1
StrCmp $R2 0 client_drivers_done

nsExec::ExecToLog '"$ClientDriverPowerShell" -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\scripts\prepare-driver-setup.ps1"'
Pop $R2
StrCmp $R2 0 client_drivers_ready
DetailPrint "Optional driver setup skipped: close Sunshine and run setup again."
MessageBox MB_OK|MB_ICONEXCLAMATION "Optional drivers could not be installed. Close Sunshine, then run setup again and select the desired drivers. The main application will still be installed." /SD IDOK
SetErrorLevel 1
StrCpy $ClientDriverExitCode 1
Goto client_drivers_done

client_drivers_ready:
; nsExec may use scratch registers; read the component flags again.
SectionGetFlags ${vbcable} $R0
IntOp $R0 $R0 & ${SF_SELECTED}
StrCmp $R0 0 client_drivers_dualsense
IfSilent client_drivers_vbcable_silent client_drivers_vbcable_interactive
client_drivers_vbcable_silent:
nsExec::ExecToLog '"$ClientDriverPowerShell" -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\scripts\install-vbcable.ps1" -PackageDirectory "$INSTDIR\drivers\vbcable" -LogDirectory "$INSTDIR\config\driver-setup" -Silent'
Goto client_drivers_vbcable_result
client_drivers_vbcable_interactive:
nsExec::ExecToLog '"$ClientDriverPowerShell" -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\scripts\install-vbcable.ps1" -PackageDirectory "$INSTDIR\drivers\vbcable" -LogDirectory "$INSTDIR\config\driver-setup"'
client_drivers_vbcable_result:
Pop $R2
StrCmp $R2 0 client_drivers_dualsense
StrCmp $R2 3010 client_drivers_vbcable_reboot
DetailPrint "VB-CABLE setup failed or was canceled. See config\driver-setup."
MessageBox MB_OK|MB_ICONEXCLAMATION "Microphone driver setup failed or was canceled. See the logs in $INSTDIR\config\driver-setup. You can retry setup later." /SD IDOK
SetErrorLevel 1
StrCpy $ClientDriverExitCode 1
Goto client_drivers_dualsense
client_drivers_vbcable_reboot:
SetRebootFlag true
StrCmp $ClientDriverExitCode 1 +2
StrCpy $ClientDriverExitCode 3010
DetailPrint "VB-CABLE installed. Restart Windows before forwarding a microphone."
Goto client_drivers_dualsense

client_drivers_dualsense:
SectionGetFlags ${dualsense} $R1
IntOp $R1 $R1 & ${SF_SELECTED}
StrCmp $R1 0 client_drivers_done
nsExec::ExecToLog '"$ClientDriverPowerShell" -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\scripts\install-dualsense.ps1" -PayloadDirectory "$INSTDIR\tools\sunshine-ds5-component\payload" -ComponentRoot "$INSTDIR\tools\sunshine-ds5-component" -LogDirectory "$INSTDIR\config\driver-setup"'
Pop $R2
StrCmp $R2 0 client_drivers_done
StrCmp $R2 3010 client_drivers_dualsense_reboot
DetailPrint "DualSense setup failed. See config\driver-setup."
MessageBox MB_OK|MB_ICONEXCLAMATION "DualSense driver setup did not complete. See the logs in $INSTDIR\config\driver-setup. Ordinary streaming remains available; run setup again to retry." /SD IDOK
SetErrorLevel 1
StrCpy $ClientDriverExitCode 1
Goto client_drivers_done
client_drivers_dualsense_reboot:
SetRebootFlag true
StrCmp $ClientDriverExitCode 1 +2
StrCpy $ClientDriverExitCode 3010
client_drivers_done:
