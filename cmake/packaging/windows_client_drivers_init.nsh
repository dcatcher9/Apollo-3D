; /S controls UI; these explicit switches select the optional system changes.
; /CLIENTDRIVERS selects both, or use /MICROPHONE and /DUALSENSE independently.
StrCpy $ClientDriverExitCode 0
; NSIS normally runs as a 32-bit process; launch native PowerShell explicitly.
StrCpy $ClientDriverPowerShell "$WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe"
IfFileExists "$WINDIR\Sysnative\WindowsPowerShell\v1.0\powershell.exe" 0 +2
StrCpy $ClientDriverPowerShell "$WINDIR\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
${GetParameters} $R0
ClearErrors
${GetOptions} $R0 "/CLIENTDRIVERS" $R1
IfErrors client_drivers_init_microphone
!insertmacro SelectSection ${vbcable}
!insertmacro SelectSection ${dualsense}
client_drivers_init_microphone:
ClearErrors
${GetOptions} $R0 "/MICROPHONE" $R1
IfErrors client_drivers_init_dualsense
!insertmacro SelectSection ${vbcable}
client_drivers_init_dualsense:
ClearErrors
${GetOptions} $R0 "/DUALSENSE" $R1
IfErrors client_drivers_init_done
!insertmacro SelectSection ${dualsense}
client_drivers_init_done:
!insertmacro LoadSectionSelectedIntoVar vbcable vbcable_selected
!insertmacro LoadSectionSelectedIntoVar dualsense dualsense_selected
ClearErrors
