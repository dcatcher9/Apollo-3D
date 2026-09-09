# Derive the template from the installed CMake copy, keeping its upgrade UI.
# Capture the old directory before its uninstaller removes the registry entry.
# Migration runs for every package, including builds without optional drivers.
file(READ "${CMAKE_ROOT}/Modules/Internal/CPack/NSIS.template.in" _nsis_template)
string(ASCII 239 187 191 _utf8_bom)
string(REGEX REPLACE "^${_utf8_bom}" "" _nsis_template "${_nsis_template}")

function(_sunshine_nsis_replace_once needle replacement)
    string(FIND "${_nsis_template}" "${needle}" _first)
    if(_first EQUAL -1)
        message(FATAL_ERROR "Unsupported CPack NSIS template: missing ${needle}")
    endif()
    string(LENGTH "${needle}" _length)
    math(EXPR _after "${_first} + ${_length}")
    string(SUBSTRING "${_nsis_template}" ${_after} -1 _remaining)
    string(FIND "${_remaining}" "${needle}" _second)
    if(NOT _second EQUAL -1)
        message(FATAL_ERROR "Unsupported CPack NSIS template: ambiguous ${needle}")
    endif()
    string(REPLACE "${needle}" "${replacement}" _updated "${_nsis_template}")
    set(_nsis_template "${_updated}" PARENT_SCOPE)
endfunction()

_sunshine_nsis_replace_once("Function .onInit\n" [=[Function .onInit
  StrCpy $PreviousInstallDirectory ""
  ReadRegStr $PreviousInstallDirectory HKLM "Software\@CPACK_PACKAGE_VENDOR@\@CPACK_PACKAGE_INSTALL_REGISTRY_KEY@" ""
  StrCmp $PreviousInstallDirectory "" 0 +2
  StrCpy $PreviousInstallDirectory "$PROGRAMFILES64\Apollo"
  ReadRegStr $PendingConfigSource HKLM "Software\@CPACK_PACKAGE_VENDOR@\Sunshine3D Setup" "ConfigurationSource"
  StrCmp $PendingConfigSource "" +2
  StrCpy $PreviousInstallDirectory $PendingConfigSource
  StrCpy $UpgradePowerShell "$WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe"
  IfFileExists "$WINDIR\Sysnative\WindowsPowerShell\v1.0\powershell.exe" 0 +2
  StrCpy $UpgradePowerShell "$WINDIR\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
]=])
set(_old_directory_hook [=[  StrCpy $3 $0 -$2 # remove "\@CPACK_NSIS_UNINSTALL_NAME@.exe" from UninstallString to get path]=])
_sunshine_nsis_replace_once("${_old_directory_hook}"
    "${_old_directory_hook}\n  StrCmp $PendingConfigSource \"\" 0 +2\n  StrCpy $PreviousInstallDirectory $3")
string(PREPEND _nsis_template "Var PreviousInstallDirectory\nVar PendingConfigSource\nVar UpgradePowerShell\n")

if(SUNSHINE_PACKAGE_CLIENT_DRIVERS)
    # CPack's preinstall hook runs after extraction. Select optional components
    # during .onInit, before extraction, including unattended setup.
    set(_selection_hook "  !insertmacro SectionList \"InitSection\"")
    file(READ "${CMAKE_CURRENT_LIST_DIR}/windows_client_drivers_init.nsh" _client_driver_init)
    _sunshine_nsis_replace_once("${_selection_hook}" "${_selection_hook}\n${_client_driver_init}")
    string(PREPEND _nsis_template "!include FileFunc.nsh\nVar ClientDriverExitCode\nVar ClientDriverPowerShell\n")
    string(APPEND _nsis_template "\nFunction .onInstSuccess\n  StrCmp $ClientDriverExitCode 0 +2\n  SetErrorLevel $ClientDriverExitCode\nFunctionEnd\n")
endif()

set(_sunshine_nsis_templates "${CMAKE_BINARY_DIR}/sunshine-nsis-cpack")
file(MAKE_DIRECTORY "${_sunshine_nsis_templates}")
file(WRITE "${_sunshine_nsis_templates}/NSIS.template.in" "${_nsis_template}")
set(CPACK_MODULE_PATH "${_sunshine_nsis_templates};${CMAKE_MODULE_PATH}")
