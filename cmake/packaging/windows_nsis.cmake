# NSIS Packaging
# see options at: https://cmake.org/cmake/help/latest/cpack_gen/nsis.html

# The complete TensorRT payload exceeds NSIS's uncompressed solid-block limit.
# Its optional inner archive supplies solid compression; keep the outer stream
# non-solid and inexpensive when it carries already compressed archive data.
if(SUNSHINE_PACKAGE_TENSORRT_ARCHIVE)
    set(CPACK_NSIS_COMPRESSOR "zlib")
else()
    set(CPACK_NSIS_COMPRESSOR "lzma")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/windows_nsis_template.cmake")

set(CPACK_NSIS_INSTALLED_ICON_NAME "${PROJECT__DIR}\\\\${PROJECT_EXE}")

set(SUNSHINE_NSIS_TENSORRT_SETUP "")
if(SUNSHINE_PACKAGE_TENSORRT_ARCHIVE)
    set(SUNSHINE_NSIS_TENSORRT_SETUP
        "nsExec::ExecToLog '\\\"$UpgradePowerShell\\\" -NoProfile -ExecutionPolicy Bypass -File \\\"$INSTDIR\\\\scripts\\\\install-tensorrt-bundle.ps1\\\" -ArchivePath \\\"$INSTDIR\\\\tools\\\\tensorrt\\\\runtime.7z\\\" -ManifestPath \\\"$INSTDIR\\\\scripts\\\\tensorrt-runtime.json\\\" -SevenZipPath \\\"$INSTDIR\\\\tools\\\\tensorrt\\\\7zip\\\\7z.exe\\\" -DestinationDirectory \\\"$INSTDIR\\\"'
        Pop $R2
        StrCmp $R2 0 tensorrt_ready
        MessageBox MB_OK|MB_ICONSTOP \\\"TensorRT could not be unpacked or verified. Setup will not start the host service. Close Sunshine and run setup again.\\\" /SD IDOK
        SetErrorLevel 1
        Abort
        tensorrt_ready:")
endif()

set(_sunshine_gamepad_setup "nsExec::ExecToLog 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File \\\"$INSTDIR\\\\scripts\\\\install-gamepad.ps1\\\"'")
if(SUNSHINE_PACKAGE_CLIENT_DRIVERS)
    # The optional-driver sequence also handles ViGEm selection, native
    # PowerShell, and its synchronous failure/reboot result.
    set(_sunshine_gamepad_setup "")
endif()

# Extra install commands
# Restores permissions on the install directory
# Migrates config files from the root into the new config folder
# Install service
SET(CPACK_NSIS_EXTRA_INSTALL_COMMANDS
        "${CPACK_NSIS_EXTRA_INSTALL_COMMANDS}
        IfSilent +2 0
        # ExecShell 'open' 'https://docs.lizardbyte.dev/projects/sunshine'
        nsExec::ExecToLog 'icacls \\\"$INSTDIR\\\" /reset'
        ClearErrors
        WriteRegStr HKLM \\\"Software\\\\${CPACK_PACKAGE_VENDOR}\\\\Sunshine3D Setup\\\" \\\"ConfigurationSource\\\" \\\"$PreviousInstallDirectory\\\"
        IfErrors config_migration_failed
        nsExec::ExecToLog '\\\"$UpgradePowerShell\\\" -NoProfile -ExecutionPolicy Bypass -File \\\"$INSTDIR\\\\scripts\\\\migrate-install-config.ps1\\\" -SourceDirectory \\\"$PreviousInstallDirectory\\\" -DestinationDirectory \\\"$INSTDIR\\\"'
        Pop $R2
        StrCmp $R2 0 config_migrated
        config_migration_failed:
        MessageBox MB_OK|MB_ICONSTOP \\\"Configuration migration failed. Your previous configuration remains in its original folder. Setup will not start the host service.\\\" /SD IDOK
        SetErrorLevel 1
        Abort
        config_migrated:
        DeleteRegValue HKLM \\\"Software\\\\${CPACK_PACKAGE_VENDOR}\\\\Sunshine3D Setup\\\" \\\"ConfigurationSource\\\"
        DeleteRegKey /ifempty HKLM \\\"Software\\\\${CPACK_PACKAGE_VENDOR}\\\\Sunshine3D Setup\\\"
        ${SUNSHINE_NSIS_TENSORRT_SETUP}
        nsExec::ExecToLog '\\\"$INSTDIR\\\\scripts\\\\update-path.bat\\\" add'
        nsExec::ExecToLog '\\\"$INSTDIR\\\\drivers\\\\sudovda\\\\install.bat\\\"'
        nsExec::ExecToLog '\\\"$INSTDIR\\\\scripts\\\\migrate-config.bat\\\"'
        nsExec::ExecToLog '\\\"$INSTDIR\\\\scripts\\\\add-firewall-rule.bat\\\"'
        ${_sunshine_gamepad_setup}
        ${SUNSHINE_NSIS_CLIENT_DRIVER_SETUP}
        nsExec::ExecToLog '\\\"$INSTDIR\\\\scripts\\\\install-service.bat\\\"'
        nsExec::ExecToLog '\\\"$INSTDIR\\\\scripts\\\\autostart-service.bat\\\"'
        NoController:
        ")

# Extra uninstall commands
# Uninstall service
set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS
        "${CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS}
        nsExec::ExecToLog '\\\"$INSTDIR\\\\scripts\\\\delete-firewall-rule.bat\\\"'
        nsExec::ExecToLog '\\\"$INSTDIR\\\\scripts\\\\uninstall-service.bat\\\"'
        nsExec::ExecToLog '\\\"$INSTDIR\\\\sunshine.exe\\\" --restore-nvprefs-undo'
        ${SUNSHINE_NSIS_TENSORRT_UNINSTALL}
        MessageBox MB_YESNO|MB_ICONQUESTION \
            'Do you want to remove Virtual Gamepad?' \
            /SD IDNO IDNO NoGamepad
            nsExec::ExecToLog \
              'powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
                \\\"$INSTDIR\\\\scripts\\\\uninstall-gamepad.ps1\\\"'; \
              skipped if no
        NoGamepad:
        MessageBox MB_YESNO|MB_ICONQUESTION \
            'Do you want to remove SudoVDA Virtual Display Driver?' \
            /SD IDNO IDNO NoSudoVDA
            nsExec::ExecToLog '\\\"$INSTDIR\\\\drivers\\\\sudovda\\\\uninstall.bat\\\"'; skipped if no
        NoSudoVDA:
        MessageBox MB_YESNO|MB_ICONQUESTION \
            'Do you want to remove $INSTDIR (this includes the configuration, cover images, and settings)?' \
            /SD IDNO IDNO NoDelete
            RMDir /r \\\"$INSTDIR\\\"; skipped if no
        nsExec::ExecToLog '\\\"$INSTDIR\\\\scripts\\\\update-path.bat\\\" remove'
        NoDelete:
        ")

# Adding an option for the start menu
set(CPACK_NSIS_MODIFY_PATH OFF)
set(CPACK_NSIS_EXECUTABLES_DIRECTORY ".")
# This will be shown on the installed apps Windows settings
set(CPACK_NSIS_INSTALLED_ICON_NAME "sunshine.exe")
set(CPACK_NSIS_CREATE_ICONS_EXTRA
        "${CPACK_NSIS_CREATE_ICONS_EXTRA}
        SetOutPath '\$INSTDIR'
        Delete '\$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\${CMAKE_PROJECT_NAME}.lnk'
        CreateShortCut '\$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\${PROJECT_DISPLAY_NAME}.lnk' \
            '\$INSTDIR\\\\sunshine.exe' '--shortcut'
        ")
set(CPACK_NSIS_DELETE_ICONS_EXTRA
        "${CPACK_NSIS_DELETE_ICONS_EXTRA}
        Delete '\$SMPROGRAMS\\\\$MUI_TEMP\\\\${PROJECT_DISPLAY_NAME}.lnk'
        Delete '\$SMPROGRAMS\\\\$MUI_TEMP\\\\${CMAKE_PROJECT_NAME}.lnk'
        ")

# Checking for previous installed versions
set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL "ON")

# set(CPACK_NSIS_HELP_LINK "https://docs.lizardbyte.dev/projects/sunshine/latest/md_docs_2getting__started.html")
# set(CPACK_NSIS_URL_INFO_ABOUT "${CMAKE_PROJECT_HOMEPAGE_URL}")
# set(CPACK_NSIS_CONTACT "${CMAKE_PROJECT_HOMEPAGE_URL}/support")

# set(CPACK_NSIS_MENU_LINKS
#         "https://docs.lizardbyte.dev/projects/sunshine" "Sunshine documentation"
#         "https://app.lizardbyte.dev" "LizardByte Web Site"
#         "https://app.lizardbyte.dev/support" "LizardByte Support")
set(CPACK_NSIS_MANIFEST_DPI_AWARE true)
