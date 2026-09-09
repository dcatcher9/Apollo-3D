# Optional end-user driver setup. Keep the .NET SDK out of the native build:
# packagers supply a component built by scripts/build-ds5-sidecar.ps1.
option(SUNSHINE_PACKAGE_CLIENT_DRIVERS "Include optional microphone and DualSense setup in Windows packages" OFF)
set(SUNSHINE_DS5_COMPONENT_DIR "" CACHE PATH "Verified component produced by scripts/build-ds5-sidecar.ps1")
set(SUNSHINE_VBCABLE_ARCHIVE "" CACHE FILEPATH "Optional local copy of the pinned VB-CABLE Pack45 ZIP")

if(NOT SUNSHINE_PACKAGE_CLIENT_DRIVERS)
    return()
endif()

if(NOT IS_DIRECTORY "${SUNSHINE_DS5_COMPONENT_DIR}")
    message(FATAL_ERROR "SUNSHINE_PACKAGE_CLIENT_DRIVERS requires SUNSHINE_DS5_COMPONENT_DIR. Build it with scripts/build-ds5-sidecar.ps1 first.")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/validate_ds5_package.cmake")
sunshine_validate_ds5_package("${SUNSHINE_DS5_COMPONENT_DIR}")

if(NOT SUNSHINE_VBCABLE_ARCHIVE)
    set(SUNSHINE_VBCABLE_ARCHIVE "${CMAKE_BINARY_DIR}/client-driver-cache/VBCABLE_Driver_Pack45.zip")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/client-driver-cache")
endif()
set(_vbcable_sha256 b950e39f01af1d04ea623c8f6d8eb9b6ea5c477c637295fabf20631c85116bfb)
if(NOT EXISTS "${SUNSHINE_VBCABLE_ARCHIVE}")
    file(DOWNLOAD "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"
        "${SUNSHINE_VBCABLE_ARCHIVE}" EXPECTED_HASH "SHA256=${_vbcable_sha256}"
        TLS_VERIFY ON TIMEOUT 120 STATUS _vbcable_download)
    list(GET _vbcable_download 0 _vbcable_download_code)
    if(NOT _vbcable_download_code EQUAL 0)
        message(FATAL_ERROR "VB-CABLE download failed: ${_vbcable_download}")
    endif()
endif()
file(SHA256 "${SUNSHINE_VBCABLE_ARCHIVE}" _vbcable_actual)
if(NOT _vbcable_actual STREQUAL _vbcable_sha256)
    message(FATAL_ERROR "VB-CABLE package is not the pinned official Pack45 archive.")
endif()

set(_client_driver_scripts "${SUNSHINE_SOURCE_ASSETS_DIR}/windows/misc/client-drivers")
install(FILES "${SUNSHINE_VBCABLE_ARCHIVE}" DESTINATION "drivers/vbcable"
    RENAME "VBCABLE_Driver_Pack45.zip" COMPONENT vbcable)
install(FILES "${_client_driver_scripts}/VB-CABLE-NOTICE.txt" "${_client_driver_scripts}/VB-CABLE-SETUP-CONTRACT.md"
    DESTINATION "drivers/vbcable" COMPONENT vbcable)
install(FILES "${_client_driver_scripts}/install-vbcable.ps1"
    DESTINATION "scripts" COMPONENT vbcable)
install(FILES "${_client_driver_scripts}/install-dualsense.ps1"
    DESTINATION "scripts" COMPONENT dualsense)
install(FILES "${_client_driver_scripts}/prepare-driver-setup.ps1"
    DESTINATION "scripts" COMPONENT assets)

# The complete upstream Core embeds SDK/WDK tools whose redistribution terms
# have not been established. Only our helper/runtime and notices are packaged.
# Setup downloads the exact upstream Core directly on the end user's machine.
# Retain the full manifest so activation requires the downloaded file's hash.
file(GLOB _ds5_payload LIST_DIRECTORIES false "${SUNSHINE_DS5_COMPONENT_DIR}/*")
list(FILTER _ds5_payload EXCLUDE REGEX "/HIDMaestro[.]Core[.]dll$")
install(FILES ${_ds5_payload}
    DESTINATION "tools/sunshine-ds5-component/payload" COMPONENT dualsense)
# Repeat validation at package time so changing the supplied directory after
# configure cannot silently create an incomplete or unreviewed release payload.
install(CODE "include(\"${CMAKE_CURRENT_LIST_DIR}/validate_ds5_package.cmake\")\nsunshine_validate_ds5_package(\"${SUNSHINE_DS5_COMPONENT_DIR}\")"
    COMPONENT dualsense)

set(CPACK_COMPONENT_VBCABLE_DISPLAY_NAME "Microphone forwarding (VB-CABLE)")
set(CPACK_COMPONENT_VBCABLE_DESCRIPTION "Install VB-Audio VB-CABLE for client microphone forwarding. Donationware: vb-audio.com/Cable/; contributions welcome. Requires a reboot and a compatible client.")
set(CPACK_COMPONENT_VBCABLE_GROUP "Drivers")
set(CPACK_COMPONENT_VBCABLE_DISABLED TRUE)
set(CPACK_COMPONENT_VBCABLE_DEPENDS application assets)
set(CPACK_COMPONENT_DUALSENSE_DISPLAY_NAME "DualSense haptics (Internet required)")
set(CPACK_COMPONENT_DUALSENSE_DESCRIPTION "Download HIDMaestro 1.6.2 and install its virtual-controller drivers and bundled USB/IP transport. Adds a local driver-signing certificate; USB devices may briefly reconnect. Requires a compatible client.")
set(CPACK_COMPONENT_DUALSENSE_GROUP "Drivers")
set(CPACK_COMPONENT_DUALSENSE_DISABLED TRUE)
set(CPACK_COMPONENT_DUALSENSE_DEPENDS application assets)

# CPack writes these variables into a second CMake file before NSIS sees them.
file(READ "${CMAKE_CURRENT_LIST_DIR}/windows_client_drivers_install.nsh" _client_driver_nsis)
string(REPLACE "\\" "\\\\" _client_driver_nsis "${_client_driver_nsis}")
string(REPLACE "$" "\\$" _client_driver_nsis "${_client_driver_nsis}")
string(REPLACE "\"" "\\\"" SUNSHINE_NSIS_CLIENT_DRIVER_SETUP "${_client_driver_nsis}")
