# The Game 3D text helper is loaded into 32-bit text games, so it is built by a 32-bit compiler
# that the 64-bit host build does not otherwise need. Without one the host still builds and
# reports that it has no text helper; see docs/building.md.
set(SUNSHINE_TEXT_HOOK_MINGW32_DIR "C:/msys64/mingw32/bin" CACHE PATH
    "Directory of the 32-bit MinGW-w64 compiler used for the Game 3D text helper")
find_program(SUNSHINE_TEXT_HOOK_GCC32 NAMES i686-w64-mingw32-gcc
    HINTS "${SUNSHINE_TEXT_HOOK_MINGW32_DIR}" NO_DEFAULT_PATH)
find_program(SUNSHINE_TEXT_HOOK_GXX32 NAMES i686-w64-mingw32-g++
    HINTS "${SUNSHINE_TEXT_HOOK_MINGW32_DIR}" NO_DEFAULT_PATH)

set(SUNSHINE_TEXT_HOOK_BUILD_DIR "${CMAKE_BINARY_DIR}/text-hook-package")
set(SUNSHINE_TEXT_HOOK_FILES "")

if(SUNSHINE_TEXT_HOOK_GCC32 AND SUNSHINE_TEXT_HOOK_GXX32)
    set(_text_hook_source_dir "${CMAKE_SOURCE_DIR}/tools/text_hook")
    set(_text_hook_library "${SUNSHINE_TEXT_HOOK_BUILD_DIR}/SunshineTextHook32.dll")
    set(_text_hook_injector "${SUNSHINE_TEXT_HOOK_BUILD_DIR}/sunshine-text-hook-inject32.exe")
    # The compiler's own DLLs sit beside it. Its directory is the whole search path, so the
    # 64-bit toolchain's libraries of the same names can never be picked up instead.
    set(_text_hook_env "${CMAKE_COMMAND}" -E env "PATH=${SUNSHINE_TEXT_HOOK_MINGW32_DIR}")
    set(_text_hook_flags -std=c++20 -O2 -Wall -Wextra -municode -fno-exceptions -fno-rtti -static-libgcc)

    # Loaded into another application, so it must not need a C++ runtime DLL there: it is
    # compiled without exceptions or RTTI and linked by the C driver.
    add_custom_command(OUTPUT "${_text_hook_library}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SUNSHINE_TEXT_HOOK_BUILD_DIR}"
        COMMAND ${_text_hook_env} "${SUNSHINE_TEXT_HOOK_GCC32}" -x c++ ${_text_hook_flags} -shared
            -o "${_text_hook_library}" "${_text_hook_source_dir}/text_hook.cpp"
            -ldwrite -ld2d1 -lgdi32 -lole32 -luuid
        DEPENDS "${_text_hook_source_dir}/text_hook.cpp" "${_text_hook_source_dir}/text_hook_shared.h"
        COMMENT "Building the 32-bit Game 3D text helper"
        VERBATIM)
    add_custom_command(OUTPUT "${_text_hook_injector}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SUNSHINE_TEXT_HOOK_BUILD_DIR}"
        COMMAND ${_text_hook_env} "${SUNSHINE_TEXT_HOOK_GXX32}" ${_text_hook_flags} -static
            -o "${_text_hook_injector}" "${_text_hook_source_dir}/text_hook_inject.cpp"
        DEPENDS "${_text_hook_source_dir}/text_hook_inject.cpp"
        COMMENT "Building the 32-bit Game 3D text helper loader"
        VERBATIM)
    set(SUNSHINE_TEXT_HOOK_FILES "${_text_hook_library}" "${_text_hook_injector}")

    add_custom_target(text_hook_bundle ALL
        COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:sunshine>/tools/text-hook"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${SUNSHINE_TEXT_HOOK_FILES}
            "$<TARGET_FILE_DIR:sunshine>/tools/text-hook"
        DEPENDS ${SUNSHINE_TEXT_HOOK_FILES}
        COMMENT "Bundling the Game 3D text helper"
        VERBATIM)
    add_dependencies(sunshine text_hook_bundle)
else()
    message(WARNING "No 32-bit MinGW-w64 compiler in ${SUNSHINE_TEXT_HOOK_MINGW32_DIR}: "
        "the Game 3D text helper for 32-bit text games is not built.")
endif()
