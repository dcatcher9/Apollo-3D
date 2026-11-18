# windows specific target definitions
include("${CMAKE_CURRENT_LIST_DIR}/windows_game3d.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/windows_text_hook.cmake")

# Compile the source-pinned Host SBS shaders at build time using the same source
# snapshots and D3D compiler as the live runtime. Embed the generated resources in each consumer.
add_executable(embed-host-sbs-shaders EXCLUDE_FROM_ALL
        "${CMAKE_SOURCE_DIR}/tools/embed_host_sbs_shaders.cpp"
        "${CMAKE_SOURCE_DIR}/src/host_sbs_shader_cache.cpp"
        "${CMAKE_SOURCE_DIR}/src/crypto.cpp"
        "${CMAKE_SOURCE_DIR}/src/logging.cpp")
set_target_properties(embed-host-sbs-shaders PROPERTIES CXX_STANDARD 23 LINK_SEARCH_START_STATIC 1)
target_compile_options(embed-host-sbs-shaders PRIVATE ${SUNSHINE_COMPILE_OPTIONS})
target_link_options(embed-host-sbs-shaders PRIVATE -municode)
target_link_libraries(embed-host-sbs-shaders PRIVATE
        Boost::log Boost::format libdisplaydevice::display_device D3DCompiler
        libcrypto.a crypt32 bcrypt ws2_32 z)
add_dependencies(embed-host-sbs-shaders verify_depth_coordinate_v2_contract)

set(HOST_SBS_SHADER_RESOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/host-sbs-shaders")
set(HOST_SBS_SHADER_RESOURCE "${HOST_SBS_SHADER_RESOURCE_DIR}/bytecode.rc")
set(HOST_SBS_SHADER_RESOURCE_DEPFILE "${HOST_SBS_SHADER_RESOURCE_DIR}/bytecode.d")
add_custom_command(
        OUTPUT "${HOST_SBS_SHADER_RESOURCE}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${HOST_SBS_SHADER_RESOURCE_DIR}"
        COMMAND embed-host-sbs-shaders
                "${SUNSHINE_SOURCE_ASSETS_DIR}/windows/assets/shaders/directx"
                "${HOST_SBS_SHADER_RESOURCE}" "${HOST_SBS_SHADER_RESOURCE_DEPFILE}"
        DEPENDS embed-host-sbs-shaders
        DEPFILE "${HOST_SBS_SHADER_RESOURCE_DEPFILE}"
        COMMENT "Preparing authenticated Host SBS shader resources"
        VERBATIM)
add_library(host_sbs_shader_resources OBJECT "${HOST_SBS_SHADER_RESOURCE}")
set_target_properties(host_sbs_shader_resources PROPERTIES INCLUDE_DIRECTORIES "")

set_target_properties(sunshine PROPERTIES LINK_SEARCH_START_STATIC 1)
set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll")
find_library(ZLIB ZLIB1)
list(APPEND SUNSHINE_EXTERNAL_LIBRARIES
        $<TARGET_OBJECTS:sunshine_rc_object>
        $<TARGET_OBJECTS:host_sbs_shader_resources>
        Windowsapp.lib
        Wtsapi32.lib)
