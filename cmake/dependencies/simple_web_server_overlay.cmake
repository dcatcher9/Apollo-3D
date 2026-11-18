# Patch outside the compiler include directory, then publish only changed bytes. Reconfiguring
# must validate the current inputs without making every HTTP consumer look out of date.
function(sunshine_prepare_simple_web_server_overlay source_dir patch_file output_dir)
    find_package(Git REQUIRED)
    set(staging_dir "${CMAKE_CURRENT_BINARY_DIR}/simple-web-server-overlay-staging")
    file(MAKE_DIRECTORY "${staging_dir}" "${output_dir}")
    set(headers
        asio_compatibility.hpp
        client_http.hpp
        client_https.hpp
        crypto.hpp
        mutex.hpp
        server_http.hpp
        server_https.hpp
        status_code.hpp
        utility.hpp)
    foreach(header IN LISTS headers)
        configure_file("${source_dir}/${header}" "${staging_dir}/${header}" COPYONLY)
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${patch_file}")

    file(REMOVE_RECURSE "${staging_dir}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" init --quiet
        WORKING_DIRECTORY "${staging_dir}"
        RESULT_VARIABLE init_result
        OUTPUT_VARIABLE init_output
        ERROR_VARIABLE init_error)
    if(NOT init_result EQUAL 0)
        message(FATAL_ERROR
                "Could not initialize the Simple-Web-Server overlay.\n"
                "${init_output}${init_error}")
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --whitespace=nowarn "${patch_file}"
        WORKING_DIRECTORY "${staging_dir}"
        RESULT_VARIABLE patch_result
        OUTPUT_VARIABLE patch_output
        ERROR_VARIABLE patch_error)
    file(REMOVE_RECURSE "${staging_dir}/.git")
    if(NOT patch_result EQUAL 0)
        message(FATAL_ERROR
                "Could not build the Simple-Web-Server request-limit overlay. "
                "The upstream header may have changed.\n"
                "${patch_output}${patch_error}")
    endif()

    foreach(header IN LISTS headers)
        configure_file("${staging_dir}/${header}" "${output_dir}/${header}" COPYONLY)
    endforeach()
endfunction()
