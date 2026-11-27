# common macros

# override find_package function
macro(find_package)  # cmake-lint: disable=C0103
    string(TOLOWER "${ARGV0}" ARGV0_LOWER)
    if(("${ARGV0_LOWER}" STREQUAL "boost") AND DEFINED FETCH_CONTENT_BOOST_USED)
        # Do nothing, as the package has already been fetched
    else()
        # Call the original find_package function
        _find_package(${ARGV})
    endif()
endmacro()
