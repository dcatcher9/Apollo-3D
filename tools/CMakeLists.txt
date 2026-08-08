cmake_minimum_required(VERSION 3.20)

project(sunshine_tools)

include_directories(${CMAKE_SOURCE_DIR})

add_executable(dxgi-info dxgi.cpp)
set_target_properties(dxgi-info PROPERTIES CXX_STANDARD 23)
target_link_libraries(dxgi-info
        ${CMAKE_THREAD_LIBS_INIT}
        dxgi
        ${PLATFORM_LIBRARIES})
target_compile_options(dxgi-info PRIVATE ${SUNSHINE_COMPILE_OPTIONS})

add_executable(audio-info audio.cpp utils.cpp)
set_target_properties(audio-info PROPERTIES CXX_STANDARD 23)
target_link_libraries(audio-info
        ${Boost_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ksuser
        ${PLATFORM_LIBRARIES})
target_compile_options(audio-info PRIVATE ${SUNSHINE_COMPILE_OPTIONS})

add_executable(sunshinesvc sunshinesvc.cpp)
set_target_properties(sunshinesvc PROPERTIES CXX_STANDARD 23)
target_link_libraries(sunshinesvc
        ${CMAKE_THREAD_LIBS_INIT}
        wtsapi32
        ${PLATFORM_LIBRARIES})
target_compile_options(sunshinesvc PRIVATE ${SUNSHINE_COMPILE_OPTIONS})

if(WIN32)
    # Out-of-process Chromium IAccessible2 tag:video geometry helper. Keeping
    # COM traversal outside Sunshine contains slow or hung providers.
    add_executable(video-dom-info video_dom_probe/main.cpp)
    set_target_properties(video-dom-info PROPERTIES CXX_STANDARD 23)
    target_link_libraries(video-dom-info
            dwmapi
            ole32
            oleacc
            oleaut32
            user32
            uuid)
    target_compile_options(video-dom-info PRIVATE ${SUNSHINE_COMPILE_OPTIONS})
    install(TARGETS video-dom-info RUNTIME DESTINATION "tools" COMPONENT application)
endif()
