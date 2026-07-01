# TensorRT Dependency configuration

set(TENSORRT_DIR $ENV{TENSORRT_DIR} CACHE PATH "Path to the TensorRT installation directory")

if(NOT TENSORRT_DIR)
    message(FATAL_ERROR "TENSORRT_DIR environment variable is not set. Please download the TensorRT C++ Windows ZIP from developer.nvidia.com, extract it, and set TENSORRT_DIR to the extracted folder. (e.g. C:/TensorRT-10.3.0.26)")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS "${TENSORRT_DIR}/include")
find_library(TENSORRT_LIBRARY NAMES nvinfer nvinfer_11 nvinfer_10 HINTS "${TENSORRT_DIR}/lib")
find_library(TENSORRT_PLUGIN_LIBRARY NAMES nvinfer_plugin nvinfer_plugin_11 nvinfer_plugin_10 HINTS "${TENSORRT_DIR}/lib")
find_library(TENSORRT_PARSERS_LIBRARY NAMES nvonnxparser nvonnxparser_11 nvonnxparser_10 HINTS "${TENSORRT_DIR}/lib")

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    message(STATUS "Found TensorRT: ${TENSORRT_LIBRARY}")
    
    # Create the imported target
    add_library(TensorRT::TensorRT SHARED IMPORTED)
    set_target_properties(TensorRT::TensorRT PROPERTIES
        IMPORTED_IMPLIB "${TENSORRT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR};${CMAKE_SOURCE_DIR}/third-party/stub_cuda"
    )

    if(TENSORRT_PLUGIN_LIBRARY)
        add_library(TensorRT::NvInferPlugin SHARED IMPORTED)
        set_target_properties(TensorRT::NvInferPlugin PROPERTIES
            IMPORTED_IMPLIB "${TENSORRT_PLUGIN_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR};${CMAKE_SOURCE_DIR}/third-party/stub_cuda"
        )
    endif()

    if(TENSORRT_PARSERS_LIBRARY)
        add_library(TensorRT::NvOnnxParser SHARED IMPORTED)
        set_target_properties(TensorRT::NvOnnxParser PROPERTIES
            IMPORTED_IMPLIB "${TENSORRT_PARSERS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR};${CMAKE_SOURCE_DIR}/third-party/stub_cuda"
        )
    endif()
    
    # We will copy all DLLs from the TensorRT lib directory to the output directory
    file(GLOB TENSORRT_DLLS "${TENSORRT_DIR}/lib/*.dll")
    
    # Register the DLLs to be copied post-build
    set(PROJECT_TENSORRT_DLLS ${TENSORRT_DLLS} CACHE INTERNAL "TensorRT DLLs to package")
else()
    message(FATAL_ERROR "TensorRT headers or library not found in TENSORRT_DIR: ${TENSORRT_DIR}.")
endif()
