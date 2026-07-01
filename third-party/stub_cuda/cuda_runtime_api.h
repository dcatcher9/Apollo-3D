#pragma once

#include <stddef.h>

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
typedef struct CUuuid_st {
    char bytes[16];
} cudaUUID_t;

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1
} cudaError_t;

// Some basic enums TensorRT might need
typedef enum cudaDataType {
    CUDA_R_16F = 2,
    CUDA_C_16F = 6,
    CUDA_R_32F = 0,
    CUDA_C_32F = 4,
    CUDA_R_64F = 1,
    CUDA_C_64F = 5,
    CUDA_R_8I  = 3,
    CUDA_C_8I  = 7,
    CUDA_R_8U  = 8,
    CUDA_C_8U  = 9,
    CUDA_R_32I = 10,
    CUDA_C_32I = 11,
    CUDA_R_32U = 12,
    CUDA_C_32U = 13
} cudaDataType_t;

#define __host__
#define __device__
