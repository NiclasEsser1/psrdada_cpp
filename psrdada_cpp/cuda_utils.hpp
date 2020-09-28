#ifndef PSRDADA_CPP_CUDA_UTILS_HPP
#define PSRDADA_CPP_CUDA_UTILS_HPP

#if ENABLE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <sstream>
#include <stdexcept>


/**
 * @brief Prefix for aligned message output in unittests when using cout
 */
#define CU_MSG "[ cuda msg ] "

/**
 * @brief Macro function for error checking on cuda calls that return cudaError_t values
 * @details This macro wrapps the cuda_assert_success function which raises a
 *  std::runtime_error upon receiving any cudaError_t value that is not cudaSuccess.
 * @example CUDA_ERROR_CHECK(cudaDeviceSynchronize());
 *  CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
 */
#define CUDA_ERROR_CHECK(ans) { cuda_assert_success((ans), __FILE__, __LINE__); }

/**
 * @brief Function that raises an error on receipt of any cudaError_t
 *  value that is not cudaSuccess
 */
//inline void cuda_assert_success(cudaError_t code, const char *file, int line)
inline void cuda_assert_success(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        /* Ewan note 28/07/2015:
         * This stringstream needs to be made safe.
         * Error message formatting needs to be defined.
         */
        std::stringstream error_msg;
        error_msg << "CUDA failed with error: "
              << cudaGetErrorString(code) << std::endl
              << "File: " << file << std::endl
              << "Line: " << line << std::endl;
        throw std::runtime_error(error_msg.str());
    }
}

/**
 * @brief Macro function for error checking on cufft calls that return cufftResult values
 */
#define CUFFT_ERROR_CHECK(ans) { cufft_assert_success((ans), __FILE__, __LINE__); }

/**
 * @brief Function that raises an error on receipt of any cufftResult
 * value that is not CUFFT_SUCCESS
 */
inline void cufft_assert_success(cufftResult code, const char *file, int line)
{
    if (code != CUFFT_SUCCESS)
    {
        std::stringstream error_msg;
        error_msg << "CUFFT failed with error: ";
        switch (code)
        {
        case CUFFT_INVALID_PLAN:
            error_msg <<  "CUFFT_INVALID_PLAN";
            break;

        case CUFFT_ALLOC_FAILED:
            error_msg <<  "CUFFT_ALLOC_FAILED";
            break;

        case CUFFT_INVALID_TYPE:
            error_msg <<  "CUFFT_INVALID_TYPE";
            break;

        case CUFFT_INVALID_VALUE:
            error_msg <<  "CUFFT_INVALID_VALUE";
            break;

        case CUFFT_INTERNAL_ERROR:
            error_msg <<  "CUFFT_INTERNAL_ERROR";
            break;

        case CUFFT_EXEC_FAILED:
            error_msg <<  "CUFFT_EXEC_FAILED";
            break;

        case CUFFT_SETUP_FAILED:
            error_msg <<  "CUFFT_SETUP_FAILED";
            break;

        case CUFFT_INVALID_SIZE:
            error_msg <<  "CUFFT_INVALID_SIZE";
            break;

        case CUFFT_UNALIGNED_DATA:
            error_msg <<  "CUFFT_UNALIGNED_DATA";
            break;

        default:
            error_msg <<  "CUFFT_UNKNOWN_ERROR";
        }
        error_msg << std::endl
              << "File: " << file << std::endl
              << "Line: " << line << std::endl;
        throw std::runtime_error(error_msg.str());
    }
}


/**
* @brief Macro function for error checking on cublas calls that return cublasStatus_t values
*/
#define CUBLAS_ERROR_CHECK(ans) {cublas_error_checker(ans, __FILE__, __LINE__);}

/**
* @brief Function that raises an error on receipt of any cublasStatus_t
* value that is not CUBLAS_STATUS_SUCCESS
*/
inline void cublas_error_checker(int code, const char* file, int line)
{
     if (code != CUBLAS_STATUS_SUCCESS)
     {
        std::stringstream error_msg;
        error_msg << "CUBLAS failed with error: ";
        switch (code)
        {
        /**
        * The cuBLAS library was not initialized. This is usually caused by the lack of a prior cublasCreate() call, an error in the CUDA Runtime API called by the cuBLAS routine, or an error in the hardware setup.
        *
        * To correct: call cublasCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.
        */
        case CUBLAS_STATUS_NOT_INITIALIZED:
            error_msg <<  "CUBLAS_STATUS_NOT_INITIALIZED";
            break;
        /**
        * Resource allocation failed inside the cuBLAS library. This is usually caused by a cudaMalloc() failure.
        *
        * To correct: prior to the function call, deallocate previously allocated memory as much as possible.
        */
        case CUBLAS_STATUS_ALLOC_FAILED:
            error_msg <<  "CUBLAS_STATUS_ALLOC_FAILED";
            break;
        /**
        * An unsupported value or parameter was passed to the function (a negative vector size, for example).
        *
        * To correct: ensure that all the parameters being passed have valid values.
        */
        case CUBLAS_STATUS_INVALID_VALUE:
            error_msg <<  "CUBLAS_STATUS_INVALID_VALUE";
            break;
        /**
        * The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.
        *
        * To correct: compile and run the application on a device with appropriate compute capability, which is 1.3 for double precision.
        */
        case CUBLAS_STATUS_ARCH_MISMATCH:
            error_msg <<  "CUBLAS_STATUS_ARCH_MISMATCH";
            break;
        /**
        * An access to GPU memory space failed, which is usually caused by a failure to bind a texture.
        *
        * To correct: prior to the function call, unbind any previously bound textures.
        */
        case CUBLAS_STATUS_MAPPING_ERROR:
            error_msg <<  "CUBLAS_STATUS_MAPPING_ERROR";
            break;
        /**
        * The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.
        *
        * To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.
        */
        case CUBLAS_STATUS_EXECUTION_FAILED:
            error_msg <<  "CUBLAS_STATUS_EXECUTION_FAILED";
            break;
        /**
        * An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.
        *
        * To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.
        */
        case CUBLAS_STATUS_INTERNAL_ERROR:
            error_msg <<  "CUBLAS_STATUS_INTERNAL_ERROR";
            break;
        /**
        * The functionality requested is not supported
        */
        case CUBLAS_STATUS_NOT_SUPPORTED:
            error_msg <<  "CUBLAS_STATUS_NOT_SUPPORTED";
            break;
        /**
        * The functionality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly.
        */
        case CUBLAS_STATUS_LICENSE_ERROR:
            error_msg <<  "CUBLAS_STATUS_LICENSE_ERROR";
            break;

        default:
            error_msg <<  "CUBLAS_UNKNOWN_ERROR";
        }
        error_msg << std::endl
           << "File: " << file << std::endl
           << "Line: " << std::to_string(line) << std::endl;
        throw std::runtime_error(error_msg.str());
     }
}


#endif //ENABLE_CUDA
#endif //PSRDADA_CPP_CUDA_UTILS_HPP
