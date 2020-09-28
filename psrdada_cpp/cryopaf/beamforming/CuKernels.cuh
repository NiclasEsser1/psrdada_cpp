#ifndef CUKERNELS_CUH_
#define CUKERNELS_CUH_

#include <stdio.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "psrdada_cpp/cryopaf/types.cuh"


#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{


__device__
__half2 __hCmul2(__half2 a, __half2 b);


template<typename T, typename U>__global__
void simple_bf_tafpt_power(const T *idata, U *odata, const T *weights, const bf_config_t conf);

template<typename T, typename U>__global__
void bf_tafpt_power(const T *idata, U *odata, const cudaTextureObject_t weights, const bf_config_t conf);

template<typename T>__global__
void simple_bf_tafpt_voltage(const T *idata, T *odata, const T *weights, const bf_config_t conf);

template<typename T>__global__
void bf_tfpat_voltage(const T *idata, T *odata, const T *weights, const bf_config_t conf);

#include "psrdada_cpp/cryopaf/beamforming/src/CuKernels.cu"

} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp


#endif /* CUKERNELS_CUH_ */
