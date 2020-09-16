#ifndef CUKERNELS_H_
#define CUKERNELS_H_

#include <cuda.h>
#include <cuComplex.h>
#include <cuda_fp16.h>

#include "psrdada_cpp/cryopaf/types.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{

__device__
__half2 __hCmul2(__half2 a, __half2 b);

template<typename T, typename U>__global__
void simple_bf_tafpt_power(const T *idata, U *odata, const T *weights, const bf_config_t conf);

template<typename T, typename U>__global__
void bf_tafpt_power(const T *idata, U *odata, const T *weights, const bf_config_t conf);

template<typename T>__global__
void simple_bf_tafpt_voltage(const T *idata, T *odata, const T *weights, const bf_config_t conf);

template<typename T>__global__
void bf_tfpat_voltage(const T *idata, T *odata, const T *weights, const bf_config_t conf);

// __global__
// void bf_tfpat_voltage(const __half2 *idata, __half2 *odata, const __half2 *weights, const bf_config_t *conf);

} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/beamforming/src/cu_kernels.cu"

#endif /* CUKERNELS_H_ */
