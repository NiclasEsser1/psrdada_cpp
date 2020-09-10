#ifndef CUKERNELS_H_
#define CUKERNELS_H_

#include <cuda.h>
#include <thrust/complex.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include "psrdada_cpp/cryopaf/types.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{


template<typename T>__global__
void simple_bf_tafpt_power(
  const thrust::complex<T> *idata, T *odata, const thrust::complex<T> *weights, const bf_config_t *conf);

template<typename T>__global__
void bf_tafpt_power(
  const thrust::complex<T> *idata, T *odata, const thrust::complex<T> *weights, const bf_config_t *conf);

template<typename T>__global__
void simple_bf_tafpt_voltage(
  const thrust::complex<T> *idata, thrust::complex<T> *odata, const thrust::complex<T> *weights, const bf_config_t *conf);

template<typename T>__global__
void bf_tfpat_voltage(
  const thrust::complex<T> *idata, thrust::complex<T> *odata, const thrust::complex<T> *weights, const bf_config_t *conf);


} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp


#endif /* CUKERNELS_H_ */
