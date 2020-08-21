#ifndef CUKERNELS_H_
#define CUKERNELS_H_

#include <cuda.h>
#include <cuComplex.h>
#include <stdio.h>

#include "psrdada_cpp/cryopaf/cryopaf_conf.hpp"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{


enum{LOG, LINEAR};

template<typename T> __device__
T complex_mul(const T a, const T b);

template<typename T, typename U>__global__
void simple_bf_tafpt_stokes_I(const T *idata, U *odata, const T *weights, const bf_config_t *conf);

template<typename T>__global__
void simple_bf_tafpt(const T *idata, T *odata, const T *weights, const bf_config_t *conf);


} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#endif /* CUKERNELS_H_ */
