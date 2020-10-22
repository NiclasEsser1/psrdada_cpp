#ifndef CUKERNELS_CUH_
#define CUKERNELS_CUH_

#include <stdio.h>
#include <cuda.h>
#include <cuComplex.h>
#include <cuda_fp16.h>



namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{

// namespace cg = cooperative_groups;

__device__ __half2 __hCmul2(__half2 a, __half2 b);

template<typename T>
__host__ __device__ T cmadd(T a, T b, T c = 0);

template<typename T>
__host__ __device__ T cadd(T a, T b);

template <typename T>
__device__ void warp_reduce32(T* idata, int warp_id, int tid);

template<typename T, typename U>
__device__ void warp_reduce_v2p(T *s_odata, U *s_idata, int warp_idx);



#include "psrdada_cpp/cryopaf/details/DeviceFunc.cu"

} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp


#endif /* CUKERNELS_CUH_ */
