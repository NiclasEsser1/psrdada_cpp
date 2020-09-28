/*
 * cu_texture.cuh
 *
 *  Created on: Aug 4, 2020
 *      Author: niclas
 */

#ifndef CU_TEXTURE_CUH_
#define CU_TEXTURE_CUH_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "psrdada_cpp/cuda_utils.hpp"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{

template <class T>
class CudaTexture {
public:
  CudaTexture(std::size_t width, std::size_t height=1, std::size_t depth=1, int dev_id = 0);
  CudaTexture(cudaExtent volume);
  ~CudaTexture();

	void set(thrust::device_vector<T> vec);
  void set(thrust::host_vector<T> vec);

  void resize(std::size_t width, std::size_t height=1, std::size_t depth=1);

  void free_texture();

  void locked(bool val){lock = val;}
  bool locked(){return lock;}

	cudaTextureObject_t getTexture(){return tex;}

	void print_layout();
private:
  std::size_t byte_size;
  std::size_t dim = 3;
  std::size_t x;
  std::size_t y;
  std::size_t z;

	int id;

	cudaDeviceProp prop;

  bool lock = false;
  cudaPitchedPtr pitched_ptr;
  cudaTextureObject_t tex;
  cudaChannelFormatDesc chan_desc;
	cudaResourceDesc res_desc;
	cudaTextureDesc tex_desc;
  cudaExtent vol;
  cudaArray_t array;
  cudaMemcpy3DParms copy_params = {0};

private:
	void create_texture();
};

template class CudaTexture<__half2>;
template class CudaTexture<float2>;
// template class CudaBeamformer<double2>;


} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/beamforming/src/CuTexture.cu"
#endif /* CU_TEXTURE_CUH_ */
