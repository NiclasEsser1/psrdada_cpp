

#ifndef POWER_BEAMFORMER_CUH_
#define POWER_BEAMFORMER_CUH_

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuComplex.h>

#include "psrdada_cpp/cryopaf/types.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/cryopaf/beamforming/CuKernels.cuh"
#include "psrdada_cpp/cryopaf/beamforming/CuTexture.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{

template <class T, class U>
class PowerBeamformer {
public:
	/**
	* @brief			Constructs a PowerBeamformer object
	*
	* @param
	*/
	PowerBeamformer(bf_config_t *conf, int device_id = 1);
	virtual ~PowerBeamformer();

	/**
	* @brief
	*
	* @param			in  TAFPT
	*							out BTF
	* 						weights BAFP
	*/
	void init(bf_config_t *conf = nullptr);

	/**
	* @brief
	*
	* @param			in  TAFPT
	*							out BTF
	* 						weights BAFP
	*/
	void process(
		const thrust::device_vector<T>& in,
		thrust::device_vector<U>& out,
		const thrust::device_vector<T>& weights,
		cudaStream_t stream = NULL);

	void upload_weights(thrust::device_vector<T> weights);
	void upload_weights(thrust::host_vector<T> weights);

	void print_layout();

private:
	int id;
	bool success = true;
	std::size_t shared_mem_bytes;
	bf_config_t *_conf;	// TODO: non pointer
	dim3 grid_layout;
	dim3 block_layout;
	cudaStream_t _stream;
	cudaDeviceProp prop;
	CudaTexture<T> *texture = nullptr;
};


template class PowerBeamformer<__half2, __half>;
template class PowerBeamformer<float2, float>;
// template class PowerBeamformer<double2>;


} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/beamforming/src/PowerBeamformer.cu"
#endif /* POWER_BEAMFORMER_CUH_ */
