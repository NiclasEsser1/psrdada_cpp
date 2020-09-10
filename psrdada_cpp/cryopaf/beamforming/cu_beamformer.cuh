/*
 * cuda_beamformer.cuh
 *
 *  Created on: Aug 4, 2020
 *      Author: niclas
 */

#ifndef CUDA_BEAMFORMER_CUH_
#define CUDA_BEAMFORMER_CUH_

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/complex.h>
#include <cuda_fp16.h>
#include <cuda.h>

#include "psrdada_cpp/cryopaf/types.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/cryopaf/beamforming/cu_kernels.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{

template <class T>
class CudaBeamformer {
public:
	/**
	* @brief			Constructs a CudaBeamformer object
	*
	* @param
	*/
	CudaBeamformer(bf_config_t *conf, int device_id = 0);
	virtual ~CudaBeamformer();

	/**
	* @brief
	*
	* @param			in  TAFPT
	*							out BTF
	* 						weights BAFPT
	*/
	void init(bf_config_t *conf = nullptr);

	/**
	* @brief
	*
	* @param			in  TAFPT
	*							out BTF
	* 						weights BAFPT
	*/
	void process(
		const thrust::device_vector<thrust::complex<T>>& in,
		thrust::device_vector<T>& out,
		const thrust::device_vector<thrust::complex<T>>& weights,
		cudaStream_t stream = NULL);

	/**
	* @brief
	*
	* @param			in  TAFPT
	*							out BTF
	* 						weights BAFPT
	*/
	void process(
		const thrust::device_vector<thrust::complex<T>>& in,
		thrust::device_vector<thrust::complex<T>>& out,
		const thrust::device_vector<thrust::complex<T>>& weights,
		cudaStream_t stream = NULL);

private:
	/**
	* @brief
	*
	* @param			in  TAFPT
	*							out BTF
	* 						weights BAFPT
	*/
	void kernel_layout();

private:
	int id;
	bool success = true;
	cudaDeviceProp prop;
	bf_config_t *_conf;
	bf_config_t *_conf_device;
	cudaStream_t _stream;
	dim3 grid_layout;
	dim3 block_layout;
	std::size_t shared_mem_bytes;
};


// template class CudaBeamformer<short>;
template class CudaBeamformer<float>;
template class CudaBeamformer<double>;

} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/beamforming/src/cu_beamformer.cu"
#endif /* CUDA_BEAMFORMER_CUH_ */
