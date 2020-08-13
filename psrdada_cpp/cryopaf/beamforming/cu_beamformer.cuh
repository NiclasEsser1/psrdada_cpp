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
#include <cuda.h>
#include <cuComplex.h>

#include "psrdada_cpp/cryopaf/cryopaf_conf.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/cryopaf/beamforming/cu_kernels.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{


class CudaBeamformer {
public:
	/**
	* @brief			Constructs a CudaBeamformer object
	*
	* @param
	*/
	CudaBeamformer(bf_config_t *conf);
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
	void process(const thrust::device_vector<cuComplex>& in,
		thrust::device_vector<float>& out,
		const thrust::device_vector<cuComplex>& weights,
		cudaStream_t stream = NULL);

	/**
	* @brief
	*
	* @param			in  TAFPT
	*							out BTF
	* 						weights BAFPT
	*/
	void process(const thrust::device_vector<cuComplex>& in,
		thrust::device_vector<cuComplex>& out,
		const thrust::device_vector<cuComplex>& weights,
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
	bf_config_t *_conf;
	bf_config_t *_conf_device;
	cudaStream_t _stream;
	dim3 grid_layout;
	dim3 block_layout;
};

} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#endif /* CUDA_BEAMFORMER_CUH_ */
