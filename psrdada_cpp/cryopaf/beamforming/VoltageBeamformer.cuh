/*
 * cuda_beamformer.cuh
 *
 *  Created on: Aug 4, 2020
 *      Author: niclas
 */

#ifndef VOLTAGE_BEAMFORMER_CUH_
#define VOLTAGE_BEAMFORMER_CUH_

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuComplex.h>

#include "psrdada_cpp/cryopaf/types.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/cryopaf/beamforming/CuKernels.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{

template <class T>
class VoltageBeamformer {
public:
	/**
	* @brief			Constructs a VoltageBeamformer object
	*
	* @param
	*/
	VoltageBeamformer(bf_config_t *conf, int device_id = 0);
	virtual ~VoltageBeamformer();

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
		thrust::device_vector<T>& out,
		const thrust::device_vector<T>& weights);

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
	cublasHandle_t blas_handle;
	cudaDataType blas_dtype;
	cudaDataType blas_ctype;
};


template class VoltageBeamformer<__half2>;
template class VoltageBeamformer<float2>;
// template class VoltageBeamformer<double2>;


} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/beamforming/src/VoltageBeamformer.cu"
#endif /* VOLTAGE_BEAMFORMER_CUH_ */
