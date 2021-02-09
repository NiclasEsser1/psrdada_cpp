/*
 * cuda_beamformer.cuh
 *
 *  Created on: Aug 4, 2020
 *      Author: niclas
 */

#ifndef POWERBEAMFORMER_CUH_
#define POWERBEAMFORMER_CUH_

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <unordered_map>

#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/multilog.hpp"

#include "psrdada_cpp/cryopaf/Types.cuh"

namespace psrdada_cpp{
namespace cryopaf{

/**
* @brief 	kernel function with a more optimized approach for power based beamforming. Allows integration as well as Stokes I detection
*
* @param	idata	raw voltages (linear 1D; format: TFAP)
* @param	weight	weight for beamforming (texture 3D; format: B-F-AP)
* @param	odata	output power / Stokes I (linear 1D; format: BFT)
*/
template<typename T, typename U>__global__
void bf_tfap_power(const T* __restrict__ idata, U *odata, const T *weights, const bf_config_t conf);


template<class HandlerType, class InputType, class WeightType, class OutputType>
class PowerBeamformer{
public:


	/**
	* @brief	Constructs a VoltageBeamform object
	*
	* @param	conf		Pointer to a configuration object of type bf_config_t (defined in typs.cuh)
	* @param	device_id	Identifier of the GPU which executes the beamforming (default 0)
	*
	* @detail	Initializes the enviroment (without allocating memory) for beamcalculation (e.g. device selection, kernel layout, and shared memory checks)
	*/
	PowerBeamformer(bf_config_t& conf, MultiLog &log, HandlerType &handler);


	/**
	* @brief	Deconstructs a VoltageBeamformer object
	*/
	~PowerBeamformer();


	/**
	* @brief	If paramaters of the configuration are changing over time, this function can be called to reinitialize the object
	*
	* @param	conf		pointer to configuration object
	*/
	void init(RawBytes &header_block);


	/**
	* @brief	Process a batched block of raw voltages. All necessary parameters are provided by bf_conf_t
	*
	* @param	dada_block			input data vector with linear alignment. The dataformat of this vector is FTAP
	*/
	bool operator()(RawBytes &dada_block);


	void process();

	void check_shared_mem();

	template<class T, class Type>
	void async_copy(thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>& vec);
	template<class T, class Type>
	void sync_copy(thrust::host_vector<T>& vec);


	InputType *input(){return _input_buffer;}
	WeightType *weight(){return _weight_buffer;}
	OutputType *output(){return _output_buffer;}

private:

	HandlerType &_handler;
	MultiLog &_log;
	bf_config_t& _conf;

	InputType *_input_buffer = nullptr;
	WeightType *_weight_buffer = nullptr;
	OutputType *_output_buffer = nullptr;

	std::size_t _shared_mem_static;
	std::size_t _shared_mem_dynamic;
	std::size_t _shared_mem_total;

	dim3 _grid_layout;
	dim3 _block_layout;

	cudaDeviceProp _prop;
	cudaStream_t _stream;
	cudaEvent_t start, stop;
	float ms;

	// typename InputType::_type input_dtype = _input_buffer->type();
};


} // namespace cryopaf
} // namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/details/utils.cu"
#include "psrdada_cpp/cryopaf/details/PowerBeamformerKernels.cu"
#include "psrdada_cpp/cryopaf/src/PowerBeamformer.cu"

#endif /* POWERBEAMFORMER_CUH_ */
