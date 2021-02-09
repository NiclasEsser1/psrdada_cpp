/*
 * cuda_beamformer.cuh
 *
 *  Created on: Aug 4, 2020
 *      Author: niclas
 */

#ifndef VOLTAGEBEAMFORMER_CUH_
#define VOLTAGEBEAMFORMER_CUH_

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <unordered_map>

#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/multilog.hpp"

#include "psrdada_cpp/cryopaf/Types.cuh"
// #include "psrdada_cpp/cryopaf/details/VoltageBeamformerKernels.cu"

namespace psrdada_cpp{
namespace cryopaf{

template<class HandlerType, class InputType, class WeightType, class OutputType>
class VoltageBeamformer{
public:


	/**
	* @brief	Constructs a VoltageBeamform object
	*
	* @param	conf		Pointer to a configuration object of type bf_config_t (defined in typs.cuh)
	* @param	device_id	Identifier of the GPU which executes the beamforming (default 0)
	*
	* @detail	Initializes the enviroment (without allocating memory) for beamcalculation (e.g. device selection, kernel layout, and shared memory checks)
	*/
	VoltageBeamformer(bf_config_t& conf, MultiLog &log, HandlerType &handler);


	/**
	* @brief	Deconstructs a PowerBeamformer object
	*/
	~VoltageBeamformer();


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



private:

	HandlerType &_handler;

	InputType *_input_buffer;
	WeightType *_weight_buffer;
	OutputType *_output_buffer;

	std::size_t _shared_mem_static;
	std::size_t _shared_mem_dynamic;
	std::size_t _shared_mem_total;

	dim3 _grid_layout;
	dim3 _block_layout;

	cudaDeviceProp _prop;
	cudaStream_t _stream;

	MultiLog &_log;
	bf_config_t& _conf;

	cutensorHandle_t _cutensor_handle;
	cutensorContractionDescriptor_t _cutensor_desc;
	cutensorContractionFind_t _cutensor_find;
	cutensorContractionPlan_t _cutensor_plan;
	cutensorComputeType_t _cutensor_type = CUTENSOR_COMPUTE_TF32;
	cutensorAlgo_t _cutensor_algo = CUTENSOR_ALGO_DEFAULT;
	cutensorOperator_t _operator = CUTENSOR_OP_IDENTITY;
	cutensorWorksizePreference_t _work_preference = CUTENSOR_WORKSPACE_RECOMMENDED;
	uint64_t _worksize = 0;
	int32_t _max_algos = 0;
	void* _work = nullptr;
};


} // namespace cryopaf
} // namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/details/utils.cu"
#include "psrdada_cpp/cryopaf/src/VoltageBeamformer.cu"

#endif /* VOLTAGEBEAMFORMER_CUH_ */
