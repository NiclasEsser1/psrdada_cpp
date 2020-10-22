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
#include "psrdada_cpp/cryopaf/DeviceFunc.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{


// NOTE: All function which are declared as __global__ are wrapped/called by the VoltageBeamformer class. Kernel implementation can be found in 'details/VoltageBeamformKernels.cu'


/**
* @brief 	kernel function with a naive approach for voltage based beamforming.
*
* @param	idata	raw voltage (format: TFAP)
* @param	weight	weight for beamforming (format: BFAP)
* @param	odata	in voltage (format: BFT)
*/
template<typename T>__global__
void simple_bf_tafp_voltage(const T *idata, T *odata, const T *weights, const bf_config_t conf);


/**
* @brief 	kernel function with a more optimized approach for voltage based beamforming.
*
* @param	idata	raw voltages (linear 1D; format: TFAP)
* @param	weight	weight for beamforming (texture 3D; format: B-F-AP)
* @param	odata	output power / Stokes I (linear 1D; format: BFT)
*/
template<typename T>__global__
void bf_tfpa_voltage(const T *idata, T *odata, const T *weights, const bf_config_t conf);

template <class T>
class VoltageBeamformer {
public:


	/**
	* @brief	Constructs a VoltageBeamform object
	*
	* @param	conf		Pointer to a configuration object of type bf_config_t (defined in typs.cuh)
	* @param	device_id	Identifier of the GPU which executes the beamforming (default 0)
	*
	* @detail	Initializes the enviroment (without allocating memory) for beamcalculation (e.g. device selection, kernel layout, and shared memory checks)
	*/
	VoltageBeamformer(bf_config_t *conf, int device_id = 0);


	/**
	* @brief	Deconstructs a PowerBeamformer object
	*/
	virtual ~VoltageBeamformer();


	/**
	* @brief	If paramaters of the configuration are changing over time, this function can be called to reinitialize the object
	*
	* @param	conf		pointer to configuration object
	*/
	void init(bf_config_t *conf = nullptr);

	/**
	* @brief	Process a batched block of raw voltages. All necessary parameters are provided by bf_conf_t
	*
	* @param	in			input data vector with linear alignment. The dataformat of this vector is FTAP
	* @param	out			output data vector with linear alignment. The dataformat of this vector is BFT
	* @param	weight		input data vector with a linear alignment. The dataformat of this vector is BFAP
	*
	*/
	void process(
		const thrust::device_vector<T>& in,
		thrust::device_vector<T>& out,
		const thrust::device_vector<T>& weights);

	/**
	* @brief	Prints the block and grid layout of a kernel (mostly used for debugging purposes)
	*/
	void print_layout();
private:


	/**
	* @brief 	internal function which checks if the required shared memory size can be provided by an GPU device
	*/
	void check_shared_mem_size();

private:
	int _device_id;
	bool _success = true;
	bf_config_t *_conf;	// TODO: non pointer


	cudaDeviceProp _prop;
	std::size_t _shared_mem_static;
	std::size_t _shared_mem_dynamic;
	std::size_t _shared_mem_total;

	dim3 _grid_layout;
	dim3 _block_layout;

	cudaStream_t _h2d_stream;
	cudaStream_t _proc_stream;
	cudaStream_t _d2h_stream;

	cublasHandle_t blas_handle;
	cudaDataType blas_dtype;
	cudaDataType blas_ctype;

	// TensorGETT<Tensor<T>, Tensor<T>, Tensor<T>> *gett;

};


template class VoltageBeamformer<__half2>;
template class VoltageBeamformer<float2>;
// template class VoltageBeamformer<double2>;


} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/details/VoltageBeamformerKernels.cu"
#include "psrdada_cpp/cryopaf/src/VoltageBeamformer.cu"
#endif /* VOLTAGE_BEAMFORMER_CUH_ */
