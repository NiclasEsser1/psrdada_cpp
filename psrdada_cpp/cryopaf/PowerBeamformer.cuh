

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
#include "psrdada_cpp/cryopaf/DeviceFunc.cuh"
#include "psrdada_cpp/cryopaf/TextureMem.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{



// NOTE: All function which are declared as __global__ are wrapped/called by the PowerBeamformer class. Kernel implementation can be found in 'details/PowerBeamformKernels.cu'


/**
* @brief 	kernel function with a naive approach for power based beamforming. (NOTE: No capability to perform integration)
*
* @param	idata	raw voltages (format: TFAP)
* @param	weight	weight for beamforming (format: BFAP)
* @param	odata	in power (format: BFT)
*/
template<typename T, typename U>__global__
void simple_bf_tfap_power(const T *idata, U *odata, const T *weights, const bf_config_t conf);


/**
* @brief 	kernel function with a more optimized approach for power based beamforming. Allows integration as well as Stokes I detection
*
* @param	idata	raw voltages (linear 1D; format: TFAP)
* @param	weight	weight for beamforming (texture 3D; format: B-F-AP)
* @param	odata	output power / Stokes I (linear 1D; format: BFT)
*/
template<typename T, typename U>__global__
void bf_tfap_power(const T* __restrict__ idata, U *odata, const T *weights, const bf_config_t conf);


/**
* @brief 	kernel function with a more optimized approach for power based beamforming. Weights are stored in texture memory. Allows integration as well as Stokes I detection
*
* @param	idata	raw voltages (linear 1D; format: TFAP)
* @param	weight	weight for beamforming (texture 3D; format: B-F-AP)
* @param	odata	output power / Stokes I (linear 1D; format: BFT)
*/
template<typename T, typename U>__global__
void bf_tfap_power(const T* __restrict__ idata, U *odata, const cudaTextureObject_t *weights, const bf_config_t conf);


/**
* @brief 	final kernel version with the best performance
*
* @param	idata	raw voltages (linear 1D; format: TFAP)
* @param	weight	weight for beamforming (linear 1D; format: B-F-AP)
* @param	odata	output power / Stokes I (linear 1D; format: BFT)
*/
template<typename T, typename U>__global__
void coherent_bf_power(T *idata, U *odata, T *weight);


/**
*	@brief 		template class for power / stoke I dection beamforming
*
*	@template 	T 			datatype of input and weight data
* 	@template	U			datatype of output data
*/
template <class T, class U>
class PowerBeamformer
{
public:


	/**
	* @brief	Constructs a PowerBeamformer object
	*
	* @param	conf		Pointer to a configuration object of type bf_config_t (defined in typs.cuh)
	* @param	device_id	Identifier of the GPU which executes the beamforming (default 0)
	*
	* @detail	Initializes the enviroment (without allocating memory) for beamcalculation (e.g. device selection, kernel layout, and shared memory checks)
	*/
	PowerBeamformer(bf_config_t *conf, int device_id = 0);


	/**
	* @brief	Deconstructs a PowerBeamformer object
	*/
	virtual ~PowerBeamformer();


	/**
	* @brief	If paramaters of the configuration are changing over time, this function can be called to reinitialize the object
	*
	* @param	conf		pointer to configuration object
	*/
	void init(bf_config_t *conf = nullptr);


	/**
	* @brief	Process a batched block of raw voltages. All necessary parameters are provided by bf_conf_t
	*
	* @param	in			input data vector with linear alignment. The dataformat of this vector is TFAP
	* @param	out			output data vector with linear alignment. The dataformat of this vector is BFT
	* @param	weight		input data vector with a linear alignment. The dataformat of this vector is BFAP
	*
	*/
	void process(
		const thrust::device_vector<T>& in,
		thrust::device_vector<U>& out,
		const thrust::device_vector<T>& weights,
		cudaStream_t stream = NULL);


	/**
	* @brief	Overloaded function to upload weights to texture memory (may be deprecated in future). Accepts either thrust::device_vector<T> or thrust::host_vector<T>. This function instantiates an object of TextureMem<T> and loads the vector in a 3D texture array.
	*
	* @param	weights		thrust vector which contains the weights for beamforming
	*
	* @detail	Previously used to evaluate performance of using texture memory, instead using a very limited shared memory or slow global memory accesses. It was figured out that no performance improvements results in using the texture memory, therefore this function will maybe removed. On the other hand, since the texture memory has other advantages (e.g. an interleaved mode), it may possible that the texture can be used for an on-the-fly calibration. In order to do so more research has to be investigated.
	*/
	void upload_weights(thrust::device_vector<T> weights);
	void upload_weights(thrust::host_vector<T> weights);

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

	// DadaBufferLayout _dadaBufferLayout;
	// InputType input;
	// WeightType weight;
	// OutputType output

	TextureMem<T> *texture = nullptr;
};


template class PowerBeamformer<__half2, __half>;
template class PowerBeamformer<float2, float>;
// template class PowerBeamformer<double2>;


} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/details/PowerBeamformerKernels.cu"
#include "psrdada_cpp/cryopaf/src/PowerBeamformer.cu"
#endif /* POWER_BEAMFORMER_CUH_ */
