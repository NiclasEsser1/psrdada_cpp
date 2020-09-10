// #include "psrdada_cpp/cryopaf/beamforming/cu_beamformer.cuh"

#ifdef CUDA_BEAMFORMER_CUH_

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{


template<class T>
CudaBeamformer<T>::CudaBeamformer(bf_config_t *conf, int device_id)
	: _conf(conf), id(device_id)
{
	// Set device to use
	CUDA_ERROR_CHECK(cudaSetDevice(id))
	// Retrieve device properties
	CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, id))
	// initialize beamformer enviroment
	init();
}


template<class T>
CudaBeamformer<T>::~CudaBeamformer()
{
	// std::cout << "Destroying instance of CudaBeamformer" << std::endl;
}


template<class T>
void CudaBeamformer<T>::kernel_layout()
{
	switch(_conf->bf_type){
		case SIMPLE_BF_TAFPT:
			grid_layout.x = (_conf->n_samples < NTHREAD) ? 1 : _conf->n_samples/NTHREAD;
			grid_layout.y = _conf->n_beam;
			grid_layout.z = _conf->n_channel;
			block_layout.x = NTHREAD; //(_conf->n_samples < NTHREAD) ? _conf->n_samples : NTHREAD;
			break;
		case BF_TFAP:
			// Term is multiplied by WARP_SIZE since each block stores 32 samples of time data and + 1 for beam weights
			shared_mem_bytes = sizeof(thrust::complex<T>) * (_conf->n_antenna * _conf->n_pol * (WARP_SIZE + 1)	+ NTHREAD); // TODO: This is not true for power /stokes I
			std::cout << "Required shared memory: " << std::to_string(shared_mem_bytes) << " Bytes" << std::endl;
			if(prop.sharedMemPerBlock < shared_mem_bytes)
			{
				std::cout << "The requested size for shared memory per block exceeds the size provided by device "
					<< std::to_string(id) << std::endl
					<< "! Warning: Kernel will not get launched !" << std::endl;
					success = false;
					return;
			}
			else
			{
				success = true;
			}
			grid_layout.x = _conf->n_samples * WARP_SIZE / (NTHREAD);
			grid_layout.z = _conf->n_channel;
			grid_layout.y = _conf->n_beam;
			block_layout.x = NTHREAD; //(_conf->n_samples < NTHREAD) ? _conf->n_samples : NTHREAD;
			break;

		default:
			std::cout << "Beamform type not known..." << std::endl;
			break;
	}
	std::cout << " Kernel layout: " << std::endl
		<< " g.x = " << std::to_string(grid_layout.x) << std::endl
		<< " g.y = " << std::to_string(grid_layout.y) << std::endl
		<< " g.z = " << std::to_string(grid_layout.z) << std::endl
		<< " b.x = " << std::to_string(block_layout.x)<< std::endl
		<< " b.y = " << std::to_string(block_layout.y)<< std::endl
		<< " b.z = " << std::to_string(block_layout.z)<< std::endl;
}


template<class T>
void CudaBeamformer<T>::init(bf_config_t *conf)
{
	// If new configuration is passed
	if(conf){_conf = conf;}
	// Copy configuration struct to device
	CUDA_ERROR_CHECK(cudaMalloc((void**)&_conf_device, sizeof(bf_config_t)));
	CUDA_ERROR_CHECK(cudaMemcpy(_conf_device, _conf, sizeof(bf_config_t), cudaMemcpyHostToDevice));
	// Make kernel layout for GPU-Kernel
	kernel_layout();
}


template<class T>
void CudaBeamformer<T>::process(
	const thrust::device_vector<thrust::complex<T>>& in,
	thrust::device_vector<T>& out,
	const thrust::device_vector<thrust::complex<T>>& weights,
	cudaStream_t stream)
{
	if(!success){return;}
	// Cast raw data pointer for passing to CUDA kernel
	const thrust::complex<T> *p_in = thrust::raw_pointer_cast(in.data());
	const thrust::complex<T> *p_weights = thrust::raw_pointer_cast(weights.data());
	T *p_out = thrust::raw_pointer_cast(out.data());

	// Switch to desired CUDA kernel
	switch(_conf->bf_type)
	{

		// Simple beamforming approach, for more information see kernel description
		case SIMPLE_BF_TAFPT:
		{
			// Launch kernel
			std::cout << "Beamform Stokes I type: simple TFAPT" << std::endl;
			simple_bf_tafpt_power<<<grid_layout, block_layout>>>(p_in, p_out, p_weights, _conf_device);
			break;
		}
		case BF_TFAP:
		{
			std::cout << "Beamform Stokes I type: Not implemented yet" << std::endl;
			bf_tafpt_power<<<grid_layout, block_layout>>>(p_in, p_out, p_weights, _conf_device);
			break;
		}
		default:
			std::cout << "Beamform  Stokes I type with identifier " << std::to_string(_conf->bf_type) << " not known..." << std::endl;
			break;
	}
}


template <class T>
void CudaBeamformer<T>::process(
	const thrust::device_vector<thrust::complex<T>>& in,
	thrust::device_vector<thrust::complex<T>>& out,
	const thrust::device_vector<thrust::complex<T>>& weights,
	cudaStream_t stream)
{
	if(!success){return;}
	// Cast raw data pointer for passing to CUDA kernel
	const thrust::complex<T> *p_in = thrust::raw_pointer_cast(in.data());
	const thrust::complex<T> *p_weights = thrust::raw_pointer_cast(weights.data());
	thrust::complex<T> *p_out = thrust::raw_pointer_cast(out.data());

	// Switch to desired CUDA kernel
	switch(_conf->bf_type){

		// Simple beamforming approach, for more information see kernel description
		case SIMPLE_BF_TAFPT:
		{
			// Launch kernel
			std::cout << "Beamform type: simple TAFPT" << std::endl;
			simple_bf_tafpt_voltage<<<grid_layout, block_layout>>>(p_in, p_out, p_weights, _conf_device);
			break;
		}
		case BF_TFAP:
		{
			std::cout << "Beamform type: TFAPT shared idata" << std::endl;
			bf_tfpat_voltage<<<grid_layout, block_layout, shared_mem_bytes>>>(p_in, p_out, p_weights, _conf_device);
			break;
		}

		default:
			std::cout << "Beamform type not known..." << std::endl;
			break;
	}
}


} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#endif
