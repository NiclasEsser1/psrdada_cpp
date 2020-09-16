// #include "psrdada_cpp/cryopaf/beamforming/cu_beamformer.cuh"

#ifdef CUDA_BEAMFORMER_CUH_

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{


template<class T>
CudaBeamformer<T>::CudaBeamformer(bf_config_t *conf, int device_id)
	: _conf(conf), id(device_id)
{
	// std::cout << "Creating instance of CudaBeamformer" << std::endl;
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
			// shared_mem_bytes = sizeof(T) * (_conf->n_antenna * _conf->n_pol * (WARPS + 1)); // TODO: This is not true for power /stokes I
			shared_mem_bytes = sizeof(T) * (_conf->n_antenna * _conf->n_pol * (WARPS + 1)); // TODO: This is not true for power /stokes I
			std::cout << "Required shared memory: " << std::to_string(shared_mem_bytes) << " Bytes" << std::endl;
			if(prop.sharedMemPerBlock < shared_mem_bytes + NTHREAD)
			{
				std::cout << "The requested size for shared memory per block exceeds the size provided by device "
					<< std::to_string(id) << std::endl
					<< "! Warning: Kernel will not get launched !" << std::endl;
					success = false;
					return;
			}else{
				success = true;
			}
			grid_layout.x = _conf->n_samples * WARP_SIZE / (NTHREAD);
			grid_layout.y = _conf->n_beam;
			grid_layout.z = _conf->n_channel;
			block_layout.x = NTHREAD;
			break;

		default:
			std::cout << "Beamform type not known..." << std::endl;
			break;
	}
}


template<class T>
void CudaBeamformer<T>::init(bf_config_t *conf)
{
	// If new configuration is passed
	if(conf){_conf = conf;}
	// Make kernel layout for GPU-Kernel
	kernel_layout();
	// if(success)
	// 	print_layout();
}


template<class T>
template<class U>
void CudaBeamformer<T>::process(
	const thrust::device_vector<T>& in,
	thrust::device_vector<U>& out,
	const thrust::device_vector<T>& weights,
	cudaStream_t stream)
{
	if(!success){return;}
	// Cast raw data pointer for passing to CUDA kernel
	const T *p_in = thrust::raw_pointer_cast(in.data());
	const T *p_weights = thrust::raw_pointer_cast(weights.data());
	U *p_out = thrust::raw_pointer_cast(out.data());

	if constexpr (!std::is_same<T, U>::value)
	{
		// Switch to desired CUDA kernel
		switch(_conf->bf_type)
		{
			case SIMPLE_BF_TAFPT:
			{
				std::cout << "Power beamformer (Stokes I): simple TFAPT" << std::endl;
				simple_bf_tafpt_power<<<grid_layout, block_layout>>>(p_in, p_out, p_weights, *_conf);
				break;
			}
			case BF_TFAP:
			{
				std::cout << "Power beamformer (Stokes I): optimized TFAPT" << std::endl;
				bf_tafpt_power<<<grid_layout, block_layout>>>(p_in, p_out, p_weights, *_conf);
				break;
			}
			default:
				std::cout << "Beamform type " << std::to_string(_conf->bf_type) << " not known..." << std::endl;
				break;
		}
	}
	else
	{
		// Switch to desired CUDA kernel
		switch(_conf->bf_type)
		{
			case SIMPLE_BF_TAFPT:
			{
				std::cout << "Voltage beamformer: simple TAFPT" << std::endl;
				simple_bf_tafpt_voltage<<<grid_layout, block_layout>>>(p_in, p_out, p_weights, *_conf);
				break;
			}
			case BF_TFAP:
			{
				std::cout << "Voltage beamformer: optimzed TFAPT" << std::endl;
					bf_tfpat_voltage<<<grid_layout, block_layout, shared_mem_bytes>>>(p_in, p_out, p_weights, *_conf);
				break;
			}

			default:
				std::cout << "Beamform type " << std::to_string(_conf->bf_type) << "not known..." << std::endl;
				break;
		}
	}
}


template<class T>
void CudaBeamformer<T>::print_layout()
{
	std::cout << " Kernel layout: " << std::endl
		<< " g.x = " << std::to_string(grid_layout.x) << std::endl
		<< " g.y = " << std::to_string(grid_layout.y) << std::endl
		<< " g.z = " << std::to_string(grid_layout.z) << std::endl
		<< " b.x = " << std::to_string(block_layout.x)<< std::endl
		<< " b.y = " << std::to_string(block_layout.y)<< std::endl
		<< " b.z = " << std::to_string(block_layout.z)<< std::endl;
}

} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#endif
