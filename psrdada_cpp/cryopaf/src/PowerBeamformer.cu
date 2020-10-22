#ifdef POWER_BEAMFORMER_CUH_

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{



template<class T, class U>
PowerBeamformer<T, U>::PowerBeamformer(bf_config_t *conf, int device_id)
	: _conf(conf), _device_id(device_id)
{
	// std::cout << "Creating instance of PowerBeamformer" << std::endl;
	// Set device to use
	CUDA_ERROR_CHECK(cudaSetDevice(_device_id));
	// Retrieve device properties
	CUDA_ERROR_CHECK(cudaGetDeviceProperties(&_prop, _device_id));
	// initialize beamformer enviroment
	init();

}

template<class T, class U>
PowerBeamformer<T, U>::~PowerBeamformer()
{
	// std::cout << "Destroying instance of PowerBeamformer" << std::endl;
}



template<class T, class U>
void PowerBeamformer<T, U>::init(bf_config_t *conf)
{
	// If new configuration is passed
	if(conf){_conf = conf;}
	// Make kernel layout for GPU-Kernel
	switch(_conf->bf_type){
		case SIMPLE_BF_TAFPT:
			_grid_layout.x = (_conf->n_samples < NTHREAD) ? 1 : _conf->n_samples/NTHREAD;
			_grid_layout.y = _conf->n_beam;
			_grid_layout.z = _conf->n_channel;
			_block_layout.x = NTHREAD; //(_conf->n_samples < NTHREAD) ? _conf->n_samples : NTHREAD;
			break;
		case BF_TFAP:
			_shared_mem_static = (SHARED_IDATA + WARPS * WARP_SIZE) * sizeof(T);
			_shared_mem_dynamic = _conf->n_beam * sizeof(U) + _conf->n_antenna * _conf->n_pol * sizeof(T);
			// _shared_mem_dynamic = _conf->n_beam * sizeof(U) + NTHREAD * sizeof(T) ;
			_shared_mem_total = _shared_mem_static + _shared_mem_dynamic;
			check_shared_mem_size();
			_grid_layout.x = _conf->n_samples / _conf->interval;
			_grid_layout.y = _conf->n_channel;
			_block_layout.x = NTHREAD;
			break;
		case BF_TFAP_TEX:
			_shared_mem_static = SHARED_IDATA * sizeof(T);
			_shared_mem_dynamic = _conf->n_beam * sizeof(U) + NTHREAD * sizeof(T) + _conf->n_antenna * _conf->n_pol * sizeof(T);
			_shared_mem_total = _shared_mem_static + _shared_mem_dynamic;
			check_shared_mem_size();
			_grid_layout.x = _conf->n_samples / _conf->interval;
			_grid_layout.y = _conf->n_channel;
			_block_layout.x = NTHREAD;
			break;
		case BF_TFAP_V2:
			_shared_mem_static = (N_ELEMENTS_CB * (1 + WARPS_CB) + WARPS_CB * WARP_SIZE) * sizeof(T)
			 	+ (WARPS_CB + WARP_SIZE) * sizeof(U);
			_shared_mem_dynamic = 0;
			_shared_mem_total = _shared_mem_static + _shared_mem_dynamic;
			check_shared_mem_size();
			_grid_layout.x = _conf->n_beam / WARPS_CB;
			_grid_layout.y = _conf->n_channel;
			_block_layout.x = N_THREAD_CB;
			break;
		default:
			std::cout << "Beamform type not known..." << std::endl;
			break;
	}
	// print_layout();
}


template<class T, class U>
void PowerBeamformer<T, U>::process(
	const thrust::device_vector<T>& in,
	thrust::device_vector<U>& out,
	const thrust::device_vector<T>& weights,
	cudaStream_t stream)
{
	if(!_success){return;}
	// Cast raw data pointer for passing to CUDA kernel
	const T *p_in = thrust::raw_pointer_cast(in.data());
	const T *p_weights = thrust::raw_pointer_cast(weights.data());
	U *p_out = thrust::raw_pointer_cast(out.data());

	// Switch to desired CUDA kernel
	switch(_conf->bf_type)
	{
		case SIMPLE_BF_TAFPT:
			std::cout << "Power beamformer (Stokes I): simple TFAPT" << std::endl;
			simple_bf_tfap_power<<<_grid_layout, _block_layout>>>(p_in, p_out, p_weights, *_conf);
			break;
		case BF_TFAP:
			if constexpr (std::is_same<T, half2>::value)
			{
				throw std::runtime_error("Not implemented yet.");
			}
			std::cout << "Power beamformer (Stokes I): optimized TFAP" << std::endl;
			bf_tfap_power<<<_grid_layout, _block_layout, _shared_mem_dynamic>>>(p_in, p_out, p_weights, *_conf);
			break;
		case BF_TFAP_TEX:
			if constexpr (std::is_same<T, half2>::value)
			{
				throw std::runtime_error("Not implemented yet.");
			}
			std::cout << "Power beamformer (Stokes I): optimized TFAP with texture mem" << std::endl;
			this->upload_weights(weights);
			bf_tfap_power<<<_grid_layout, _block_layout, _shared_mem_dynamic>>>(p_in, p_out, texture->getTexture(), *_conf);
			break;
		case BF_TFAP_V2:
			if constexpr (std::is_same<T, float2>::value)
			{
				throw std::runtime_error("Not implemented yet.");
			}
			std::cout << "Power beamformer (Stokes I): optimized TFAPv2" << std::endl;
			coherent_bf_power<<<_grid_layout, _block_layout>>>(p_in, p_out, p_weights);
			break;
		default:
			std::cout << "Beamform type " << std::to_string(_conf->bf_type) << " not known..." << std::endl;
			break;
	}

}

template<class T, class U>
void PowerBeamformer<T, U>::upload_weights(thrust::device_vector<T> weights)
{
	texture = new TextureMem<T>(_conf->n_antenna * _conf->n_pol, _conf->n_channel, _conf->n_beam, _device_id);
	texture->set(weights);
}


template<class T, class U>
void PowerBeamformer<T, U>::upload_weights(thrust::host_vector<T> weights)
{
	texture = new TextureMem<T>(_conf->n_antenna * _conf->n_pol, _conf->n_channel, _conf->n_beam, _device_id);
	texture->set(weights);
}

template<class T, class U>
void PowerBeamformer<T, U>::print_layout()
{
	std::cout << " Kernel layout: " << std::endl
		<< " g.x = " << std::to_string(_grid_layout.x) << std::endl
		<< " g.y = " << std::to_string(_grid_layout.y) << std::endl
		<< " g.z = " << std::to_string(_grid_layout.z) << std::endl
		<< " b.x = " << std::to_string(_block_layout.x)<< std::endl
		<< " b.y = " << std::to_string(_block_layout.y)<< std::endl
		<< " b.z = " << std::to_string(_block_layout.z)<< std::endl;
}

template<class T, class U>
void PowerBeamformer<T, U>::check_shared_mem_size()
{
	std::cout << "Required shared memory: " << std::to_string(_shared_mem_total) << " Bytes" << std::endl;
	if(_prop.sharedMemPerBlock < _shared_mem_total)
	{
		std::cout << "Attempting to increase shared memory per block size..." << std::endl;
		CUDA_ERROR_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
		// if(_prop.sharedMemPerBlock < _shared_mem_total)
		// {
		// 	std::cout << "The requested size for shared memory per block exceeds " << std::to_string(_prop.sharedMemPerBlock) << " of device "
		// 		<< std::to_string(_device_id) << std::endl
		// 		<< "Warning: Kernel will not get launched !" << std::endl;
		// 		_success = false;
		// }
	}else{
		_success = true;
	}
}

} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#endif
