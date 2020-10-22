#ifdef VOLTAGE_BEAMFORMER_CUH_

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{


template<class T>
VoltageBeamformer<T>::VoltageBeamformer(bf_config_t *conf, int device_id)
	: _conf(conf), _device_id(device_id)
{
	// std::cout << "Creating instance of VoltageBeamformer" << std::endl;
	// Set device to use
	CUDA_ERROR_CHECK(cudaSetDevice(_device_id))
	// Retrieve device properties
	CUDA_ERROR_CHECK(cudaGetDeviceProperties(&_prop, _device_id))
	// initialize beamformer enviroment
	init();

}


template<class T>
VoltageBeamformer<T>::~VoltageBeamformer()
{
	// std::cout << "Destroying instance of VoltageBeamformer" << std::endl;
}



template<class T>
void VoltageBeamformer<T>::init(bf_config_t *conf)
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
			// shared_mem_bytes = sizeof(T) * (_conf->n_antenna * _conf->n_pol * (WARPS + 1)); // TODO: This is not true for power /stokes I
			_shared_mem_static = 0;
			_shared_mem_dynamic = sizeof(T) * (_conf->n_antenna * _conf->n_pol * (WARPS + 1));
			_shared_mem_total = _shared_mem_static + _shared_mem_dynamic;
			check_shared_mem_size();
			_grid_layout.x = _conf->n_samples * WARP_SIZE / (NTHREAD);
			_grid_layout.y = _conf->n_beam;
			_grid_layout.z = _conf->n_channel;
			_block_layout.x = NTHREAD;
			break;
		case CUTENSOR_BF_TFAP:
		{
			// std::vector<int> modeA{'F','T','A'};
		    // std::vector<int> modeB{'B','F','A'};
		    // std::vector<int> modeC{'B','F','T'};
			// std::unordered_map<int, int64_t> extent;
		    // extent['F'] = N_CHANNEL;
		    // extent['T'] = N_TIMESTAMPS;
		    // extent['A'] = N_ELEMENTS;
		    // // extent['P'] = N_POL;
		    // extent['B'] = N_BEAM;
			// Tensor<T> input()
			break;
		}
		default:
			std::cout << "Beamform type not known..." << std::endl;
			break;
	}
}


template<class T>
void VoltageBeamformer<T>::process(
	const thrust::device_vector<T>& in,
	thrust::device_vector<T>& out,
	const thrust::device_vector<T>& weights)
{
	if(!_success){return;}
	// Cast raw data pointer for passing to CUDA kernel
	const T *p_in = thrust::raw_pointer_cast(in.data());
	const T *p_weights = thrust::raw_pointer_cast(weights.data());
	T *p_out = thrust::raw_pointer_cast(out.data());
	// Switch to desired CUDA kernel
	switch(_conf->bf_type)
	{
		case SIMPLE_BF_TAFPT:
		{
			std::cout << "Voltage beamformer: simple TAFP" << std::endl;
			simple_bf_tafp_voltage<<<_grid_layout, _block_layout>>>(p_in, p_out, p_weights, *_conf);
			break;
		}
		case BF_TFAP:
		{
			std::cout << "Voltage beamformer: optimzed TFAP" << std::endl;
				bf_tfpa_voltage<<<_grid_layout, _block_layout, _shared_mem_dynamic>>>(p_in, p_out, p_weights, *_conf);
			break;
		}
		case CUTENSOR_BF_TFAP:
		{
			std::cout << "Voltage beamformer: CUTENSOR TFAP" << std::endl;

			break;
		}
		default:
		{
			std::cout << "Beamform type " << std::to_string(_conf->bf_type) << " not known.." << std::endl;
			break;
		}
	}
}



template<class T>
void VoltageBeamformer<T>::print_layout()
{
	std::cout << " Kernel layout: " << std::endl
		<< " g.x = " << std::to_string(_grid_layout.x) << std::endl
		<< " g.y = " << std::to_string(_grid_layout.y) << std::endl
		<< " g.z = " << std::to_string(_grid_layout.z) << std::endl
		<< " b.x = " << std::to_string(_block_layout.x)<< std::endl
		<< " b.y = " << std::to_string(_block_layout.y)<< std::endl
		<< " b.z = " << std::to_string(_block_layout.z)<< std::endl;
}


template<class T>
void VoltageBeamformer<T>::check_shared_mem_size()
{
	std::cout << "Required shared memory: " << std::to_string(_shared_mem_total) << " Bytes" << std::endl;
	if(_prop.sharedMemPerBlock < _shared_mem_total)
	{
		std::cout << "The requested size for shared memory per block exceeds the size provided by device "
			<< std::to_string(_device_id) << std::endl
			<< "! Warning: Kernel will not get launched !" << std::endl;
			_success = false;
			return;
	}else{
		_success = true;
	}
}




} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#endif
