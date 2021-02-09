#ifdef POWERBEAMFORMER_CUH_

namespace psrdada_cpp{
namespace cryopaf{


template<class HandlerType, class InputType, class WeightType, class OutputType>
PowerBeamformer<HandlerType, InputType, WeightType, OutputType>::PowerBeamformer
	(bf_config_t& conf, MultiLog &log, HandlerType &handler)
	: _conf(conf), _log(log), _handler(handler)
{
	CUDA_ERROR_CHECK(cudaGetDeviceProperties(&_prop, _conf.device_id));

  CUDA_ERROR_CHECK( cudaEventCreate(&start) );
	CUDA_ERROR_CHECK( cudaEventCreate(&stop) );
	std::vector<int> input_dim{'T','F','A', 'P'};
	std::vector<int> weight_dim{'B','F','A', 'P'};
	std::vector<int> output_dim{'B','F','t'};

	std::unordered_map<int, int64_t> extent;

	extent['T'] = conf.n_samples;
	extent['F'] = conf.n_channel;
	extent['A'] = conf.n_antenna;
	extent['P'] = conf.n_pol;
	extent['B'] = conf.n_beam;
	extent['t'] = (int)conf.n_samples / conf.interval;


	_input_buffer = new InputType(input_dim, extent);
	_weight_buffer = new WeightType(weight_dim, extent);
	_output_buffer = new OutputType(output_dim, extent);

	_shared_mem_static = (SHARED_IDATA) * sizeof(*_input_buffer->a_ptr());
	// _shared_mem_dynamic = _conf.n_beam * sizeof(*_output_buffer->a_ptr()) + _conf.n_antenna * _conf.n_pol * sizeof(*_input_buffer->a_ptr());
	_shared_mem_dynamic = _conf.n_beam * sizeof(*_output_buffer->a_ptr())
		+ NTHREAD * sizeof(*_input_buffer->a_ptr())
		+ _conf.n_antenna * _conf.n_pol * sizeof(*_input_buffer->a_ptr());
	_shared_mem_total = _shared_mem_static + _shared_mem_dynamic;

  check_shared_mem();

	_grid_layout.x = _conf.n_samples / _conf.interval;
	_grid_layout.y = _conf.n_channel;
	_block_layout.x = NTHREAD;

	CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}


template<class HandlerType, class InputType, class WeightType, class OutputType>
PowerBeamformer<HandlerType, InputType, WeightType, OutputType>::~PowerBeamformer()
{
	  CUDA_ERROR_CHECK( cudaEventDestroy(start) );
	  CUDA_ERROR_CHECK( cudaEventDestroy(stop) );
		if(_stream)
		{
			CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
		}

		if(_input_buffer)
		{
			delete _input_buffer;
		}
		if(_weight_buffer)
		{
			delete _weight_buffer;
		}
		if(_output_buffer)
		{
			delete _output_buffer;
		}
}



template<class HandlerType, class InputType, class WeightType, class OutputType>
void PowerBeamformer<HandlerType, InputType, WeightType, OutputType>::init(RawBytes &header_block)
{
		std::size_t bytes = header_block.total_bytes();
		_handler.init(header_block);
}

template<class HandlerType, class InputType, class WeightType, class OutputType>
bool PowerBeamformer<HandlerType, InputType, WeightType, OutputType>::operator()(RawBytes &dada_input)
{
	if(dada_input.used_bytes() > _input_buffer->total_bytes())
	{
		BOOST_LOG_TRIVIAL(error) << "Unexpected Buffer Size - Got "
       << dada_input.used_bytes() << " byte, expected "
       << _input_buffer->total_bytes() << " byte)";
		CUDA_ERROR_CHECK(cudaDeviceSynchronize());
		return true;
	}
	CUDA_ERROR_CHECK( cudaEventRecord(start,0) );
	_input_buffer->swap();
	_input_buffer->sync_cpy(dada_input.ptr(), dada_input.used_bytes());
	process();
	BOOST_LOG_TRIVIAL(debug) << "Took " << ms << " ms to process stream";


	// Wrap in a RawBytes object here;
	RawBytes dada_output(reinterpret_cast<char*>(_output_buffer->a_ptr()),
		_output_buffer->total_bytes(),
		_output_buffer->total_bytes(), true);

	_handler(dada_output);
  CUDA_ERROR_CHECK(cudaEventRecord(stop, 0));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&ms, start, stop));
	return false;
}


template<class HandlerType, class InputType, class WeightType, class OutputType>
void PowerBeamformer<HandlerType, InputType, WeightType, OutputType>::process()
{
	bf_tfap_power<<<_grid_layout, _block_layout, _shared_mem_dynamic, _stream>>>
		(_input_buffer->a_ptr(), _output_buffer->a_ptr(), _weight_buffer->a_ptr(), _conf);
}

template<class HandlerType, class InputType, class WeightType, class OutputType>
template<class T, class Type>
void PowerBeamformer<HandlerType, InputType, WeightType, OutputType>::async_copy(thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>& vec)
{
	if(std::is_same<Type, InputType>::value)
	{
		_input_buffer->async_cpy(thrust::raw_pointer_cast(vec.data()), vec.size()*sizeof(T), _stream);
		_input_buffer->synchronize();
	}
	else if(std::is_same<Type, WeightType>::value)
	{
		_weight_buffer->async_cpy(thrust::raw_pointer_cast(vec.data()), vec.size()*sizeof(T), _stream);
		_weight_buffer->synchronize();
	}
	else if(std::is_same<Type, OutputType>::value)
	{
		_output_buffer->async_cpy(thrust::raw_pointer_cast(vec.data()), vec.size()*sizeof(T), _stream);
		_output_buffer->synchronize();
	}
	else
	{
		BOOST_LOG_TRIVIAL(error) << "Type not known";
	}
}

template<class HandlerType, class InputType, class WeightType, class OutputType>
template<class T, class Type>
void PowerBeamformer<HandlerType, InputType, WeightType, OutputType>::sync_copy(thrust::host_vector<T>& vec)
{
	if(std::is_same<Type, InputType>::value)
	{
		_input_buffer->sync_cpy(vec.data(), vec.size()*sizeof(T));
	}
	else if(std::is_same<Type, WeightType>::value)
	{
		_weight_buffer->sync_cpy(vec.data(), vec.size()*sizeof(T));
	}
	else if(std::is_same<Type, OutputType>::value)
	{
		_output_buffer->sync_cpy(vec.data(), vec.size()*sizeof(T));
	}
	else
	{
		BOOST_LOG_TRIVIAL(error) << "Type not known";
	}
}

template<class HandlerType, class InputType, class WeightType, class OutputType>
void PowerBeamformer<HandlerType, InputType, WeightType, OutputType>::check_shared_mem()
{
	BOOST_LOG_TRIVIAL(debug) << "Required shared memory: " << std::to_string(_shared_mem_total) << " Bytes";
	if(_prop.sharedMemPerBlock < _shared_mem_total)
	{
		throw std::runtime_error("Not enough shared memory per block.");
	}
}


} // namespace cryopaf
} // namespace psrdada_cpp

#endif
