#ifdef VOLTAGEBEAMFORMER_CUH_

namespace psrdada_cpp{
namespace cryopaf{


template<class HandlerType, class InputType, class WeightType, class OutputType>
VoltageBeamformer<HandlerType, InputType, WeightType, OutputType>::VoltageBeamformer
	(bf_config_t& conf, MultiLog &log, HandlerType &handler)
	: _conf(conf), _handler(handler), _log(log)
{
	CUDA_ERROR_CHECK(cudaGetDeviceProperties(&_prop, _conf.device_id));

	std::vector<int> input_dim{'T','F','A', 'P'};
	std::vector<int> weight_dim{'B','F','A', 'P'};
	std::vector<int> output_dim{'B','F','T', 'P'};

	std::unordered_map<int, int64_t> extent;

	extent['T'] = conf.n_samples;
	extent['F'] = conf.n_channel;
	extent['A'] = conf.n_antenna;
	extent['P'] = conf.n_pol;
	extent['B'] = conf.n_beam;

	_input_buffer = new InputType(input_dim, extent);
	_weight_buffer = new WeightType(weight_dim, extent);
	_output_buffer = new OutputType(output_dim, extent);

	CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));

	CUTENSOR_ERROR_CHECK(cutensorInit(&_cutensor_handle));

	_input_buffer->init_desc(&_cutensor_handle);
	_weight_buffer->init_desc(&_cutensor_handle);
	_output_buffer->init_desc(&_cutensor_handle);
	_input_buffer->init_alignment(&_cutensor_handle, _input_buffer->a_ptr());
	_weight_buffer->init_alignment(&_cutensor_handle, _weight_buffer->a_ptr());
	_output_buffer->init_alignment(&_cutensor_handle, _output_buffer->a_ptr());

	CUTENSOR_ERROR_CHECK(cutensorInitContractionDescriptor(&_cutensor_handle, &_cutensor_desc,
			&_input_buffer->desc(), _input_buffer->mode().data(), _input_buffer->alignment(),
			&_weight_buffer->desc(), _weight_buffer->mode().data(), _weight_buffer->alignment(),
			&_output_buffer->desc(), _output_buffer->mode().data(), _output_buffer->alignment(),
			&_output_buffer->desc(), _output_buffer->mode().data(), _output_buffer->alignment(),
			_cutensor_type));
	CUTENSOR_ERROR_CHECK(cutensorInitContractionFind(&_cutensor_handle, &_cutensor_find, _cutensor_algo));

	CUTENSOR_ERROR_CHECK(cutensorContractionGetWorkspace(&_cutensor_handle, &_cutensor_desc,
			&_cutensor_find, _work_preference, &_worksize));

	if(_worksize > 0) {CUDA_ERROR_CHECK(cudaMalloc(&_work, _worksize));}
	CUTENSOR_ERROR_CHECK(cutensorContractionMaxAlgos(&_max_algos));
	CUTENSOR_ERROR_CHECK(cutensorInitContractionPlan(&_cutensor_handle, &_cutensor_plan, &_cutensor_desc, &_cutensor_find, _worksize));
}


template<class HandlerType, class InputType, class WeightType, class OutputType>
VoltageBeamformer<HandlerType, InputType, WeightType, OutputType>::~VoltageBeamformer()
{

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
void VoltageBeamformer<HandlerType, InputType, WeightType, OutputType>::init(RawBytes &header_block)
{
		std::size_t bytes = header_block.total_bytes();
		_handler.init(header_block);
}

template<class HandlerType, class InputType, class WeightType, class OutputType>
bool VoltageBeamformer<HandlerType, InputType, WeightType, OutputType>::operator()(RawBytes &dada_block)
{
	if(dada_block.used_bytes() != _input_buffer->total_bytes())
	{
		BOOST_LOG_TRIVIAL(warning) << "Stopped reading from dada stream " << dada_block.used_bytes() << "    " << _input_buffer->total_bytes();
		CUDA_ERROR_CHECK(cudaDeviceSynchronize());
		return true;
	}
	// _input_buffer->synchronize();
	_input_buffer->sync_cpy(dada_block.ptr(), dada_block.used_bytes());
	_input_buffer->swap();
	process();

	// Wrap in a RawBytes object here;
	RawBytes dada_output((char*)_output_buffer->a_ptr(), _output_buffer->total_bytes(), _output_buffer->total_bytes(), true);
	_output_buffer->swap();
	_handler(dada_output);
	return false;
}

template<class HandlerType, class InputType, class WeightType, class OutputType>
template<class T, class Type>
void VoltageBeamformer<HandlerType, InputType, WeightType, OutputType>::async_copy(thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>& vec)
{
	if(std::is_same<Type, InputType>::value)
	{
		_input_buffer->synchronize();
		_input_buffer->async_cpy(vec.data(), vec.size()*sizeof(T));
	}
	else if(std::is_same<Type, WeightType>::value)
	{
		_weight_buffer->synchronize();
		_weight_buffer->async_cpy(vec.data(), vec.size()*sizeof(T));
	}
	else if(std::is_same<Type, OutputType>::value)
	{
		_output_buffer->synchronize();
		_output_buffer->async_cpy(vec.data(), vec.size()*sizeof(T));
	}
	else
	{
		BOOST_LOG_TRIVIAL(error) << "Type not know";
	}
}

template<class HandlerType, class InputType, class WeightType, class OutputType>
template<class T, class Type>
void VoltageBeamformer<HandlerType, InputType, WeightType, OutputType>::sync_copy(thrust::host_vector<T>& vec)
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
void VoltageBeamformer<HandlerType, InputType, WeightType, OutputType>::process()
{
	float alpha = 1.0;
	float beta = 0;
	BOOST_LOG_TRIVIAL(debug) << "Voltage beamformer: CUTENSOR" << std::endl;
	CUTENSOR_ERROR_CHECK(cutensorContraction(&_cutensor_handle, &_cutensor_plan,
			(void*)&alpha, (void*)_input_buffer->a_ptr(), (void*)_weight_buffer->a_ptr(),
			(void*)&beta, (void*)_output_buffer->a_ptr(), (void*)_output_buffer->a_ptr(),
			_work, _worksize, _stream));
	CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}


} // namespace cryopaf
} // namespace psrdada_cpp

#endif
