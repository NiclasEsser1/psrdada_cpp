#ifdef PIPELINE_CUH_

namespace psrdada_cpp{
namespace cryopaf{

template<class HandlerType, class ComputeType, class ResultType>
Pipeline<HandlerType, ComputeType, ResultType>::Pipeline
	(PipelineConfig& conf, MultiLog &log, HandlerType &handler)
	: _conf(conf), _log(log), _handler(handler)
{
	// Check if passed template types are valid
	if(  !(std::is_same<ComputeType,  float2>::value)
		&& !(std::is_same<ComputeType, __half2>::value))
	{
		BOOST_LOG_TRIVIAL(error) << "PipelineError: Template type not supported";
		exit(1);
	}
#ifdef DEBUG
	// Cuda events for profiling
  CUDA_ERROR_CHECK( cudaEventCreate(&start) );
	CUDA_ERROR_CHECK( cudaEventCreate(&stop) );
#endif
	// Create streams
	CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_stream));
	CUDA_ERROR_CHECK(cudaStreamCreate(&_prc_stream));
	CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_stream));

	// Instantiate processor objects
	unpacker = new Unpacker<ComputeType>(_prc_stream,
		conf.n_samples,
		conf.n_channel,
		conf.n_elements,
		conf.protocol);
	beamformer = new Beamformer<ComputeType>(_prc_stream,
		_conf.n_samples,
		_conf.n_channel,
		_conf.n_elements,
		_conf.n_beam,
		_conf.integration);

	// Calculate buffer sizes
  std::size_t raw_input_size = _conf.n_samples
		* _conf.n_channel
		* _conf.n_elements
		* _conf.n_pol
		* unpacker->sample_size();
  std::size_t transpose_size = _conf.n_samples
		* _conf.n_channel
		* _conf.n_elements
		* _conf.n_pol;
  std::size_t weight_size = _conf.n_beam
		* _conf.n_channel
		* _conf.n_elements
		* _conf.n_pol;
  std::size_t output_size = 0;
	if(conf.mode == "power")
	{
		output_size = _conf.n_beam
			* _conf.n_channel
			* _conf.n_samples
			/ _conf.integration;
	}
  else if (conf.mode == "voltage")
	{
  	output_size = _conf.n_beam
			* _conf.n_channel
			* _conf.n_samples
			* _conf.n_pol;
  }
  else
  {
    BOOST_LOG_TRIVIAL(debug) << "Beamform mode not known";
		exit(1);
  }
	std::size_t total_size = (raw_input_size
		+ transpose_size * sizeof(ComputeType)
		+ weight_size * sizeof(ComputeType)
		+ output_size * sizeof(ComputeType)) * 2; // Multiplied by two, since each buffer is double buffer
	BOOST_LOG_TRIVIAL(info) << "Requested global device memory: " << total_size;

	// Instantiate buffers
	_raw_input_buffer = new RawInputType(raw_input_size);
	_input_buffer = new InputType(transpose_size);
	_weight_buffer = new WeightType(weight_size);
	_output_buffer = new OutputType(output_size);

}


template<class HandlerType, class ComputeType, class ResultType>
Pipeline<HandlerType, ComputeType, ResultType>::~Pipeline()
{

#ifdef DEBUG
	  CUDA_ERROR_CHECK( cudaEventDestroy(start) );
	  CUDA_ERROR_CHECK( cudaEventDestroy(stop) );
#endif
		if(_h2d_stream)
		{
			CUDA_ERROR_CHECK(cudaStreamDestroy(_h2d_stream));
		}
		if(_prc_stream)
		{
			CUDA_ERROR_CHECK(cudaStreamDestroy(_prc_stream));
		}
		if(_d2h_stream)
		{
			CUDA_ERROR_CHECK(cudaStreamDestroy(_d2h_stream));
		}

		if(_raw_input_buffer)
		{
			delete _raw_input_buffer;
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



template<class HandlerType, class ComputeType, class ResultType>
void Pipeline<HandlerType, ComputeType, ResultType>::init(RawBytes &header_block)
{
		std::size_t bytes = header_block.total_bytes();
		_handler.init(header_block);
}

template<class HandlerType, class ComputeType, class ResultType>
bool Pipeline<HandlerType, ComputeType, ResultType>::operator()(RawBytes &dada_input)
{
	_call_cnt += 1;
	BOOST_LOG_TRIVIAL(debug) << "Processing "<< _call_cnt << " dada block..";
	if(dada_input.used_bytes() > _raw_input_buffer->total_bytes())
	{
		BOOST_LOG_TRIVIAL(error) << "Unexpected Buffer Size - Got "
       << dada_input.used_bytes() << " byte, expected "
       << _input_buffer->total_bytes() << " byte)";
		CUDA_ERROR_CHECK(cudaDeviceSynchronize());
		return true;
	}

#ifdef DEBUG
	CUDA_ERROR_CHECK( cudaEventRecord(start,0) );
#endif

	CUDA_ERROR_CHECK(cudaMemcpyAsync(_raw_input_buffer->b_ptr(), dada_input.ptr(), dada_input.used_bytes(), cudaMemcpyHostToDevice, _h2d_stream));
	BOOST_LOG_TRIVIAL(debug) << "Copied dada block from host to device";
	if(_call_cnt == 1)
	{
		_raw_input_buffer->swap();
		return false;
	}

	unpacker->unpack(_raw_input_buffer->a_ptr(), _input_buffer->b_ptr());
	_raw_input_buffer->swap();
	BOOST_LOG_TRIVIAL(debug) << "Unpacked data";

	if(_call_cnt == 2)
	{
		_input_buffer->swap();
		return false;
	}

	beamformer->process(_input_buffer->a_ptr(), _weight_buffer->a_ptr(), _output_buffer->b_ptr());
	_input_buffer->swap();
	BOOST_LOG_TRIVIAL(debug) << "Beamformed data";

	if(_call_cnt == 3)
	{
		_output_buffer->swap();
		return false;
	}
	_output_buffer->async_copy(_d2h_stream);
	BOOST_LOG_TRIVIAL(debug) << "Copied processed dada block from device to host";

	// Wrap in a RawBytes object here;
	RawBytes dada_output(static_cast<char*>((void*)_output_buffer->host.a_ptr()),
		_output_buffer->total_bytes(),
		_output_buffer->total_bytes());

	_output_buffer->swap();
	_output_buffer->host.swap();

	_handler(dada_output);
#ifdef DEBUG
	CUDA_ERROR_CHECK(cudaEventRecord(stop, 0));
	CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
	CUDA_ERROR_CHECK(cudaEventElapsedTime(&ms, start, stop));
	BOOST_LOG_TRIVIAL(debug) << "Took " << ms << " ms to process stream";
#endif
	return false;
}

} // namespace cryopaf
} // namespace psrdada_cpp

#endif
