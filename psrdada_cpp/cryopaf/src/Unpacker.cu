#ifdef UNPACKER_CUH

namespace psrdada_cpp {
namespace cryopaf {

template<class HandlerType, class InputType, class OutputType>
Unpacker<HandlerType, InputType, OutputType>::Unpacker(bf_config_t& config, MultiLog &logger, HandlerType &handler)
    : conf(config), log(logger), _handler(handler)
{
  if(conf.n_channel % 7)
  {
      BOOST_LOG_TRIVIAL(error) << "Unpacker expects a multiple of 7 channels";
      exit(1);
  }
  if(conf.n_pol != 2)
  {
      BOOST_LOG_TRIVIAL(error) << "Unpacker expects exactly 2 polarizations (XY)";
      exit(1);
  }
  std::vector<int> input_dim{'T', 'A', 't', 'F'}; // Since P-dimension is included in a int64 sample, we don't need to refer it here
  std::vector<int> output_dim{'T', 't', 'F', 'A', 'P'};

  std::unordered_map<int, int64_t> extent;

  extent['T'] = conf.n_samples / NSAMP_DF;
  extent['t'] = NSAMP_DF;
  extent['F'] = conf.n_channel;
  extent['A'] = conf.n_antenna;
  extent['P'] = conf.n_pol;

  _input_buffer = new InputType(input_dim, extent);
  _output_buffer = new OutputType(output_dim, extent);
  _grid_layout.x = conf.n_samples / NSAMP_DF;
  _grid_layout.y = conf.n_antenna;
  _grid_layout.z = conf.n_channel / NCHAN_CHK;
  _block_layout.x = NSAMP_DF;
  _block_layout.z = NCHAN_CHK;

	CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

template<class HandlerType, class InputType, class OutputType>
Unpacker<HandlerType, InputType, OutputType>::~Unpacker()
{
  BOOST_LOG_TRIVIAL(debug) << "Destroying Unpacker object";
  if(_stream)
  {
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
  }
  if(_input_buffer)
  {
    delete _input_buffer;
  }
  if(_output_buffer)
  {
    delete _output_buffer;
  }
}

template<class HandlerType, class InputType, class OutputType>
void Unpacker<HandlerType, InputType, OutputType>::init(RawBytes& header_block)
{
		std::size_t bytes = header_block.total_bytes();
		_handler.init(header_block);
}

template<class HandlerType, class InputType, class OutputType>
bool Unpacker<HandlerType, InputType, OutputType>::operator()(RawBytes& dada_block)
{
  if(dada_block.used_bytes() > _input_buffer->total_bytes())
	{
		BOOST_LOG_TRIVIAL(error) << "Unexpected Buffer Size - Got "
       << dada_block.used_bytes() << " byte, expected "
       << _input_buffer->total_bytes() << " byte)";
		CUDA_ERROR_CHECK(cudaDeviceSynchronize());
		return true;
	}

  BOOST_LOG_TRIVIAL(debug) << "Unpacking PAF data";
	_input_buffer->swap();
	_input_buffer->sync_cpy(dada_block.ptr(), dada_block.used_bytes());

  process();

	// Wrap in a RawBytes object here;
	RawBytes dada_output(reinterpret_cast<char*>(_output_buffer->a_ptr()),
		_output_buffer->total_bytes(),
		_output_buffer->total_bytes(), true);

  _handler(dada_output);
  return false;
}

template<class HandlerType, class InputType, class OutputType>
void Unpacker<HandlerType, InputType, OutputType>::process()
{
  unpack_codif_to_float32<<< _grid_layout, _block_layout, 0, _stream>>>
    (_input_buffer->a_ptr(), _output_buffer->a_ptr());
  CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
  // CUDA_ERROR_CHECK(cudaDeviceSynchronize());

}

template<class HandlerType, class InputType, class OutputType>
template<class T, class Type>
void Unpacker<HandlerType, InputType, OutputType>::sync_copy(thrust::host_vector<T>& vec, cudaMemcpyKind kind)
{
	if(std::is_same<Type, InputType>::value)
	{
		_input_buffer->sync_cpy(vec.data(), vec.size()*sizeof(T), kind);
	}
	else if(std::is_same<Type, OutputType>::value)
	{
		_output_buffer->sync_cpy(vec.data(), vec.size()*sizeof(T), kind);
	}
	else
	{
		BOOST_LOG_TRIVIAL(error) << "Type not known";
	}
}

template<class HandlerType, class InputType, class OutputType>
void Unpacker<HandlerType, InputType, OutputType>::print_layout()
{
  std::cout << "Grid.x " << _grid_layout.x << std::endl;
  std::cout << "Grid.y " << _grid_layout.y << std::endl;
  std::cout << "Grid.z " << _grid_layout.z << std::endl;
  std::cout << "block.x " << _block_layout.x << std::endl;
  std::cout << "block.z " << _block_layout.z << std::endl;
}

} //namespace cryopaf
} //namespace psrdada_cpp

#endif
