#ifdef UNPACKER_CUH

namespace psrdada_cpp {
namespace cryopaf {

template<typename T>
Unpacker<T>::Unpacker(cudaStream_t& stream,
  std::size_t nsamples,
  std::size_t nchannels,
  std::size_t nelements,
  std::string protocol)
    : _stream(stream), _protocol(protocol)
{
  if(  !(std::is_same<T,  float2>::value)
    && !(std::is_same<T, __half2>::value))
  {
    BOOST_LOG_TRIVIAL(error) << "UnpackerError: Template type not supported";
    exit(1);
  }
  if(_protocol == "codif")
  {
    if(nchannels % 7)
    {
        BOOST_LOG_TRIVIAL(error) << "UnpackerError: Unpacker expects a multiple of 7 channels";
        exit(1);
    }

    grid.x = nsamples / NSAMP_DF;
    grid.y = nelements;
    grid.z = nchannels / NCHAN_CHK;
    block.x = NSAMP_DF;
    block.z = NCHAN_CHK;
  }
  else if(_protocol == "spead")
  {
    grid.x = nsamples / NSAMP_PER_HEAP;
    grid.y = nelements;
    grid.z = nchannels;
    block.x = NSAMP_PER_HEAP;
  }
  else if(_protocol == "dummy_input")
  {
    BOOST_LOG_TRIVIAL(warning) << "UnpackerWarning: Skipping unpacking process";
  }
  else
  {
    BOOST_LOG_TRIVIAL(error) << "UnpackerError: Protocol " << _protocol << " not implemented";
    exit(1);
  }
}


template<typename T>
Unpacker<T>::~Unpacker()
{
  BOOST_LOG_TRIVIAL(debug) << "Destroy Unpacker object";
}


template<typename T>
void Unpacker<T>::unpack(char* input, T* output)
{
  if(_protocol == "codif")
  {
    BOOST_LOG_TRIVIAL(debug) << "Unpack CODIF data";
    unpack_codif_to_fpte<<< grid, block, 0, _stream>>>((uint64_t*)input, output);
  }
  else if(_protocol == "spead")
  {
    BOOST_LOG_TRIVIAL(debug) << "Unpack SPEAD2 data";
    // unpack_spead_to_fpte<<< grid, block, 0, _stream>>>(input, output);
  }
}



template<typename T>
void Unpacker<T>::print_layout()
{
  std::cout << "Grid.x " << grid.x << std::endl;
  std::cout << "Grid.y " << grid.y << std::endl;
  std::cout << "Grid.z " << grid.z << std::endl;
  std::cout << "block.x " << block.x << std::endl;
  std::cout << "block.z " << block.z << std::endl;
}

} //namespace cryopaf
} //namespace psrdada_cpp

#endif
