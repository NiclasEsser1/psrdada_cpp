#ifdef BEAMFORMER_CUH_

namespace psrdada_cpp{
namespace cryopaf{

template<class ComputeType>
Beamformer<ComputeType>::Beamformer(
  cudaStream_t& stream,
  std::size_t sample,
  std::size_t channel,
  std::size_t element,
  std::size_t beam,
  std::size_t integration)
  : _stream(stream),
    _sample(sample),
    _channel(channel),
    _element(element),
    _beam(beam),
    _integration(integration)
{
	grid.x = ceil(_beam / (double)TILE_SIZE);
	grid.y = ceil(_sample / (double)TILE_SIZE);
	grid.z = _channel;
	block.x = TILE_SIZE;
	block.y = TILE_SIZE;
}

template<class ComputeType>
Beamformer<ComputeType>::~Beamformer()
{
  BOOST_LOG_TRIVIAL(debug) << "Destroy Beamformer object";
}

template<class ComputeType>
void Beamformer<ComputeType>::process(const ComputeType* input, const ComputeType* weights, ResultType* output)
{
  BOOST_LOG_TRIVIAL(debug) << "Power BF";
	beamformer_power_fpte_fpbe_ftb<<<grid, block, 0, _stream>>>
		(input, weights, output, _sample, _element, _beam, _integration);
}

template<class ComputeType>
void Beamformer<ComputeType>::process(const ComputeType* input, const ComputeType* weights, ComputeType* output)
{
  BOOST_LOG_TRIVIAL(debug) << "Voltage BF";
  beamformer_voltage_fpte_fpbe_fptb<<<grid, block, 0, _stream>>>
    (input, weights, output, _sample, _element, _beam);
}

template<class ComputeType>
void Beamformer<ComputeType>::print_layout()
{
  std::cout << " Kernel layout: " << std::endl
    << " g.x = " << std::to_string(grid.x) << std::endl
    << " g.y = " << std::to_string(grid.y) << std::endl
    << " g.z = " << std::to_string(grid.z) << std::endl
    << " b.x = " << std::to_string(block.x)<< std::endl
    << " b.y = " << std::to_string(block.y)<< std::endl
    << " b.z = " << std::to_string(block.z)<< std::endl;
}


} // namespace cryopaf
} // namespace psrdada_cpp

#endif
