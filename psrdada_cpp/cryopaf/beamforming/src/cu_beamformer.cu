#include "psrdada_cpp/cryopaf/beamforming/cu_beamformer.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{



CudaBeamformer::CudaBeamformer(bf_config_t *conf)
	: _conf(conf)
{
	// std::cout << "Building instance of CudaBeamformer" << std::endl;
	init();
}



CudaBeamformer::~CudaBeamformer()
{
	// std::cout << "Destroying instance of CudaBeamformer" << std::endl;
}



void CudaBeamformer::kernel_layout()
{
	switch(_conf->bf_type){
		case SIMPLE_BF_TAFPT:
			grid_layout.x = (_conf->n_samples < NTHREAD) ? 1 : _conf->n_samples/NTHREAD;
			grid_layout.y = _conf->n_beam;
			grid_layout.z = _conf->n_channel;
			block_layout.x = NTHREAD; //(_conf->n_samples < NTHREAD) ? _conf->n_samples : NTHREAD;
			break;

		default:
			std::cout << "Beamform type not known..." << std::endl;
			break;
	}
	// std::cout << " Kernel layout: " << std::endl
	// 	<< " g.x = " << std::to_string(grid_layout.x) << std::endl
	// 	<< " g.y = " << std::to_string(grid_layout.y) << std::endl
	// 	<< " g.z = " << std::to_string(grid_layout.z) << std::endl
	// 	<< " b.x = " << std::to_string(block_layout.x)<< std::endl
	// 	<< " b.y = " << std::to_string(block_layout.y)<< std::endl
	// 	<< " b.z = " << std::to_string(block_layout.z)<< std::endl;
}



void CudaBeamformer::init(bf_config_t *conf)
{
	if(conf){_conf = conf;}
	CUDA_ERROR_CHECK(cudaMalloc((void**)&_conf_device, sizeof(bf_config_t)));
	CUDA_ERROR_CHECK(cudaMemcpy(_conf_device, _conf, sizeof(bf_config_t), cudaMemcpyHostToDevice));
	kernel_layout();
}



void CudaBeamformer::process(const thrust::device_vector<cuComplex>& in,
	thrust::device_vector<float>& out,
	const thrust::device_vector<cuComplex>& weights,
	cudaStream_t stream)
{
	// Cast raw data pointer for passing to CUDA kernel
	const cuComplex *p_in = thrust::raw_pointer_cast(in.data());
	const cuComplex *p_weights = thrust::raw_pointer_cast(weights.data());
	float *p_out = thrust::raw_pointer_cast(out.data());

	// Switch to desired CUDA kernel
	switch(_conf->bf_type){

		// Simple beamforming approach, for more information see kernel description
		case SIMPLE_BF_TAFPT:
			// Launch kernel
			simple_bf_tafpt_stokes_I<<<grid_layout, block_layout>>>(p_in, p_out, p_weights, _conf_device);
			break;

		default:
			std::cout << "Beamform type not known..." << std::endl;
			break;
	}
}



void CudaBeamformer::process(const thrust::device_vector<cuComplex>& in,
	thrust::device_vector<cuComplex>& out,
	const thrust::device_vector<cuComplex>& weights,
	cudaStream_t stream)
{
	// Cast raw data pointer for passing to CUDA kernel
	const cuComplex *p_in = thrust::raw_pointer_cast(in.data());
	const cuComplex *p_weights = thrust::raw_pointer_cast(weights.data());
	cuComplex *p_out = thrust::raw_pointer_cast(out.data());

	// Switch to desired CUDA kernel
	switch(_conf->bf_type){

		// Simple beamforming approach, for more information see kernel description
		case SIMPLE_BF_TAFPT:
			// Launch kernel
			simple_bf_tafpt<<<grid_layout, block_layout>>>(p_in, p_out, p_weights, _conf_device);
			break;

		default:
			std::cout << "Beamform type not known..." << std::endl;
			break;
	}
}




} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp
