#ifdef VOLTAGE_BEAMFORMER_CUH_

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{


template<class T>
VoltageBeamformer<T>::VoltageBeamformer(bf_config_t *conf, int device_id)
	: _conf(conf), id(device_id)
{
	// std::cout << "Creating instance of VoltageBeamformer" << std::endl;
	// Set device to use
	CUDA_ERROR_CHECK(cudaSetDevice(id))
	// Retrieve device properties
	CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, id))
	// initialize beamformer enviroment
	init();

}


template<class T>
VoltageBeamformer<T>::~VoltageBeamformer()
{
	if(_conf->bf_type == CUBLAS_BF_TFAP)
		CUBLAS_ERROR_CHECK(cublasDestroy(blas_handle));
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
		case CUBLAS_BF_TFAP:
		{
			CUBLAS_ERROR_CHECK(cublasCreate_v2(&blas_handle));
			// Set the math mode to allow cuBLAS to use Tensor Cores:
			CUBLAS_ERROR_CHECK(cublasSetMathMode(blas_handle, CUBLAS_DEFAULT_MATH));
			if constexpr (std::is_same<T, __half2>::value)
			{
				blas_dtype = CUDA_C_16F;
				blas_ctype = CUDA_C_16F;
			}
			else if constexpr (std::is_same<T, float2>::value)
			{
				blas_dtype = CUDA_C_32F;
				blas_ctype = CUDA_C_32F;
			}
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
	if(!success){return;}
	// Cast raw data pointer for passing to CUDA kernel
	const T *p_in = thrust::raw_pointer_cast(in.data());
	const T *p_weights = thrust::raw_pointer_cast(weights.data());
	T *p_out = thrust::raw_pointer_cast(out.data());
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
		case CUBLAS_BF_TFAP:
		{
			std::cout << "Voltage beamformer: cuBLAS TFAPT" << std::endl;
			// float alpha = 1.0;
			// float beta = .0;
			// const T** pp_in = (const T**)malloc(_conf->n_channel*sizeof(T**));
			// const T** pp_weights = (const T**)malloc(_conf->n_channel*sizeof(T**));
			// T** pp_out = (T**)malloc(_conf->n_channel*sizeof(T**));
			// for(int i = 0; i < _conf->n_channel; i++){
			// 	pp_in[i] = &p_in[_conf->n_samples * _conf->n_antenna * _conf->n_pol * i];
			// 	pp_weights[i] = &p_weights[_conf->n_beam * _conf->n_antenna * _conf->n_pol * i];
			// 	pp_out[i] = &p_out[_conf->n_samples * _conf->n_beam * i];
			// }
			// int m = (int)(_conf->n_antenna * _conf->n_pol);	// number of rows of matrix op(A) and C.
			// int n = (int)(_conf->n_beam);			// number of columns of matrix op(B) and C
			// int k = (int)(_conf->n_samples);					// number of columns of op(A) and rows of op(B).
			// int lda = m;
			// int ldb = n;
			// int ldc = k;
			// int batch = _conf->n_channel;
			// CUBLAS_ERROR_CHECK(cublasGemmBatchedEx(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
			// 	m, n, k,
			// 	(void*)&alpha, (void**)pp_in, blas_dtype, lda,
      //   (void**)pp_weights, blas_dtype, ldb, (void*)&beta,
			// 	(void**)pp_out, blas_dtype, ldc, batch, CUDA_C_32F, CUBLAS_GEMM_DEFAULT));

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
