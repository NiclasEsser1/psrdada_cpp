#ifdef VOLTAGEBEAMFORM_TESTER_H_

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{
namespace test{




VoltageBeamformTester::VoltageBeamformTester()
  : ::testing::TestWithParam<bf_config_t>()
  , _conf(GetParam()) // GetParam() is inherited from base TestWithParam
  , _stream(0)
{
  // std::cout << "Creating instance of VoltageBeamformTester" << std::endl;
}


VoltageBeamformTester::~VoltageBeamformTester()
{
  // std::cout << "Destroying instance of VoltageBeamformTester" << std::endl;
}


void VoltageBeamformTester::SetUp()
{
    //CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}




void VoltageBeamformTester::TearDown()
{
    //CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}


template<typename T>
void VoltageBeamformTester::test(int device_id)
{
  // Set up normal distributed sample and weight generator
  const float input_level = 2.0f;
  const double pi = std::acos(-1);
  std::default_random_engine generator;
  std::normal_distribution<float> normal_dist(0.0, input_level);
  std::uniform_real_distribution<float> uniform_dist(0.0, 2*pi);

	id = device_id;
	CUDA_ERROR_CHECK(cudaSetDevice(id));

  // Calulate memory size for input, weights and output
  std::size_t input_size = _conf.n_samples * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  std::size_t weight_size =  _conf.n_beam * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  std::size_t output_size = _conf.n_samples * _conf.n_beam * _conf.n_channel * _conf.n_pol;
  std::size_t required_mem = input_size * sizeof(T)
    + weight_size * sizeof(T)
    + output_size * sizeof(T);

  std::cout << "Required device memory: " << std::to_string(required_mem / (1024*1024)) << "MiB" << std::endl;
  std::cout << "Required host memory: " << std::to_string((required_mem
		+ (input_size + weight_size) * sizeof(T)) / (1024*1024)) << "MiB" << std::endl;

  // Allocate host vectors
  thrust::host_vector<T> host_output(output_size);
  thrust::host_vector<T> host_input(input_size, {0, 0});
  thrust::host_vector<T> host_weights(weight_size, {0, 0});

  // Generate test samples / normal distributed noise for input signal
  for (size_t i = 0; i < host_input.size(); i++)
  {
      if constexpr (std::is_same<T, __half2>::value)
        host_input[i] = __float22half2_rn({normal_dist(generator), normal_dist(generator)});
      else
        host_input[i] = {i,i};//{normal_dist(generator), normal_dist(generator)};
  }
  // Build complex weight as C * exp(i * theta).
  for (size_t i = 0; i < host_weights.size(); i++)
  {
      if constexpr (std::is_same<T, __half2>::value)
        host_weights[i] = __float22half2_rn({normal_dist(generator), normal_dist(generator)});
      else
        host_weights[i] = {1,1};//{normal_dist(generator), normal_dist(generator)};
  }

  // Allocate device memory & assign test samples
  thrust::device_vector<T> dev_input = host_input;
  thrust::device_vector<T> dev_weights = host_weights;
  thrust::device_vector<T> dev_output(output_size);

	// TODO: print elapsed time
	// launches cpu beamforming
  cpu_process(host_input, host_output, host_weights);
	// launches CUDA kernel
  gpu_process(dev_input, dev_output, dev_weights);
  // compare results of both outputs
  compare(host_output, dev_output);

}


template <typename T>
void VoltageBeamformTester::compare(
  const thrust::host_vector<T> cpu,
  const thrust::device_vector<T> gpu,
  const float tol)
{
  // Check if vectors have the same size
  ASSERT_TRUE(cpu.size() == gpu.size())
    << "Host and device vector size not equal" << std::endl;

  // Copy CUDA results to host
	thrust::host_vector<T> host_gpu_result = gpu;
	float abs_diff;
	int arg_max_deviation = 0;

  // Check each element of CPU and GPU implementation

	float2 gpu_element;
	float2 cpu_element;
  float2 diff;
  float2 max_deviation = {0,0};
	for(std::size_t i = 0; i < gpu.size(); i++)
  {
		if constexpr (std::is_same<T, __half2>::value)
		{
			gpu_element = __half22float2(host_gpu_result[i]);
			cpu_element = __half22float2(cpu[i]);
		}
		else
		{
			gpu_element = host_gpu_result[i];
			cpu_element = cpu[i];
		}

    diff = cuCsubf(cpu_element, gpu_element);
    abs_diff = cuCabsf(diff);

    if(abs_diff > cuCabsf(max_deviation))
		{
        arg_max_deviation = i;
        max_deviation = diff;
    }

		/** DEBUG **/
		if(_conf.bf_type == CUBLAS_BF_TFAP){
			printf("CPU[%d] = %f + i*%f\nGPU[%d] = %f + i*%f\n", i, cpu_element.x, cpu_element.y, i, gpu_element.x, gpu_element.y);
		}
		/**/
    ASSERT_TRUE(abs_diff <= 1)  << "Beamformer: CPU and GPU result is unequal for element " << std::to_string(i) << std::endl
      << "  CPU result: " << std::to_string(cpu_element.x) << "+ i*" << std::to_string(cpu_element.y) << std::endl
      << "  GPU result: " << std::to_string(gpu_element.x) << "+ i*" << std::to_string(gpu_element.y) << std::endl;
  }

  std::cout << "Maximum deviation detected for element " << std::to_string(arg_max_deviation) << std::endl
    << "Deviation abs(cpu[" << std::to_string(arg_max_deviation)
    << "] - gpu[" << std::to_string(arg_max_deviation) << "]) = "
    << std::to_string(max_deviation.x) << " + i* " << std::to_string(max_deviation.y) << std::endl << std::endl;
}


template<typename T>
void VoltageBeamformTester::gpu_process(
  thrust::device_vector<T>& in,
  thrust::device_vector<T>& out,
  thrust::device_vector<T>& weights)
{
  VoltageBeamformer<T> bf(&_conf, id);
  bf.process(in, out, weights);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}


template<typename T>
void VoltageBeamformTester::cpu_process(
  thrust::host_vector<T>& in,
  thrust::host_vector<T>& out,
  thrust::host_vector<T>& weights)
{
	int pt = _conf.n_pol;
	int fpt = _conf.n_channel * pt;
	int afpt = _conf.n_antenna * fpt;
	int out_idx, in_idx, weight_idx;

	for(int t = 0; t < _conf.n_samples; t++)
	{
		for(int b = 0; b < _conf.n_beam; b++)
		{
			for(int a = 0; a < _conf.n_antenna; a++)
			{
				for(int f = 0; f < _conf.n_channel; f++)
				{
					for(int p = 0; p < _conf.n_pol; p++)
					{
						/*
						* INPUT: TAFP(t)
						* WEIGHTS: BFAP
						* OUTPUT: TBF
						*/
						if(_conf.bf_type == SIMPLE_BF_TAFPT)
						{
							out_idx = b * _conf.n_samples * fpt + t * fpt + f * pt + p;
							in_idx = t * afpt + a * fpt + f * pt + p;
							weight_idx = b * afpt + a * fpt + f * pt + p;
							// out[out_idx] = cuCmulf(in[in_idx], weights[weight_idx]);
						}
						/* ! Slightly different output dataproduct !
						* INPUT: TFAP(t)
						* WEIGHTS: BFAP
						* OUTPUT: BFT
						*/
						else
						{
							out_idx = b * _conf.n_samples * _conf.n_channel * _conf.n_pol
								+ f * _conf.n_samples * _conf.n_pol + t * _conf.n_pol + p;
							in_idx = t * _conf.n_channel * _conf.n_antenna * _conf.n_pol
								+ f * _conf.n_antenna * _conf.n_pol + a * _conf.n_pol + p;
							weight_idx = b * _conf.n_channel * _conf.n_antenna * _conf.n_pol
								+ f * _conf.n_antenna * _conf.n_pol + a * _conf.n_pol + p;
						}
						if constexpr (std::is_same<T, float2>::value)
							out[out_idx] = cuCaddf(out[out_idx], cuCmulf(in[in_idx], weights[weight_idx]));
						else if constexpr (std::is_same<T, __half2>::value)
							out[out_idx] = __float22half2_rn(cuCaddf(
								__half22float2(out[out_idx]),
								cuCmulf(__half22float2(in[in_idx]), __half22float2(weights[weight_idx]))));
					}
				}
			}
		}
	}

}


/**
* Testing with Google Test Framework
*/
TEST_P(VoltageBeamformTester, BeamformerVoltageHalf){
  std::cout << std::endl
    << "-------------------------------------------------------------" << std::endl
    << " Testing voltage mode with T=__half2 (half precision 2x16bit)" << std::endl
    << "-------------------------------------------------------------" << std::endl << std::endl;
  test<__half2>();
}

TEST_P(VoltageBeamformTester, BeamformerVoltageSingle){
  std::cout << std::endl
    << "-------------------------------------------------------------" << std::endl
    << "Testing voltage mode with T=float2 (single precision 2x32bit)" << std::endl
    << "-------------------------------------------------------------" << std::endl << std::endl;
  test<float2>();
}


INSTANTIATE_TEST_CASE_P(BeamformerTesterInstantiation, VoltageBeamformTester, ::testing::Values(

  // samples | channels | antenna | polarisation | beam | interval/integration | beamformer type
  bf_config_t{32, 1, 16, 2, 1, 0, BF_TFAP},
  bf_config_t{2048, 1, 16, 2, 4, 0, BF_TFAP},
  bf_config_t{1024, 64, 32, 2, 32, 0, BF_TFAP},
  bf_config_t{1024, 64, 32, 1, 64, 0, BF_TFAP},
  bf_config_t{1024, 64, 64, 2, 32, 0, BF_TFAP},
	bf_config_t{1024, 64, 64, 2, 32, 0, SIMPLE_BF_TAFPT}
));


} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test

#endif
