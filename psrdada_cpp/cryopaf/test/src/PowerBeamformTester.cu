#ifdef POWERBEAMFORMER_TESTER_H_

namespace psrdada_cpp{
namespace cryopaf{
namespace test{




PowerBeamformTester::PowerBeamformTester()
  : ::testing::TestWithParam<bf_config_t>()
  , _conf(GetParam()) // GetParam() is inherited from base TestWithParam
  , _stream(0)
{
  BOOST_LOG_TRIVIAL(debug) << "Creating instance of PowerBeamformTester";
}


PowerBeamformTester::~PowerBeamformTester()
{
  BOOST_LOG_TRIVIAL(debug) << "Destroying instance of PowerBeamformTester";
}


void PowerBeamformTester::SetUp()
{
}


void PowerBeamformTester::TearDown()
{
}


template<typename T, typename U>
void PowerBeamformTester::test(int device_id)
{
  // Set up normal distributed sample and weight generator
  const float input_level = 1.0f;
  const double pi = std::acos(-1);
  std::default_random_engine generator;
  std::normal_distribution<float> normal_dist(0.0, input_level);
	id = device_id;

	CUDA_ERROR_CHECK(cudaSetDevice(id));

  // Calulate memory size for input, weights and output
  std::size_t input_size = _conf.n_samples * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  std::size_t weight_size =  _conf.n_beam * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  std::size_t output_size = _conf.n_samples / _conf.interval * _conf.n_beam * _conf.n_channel;
  std::size_t required_mem = input_size * sizeof(T)
    + weight_size * sizeof(T)
    + output_size * sizeof(U);
  BOOST_LOG_TRIVIAL(debug) << "Required device memory: " << std::to_string(required_mem * 2 / (1024*1024)) << "MiB";
  BOOST_LOG_TRIVIAL(debug) << "Required host memory: " << std::to_string((required_mem + output_size * sizeof(U)) / (1024*1024)) << "MiB";

  // Allocate host vectors
  thrust::host_vector<U> host_output(output_size, 0);
	thrust::host_vector<U> dev_output(output_size, 0);
	thrust::host_vector<T> host_input(input_size, {0, 0});
	thrust::host_vector<T> host_weights(weight_size, {0, 0});

  // Generate test samples / normal distributed noise for input signal
	for (size_t i = 0; i < host_input.size(); i++)
  {
      if constexpr (std::is_same<T, __half2>::value)
        host_input[i] = __float22half2_rn({normal_dist(generator), normal_dist(generator)});
      else
				host_input[i] = {normal_dist(generator), normal_dist(generator)};
	}

	// Generate test weights
  for (size_t i = 0; i < host_weights.size(); i++)
  {
      if constexpr (std::is_same<T, __half2>::value)
        host_weights[i] = __float22half2_rn({normal_dist(generator), normal_dist(generator)});
			else
				host_weights[i] = {normal_dist(generator), normal_dist(generator)};
  }

	// launches CUDA kernel
	gpu_process(host_input, dev_output, host_weights);

	// launches cpu beamforming
	cpu_process(host_input, host_output, host_weights);

  // compare results of both outputs
	compare(host_output, dev_output);

}


template <typename T>
void PowerBeamformTester::compare(
  const thrust::host_vector<T> cpu,
  const thrust::host_vector<T> gpu,
  const float tol)
{
  // Check if vectors have the same size
  ASSERT_TRUE(cpu.size() == gpu.size())
    << "Host and device vector size not equal" << std::endl;

  // Copy CUDA results to host
	float abs_diff;
	int arg_max_deviation = 0;

	float gpu_element;
	float cpu_element;
	float max_deviation = .0;

  for(std::size_t i = 0; i < gpu.size(); i++)
  {
			if constexpr (std::is_same<T, __half>::value)
			{
				gpu_element = __half2float(gpu[i]);
				cpu_element = __half2float(cpu[i]);
			}
			else
			{
				gpu_element = gpu[i];
				cpu_element = cpu[i];
			}

			abs_diff = std::abs(cpu_element - gpu_element);
			if(abs_diff > max_deviation)
			{
					arg_max_deviation = i;
					max_deviation = abs_diff;
			}

      ASSERT_TRUE(std::abs(cpu_element - gpu_element) <= std::abs(gpu_element)*tol/2)
      	<< "Beamformer with Stokes I: CPU and GPU results are unequal for element " << std::to_string(i) << std::endl
        << "  CPU result: " << std::to_string(cpu_element) << std::endl
        << "  GPU result: " << std::to_string(gpu_element) << std::endl;
  }
	BOOST_LOG_TRIVIAL(debug) << "Maximum deviation detected for element " << std::to_string(arg_max_deviation) << std::endl
		<< "Deviation abs(cpu[" << std::to_string(arg_max_deviation)
		<< "] - gpu[" << std::to_string(arg_max_deviation) << "]) = "
		<< std::to_string(max_deviation) << std::endl;
}


template<typename T, typename U>
void PowerBeamformTester::gpu_process(
  thrust::host_vector<T>& in,
  thrust::host_vector<U>& out,
  thrust::host_vector<T>& weights)
{
  MultiLog log(_conf.logname);
  PowerBeamformer<decltype(*this), RawVoltage<T>, Weights<T>, PowerBeam<U>> beamformer(_conf, log, *this);

  beamformer.template sync_copy<T, RawVoltage<T>>(in);
  beamformer.template sync_copy<T, Weights<T>>(weights);
  beamformer.process();
  beamformer.template sync_copy<U, PowerBeam<U>>(out);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}


template<typename T, typename U>
void PowerBeamformTester::cpu_process(
  thrust::host_vector<T>& in,
  thrust::host_vector<U>& out,
  thrust::host_vector<T>& weights)
{
	BOOST_LOG_TRIVIAL(debug) << "Calculating CPU results... " << std::flush;
	int ap = _conf.n_antenna * _conf.n_pol;
	int fap = _conf.n_channel * ap;
	int out_idx, in_idx, weight_idx;
	float2 voltage = {0,0};
	float power;
	for(int t = 0; t < _conf.n_samples; t+=_conf.interval)
	{
		for(int b = 0; b < _conf.n_beam; b++)
		{
			for(int f = 0; f < _conf.n_channel; f++)
			{
				out_idx = b * _conf.n_channel * _conf.n_samples/_conf.interval + f * _conf.n_samples/_conf.interval + t/_conf.interval;
				power = .0;
				for(int i = 0; i < _conf.interval; i++)
				{
					for(int p = 0; p < _conf.n_pol; p++)
					{
						for(int a = 0; a < _conf.n_antenna; a++)
						{
							in_idx = (t+i) * fap + f * ap + a * _conf.n_pol + p;
							weight_idx = b * fap + f * ap + a * _conf.n_pol + p;

							if constexpr(std::is_same<T, float2>::value)
							{
								voltage = cmadd(in[in_idx], weights[weight_idx], voltage);
							}
							else if constexpr(std::is_same<T, __half2>::value)
							{
								voltage = cmadd(__half22float2(in[in_idx]), __half22float2(weights[weight_idx]), voltage);
							}
						}
						power += cabs(voltage)*cabs(voltage) / _conf.interval;
						voltage = {0,0};
					}
				}
				if constexpr (std::is_same<T, float2>::value)
					out[out_idx] = power;
				else if constexpr (std::is_same<T, __half2>::value)
					out[out_idx] = __float2half(power);
			}
		}
	}
}



/**
* Testing with Google Test Framework
*/
TEST_P(PowerBeamformTester, BeamformerPowerSingle){
  std::cout << std::endl
    << "\n-------------------------------------------------------------" << std::endl
    << " Testing power mode with T=float2 (single precision 2x32bit) " << std::endl
    << "-------------------------------------------------------------\n" << std::endl;
  test<float2, float>();
}


// TEST_P(PowerBeamformTester, BeamformerPowerHalf)
// {
//   std::cout << std::endl
//     << "-------------------------------------------------------------" << std::endl
//     << " Testing power mode with T=__half2 (half precision 2x16bit)  " << std::endl
//     << "-------------------------------------------------------------" << std::endl << std::endl;
//   test<__half2, __half>();
// }


INSTANTIATE_TEST_CASE_P(BeamformerTesterInstantiation, PowerBeamformTester, ::testing::Values(

	// psrdada input key | psrdada output key | device ID | logname | samples | channels | antenna | polarisation | beam | interval/integration | beamformer type
	bf_config_t{0xdada, 0xdadd, 0, "test.log", 1024, 32, 512, 2, 64, 32, POWER_BF},
	bf_config_t{0xdada, 0xdadd, 0, "test.log", 24, 1, 512, 2, 32, 8, POWER_BF},
	bf_config_t{0xdada, 0xdadd, 0, "test.log", 256, 8, 512, 2, 64, 64, POWER_BF},
  bf_config_t{0xdada, 0xdadd, 0, "test.log", 4096, 32, 512, 2, 1024, 64, POWER_BF},
  bf_config_t{0xdada, 0xdadd, 0, "test.log", 128, 256, 512, 2, 64, 64, POWER_BF},
  bf_config_t{0xdada, 0xdadd, 0, "test.log", 4096, 7, 32, 2, 64, 64, POWER_BF},
	bf_config_t{0xdada, 0xdadd, 0, "test.log", 262144, 7, 32, 2, 256, 64, POWER_BF}
));



} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace test

#endif
