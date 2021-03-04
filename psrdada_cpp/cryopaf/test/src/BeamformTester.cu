#ifdef BEAMFORMER_TESTER_H_

namespace psrdada_cpp{
namespace cryopaf{
namespace test{




BeamformTester::BeamformTester()
  : ::testing::TestWithParam<BeamformTestConfig>(),
  conf(GetParam()), // GetParam() is inherited from base TestWithParam
  _stream(0)
{
  BOOST_LOG_TRIVIAL(debug) << "Creating instance of BeamformTester";
	CUDA_ERROR_CHECK(cudaSetDevice(conf.device_id));
}


BeamformTester::~BeamformTester()
{
  BOOST_LOG_TRIVIAL(debug) << "Destroying instance of BeamformTester";
}


void BeamformTester::SetUp()
{
}


void BeamformTester::TearDown()
{
}


template<typename T, typename U>
void BeamformTester::test(float tol)
{
  // Set up normal distributed sample and weight generator
  const float input_level = 1.0f;
  const double pi = std::acos(-1);
  std::default_random_engine generator;
  std::normal_distribution<float> normal_dist(0.0, input_level);

  // Calulate memory size for input, weights and output
  std::size_t input_size = conf.n_samples
    * conf.n_elements
    * conf.n_channel
    * conf.n_pol;
  std::size_t weight_size =  conf.n_beam
    * conf.n_elements
    * conf.n_channel
    * conf.n_pol;
  std::size_t output_size;
  if constexpr (std::is_same<T, U>::value)
  {
    output_size = conf.n_samples
      * conf.n_beam
      * conf.n_channel
      * conf.n_pol;
  }
  else
  {
    output_size = conf.n_samples
      * conf.n_beam
      * conf.n_channel
      / conf.integration;
  }
  std::size_t required_mem = input_size * sizeof(T)
    + weight_size * sizeof(T)
    + output_size * sizeof(U);
  BOOST_LOG_TRIVIAL(debug) << "Required device memory: " << std::to_string(required_mem / (1024*1024)) << "MiB";
  BOOST_LOG_TRIVIAL(debug) << "Required host memory: " << std::to_string((required_mem + output_size * sizeof(U)) / (1024*1024)) << "MiB";

  // Allocate host vectors
  thrust::host_vector<U> host_output(output_size);
  thrust::host_vector<U> dev_output(output_size);
	thrust::host_vector<T> input(input_size);
	thrust::host_vector<T> weights(weight_size);

  // Generate test samples / normal distributed noise for input signal
	for (size_t i = 0; i < input.size(); i++)
  {
      if constexpr (std::is_same<T, __half2>::value)
      {
        input[i] = __float22half2_rn({normal_dist(generator), normal_dist(generator)});
      }
      else
      {
        input[i] = {normal_dist(generator), normal_dist(generator)};
      }
	}

	// Generate test weights
  for (size_t i = 0; i < weights.size(); i++)
  {
      if constexpr (std::is_same<T, __half2>::value)
      {
        weights[i] = __float22half2_rn({normal_dist(generator), normal_dist(generator)});
      }
			else
      {
        weights[i] = {normal_dist(generator), normal_dist(generator)};
      }
  }

	// launches CUDA kernel
	gpu_process(input, weights, dev_output);

  if constexpr (std::is_same<T, U>::value)
  {
    // launches cpu beamforming
    cpu_process_voltage(input, weights, host_output);
    // compare results of both outputs
    compare_voltage(host_output, dev_output, tol);
  }
  else
  {
  	// launches cpu beamforming
  	cpu_process_power(input, weights, host_output);
    // compare results of both outputs
  	compare_power(host_output, dev_output, tol);
  }
}


template <typename T>
void BeamformTester::compare_power(
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


template <typename T>
void BeamformTester::compare_voltage(
  const thrust::host_vector<T> cpu,
  const thrust::host_vector<T> gpu,
  const float tol)
{
  BOOST_LOG_TRIVIAL(debug) << "Comparing results ..";
	// Check if vectors have the same size
	ASSERT_TRUE(cpu.size() == gpu.size())
		<< "Host and device vector size not equal" << std::endl;

	int arg_max = 0;
  float absolute;

  float2 max_deviation = {0,0};
	float2 diff;
  float2 gpu_element;
	float2 cpu_element;
  // Check each element of CPU and GPU implementation
	for(std::size_t i = 0; i < gpu.size(); i++)
	{
    if constexpr (std::is_same<T, __half2>::value)
    {
      gpu_element = __half22float2(gpu[i]);
      cpu_element = __half22float2(cpu[i]);
    }
    else
    {
      gpu_element = gpu[i];
      cpu_element = cpu[i];
    }
    diff = csub(gpu_element, cpu_element);
    absolute = cabs(diff);

    if(absolute > cabs(max_deviation))
	  {
        arg_max = i;
        max_deviation = diff;
    }
    ASSERT_TRUE(absolute <= tol)  << "Beamformer: CPU and GPU result is unequal for element " << std::to_string(i) << std::endl
      << "  CPU result: " << std::to_string(cpu_element.x) << "+ i*" << std::to_string(cpu_element.y) << std::endl
      << "  GPU result: " << std::to_string(gpu_element.x) << "+ i*" << std::to_string(gpu_element.y) << std::endl;
  }

  BOOST_LOG_TRIVIAL(debug) << "Maximum deviation detected for element " << std::to_string(arg_max) << std::endl
    << "Deviation abs(cpu[" << std::to_string(arg_max)
    << "] - gpu[" << std::to_string(arg_max) << "]) = "
    << std::to_string(max_deviation.x) << " + i* " << std::to_string(max_deviation.y);
}


template<typename T, typename U>
void BeamformTester::gpu_process(
  thrust::host_vector<T>& in,
  thrust::host_vector<T>& weights,
  thrust::host_vector<U>& out)
{
  thrust::device_vector<T> dev_input = in;
  thrust::device_vector<T> dev_weights = weights;
  thrust::device_vector<U> dev_output(out.size());

  cudaStream_t stream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&stream));

  Beamformer<T> beamformer(
    stream,
    conf.n_samples,
    conf.n_channel,
    conf.n_elements,
    conf.n_beam,
    conf.integration);

  beamformer.process(
    thrust::raw_pointer_cast(dev_input.data()),
    thrust::raw_pointer_cast(dev_weights.data()),
    thrust::raw_pointer_cast(dev_output.data()));

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaStreamDestroy(stream));
  // Copy device 2 host (output)
  out = dev_output;
}


template<typename T, typename U>
void BeamformTester::cpu_process_power(
  thrust::host_vector<T>& in,       // Dataprodcut: F-P-T-E
  thrust::host_vector<T>& weights,  // Dataproduct: F-P-B-E
  thrust::host_vector<U>& out)      // Dataprodcut: F-P-T-B
{
	BOOST_LOG_TRIVIAL(debug) << "Calculating CPU results... " << std::flush;
  int te = conf.n_samples * conf.n_elements;
  int be = conf.n_beam * conf.n_elements;
  int tb = conf.n_beam * conf.n_samples / conf.integration;
  int pte = conf.n_pol * te;
  int pbe = conf.n_pol * be;
	int ou_idx, in_idx, wg_idx;
	float2 voltage = {0,0};
	float power;
  for(int f = 0; f < conf.n_channel; f++)
	{
    for(int p = 0; p < conf.n_pol; p++)
    {
  		for(int b = 0; b < conf.n_beam; b++)
  		{
        for(int t = 0; t < conf.n_samples / conf.integration; t++)
        {
          ou_idx = f * tb + conf.n_beam * t + b;
          power = 0;
          for(int i = 0; i < conf.integration; i++)
          {
            voltage = {0,0};
						for(int e = 0; e < conf.n_elements; e++)
						{
							in_idx = f * pte + p * te + conf.n_elements * (t * conf.integration + i) + e;
							wg_idx = f * pbe + p * be + conf.n_elements *  b + e;

							if constexpr(std::is_same<T, __half2>::value)
							{
								voltage = cmadd(__half22float2(in[in_idx]), __half22float2(weights[wg_idx]), voltage);
							}
              else
              {
								voltage = cmadd(in[in_idx], weights[wg_idx], voltage);
							}
						}
            power += cabs(voltage)*cabs(voltage) / conf.integration;
          }
    			if constexpr (std::is_same<T, __half2>::value)
    			{
            out[ou_idx] = __float2half(power + __half2float(out[ou_idx]));
          }
          else
          {
            out[ou_idx] += power;
          }
				}
      }
		}
	}
}


template<typename T>
void BeamformTester::cpu_process_voltage(
  thrust::host_vector<T>& in,       // Dataprodcut: F-P-T-E
  thrust::host_vector<T>& weights,  // Dataproduct: F-P-B-E
  thrust::host_vector<T>& out)      // Dataprodcut: F-P-T-B
{
  	BOOST_LOG_TRIVIAL(debug) << "Calculating CPU results... " << std::flush;
    int te = conf.n_samples * conf.n_elements;
    int be = conf.n_beam * conf.n_elements;
    int tb = conf.n_beam * conf.n_samples;
  	int ptb = conf.n_pol * tb;
    int pte = conf.n_pol * te;
  	int pbe = conf.n_pol * be;
  	int ou_idx, in_idx, wg_idx;
  	float2 voltage = {0,0};
    for(int f = 0; f < conf.n_channel; f++)
  	{
      for(int p = 0; p < conf.n_pol; p++)
      {
        for(int t = 0; t < conf.n_samples; t++)
    		{
          for(int b = 0; b < conf.n_beam; b++)
          {
            ou_idx = f * ptb + p * tb + conf.n_beam * t + b;
            voltage = {0,0};
						for(int e = 0; e < conf.n_elements; e++)
						{
							in_idx = f * pte + p * te + conf.n_elements * t + e;
							wg_idx = f * pbe + p * be + conf.n_elements * b + e;

							if constexpr(std::is_same<T, __half2>::value)
							{
								voltage = cmadd(__half22float2(in[in_idx]), __half22float2(weights[wg_idx]), voltage);
							}
              else
              {
                voltage = cmadd(in[in_idx], weights[wg_idx], voltage);
              }
						}
      			if constexpr (std::is_same<T, __half2>::value)
      			{
              out[ou_idx] = __float22half2_rn(voltage);
            }
            else
            {
              out[ou_idx] = voltage;
            }
  				}
        }
  		}
  	}
}

/**
* Testing with Google Test Framework
*/

TEST_P(BeamformTester, BeamformerVoltageHalf)
{
  std::cout << std::endl
    << "-------------------------------------------------------------" << std::endl
    << " Testing voltage mode with T=__half2 (half precision 2x16bit)  " << std::endl
    << "-------------------------------------------------------------" << std::endl << std::endl;
  test<__half2, __half2>(5); // High inaccuarcy due to half to float casting and vice versa, we need to accept a high tolerance
}

TEST_P(BeamformTester, BeamformerVoltageSingle){
  std::cout << std::endl
    << "\n-------------------------------------------------------------" << std::endl
    << " Testing voltage mode with T=float2 (single precision 2x32bit) " << std::endl
    << "---------------------------------------------------------------\n" << std::endl;
  test<float2, float2>(0.1);
}

TEST_P(BeamformTester, BeamformerPowerHalf)
{
  std::cout << std::endl
    << "-------------------------------------------------------------" << std::endl
    << " Testing power mode with T=__half2 (half precision 2x16bit)  " << std::endl
    << "-------------------------------------------------------------" << std::endl << std::endl;
  test<__half2, __half>(.1);  // High inaccuarcy due to half to float casting and vice versa between CPU and GPU. We need to accept a high tolerance.
}

TEST_P(BeamformTester, BeamformerPowerSingle){
  std::cout << std::endl
    << "\n-----------------------------------------------------------" << std::endl
    << " Testing power mode with T=float2 (single precision 2x32bit) " << std::endl
    << "-------------------------------------------------------------\n" << std::endl;
  test<float2, float>(.1);
}


INSTANTIATE_TEST_CASE_P(BeamformerTesterInstantiation, BeamformTester, ::testing::Values(

	// device ID | samples | channels | elements | polarisation | beam | integration/integration
  BeamformTestConfig{0, 8, 1, 4, 2, 4, 1},
  BeamformTestConfig{0, 128, 3, 32, 2, 32, 1},
  BeamformTestConfig{0, 512, 6, 67, 2, 97, 1},
  BeamformTestConfig{0, 1024, 17, 512, 2, 32, 1},
  BeamformTestConfig{0, 64, 1, 32, 2, 32, 2},
  BeamformTestConfig{0, 128, 42, 32, 2, 32, 4},
  BeamformTestConfig{0, 256, 7, 32, 2, 32, 8},
  BeamformTestConfig{0, 512, 10, 32, 2, 32, 16},
  BeamformTestConfig{0, 1024, 1, 32, 2, 32, 32},
  BeamformTestConfig{0, 2048, 7, 15, 2, 16, 1},
  BeamformTestConfig{0, 4096, 3, 33, 2, 17, 2},
  BeamformTestConfig{0, 64, 12, 32, 2, 32, 4},
  BeamformTestConfig{0, 64, 3, 1024, 2, 1024, 8},
  BeamformTestConfig{0, 256, 14, 36, 2, 2, 16},
  BeamformTestConfig{0, 1024, 7, 43, 2, 36, 32}
));



} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace test

#endif
