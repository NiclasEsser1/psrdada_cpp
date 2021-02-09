#ifdef VOLTAGEBEAMFORM_TESTER_H_

namespace psrdada_cpp{
namespace cryopaf{
namespace test{




VoltageBeamformTester::VoltageBeamformTester()
  : ::testing::TestWithParam<bf_config_t>()
  , _conf(GetParam()) // GetParam() is inherited from base TestWithParam
  , _stream(0)
{
}


VoltageBeamformTester::~VoltageBeamformTester()
{
}

void VoltageBeamformTester::SetUp()
{
}

void VoltageBeamformTester::TearDown()
{
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

	BOOST_LOG_TRIVIAL(debug) << "Required device memory: " << std::to_string(required_mem / (1024*1024)) << 	"MiB";
	BOOST_LOG_TRIVIAL(debug) << "Required host memory: " << std::to_string((required_mem
		+ (input_size + weight_size) * sizeof(T)) / (1024*1024)) << "MiB";

	// Allocate host vectors
  thrust::host_vector<T> host_output(output_size);             // Dataproduct: B-F-T-P
	thrust::host_vector<T> dev_output(output_size);              // Dataproduct: B-F-T-P
	thrust::host_vector<T> host_input(input_size, {0, 0});       // Dataprodcut: T-F-A-P
	thrust::host_vector<T> host_weights(weight_size, {0, 0});    // Dataprodcut: B-F-A-P

	// Generate test samples / normal distributed noise for input signal
	for (size_t i = 0; i < host_input.size(); i++)
	{
	  	if constexpr (std::is_same<T, __half2>::value)
	    	host_input[i] = __float22half2_rn({normal_dist(generator), normal_dist(generator)});
	  	else
	    	host_input[i] = {normal_dist(generator), normal_dist(generator)};
	}
	// Build complex weight as C * exp(i * theta).
	for (size_t i = 0; i < host_weights.size(); i++)
	{
		if constexpr (std::is_same<T, __half2>::value)
			host_weights[i] = __float22half2_rn({normal_dist(generator), normal_dist(generator)});
		else
			host_weights[i] = {normal_dist(generator), normal_dist(generator)};
	}

	// launches cpu beamforming
	cpu_process(host_input, host_output, host_weights);
	// launches CUDA kernel
	gpu_process(host_input, dev_output, host_weights);
	// compare results of both outputs
	compare(host_output, dev_output);

}


template <typename T>
void VoltageBeamformTester::compare(
  const thrust::host_vector<T> cpu,
  const thrust::host_vector<T> gpu,
  const float tol)
{
	// Check if vectors have the same size
	ASSERT_TRUE(cpu.size() == gpu.size())
		<< "Host and device vector size not equal" << std::endl;

	int arg_max = 0;
  double absolute;

  T max_deviation = {0,0};
	T diff;

  // Check each element of CPU and GPU implementation
	for(std::size_t i = 0; i < gpu.size(); i++)
	{
    diff = csub(gpu[i], cpu[i]);
    absolute = (double)cabs(diff);

    if(absolute > cabs(max_deviation))
	  {
        arg_max = i;
        max_deviation = diff;
    }
    ASSERT_TRUE(absolute <= 1)  << "Beamformer: CPU and GPU result is unequal for element " << std::to_string(i) << std::endl
      << "  CPU result: " << std::to_string(cpu[i].x) << "+ i*" << std::to_string(cpu[i].y) << std::endl
      << "  GPU result: " << std::to_string(gpu[i].x) << "+ i*" << std::to_string(gpu[i].y) << std::endl;
  }

  BOOST_LOG_TRIVIAL(debug) << "Maximum deviation detected for element " << std::to_string(arg_max) << std::endl
    << "Deviation abs(cpu[" << std::to_string(arg_max)
    << "] - gpu[" << std::to_string(arg_max) << "]) = "
    << std::to_string(max_deviation.x) << " + i* " << std::to_string(max_deviation.y);
}


template<typename T>
void VoltageBeamformTester::gpu_process(
  thrust::host_vector<T>& in,       // Dataprodcut: T-F-A-P
  thrust::host_vector<T>& out,      // Dataprodcut: B-F-A-P
  thrust::host_vector<T>& weights)  // Dataproduct: B-F-T-P
{
  MultiLog log(_conf.logname);
	VoltageBeamformer<decltype(*this), RawVoltage<T>, Weights<T>, VoltageBeam<T>> beamformer(_conf, log, *this);
  beamformer.template sync_copy<T, RawVoltage<T>>(in);
  beamformer.template sync_copy<T, Weights<T>>(weights);
	beamformer.process();
  beamformer.template sync_copy<T, VoltageBeam<T>>(out);
	CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}


template<typename T>
void VoltageBeamformTester::cpu_process(
  thrust::host_vector<T>& in,       // Dataprodcut: T-F-A-P
  thrust::host_vector<T>& out,      // Dataprodcut: B-F-A-P
  thrust::host_vector<T>& weights)  // Dataproduct: B-F-T-P
{
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
            in_idx = t * _conf.n_channel * _conf.n_antenna * _conf.n_pol
              + f * _conf.n_antenna * _conf.n_pol + a * _conf.n_pol + p;
						weight_idx = b * _conf.n_channel * _conf.n_antenna * _conf.n_pol
							+ f * _conf.n_antenna * _conf.n_pol + a * _conf.n_pol + p;
						out_idx = b * _conf.n_channel * _conf.n_samples * _conf.n_pol
							+ f * _conf.n_samples * _conf.n_pol + t * _conf.n_pol + p;
						out[out_idx] = cmadd(in[in_idx], weights[weight_idx], out[out_idx]);
					}
				}
			}
		}
	}

}


/**
* Testing with Google Test Framework
*/
// TEST_P(VoltageBeamformTester, BeamformerVoltageDouble){
//   std::cout << std::endl
//     << "-------------------------------------------------------------" << std::endl
//     << " Testing voltage mode with T=double2 (half precision 2x64bit)" << std::endl
//     << "-------------------------------------------------------------" << std::endl << std::endl;
//   test<double2>();
// }

TEST_P(VoltageBeamformTester, BeamformerVoltageSingle){
  std::cout << std::endl
    << "-------------------------------------------------------------" << std::endl
    << "Testing voltage mode with T=float2 (single precision 2x32bit)" << std::endl
    << "-------------------------------------------------------------" << std::endl << std::endl;
  test<float2>();
}


INSTANTIATE_TEST_CASE_P(BeamformerTesterInstantiation, VoltageBeamformTester, ::testing::Values(

	// psrdada input key | psrdada output key | device ID | logname | samples | channels | antenna | polarisation | beam | interval/integration | beamformer type
	bf_config_t{0xdada, 0xdadd, 0, "test.log", 2048, 1, 36, 2, 13, 0, VOLTAGE_BF},
	bf_config_t{0xdada, 0xdadd, 0, "test.log", 2048, 2, 36, 2, 13, 0, VOLTAGE_BF},
	bf_config_t{0xdada, 0xdadd, 0, "test.log", 2048, 4, 36, 2, 13, 0, VOLTAGE_BF},
	bf_config_t{0xdada, 0xdadd, 0, "test.log", 2048, 6, 36, 1, 13, 0, VOLTAGE_BF},
	bf_config_t{0xdada, 0xdadd, 0, "test.log", 512, 7, 36, 2, 11, 0, VOLTAGE_BF},
  bf_config_t{0xdada, 0xdadd, 0, "test.log", 2048, 8, 36, 2, 13, 0, VOLTAGE_BF},
  bf_config_t{0xdada, 0xdadd, 0, "test.log", 2048, 9, 36, 2, 13, 0, VOLTAGE_BF},
  bf_config_t{0xdada, 0xdadd, 0, "test.log", 2048, 10, 36, 2, 13, 0, VOLTAGE_BF},
  bf_config_t{0xdada, 0xdadd, 0, "test.log", 128, 16, 36, 2, 13, 0, VOLTAGE_BF},
	bf_config_t{0xdada, 0xdadd, 0, "test.log", 128, 17, 36, 2, 13, 0, VOLTAGE_BF}
));


} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace test

#endif
