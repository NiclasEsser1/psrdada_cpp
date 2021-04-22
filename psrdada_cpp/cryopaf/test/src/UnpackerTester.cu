#ifdef UNPACKERTESTER_CUH

namespace psrdada_cpp {
namespace cryopaf {
namespace test {

UnpackerTester::UnpackerTester()
    : ::testing::TestWithParam<UnpackerTestConfig>(),
    conf(GetParam())
{
  CUDA_ERROR_CHECK(cudaSetDevice(conf.device_id));
}
UnpackerTester::~UnpackerTester()
{

}
void UnpackerTester::SetUp()
{
}
void UnpackerTester::TearDown()
{
}

template<typename T>
void UnpackerTester::test()
{
    std::size_t n = conf.n_samples * conf.n_channel * conf.n_elements * conf.n_pol;
    std::default_random_engine generator;
    std::uniform_int_distribution<uint64_t> distribution(0,65535);

    thrust::host_vector<uint64_t> input(n/2);
    thrust::host_vector<T> host_output(n);
    thrust::host_vector<T> dev_output(n);
    BOOST_LOG_TRIVIAL(debug) << "Required device memory: " << (n * (sizeof(T) + sizeof(uint64_t)/2)) / (1024*1024) << "MiB";
    BOOST_LOG_TRIVIAL(debug) << "Required host memory: "   << (n * (2 * sizeof(T) + sizeof(uint64_t)/2)) / (1024*1024) << "MiB";

    // Generate test samples
    BOOST_LOG_TRIVIAL(debug) << "Generating " << n << " test samples...";
    for (int i = 0; i < n/2; i++)
    {
        input[i] = (uint64_t)((distribution(generator) << 48) & 0xFFFF000000000000LL);
        input[i] += (uint64_t)((distribution(generator) << 32) & 0x0000FFFF00000000LL);
        input[i] += (uint64_t)((distribution(generator) << 16) & 0x00000000FFFF0000LL);
        input[i] += (uint64_t)((distribution(generator)) & 0x000000000000FFFFLL);
    }

    cpu_process(input, host_output);

    gpu_process(input, dev_output);

    compare(host_output, dev_output);
}

template<typename T>
void UnpackerTester::cpu_process(
  thrust::host_vector<uint64_t>& input,
  thrust::host_vector<T>& output)
{
  BOOST_LOG_TRIVIAL(debug) << "Computing CPU results...";
  uint64_t tmp;
  int in_idx, out_idx_x, out_idx_y;
  if(conf.protocol == "codif")
  {
    for (int f = 0; f < conf.n_channel; f++)
    {
      for(int t1 = 0; t1 < conf.n_samples/NSAMP_DF; t1++)
      {
        for(int a = 0; a < conf.n_elements; a++)
        {
          for(int t2 = 0; t2 < NSAMP_DF; t2++)
          {
              in_idx = t1 * conf.n_elements * conf.n_channel * NSAMP_DF
                + a * conf.n_channel * NSAMP_DF
                + t2 * conf.n_channel
                + f;
              out_idx_x = f * conf.n_pol * conf.n_samples * conf.n_elements
                + (NSAMP_DF * t1 + t2) * conf.n_elements
                + a;
              out_idx_y = f * conf.n_pol * conf.n_samples * conf.n_elements
                + conf.n_samples * conf.n_elements
                + (NSAMP_DF * t1 + t2) * conf.n_elements
                + a;
              tmp = bswap_64(input[in_idx]);
              if constexpr(std::is_same<T, __half2>::value)
              {
                output[out_idx_x].x = __float2half(static_cast<float>(tmp & 0x000000000000ffffULL));
                output[out_idx_x].y = __float2half(static_cast<float>((tmp & 0x00000000ffff0000ULL) >> 16));
                output[out_idx_y].x = __float2half(static_cast<float>((tmp & 0x0000ffff00000000ULL) >> 32));
                output[out_idx_y].y = __float2half(static_cast<float>((tmp & 0xffff000000000000ULL) >> 48));
              }
              else
              {
                output[out_idx_x].x = static_cast<decltype(T::x)>(tmp & 0x000000000000ffffULL);
                output[out_idx_x].y = static_cast<decltype(T::y)>((tmp & 0x00000000ffff0000ULL) >> 16);
                output[out_idx_y].x = static_cast<decltype(T::x)>((tmp & 0x0000ffff00000000ULL) >> 32);
                output[out_idx_y].y = static_cast<decltype(T::y)>((tmp & 0xffff000000000000ULL) >> 48);
              }
          }
        }
      }
    }
  }
}

template<typename T>
void UnpackerTester::gpu_process(
  thrust::host_vector<uint64_t>& input,
  thrust::host_vector<T>& output)
{
    cudaStream_t stream;
    CUDA_ERROR_CHECK(cudaStreamCreate(&stream));
    // Copy host 2 device (input)
    thrust::device_vector<uint64_t> dev_input = input;
    thrust::device_vector<T> dev_output(output.size());

    BOOST_LOG_TRIVIAL(debug) << "Computing GPU results...";

    Unpacker<T> unpacker(stream, conf.n_samples, conf.n_channel, conf.n_elements, conf.protocol);
    unpacker.unpack(
        (char*)thrust::raw_pointer_cast(dev_input.data()),
        (T*)thrust::raw_pointer_cast(dev_output.data()));

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaStreamDestroy(stream));
    // Copy device 2 host (output)
    output = dev_output;
}

template<typename T>
void UnpackerTester::compare(
  thrust::host_vector<T>& cpu,
  thrust::host_vector<T>& gpu)
{
    BOOST_LOG_TRIVIAL(debug) << "Comparing results...";

    float2 gpu_element;
  	float2 cpu_element;

    ASSERT_TRUE(cpu.size() == gpu.size())
  		<< "Host and device vector size not equal" << std::endl;
    for (int i = 0; i < cpu.size(); i++)
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
      ASSERT_TRUE(cpu_element.x == gpu_element.x && cpu_element.y == gpu_element.y)
        << "Unpacker: CPU and GPU results are unequal for element " << std::to_string(i) << std::endl
        << "  CPU result: " << std::to_string(cpu_element.x) << " + i*" << std::to_string(cpu_element.y) << std::endl
        << "  GPU result: " << std::to_string(gpu_element.x) << " + i*" << std::to_string(gpu_element.y) << std::endl;
    }
}

/**
* Testing with Google Test Framework
*/

TEST_P(UnpackerTester, UnpackerSinglePrecisionFPTE){
  BOOST_LOG_TRIVIAL(info)
    << "\n------------------------------------------------------------------------" << std::endl
    << " Testing unpacker with T=float2 (single precision 2x32bit), Format = FPTE " << std::endl
    << "------------------------------------------------------------------------\n" << std::endl;
  test<float2>();
}

TEST_P(UnpackerTester, UnpackerHalfPrecisionFPTE){
  BOOST_LOG_TRIVIAL(info)
    << "\n------------------------------------------------------------------------" << std::endl
    << " Testing unpacker with T=half2 (half precision 2x16bit), Format = FPTE " << std::endl
    << "------------------------------------------------------------------------\n" << std::endl;
  test<__half2>();
}

INSTANTIATE_TEST_CASE_P(UnpackerTesterInstantiation, UnpackerTester, ::testing::Values(
	// devie id | samples | channels | elements | protocol
  UnpackerTestConfig{0, 4096, 7, 36, "codif"},
  UnpackerTestConfig{0, 128, 14, 2, "codif"},
  UnpackerTestConfig{0, 256, 21, 32, "codif"},
  UnpackerTestConfig{0, 512, 21, 11, "codif"},
  UnpackerTestConfig{0, 1024, 14, 28, "codif"},
  UnpackerTestConfig{0, 2048, 21, 28, "codif"},
  UnpackerTestConfig{0, 4096, 14, 128, "codif"},
  UnpackerTestConfig{0, 1024, 14, 64, "codif"},
  UnpackerTestConfig{0, 256, 14, 17, "codif"},
  UnpackerTestConfig{0, 128, 14, 13, "codif"},
  UnpackerTestConfig{0, 896, 14, 11, "codif"}
));

} //namespace test
} // cryopaf
} //namespace psrdada_cpp

#endif
