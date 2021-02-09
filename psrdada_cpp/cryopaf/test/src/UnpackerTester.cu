#ifdef UNPACKERTESTER_CUH

namespace psrdada_cpp {
namespace cryopaf {
namespace test {

UnpackerTester::UnpackerTester()
    : ::testing::TestWithParam<bf_config_t>()
    , conf(GetParam())
{

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
    std::size_t n = conf.n_samples * conf.n_channel * conf.n_antenna * conf.n_pol;
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
    int in_idx, out_idx;
    for(int t1 = 0; t1 < conf.n_samples/NSAMP_DF; t1++)
    {
      for(int a = 0; a < conf.n_antenna; a++)
      {
        for(int t2 = 0; t2 < NSAMP_DF; t2++)
        {
          for (int f = 0; f < conf.n_channel; f++)
          {
              in_idx = t1 * conf.n_antenna * conf.n_channel * NSAMP_DF
                + a * conf.n_channel * NSAMP_DF
                + t2 * conf.n_channel
                + f;
              out_idx = (NSAMP_DF * t1 + t2) * conf.n_channel * conf.n_antenna * conf.n_pol
                + f * conf.n_antenna * conf.n_pol
                + a * conf.n_pol;
              tmp = bswap_64(input[in_idx]);
              output[out_idx].x = (float)(tmp & 0x000000000000ffffULL);
              output[out_idx].y = (float)((tmp & 0x00000000ffff0000ULL) >> 16);
              output[out_idx + 1].x = (float)((tmp & 0x0000ffff00000000ULL) >> 32);
              output[out_idx + 1].y = (float)((tmp & 0xffff000000000000ULL) >> 48);
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
    MultiLog log("unpack-tester");
    BOOST_LOG_TRIVIAL(debug) << "Computing GPU results...";
    Unpacker<decltype(*this), RawVoltage<uint64_t>, RawVoltage<T>> unpacker(conf, log, *this);

    unpacker.template sync_copy<uint64_t, RawVoltage<uint64_t>>(input, cudaMemcpyHostToDevice);
    unpacker.process();
    unpacker.template sync_copy<T, RawVoltage<T>>(output, cudaMemcpyDeviceToHost);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

template<typename T>
void UnpackerTester::compare(
  thrust::host_vector<T>& cpu,
  thrust::host_vector<T>& gpu)
{
    BOOST_LOG_TRIVIAL(debug) << "Comparing results...";

    ASSERT_TRUE(cpu.size() == gpu.size());
    for (int i = 0; i < cpu.size(); i++)
    {
      ASSERT_TRUE(cpu[i].x == gpu[i].x && cpu[i].y == gpu[i].y)
        << "Unpacker: CPU and GPU results are unequal for element " << i << std::endl
        << "  CPU result: " << cpu[i].x << " + i*" << cpu[i].y << std::endl
        << "  GPU result: " << gpu[i].x << " + i*" << gpu[i].y << std::endl;
    }
}

/**
* Testing with Google Test Framework
*/
TEST_P(UnpackerTester, UnpackerSinglePrecision){
  BOOST_LOG_TRIVIAL(info)
    << "\n-------------------------------------------------------------" << std::endl
    << " Testing unpacker with T=float2 (single precision 2x32bit) " << std::endl
    << "-------------------------------------------------------------\n" << std::endl;
  test<float2>();
}

INSTANTIATE_TEST_CASE_P(UnpackerTesterInstantiation, UnpackerTester, ::testing::Values(
	// psrdada input key | psrdada output key | device ID | logname | samples | channels | antenna | polarisation | beam | interval/integration | beamformer type
  bf_config_t{0xdada, 0xdadd, 0, "test.log", 4096, 7, 36, 2, 64, 64, POWER_BF},
  bf_config_t{0xdada, 0xdadd, 0, "test.log", 128, 14, 2, 2, 64, 64, POWER_BF},
  bf_config_t{0xdada, 0xdadd, 0, "test.log", 256, 21, 32, 2, 64, 64, POWER_BF},
  bf_config_t{0xdada, 0xdadd, 0, "test.log", 512, 21, 11, 2, 64, 64, POWER_BF},
  bf_config_t{0xdada, 0xdadd, 0, "test.log", 1024, 14, 28, 2, 64, 64, POWER_BF},
	bf_config_t{0xdada, 0xdadd, 0, "test.log", 262144, 7, 17, 2, 256, 64, POWER_BF}
));

} //namespace test
} // cryopaf
} //namespace psrdada_cpp

#endif
