#ifndef UNPACKERTESTER_CUH
#define UNPACKERTESTER_CUH

#include <random>
#include <gtest/gtest.h>
#include <vector>
#include <byteswap.h>
#include <thrust/host_vector.h>

#include "psrdada_cpp/cryopaf/Unpacker.cuh"

namespace psrdada_cpp {
namespace cryopaf {
namespace test {

struct UnpackerTestConfig{
   int device_id;
   std::size_t n_samples;
   std::size_t n_channel;
   std::size_t n_elements;
   std::size_t n_pol;
   std::string protocol;
   void print()
   {
     std::cout << "Test configuration" << std::endl;
     std::cout << "device_id: " << device_id << std::endl;
     std::cout << "n_samples: " << n_samples << std::endl;
     std::cout << "n_channel: " << n_channel << std::endl;
     std::cout << "n_elements: " << n_elements << std::endl;
     std::cout << "n_pol: " << n_pol << std::endl;
     std::cout << "protocol: " << protocol << std::endl;
   }
};


class UnpackerTester: public ::testing::TestWithParam<UnpackerTestConfig>
{
public:
    UnpackerTester();
    ~UnpackerTester();

    template<typename T>
    void test();

protected:
    void SetUp() override;
    void TearDown() override;

    template<typename T>
    void compare(
      thrust::host_vector<T>& cpu,
      thrust::host_vector<T>& gpu);

    template<typename T>
    void cpu_process(
      thrust::host_vector<uint64_t>& input,
      thrust::host_vector<T>& output);

    template<typename T>
    void gpu_process(
      thrust::host_vector<uint64_t>& input,
      thrust::host_vector<T>& output);

private:
  UnpackerTestConfig conf;

};

} //namespace test
} //namespace cryopaf
} //namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/test/src/UnpackerTester.cu"

#endif //PSRDADA_CPP_EFFELSBERG_PAF_UNPACKERTESTER_CUH
