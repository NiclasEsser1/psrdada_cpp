#ifndef UNPACKERTESTER_CUH
#define UNPACKERTESTER_CUH

#include <random>
#include <gtest/gtest.h>
#include <vector>
#include <byteswap.h>

#include "thrust/host_vector.h"

#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/cryopaf/Types.cuh"
#include "psrdada_cpp/cryopaf/Unpacker.cuh"

namespace psrdada_cpp {
namespace cryopaf {
namespace test {

class UnpackerTester: public ::testing::TestWithParam<bf_config_t>
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
  bf_config_t conf;

};

} //namespace test
} //namespace cryopaf
} //namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/test/src/UnpackerTester.cu"

#endif //PSRDADA_CPP_EFFELSBERG_PAF_UNPACKERTESTER_CUH
