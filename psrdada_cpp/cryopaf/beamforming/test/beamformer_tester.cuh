#ifndef BEAMFORMER_TEST_H_
#define BEAMFORMER_TEST_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <random>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include "psrdada_cpp/cryopaf/beamforming/cu_beamformer.cuh"
#include "psrdada_cpp/cryopaf/types.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{
namespace test{


class BeamformerTester : public ::testing::TestWithParam<bf_config_t> {
public:
  BeamformerTester();
  ~BeamformerTester();


  /**
  * @brief
  *
  * @param			in  TAFPT
  *							out BTF
  * 						weights BAFPT
  */
  template <typename T, typename U>
  void test();


  /**
  * @brief      POWER BF
  *
  * @param			in  TAFPT
  *							out BTF
  * 						weights BAFPT
  */
  template<typename T, typename U>
  void cpu_process(
    thrust::host_vector<T>& in,
    thrust::host_vector<U>& out,
    thrust::host_vector<T>& weights);


  /**
  * @brief      POWER BF
  *
  * @param			in  TAFPT
  *							out BTF
  * 						weights BAFPT
  */
  template<typename T, typename U>
  void gpu_process(
    thrust::device_vector<T>& in,
    thrust::device_vector<U>& out,
    thrust::device_vector<T>& weights);


  /**
	* @brief       POWER BF
	*
	* @param			in  TAFPT
	*							out BTF
	* 						weights BAFPT
	*/
  template <typename T>
  void compare(
    const thrust::host_vector<T> cpu,
    const thrust::device_vector<T> gpu,
    const float tol=0.01);


protected:
  void SetUp() override;
  void TearDown() override;

private:
  bf_config_t _conf;
  cudaStream_t _stream;
};

} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test


#endif //BEAMFORMER_TEST_H_
