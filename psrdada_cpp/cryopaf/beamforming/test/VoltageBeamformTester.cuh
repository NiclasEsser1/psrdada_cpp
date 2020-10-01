#ifndef VOLTAGEBEAMFORM_TESTER_H_
#define VOLTAGEBEAMFORM_TESTER_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <random>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include "psrdada_cpp/cryopaf/beamforming/VoltageBeamformer.cuh"
#include "psrdada_cpp/cryopaf/types.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{
namespace test{


class VoltageBeamformTester : public ::testing::TestWithParam<bf_config_t> {
public:
  VoltageBeamformTester();
  ~VoltageBeamformTester();


  /**
  * @brief
  *
  * @param			in  TAFPT
  *							out BTF
  * 						weights BAFPT
  */
  template <typename T>
  void test(int device_id=0);


  /**
  * @brief      POWER BF
  *
  * @param			in  TAFPT
  *							out BTF
  * 						weights BAFP
  */
  template<typename T>
  void cpu_process(
    thrust::host_vector<T>& in,
    thrust::host_vector<T>& out,
    thrust::host_vector<T>& weights);


  /**
  * @brief      POWER BF
  *
  * @param			in  TAFPT
  *							out BTF
  * 						weights BAFPT
  */
  template<typename T>
  void gpu_process(
    thrust::device_vector<T>& in,
    thrust::device_vector<T>& out,
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
  int id;
  bf_config_t _conf;
  cudaStream_t _stream;
};

} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test

#include "psrdada_cpp/cryopaf/beamforming/test/src/VoltageBeamformTester.cu"

#endif //VOLTAGEBEAMFORM_TESTER_H_
