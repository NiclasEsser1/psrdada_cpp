#ifndef BEAMFORMER_TEST_H_
#define BEAMFORMER_TEST_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <cmath>
#include <complex>
#include <cuComplex.h>
#include <gtest/gtest.h>

#include "psrdada_cpp/cryopaf/beamforming/cu_beamformer.cuh"
#include "psrdada_cpp/cryopaf/cryopaf_conf.hpp"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{
namespace test{


class BeamformerTester : public ::testing::TestWithParam<bf_config_t> {
public:
  BeamformerTester();
  ~BeamformerTester(){};

  /**
  * @brief
  *
  * @param			in  TAFPT
  *							out BTF
  * 						weights BAFPT
  */
  void cpu_process(thrust::host_vector<cuComplex>& in,
    thrust::host_vector<float>& out,
    thrust::host_vector<cuComplex>& weights);

  /**
  * @brief
  *
  * @param			in  TAFPT
  *							out BTF
  * 						weights BAFPT
  */
  void cpu_process(thrust::host_vector<cuComplex>& in,
    thrust::host_vector<cuComplex>& out,
    thrust::host_vector<cuComplex>& weights);


  /**
	* @brief
	*
	* @param			in  TAFPT
	*							out BTF
	* 						weights BAFPT
	*/
  template<typename T, typename U>
  void gpu_process(const thrust::device_vector<T>& in,
    thrust::device_vector<U>& out,
    const thrust::device_vector<T>& weights);

  /**
	* @brief
	*
	* @param			in  TAFPT
	*							out BTF
	* 						weights BAFPT
	*/
  void compare(thrust::host_vector<float> cpu,
    thrust::device_vector<float> gpu, float tol=0.0001);



  void compare(thrust::host_vector<cuComplex> cpu,
    thrust::device_vector<cuComplex> gpu);


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
