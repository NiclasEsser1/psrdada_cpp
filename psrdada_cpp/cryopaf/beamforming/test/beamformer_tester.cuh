#ifndef BEAMFORMER_TEST_H_
#define BEAMFORMER_TEST_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <cuda_fp16.h>
#include <random>
#include <cmath>
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
  ~BeamformerTester(){};


  /**
  * @brief
  *
  * @param			in  TAFPT
  *							out BTF
  * 						weights BAFPT
  */
  template <typename T>
  void test(T type);



  /**
  * @brief
  *
  * @param			in  TAFPT
  *							out BTF
  * 						weights BAFPT
  */
  template <typename T>
  void cpu_process(
    thrust::host_vector<thrust::complex<T>>& in,
    thrust::host_vector<T>& out,
    thrust::host_vector<thrust::complex<T>>& weights);



  /**
  * @brief
  *
  * @param			in  TAFPT
  *							out BTF
  * 						weights BAFPT
  */
  template <typename T>
  void cpu_process(
    thrust::host_vector<thrust::complex<T>>& in,
    thrust::host_vector<thrust::complex<T>>& out,
    thrust::host_vector<thrust::complex<T>>& weights);



  /**
  * @brief
  *
  * @param			in  TAFPT
  *							out BTF
  * 						weights BAFPT
  */
  template<typename T>
  void gpu_process(
    thrust::device_vector<thrust::complex<T>>& in,
    thrust::device_vector<T>& out,
    thrust::device_vector<thrust::complex<T>>& weights);



  /**
	* @brief
	*
	* @param			in  TAFPT
	*							out BTF
	* 						weights BAFPT
	*/
  template<typename T>
  void gpu_process(
    thrust::device_vector<thrust::complex<T>>& in,
    thrust::device_vector<thrust::complex<T>>& out,
    thrust::device_vector<thrust::complex<T>>& weights);



  /**
	* @brief
	*
	* @param			in  TAFPT
	*							out BTF
	* 						weights BAFPT
	*/
  template <typename T>
  void compare(
    const thrust::host_vector<T> cpu,
    const thrust::device_vector<T> gpu,
    const float tol=0.0001);



  /**
	* @brief
	*
	* @param			in  TAFPT
	*							out BTF
	* 						weights BAFPT
	*/
  template<typename T>
  void compare(
    const thrust::host_vector<thrust::complex<T>>& cpu,
    const thrust::device_vector<thrust::complex<T>>& gpu);


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
