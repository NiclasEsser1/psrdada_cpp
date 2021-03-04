#ifndef BEAMFORMER_TESTER_H_
#define BEAMFORMER_TESTER_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_fp16.h>
#include <random>
#include <cmath>
#include <gtest/gtest.h>

#include "psrdada_cpp/cryopaf/Beamformer.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace test{

struct BeamformTestConfig{
   int device_id;
   std::size_t n_samples;
   std::size_t n_channel;
   std::size_t n_elements;
   std::size_t n_pol;
   std::size_t n_beam;
   std::size_t integration;
   void print()
   {
     std::cout << "Test configuration" << std::endl;
     std::cout << "device_id: " << device_id << std::endl;
     std::cout << "n_samples: " << n_samples << std::endl;
     std::cout << "n_channel: " << n_channel << std::endl;
     std::cout << "n_elements: " << n_elements << std::endl;
     std::cout << "n_pol: " << n_pol << std::endl;
     std::cout << "n_beam: " << n_beam << std::endl;
     std::cout << "integration: " << integration << std::endl;
   }
};

class BeamformTester : public ::testing::TestWithParam<BeamformTestConfig> {
public:
  BeamformTester();
  ~BeamformTester();


  /**
  * @brief
  *
  * @param
  */
  template <typename T, typename U>
  void test(float tol=0.0001);


  /**
  * @brief
  *
  * @param
  */
  template<typename T, typename U>
  void cpu_process_power(
    thrust::host_vector<T>& in,
    thrust::host_vector<T>& weights,
    thrust::host_vector<U>& out);

  /**
  * @brief
  *
  * @param
  */
  template<typename T>
  void cpu_process_voltage(
    thrust::host_vector<T>& in,
    thrust::host_vector<T>& weights,
    thrust::host_vector<T>& out);


  /**
  * @brief
  *
  * @param
  */
  template<typename T, typename U>
  void gpu_process(
    thrust::host_vector<T>& in,
    thrust::host_vector<T>& weights,
    thrust::host_vector<U>& out);


  /**
	* @brief
	*
	* @param
	*/
  template <typename T>
  void compare_power(
    const thrust::host_vector<T> cpu,
    const thrust::host_vector<T> gpu,
    const float tol);

  /**
	* @brief
	*
	* @param
	*/
  template <typename T>
  void compare_voltage(
    const thrust::host_vector<T> cpu,
    const thrust::host_vector<T> gpu,
    const float tol);


protected:
  void SetUp() override;
  void TearDown() override;

private:
  BeamformTestConfig conf;
  cudaStream_t _stream;

};

} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace test

#include "psrdada_cpp/cryopaf/test/src/BeamformTester.cu"

#endif //POWERBEAMFORMER_TESTER_H_
