/*
* Beamformer.cuh
* Author: Niclas Esser <nesser@mpifr-bonn.mpg.de>
* Description:
*  This file consists of a single class (Beamformer<ComputeType>). An object of Beamformer
*  can be used o either perform a Stokes I detection or raw voltage beamforming
*  on a GPU.
*  Both beamforming kernels expect the same dataproduct (linear aligned in device memory)
*    Input:  F-P-T-E
*    Weight: F-P-B-E
*    Output: F-T-B-P (voltage beams)
*    Output: F-T-B   (Stokes I beams)
*/

#ifndef BEAMFORMER_CUH_
#define BEAMFORMER_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>

#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/multilog.hpp"

namespace psrdada_cpp{
namespace cryopaf{

// Constants for beamform kernels
#define NTHREAD 1024
#define TILE_SIZE 32
#define WARP_SIZE 32

/**
* @brief 	GPU kernel to perform Stokes I detection beamforming
*
* @detail Template type T has to be etiher T=float2 or T=__half2.
*         According to T, U has to be either U=float or U=__half
*
* @param	T* idata       pointer to input memory (format: F-P-T-E)
* @param	T* wdata       pointer to beam weight memory (format: F-P-B-E)
* @param	U* odata	     pointer to output memory type of U is equal to T::x (format: F-T-B)
* @param	int time	     Width of time dimension (T)
* @param	int elem       Number of elements (E)
* @param	int beam       Number of beams (B)
* @param	int integrate  Integration time, currently limited to 32 and a power of 2
*
* @TODO: Allow greater integration time
*/
template<typename T, typename U>__global__
void beamformer_power_fpte_fpbe_ftb(
  const T *idata,
  const T* wdata,
  U *odata,
  int time,
  int elem,
  int beam,
  int integrate);


/**
* @brief 	GPU kernel to perform raw voltage beamforming
*
* @detail Template type T has to be etiher T=float2 or T=__half2
*
* @param	T* idata       pointer to input memory (format: F-P-T-E)
* @param	T* wdata       pointer to beam weight memory (format: F-P-B-E)
* @param	T* odata	     pointer to output memory (format: F-T-B)
* @param	int time	     Width of time dimension (T)
* @param	int elem       Number of elements (E)
* @param	int beam       Number of beams (B)
*/
template<typename T>__global__
void beamformer_voltage_fpte_fpbe_fptb(
  const T *idata,
  const T* wdata,
  T *odata,
  int time,
  int elem,
  int beam);


template<class ComputeType>
class Beamformer{

// Internal typedefintions
private:
  typedef decltype(ComputeType::x) ResultType; // Just necessary for Stokes I beamformer

// Public functions
public:
  /**
  * @brief  constructs an object of Beamformer<ComputeType> (ComputeType=float2 or ComputeType=__half2)
  *
  * @param	cudaStream_t& stream      Object of cudaStream_t to allow parallel copy + processing (has to be created and destroyed elsewhere)
  * @param  std::size_t sample        Number of samples to process in on kernel launch (no restrictions)
  * @param  std::size_t channel       Number of channels to process in on kernel launch (no restrictions)
  * @param  std::size_t element       Number of elements to process in on kernel launch (no restrictions)
  * @param  std::size_t beam          Number of beams to process in on kernel launch (no restrictions)
  * @param  std::size_t integration   Samples to be integrated, has to be power of 2 and smaller 32
  */
  Beamformer(
    cudaStream_t& stream,
    std::size_t sample,
    std::size_t channel,
    std::size_t element,
    std::size_t beam,
    std::size_t integration = 1);

  /**
  * @brief  deconstructs an object of Beamformer<ComputeType> (ComputeType=float2 or ComputeType=__half2)
  */
  ~Beamformer();

  /**
  * @brief 	Launches voltage beamforming GPU kernel
  *
  * @param	ComputeType* input       pointer to input memory (format: F-P-T-E)
  * @param	ComputeType* weights     pointer to beam weight memory (format: F-P-B-E)
  * @param	ComputeType* output	     pointer to output memory (format: F-T-B-P)
  */
  void process(
    const ComputeType* input,
    const ComputeType* weights,
    ComputeType* output);

  /**
  * @brief 	Launches Stokes I beamforming GPU kernel
  *
  * @param	ComputeType* input       pointer to input memory (format: F-P-T-E)
  * @param	ComputeType* weights     pointer to beam weight memory (format: F-P-B-E)
  * @param	ResultType* output	     pointer to output memory (format: F-T-B)
  */
  void process(
    const ComputeType* input,
    const ComputeType* weights,
    ResultType* output);

  /**
	* @brief	Prints the block and grid layout of a kernel (used for debugging purposes)
	*/
	void print_layout();

// Private attributes
private:
  cudaStream_t& _stream;
  dim3 grid;
  dim3 block;
  std::size_t _sample;
  std::size_t _channel;
  std::size_t _element;
  std::size_t _beam;
  std::size_t _integration;
};


} // namespace cryopaf
} // namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/details/utils.cu"
#include "psrdada_cpp/cryopaf/details/BeamformerKernels.cu"
#include "psrdada_cpp/cryopaf/src/Beamformer.cu"

#endif /* BEAMFORMER_CUH_ */
