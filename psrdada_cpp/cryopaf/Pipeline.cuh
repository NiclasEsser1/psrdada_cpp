/*
* Pipeline.cuh
* Author: Niclas Esser <nesser@mpifr-bonn.mpg.de>
* Description:
*  This files consists of a single class (Pipeline<HandlerType, ComputeType, ResultType>)
*  and a configuration structure (PipelineConfig).
*  An object of Pipeline is used to access data from psrdada buffers, unpacks them,
*  performs beamforming and writes the results back to another psrdada buffer.
*  TODO:
*    - We need a packer
*    - We need a monitoring interface
*/
#ifndef PIPELINE_CUH_
#define PIPELINE_CUH_

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cuda.h>

#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/raw_bytes.hpp"

#include "psrdada_cpp/cryopaf/Unpacker.cuh"
#include "psrdada_cpp/cryopaf/Beamformer.cuh"
#include "psrdada_cpp/cryopaf/BufferTypes.cuh"

namespace psrdada_cpp{
namespace cryopaf{

struct PipelineConfig{
   key_t in_key;
   key_t out_key;
   int device_id;
   std::string logname;
   std::size_t n_samples;
   std::size_t n_channel;
   std::size_t n_elements;
   std::size_t n_beam;
   std::size_t integration;
   std::string input_type;
   std::string mode;
   std::string protocol;
   const std::size_t n_pol = 2;
   void print()
   {
     std::cout << "Pipeline configuration" << std::endl;
     std::cout << "in_key: " << in_key << std::endl;
     std::cout << "out_key: " << out_key  << std::endl;
     std::cout << "device_id: " << device_id << std::endl;
     std::cout << "logname: " << logname << std::endl;
     std::cout << "n_samples: " << n_samples << std::endl;
     std::cout << "n_channel: " << n_channel << std::endl;
     std::cout << "input_type: " << input_type << std::endl;
     std::cout << "n_elements: " << n_elements << std::endl;
     std::cout << "n_pol: " << n_pol << std::endl;
     std::cout << "n_beam: " << n_beam << std::endl;
     std::cout << "integration: " << integration << std::endl;
     std::cout << "mode: " << mode << std::endl;
   }
};


template<class HandlerType, class ComputeType, class ResultType>
class Pipeline{
// Internal type defintions
private:
  typedef RawVoltage<char> RawInputType;    // Type for received raw input data (voltage)
	typedef RawVoltage<ComputeType> InputType;// Type for unpacked raw input data (voltage)
  typedef Weights<ComputeType> WeightType;  // Type for beam weights
  typedef BeamOutput<ResultType> OutputType;// Type for beamfored output data
public:


	/**
	* @brief	Constructs an object of Pipeline
	*
  * @param	PipelineConfig conf  Pipeline configuration containing all necessary parameters (declaration can be found in Types.cuh)
  * @param	MultiLog log         Logging instance
	* @param	HandlerType	handler  Object for handling output data
	*
	* @detail	Initializes the pipeline enviroment including device memory and processor objects.
	*/
	Pipeline(PipelineConfig& conf, MultiLog &log, HandlerType &handler);


	/**
	* @brief	Deconstructs an object of Pipeline
  *
	* @detail	Destroys all objects and allocated memory
	*/
	~Pipeline();


  /**
   * @brief      Initialise the pipeline with a DADA header block
   *
   * @param      header  A RawBytes object wrapping the DADA header block
   */
	void init(RawBytes &header_block);


  /**
   * @brief      Process the data in a DADA data buffer
   *
   * @param      data  A RawBytes object wrapping the DADA data block
   */
	bool operator()(RawBytes &dada_block);
// Internal attributes
private:

	HandlerType &_handler;
	MultiLog &_log;
	PipelineConfig& _conf;
  // Processors
  Unpacker<ComputeType>* unpacker = nullptr; // Object to unpack and transpose received input data on GPU to an expected format
	Beamformer<ComputeType>* beamformer = nullptr; // Object to perform beamforming on GPU
  // Buffers
	RawInputType *_raw_input_buffer = nullptr; // Received input buffer
	InputType *_input_buffer = nullptr;  // Unpacked and transposed input buffer
	WeightType *_weight_buffer = nullptr; // Beam weights, updated through shared memory
  OutputType *_output_buffer = nullptr; // Output buffer containing processed beams

  std::size_t _call_cnt = 0; // Internal dada block counter

	cudaStream_t _h2d_stream;  // Host to device cuda stream (used for async copys)
	cudaStream_t _prc_stream;  // Processing stream
	cudaStream_t _d2h_stream;  // Device to host cuda stream (used for async copys)
#ifdef DEBUG
  // Time measurement variables (debugging only)
	cudaEvent_t start, stop;
	float ms;
#endif
};


} // namespace cryopaf
} // namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/src/Pipeline.cu"

#endif /* POWERBEAMFORMER_CUH_ */
