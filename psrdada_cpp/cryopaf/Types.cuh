/*
 * cryopaf_constants.hpp
 *
 *  Created on: Aug 3, 2020
 *      Author: Niclas Esser
 */

#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cuda.h>
#include <unordered_map>

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/double_device_buffer.cuh"
#include "psrdada_cpp/double_host_buffer.cuh"

namespace psrdada_cpp {
namespace cryopaf{

#define NTHREAD 1024
#define WARP_SIZE 32
#define WARPS (NTHREAD/WARP_SIZE)
#define SHARED_IDATA (NTHREAD)

// Enumeration for beamformer type
enum {POWER_BF, VOLTAGE_BF};
enum {SIMPLE_BF_TAFPT, BF_TFAP, BF_TFAP_TEX, CU_TENSOR, BF_TFAP_V2};

struct ComplexInt8{
  int8_t x;
  int8_t y;
};

struct bf_config_t{
   key_t in_key;
   key_t out_key;
   int device_id;
   std::string logname;
   std::size_t n_samples;
   std::size_t n_channel;
   std::size_t n_antenna;
   std::size_t n_pol;
   std::size_t n_beam;
   std::size_t interval;
   std::size_t bf_type;
   std::string kind;
   void print()
   {
     std::cout << "Beamform configuration" << std::endl;
     std::cout << "in_key: " << in_key << std::endl;
     std::cout << "out_key: " << out_key  << std::endl;
     std::cout << "device_id: " << device_id << std::endl;
     std::cout << "logname: " << logname << std::endl;
     std::cout << "n_samples: " << n_samples << std::endl;
     std::cout << "n_channel: " << n_channel << std::endl;
     std::cout << "n_antenna: " << n_antenna << std::endl;
     std::cout << "n_pol: " << n_pol << std::endl;
     std::cout << "n_beam: " << n_beam << std::endl;
     std::cout << "interval: " << interval << std::endl;
     std::cout << "bf_type: " << bf_type << std::endl;
     std::cout << "kind: " << kind << std::endl;
   }
};

template<class T>
class Tensor
{
protected:
  std::size_t _elements = 1;
  std::size_t _total_bytes;
  std::vector<int> _mode;
  std::vector<int64_t> _extent;
  std::vector<int64_t> _stride;

  cudaDataType_t _type;
  cutensorTensorDescriptor_t _desc;
  cutensorOperator_t _operator = CUTENSOR_OP_IDENTITY;
  uint32_t _alignment;
  bool complex = false;

  typedef T _dtype;
public:
  Tensor(std::vector<int> mode, std::unordered_map<int, int64_t> extent)
    : _mode(mode)
  {
    std::cout << std::endl;
    for(auto key : _mode)
    {
        _extent.push_back(extent[key]);
        _elements *= extent[key];
    }
    _total_bytes = _elements * sizeof(T);
    _stride.resize(_extent.size());
    for(int i = _stride.size() - 1; i >= 0; i--)
    {
        if(i == _stride.size() - 1)
        {
            _stride[i] = 1;
        }else{
            _stride[i] = _extent[i+1] * _stride[i+1];
        }
    }
  }
  ~Tensor()
  {
  }

  void synchronize(cudaStream_t stream)
  {
  	CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
  }
  /** Methods needed for cuTensor / cuTensorContraction **/
  void init_desc(cutensorHandle_t *handle)
  {
    parse_type();
    CUTENSOR_ERROR_CHECK(cutensorInitTensorDescriptor(handle, &_desc, _mode.size(), _extent.data(), _stride.data(), _type, _operator));
  }
  void init_alignment(cutensorHandle_t *handle, T* mem_ptr)
  {
    CUTENSOR_ERROR_CHECK(cutensorGetAlignmentRequirement(handle, (void*)mem_ptr, &_desc, &_alignment));
  }
  void parse_type()
  {
    if(std::is_same<T, __half>::value)
        _type = CUDA_R_16F;
    else if(std::is_same<T, __half2>::value)
        _type = CUDA_C_16F;
    else if(std::is_same<T, float>::value)
        _type = CUDA_R_32F;
    else if(std::is_same<T, float2>::value)
        _type = CUDA_C_32F;
    else if(std::is_same<T, double>::value)
        _type = CUDA_R_64F;
    else if(std::is_same<T, double2>::value)
        _type = CUDA_C_64F;
    else if(std::is_same<T, char>::value)
        _type = CUDA_R_8I;
    else if(std::is_same<T, char2>::value)
        _type = CUDA_C_8I;
    else if(std::is_same<T, unsigned char>::value)
        _type = CUDA_R_8U;
    // else if (std::is_same<T, unsigned char2>::value)
    //     _type = CUDA_C_8U;
    else if(std::is_same<T, int>::value)
        _type = CUDA_R_32I;
    else if(std::is_same<T, int2>::value)
        _type = CUDA_C_32I;
    else if(std::is_same<T, unsigned int>::value)
        _type = CUDA_R_32U;
    // else if (std::is_same<T, unsigned int>::value)
    //     _type = CUDA_C_32U;
    if(_type > 4 && _type != 8 && _type != 10 && _type != 12)
        complex = true;
  }
  /** GETTER **/
  std::vector<int>& mode(){return _mode;}
  std::vector<int64_t>& extent(){return _extent;}
  std::vector<int64_t>& stride(){return _stride;}
  std::size_t total_bytes(){return _total_bytes;}

  uint32_t alignment(){return _alignment;}
  cutensorTensorDescriptor_t& desc(){return _desc;}
};


template<class T>
class RawVoltage : public Tensor<T>, public DoubleBuffer<thrust::device_vector<T>>
{
public:
    RawVoltage(std::vector<int> mode, std::unordered_map<int, int64_t> extent)
      : Tensor<T>(mode, extent), DoubleBuffer<thrust::device_vector<T>>()
    {
      this->resize(this->_elements);
    }
    ~RawVoltage()
    {

    }
    void async_cpy(void* host_ptr, std::size_t size, cudaStream_t stream, cudaMemcpyKind kind = cudaMemcpyHostToDevice, std::size_t pos = 0)
    {
        CUDA_ERROR_CHECK(cudaMemcpyAsync(&this->a_ptr()[pos], host_ptr, size, kind, stream));
    }
    void sync_cpy(void* host_ptr, std::size_t size, cudaMemcpyKind kind = cudaMemcpyHostToDevice, std::size_t pos = 0)
    {
        if(kind == cudaMemcpyHostToDevice)
        {
          CUDA_ERROR_CHECK(cudaMemcpy(&this->a_ptr()[pos], host_ptr, size, kind));
        }
        else if (kind == cudaMemcpyDeviceToHost)
        {
          CUDA_ERROR_CHECK(cudaMemcpy(host_ptr, &this->a_ptr()[pos], size, kind));
        }
    }
};


template<class T>
class PowerBeam : public Tensor<T>, public DoubleBuffer<thrust::device_vector<T>>
{
private:

    DoublePinnedHostBuffer<char> _host_power;
    // DoubleBuffer<thrust::host_vector<T>>
public:
    PowerBeam(std::vector<int> mode, std::unordered_map<int, int64_t> extent)
      : Tensor<T>(mode, extent), DoubleBuffer<thrust::device_vector<T>>()
    {
      this->resize(this->_elements);
    }
    ~PowerBeam()
    {

    }
    void async_cpy(void* host_ptr, std::size_t size, cudaStream_t stream, cudaMemcpyKind kind = cudaMemcpyDeviceToHost, std::size_t pos = 0)
    {
        CUDA_ERROR_CHECK(cudaMemcpyAsync(&this->a_ptr()[pos], host_ptr, size, kind, stream));
    }
    void sync_cpy(void* host_ptr, std::size_t size, cudaMemcpyKind kind = cudaMemcpyDeviceToHost, std::size_t pos = 0)
    {
        CUDA_ERROR_CHECK(cudaMemcpy(host_ptr, &this->a_ptr()[pos], size, kind));
    }
};

template<class T>
class VoltageBeam : public Tensor<T>, public DoubleBuffer<thrust::device_vector<T>>
{
    DoublePinnedHostBuffer<char> _host_power;

public:
    VoltageBeam(std::vector<int> mode, std::unordered_map<int, int64_t> extent)
      : Tensor<T>(mode, extent), DoubleBuffer<thrust::device_vector<T>>()
    {
      this->resize(this->_elements);
    }
    ~VoltageBeam()
    {

    }
    void async_cpy(void* host_ptr, std::size_t size, cudaStream_t stream, cudaMemcpyKind kind = cudaMemcpyDeviceToHost, std::size_t pos = 0)
    {
        CUDA_ERROR_CHECK(cudaMemcpyAsync(&this->a_ptr()[pos], host_ptr, size, kind, stream));
    }
    void sync_cpy(void* host_ptr, std::size_t size, cudaMemcpyKind kind = cudaMemcpyDeviceToHost, std::size_t pos = 0)
    {
        CUDA_ERROR_CHECK(cudaMemcpy(host_ptr, &this->a_ptr()[pos], size, kind));
    }
};

template<class T>
class Weights : public Tensor<T>, public DoubleBuffer<thrust::device_vector<T>>
{
public:
    Weights(std::vector<int> mode, std::unordered_map<int, int64_t> extent)
      : Tensor<T>(mode, extent), DoubleBuffer<thrust::device_vector<T>>()
    {
      this->resize(this->_elements);
    }
    ~Weights()
    {

    }
    void async_cpy(void* host_ptr, std::size_t size, cudaStream_t stream, cudaMemcpyKind kind = cudaMemcpyHostToDevice, std::size_t pos = 0)
    {
        CUDA_ERROR_CHECK(cudaMemcpyAsync(&this->a_ptr()[pos], host_ptr, size, kind, stream));
    }
    void sync_cpy(void* host_ptr, std::size_t size, cudaMemcpyKind kind = cudaMemcpyHostToDevice, std::size_t pos = 0)
    {
        CUDA_ERROR_CHECK(cudaMemcpy(&this->a_ptr()[pos], host_ptr, size, kind));
    }
};



}
}



#endif /* TYPES_HPP_ */
