#ifndef OUTPUTTYPES_H_
#define OUTPUTTYPES_H_

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cuda.h>

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/double_device_buffer.cuh"
#include "psrdada_cpp/double_host_buffer.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace io{

template<typename T>
struct OutputType
{
    DoubleDeviceBuffer<T> d_buf;
    DoublePinnedHostBuffer<T> h_buf;
    dev2host(cudaStream_t &out_stream);
    void swap();
    void resize();
}

template<typename T>
class PowerOutputType : public OutputType<T>
{
public:
    PowerOutputType(std::size_t size): _size(size){};
private:
    std::size_t _size; // size in total amount of samples
    psrdada_cpp::effelsberg::edd::DadaBufferLayout _dada_layout;
}

}
}
}
