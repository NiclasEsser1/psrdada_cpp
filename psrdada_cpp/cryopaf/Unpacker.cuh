#ifndef UNPACKER_CUH
#define UNPACKER_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>

#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/multilog.hpp"

#define NCHAN_CHK             7
#define NSAMP_DF              128
#define NPOL_SAMP             2

namespace psrdada_cpp {
namespace cryopaf {

template<typename T>__global__
void unpack_codif_to_fpte(uint64_t const* __restrict__ idata, T* __restrict__ odata);

template<typename T>
class Unpacker
{
public:

    Unpacker(cudaStream_t& stream,
      std::size_t nsamples,
      std::size_t nchannels,
      std::size_t nelements,
      std::string protocol);
    ~Unpacker();
    Unpacker(Unpacker const&) = delete;

    void unpack(char* input, T* output);

    void print_layout();

    int sample_size(){return _sample_size;}
private:
    cudaStream_t& _stream;
    const int _sample_size = 4; // 2x uint16 in Byte
    std::string _protocol;
    dim3 grid;
    dim3 block;
};

} //namespace cryopaf
} //namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/details/UnpackerKernels.cu"
#include "psrdada_cpp/cryopaf/src/Unpacker.cu"

#endif // UNPACKER_CUH
