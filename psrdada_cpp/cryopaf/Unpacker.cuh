#ifndef UNPACKER_CUH
#define UNPACKER_CUH

#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cryopaf/Types.cuh"
#include <thrust/device_vector.h>

#define NCHAN_CHK             7
#define NSAMP_DF              128
#define NPOL_SAMP             2

namespace psrdada_cpp {
namespace cryopaf {

__global__
void unpack_codif_to_float32(uint64_t const* __restrict__ idata, float2* __restrict__ odata);

template<class HandlerType, class InputType, class OutputType>
class Unpacker
{
public:

    Unpacker(bf_config_t& config, MultiLog &logger, HandlerType &handler);
    ~Unpacker();
    Unpacker(Unpacker const&) = delete;

    void init(RawBytes& header_block);
    bool operator()(RawBytes& dada_block);
    void process();

    template<class T, class Type>
    void sync_copy(thrust::host_vector<T>& vec, cudaMemcpyKind kind);

    void print_layout();

private:
    HandlerType& _handler;
    MultiLog &log;
    bf_config_t& conf;

    InputType *_input_buffer;
    OutputType *_output_buffer;

    cudaStream_t _stream;
    dim3 _grid_layout;
    dim3 _block_layout;
};

} //namespace cryopaf
} //namespace psrdada_cpp

#include "psrdada_cpp/cryopaf/details/UnpackerKernels.cu"
#include "psrdada_cpp/cryopaf/src/Unpacker.cu"

#endif // UNPACKER_CUH
