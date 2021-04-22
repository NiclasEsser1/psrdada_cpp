#ifdef UNPACKER_CUH

namespace psrdada_cpp {
namespace cryopaf{


__device__ __forceinline__ uint64_t swap64(uint64_t x)
{
    uint64_t result;
    uint2 t;
    asm("mov.b64 {%0,%1},%2; \n\t"
        : "=r"(t.x), "=r"(t.y) : "l"(x));
    t.x = __byte_perm(t.x, 0, 0x0123);
    t.y = __byte_perm(t.y, 0, 0x0123);
    asm("mov.b64 %0,{%1,%2}; \n\t"
        : "=l"(result) : "r"(t.y), "r"(t.x));
    return result;
}


template<typename T>__global__
void unpack_codif_to_fpte(uint64_t const* __restrict__ idata, T* __restrict__ odata)
{
    int time = threadIdx.x + blockIdx.x * blockDim.x; // Time
    int elem = threadIdx.y + blockIdx.y * blockDim.y; // Elements
    int freq = threadIdx.z + blockIdx.z * blockDim.z; // Frequency
    int chan = blockDim.z * gridDim.z;

    int time_in = blockIdx.x * blockDim.x * gridDim.y * chan + threadIdx.x * chan;
    int freq_in = freq;
    int elem_in = elem * NSAMP_DF * chan ;

    int freq_out = freq * NPOL_SAMP * gridDim.x * blockDim.x * gridDim.y;
    int time_out = time * gridDim.y;

    int in_idx = time_in + freq_in + elem_in;
    int out_idx_x = freq_out + time_out + elem;
    int out_idx_y = freq_out + gridDim.x * blockDim.x * gridDim.y + time_out + elem;

    uint64_t tmp = swap64(idata[in_idx]);

    odata[out_idx_x].x = static_cast<decltype(T::x)>((tmp & 0x000000000000ffffLL));
    odata[out_idx_x].y = static_cast<decltype(T::y)>((tmp & 0x00000000ffff0000LL) >> 16);

    odata[out_idx_y].x = static_cast<decltype(T::x)>((tmp & 0x0000ffff00000000LL) >> 32);
    odata[out_idx_y].y = static_cast<decltype(T::y)>((tmp & 0xffff000000000000LL) >> 48);
}

template<typename U, typename T>__global__
void unpack_spead_ttfep_to_fpte(U const* __restrict__ idata, T* __restrict__ odata)
{
    int time = threadIdx.x; // Time
    int elem = blockIdx.y; // Elements
    int freq = blockIdx.z; // Frequency
    int heap_idx = blockIdx.x;

    int in_idx = heap_idx * NSAMP_PER_HEAP * gridDim.z * gridDim.y * NPOL_SAMP // Outer time axis
      + time * gridDim.z * gridDim.y * NPOL_SAMP // Inner time axis
      + freq * gridDim.y * NPOL_SAMP // Frequency axis
      + elem * NPOL_SAMP; // Element axis

    int out_idx_x = freq * NPOL_SAMP * gridDim.x * NSAMP_PER_HEAP * gridDim.y // Frequency axis
      + (time + blockIdx.x * blockDim.x) * gridDim.y
      + elem;
    int out_idx_y = freq * NPOL_SAMP * gridDim.x * NSAMP_PER_HEAP * gridDim.y // Frequency axis
      + gridDim.x * NSAMP_PER_HEAP * gridDim.y
      + (time + blockIdx.x * blockDim.x) * gridDim.y
      + elem;

    odata[out_idx_x].x = static_cast<decltype(T::x)>(idata[in_idx].x);
    odata[out_idx_x].y = static_cast<decltype(T::y)>(idata[in_idx].y);

    odata[out_idx_y].x = static_cast<decltype(T::x)>(idata[in_idx + 1].x);
    odata[out_idx_y].y = static_cast<decltype(T::y)>(idata[in_idx + 1].y);
}

// ######################################################
// NOTE: Kernels above are deprecated and not longer used
// ######################################################
/*
template<typename T>__global__
void unpack_codif_to_tfep(uint64_t const* __restrict__ idata, T* __restrict__ odata)
{

    int time = threadIdx.x + blockIdx.x * blockDim.x; // Time
    int elem = threadIdx.y + blockIdx.y * blockDim.y; // Elements
    int freq = threadIdx.z + blockIdx.z * blockDim.z; // Frequency
    int chan = blockDim.z * gridDim.z;

    int time_in = blockIdx.x * blockDim.x * gridDim.y * chan + threadIdx.x * chan;
    int freq_in = freq;
    int elem_in = elem * NSAMP_DF * chan ;

    int time_out = time * chan * gridDim.y * NPOL_SAMP;
    int freq_out = freq * gridDim.y * NPOL_SAMP;
    int elem_out = elem * NPOL_SAMP;

    int in_idx = time_in + freq_in + elem_in;
    int out_idx = time_out + freq_out + elem_out;

    uint64_t tmp = swap64(idata[in_idx]);

    odata[out_idx].x = static_cast<decltype(T::x)>((tmp & 0x000000000000ffffLL));
    odata[out_idx].y = static_cast<decltype(T::y)>((tmp & 0x00000000ffff0000LL) >> 16);

    odata[out_idx + 1].x = static_cast<decltype(T::x)>((tmp & 0x0000ffff00000000LL) >> 32);
    odata[out_idx + 1].y = static_cast<decltype(T::y)>((tmp & 0xffff000000000000LL) >> 48);
}
*/
}
}

#endif
