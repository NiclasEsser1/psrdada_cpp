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

__global__
void unpack_codif_to_float32(uint64_t const* __restrict__ idata, float2* __restrict__ odata)
{
    uint64_t tmp;

    int time = threadIdx.x + blockIdx.x * blockDim.x; // Time
    int elem = threadIdx.y + blockIdx.y * blockDim.y;               // Elements
    int freq = threadIdx.z + blockIdx.z * blockDim.z;                           // Frequency
    int n_ch = blockDim.z * gridDim.z;

    int time_in = blockIdx.x * blockDim.x * gridDim.y * n_ch + threadIdx.x * n_ch;
    int freq_in = freq;
    int elem_in = elem * NSAMP_DF * n_ch ;

    int time_out = time * n_ch * gridDim.y * NPOL_SAMP;
    int freq_out = freq * gridDim.y * NPOL_SAMP;
    int elem_out = elem * NPOL_SAMP;

    int in_idx = time_in + freq_in + elem_in;
    int out_idx = time_out + freq_out + elem_out;

    tmp = swap64(idata[in_idx]);

    odata[out_idx].x = (float)((tmp & 0x000000000000ffffLL));
    odata[out_idx].y = (float)((tmp & 0x00000000ffff0000LL) >> 16);

    odata[out_idx + 1].x = (float)((tmp & 0x0000ffff00000000LL) >> 32);
    odata[out_idx + 1].y = (float)((tmp & 0xffff000000000000LL) >> 48);
}


}
}

#endif
