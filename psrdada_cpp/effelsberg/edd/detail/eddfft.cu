#include "psrdada_cpp/effelsberg/edd/eddfft.cuh"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

template <class HandlerType>
SimpleFFTSpectrometer<HandlerType>::SimpleFFTSpectrometer(
    int fft_length,
    int naccumulate,
    int nbits,
    HandlerType& handler)
    : _fft_length(fft_length)
    , _naccumulate(naccumulate)
    , _nbits(nbits)
    , _handler(handler)
    , _first_block(true)
    , _nsamps(0)
    , _fft_plan(NULL)
{

}

template <class HandlerType>
SimpleFFTSpectrometer<HandlerType>::~SimpleFFTSpectrometer()
{
    if (!_first_block)
        cufftDestroy(_fft_plan);
}

template <class HandlerType>
void SimpleFFTSpectrometer<HandlerType>::init(RawBytes& block)
{
}

template <class HandlerType>
bool SimpleFFTSpectrometer<HandlerType>::operator()(RawBytes& block)
{

    int nsamps_in_block = 8 * block.used_bytes() / _nbits;
    int nchans = _fft_length / 2 + 1;

    if (_first_block)
    {
        _nsamps = nsamps_in_block;
        int n64bit_words = 3 * _nsamps / 16;
        if (_nsamps % _fft_length != 0)
        {
            throw std::runtime_error("Number of samples is not multiple of FFT size");
        }
        int batch = _nsamps/_fft_length;

        // Only do these things once
        int n[] = {_fft_length};
        CUFFT_ERROR_CHECK(cufftPlanMany(&_fft_plan, 1, n, NULL, 1, _fft_length,
            NULL, 1, _fft_length/2 + 1, CUFFT_R2C, batch));

        _edd_raw.resize(n64bit_words);
        _edd_unpacked.resize(_nsamps);
        _channelised.resize(nchans * batch);
        _detected.resize(nchans * batch / _naccumulate);
        _first_block = false;
    }

    if (_nsamps != nsamps_in_block)
    {
        throw std::runtime_error("Received incomplete block");
    }

    uint64_t* _edd_raw_ptr = thrust::raw_pointer_cast(_edd_raw.data());
    float* _edd_unpacked_ptr = thrust::raw_pointer_cast(_edd_unpacked.data());

    //Copy DADA buffer to GPU
    thrust::copy(block.ptr(), block.ptr()+block.used_bytes(), (char*) _edd_raw_ptr);

    if (_nbits == 12)
    {
        kernels::unpack_edd_12bit_to_float32<<< 64, 1024>>>(_edd_raw_ptr, _edd_unpacked_ptr, _edd_raw.size());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }
    else if (_nbits == 8)
    {
        throw std::runtime_error("Only 12-bit mode supported");
    }
    else
    {
        throw std::runtime_error("Only 12-bit mode supported");
    }

    cufftComplex* _channelised_ptr = thrust::raw_pointer_cast(_channelised.data());
    CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan, (cufftReal*)_edd_unpacked_ptr, _channelised_ptr));

    //thrust::copy(_edd_unpacked.begin(), _edd_unpacked.end(), block.ptr());
    //_handler(block);
}

} //edd
} //effelsberg
} //psrdada_cpp

