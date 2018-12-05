#include "psrdada_cpp/effelsberg/edd/eddfft.cuh"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <cuda.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

template <class HandlerType>
SimpleFFTSpectrometer<HandlerType>::SimpleFFTSpectrometer(
    int nsamps_per_block,
    int fft_length,
    int naccumulate,
    int nbits,
    HandlerType& handler)
    : _nsamps(nsamps_per_block)
    , _fft_length(fft_length)
    , _naccumulate(naccumulate)
    , _nbits(nbits)
    , _handler(handler)
    , _fft_plan(0)
    , _pass(0)
{
    BOOST_LOG_TRIVIAL(debug)
    << "Creating new SimpleFFTSpectrometer instance with parameters: \n"
    << "fft_length = " << _fft_length << "\n"
    << "naccumulate = " << _naccumulate;

    if (_nsamps % _fft_length != 0)
    {
        throw std::runtime_error("Number of samples is not multiple of FFT size");
    }

    int n64bit_words;
    if (_nbits == 12)
    {
        n64bit_words = 3 * _nsamps / 16;
    }
    else if (_nbits == 8)
    {
        n64bit_words = _nsamps / 8;
    }
    else
    {
        throw std::runtime_error("Only 12- and 8-bit modes are supported");
    }

    cudaStreamCreate(&_h2d_stream);
    cudaStreamCreate(&_proc_stream);
    cudaStreamCreate(&_d2h_stream);

    _nchans = _fft_length / 2 + 1;
    int batch = _nsamps/_fft_length;

    BOOST_LOG_TRIVIAL(debug) << "Generating FFT plan";
    int n[] = {_fft_length};
    CUFFT_ERROR_CHECK(cufftPlanMany(&_fft_plan, 1, n, NULL, 1, _fft_length,
        NULL, 1, _fft_length/2 + 1, CUFFT_R2C, batch));
    cufftSetStream(_fft_plan, _proc_stream);

    BOOST_LOG_TRIVIAL(debug) << "Allocating memory";
    _edd_raw.resize(n64bit_words);
    _edd_unpacked.resize(_nsamps);
    _channelised.resize(_nchans * batch);
    _detected.resize(_nchans * batch / _naccumulate);
    _detected_host.resize(_nchans * batch / _naccumulate);
}

template <class HandlerType>
SimpleFFTSpectrometer<HandlerType>::~SimpleFFTSpectrometer()
{
    BOOST_LOG_TRIVIAL(debug) << "Destroying SimpleFFTSpectrometer";
    if (!_fft_plan)
        cufftDestroy(_fft_plan);
    cudaStreamDestroy(_h2d_stream);
    cudaStreamDestroy(_proc_stream);
    cudaStreamDestroy(_d2h_stream);
}

template <class HandlerType>
void SimpleFFTSpectrometer<HandlerType>::init(RawBytes& block)
{
    BOOST_LOG_TRIVIAL(debug) << "SimpleFFTSpectrometer init called";
    _handler.init(block);
}

template <class HandlerType>
void SimpleFFTSpectrometer<HandlerType>::process(
    thrust::device_vector<uint64_t>* digitiser_raw,
    thrust::device_vector<char>* detected)
{

    uint64_t* digitiser_raw_ptr = thrust::raw_pointer_cast(digitiser_raw->data());
    float* digitiser_unpacked_ptr = thrust::raw_pointer_cast(_edd_unpacked.data());
    cufftComplex* channelised_ptr = thrust::raw_pointer_cast(_channelised.data());
    char* detected_ptr = thrust::raw_pointer_cast(detected->data());

    if (_nbits == 12)
    {
        BOOST_LOG_TRIVIAL(debug) << "Unpacking 12-bit data";
        int nblocks = digitiser_raw->size() / NTHREADS_UNPACK;
        kernels::unpack_edd_12bit_to_float32<<< nblocks, NTHREADS_UNPACK, 0, _proc_stream>>>(
            digitiser_raw_ptr, digitiser_unpacked_ptr, digitiser_raw->size());
    }
    else if (_nbits == 8)
    {
        BOOST_LOG_TRIVIAL(debug) << "Unpacking 8-bit data";
        int nblocks = digitiser_raw->size() / NTHREADS_UNPACK;
        kernels::unpack_edd_8bit_to_float32<<< nblocks, NTHREADS_UNPACK, 0, _proc_stream>>>(
            digitiser_raw_ptr, digitiser_unpacked_ptr, digitiser_raw->size());
    }

    BOOST_LOG_TRIVIAL(debug) << "Performing FFT";
    CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan, (cufftReal*) digitiser_unpacked_ptr, channelised_ptr));

    BOOST_LOG_TRIVIAL(debug) << "Detecting and accumulating";
    kernels::detect_and_accumulate<<<1024, 1024, 0, _proc_stream>>>(channelised_ptr, detected_ptr,
        _nchans, _nsamps/_fft_length, _naccumulate, 1.0f, 0.0f);
}

template <class HandlerType>
bool SimpleFFTSpectrometer<HandlerType>::operator()(RawBytes& block)
{
    BOOST_LOG_TRIVIAL(debug) << "SimpleFFTSpectrometer operator() called";
    int nsamps_in_block = 8 * block.used_bytes() / _nbits;
    if (_nsamps != nsamps_in_block)
    {
        throw std::runtime_error("Received expected number of samples");
    }
    BOOST_LOG_TRIVIAL(debug) << nsamps_in_block << " samples in RawBytes block";

    // Synchronize all streams
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
    _detected.swap();

    CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_stream));
    _detected_host.swap();

    // Start host to device copy
    cudaMemcpyAsync((char*) thrust::raw_pointer_cast(_edd_raw.a()->data()),
        block.ptr(), block.used_bytes(), cudaMemcpyHostToDevice, _h2d_stream);

    // Guaranteed that the previous copy is completed here
    process(_edd_raw.b(), _detected.a());

    cudaMemcpyAsync((char*) thrust::raw_pointer_cast(_detected_host.a()->data()),
        (char*) thrust::raw_pointer_cast(_detected.b()->data()),
        _detected.b()->size() * sizeof(char),
        cudaMemcpyDeviceToHost, _d2h_stream);

    //Wrap _detected_host_previous in a RawBytes object here;
    RawBytes bytes((char*) thrust::raw_pointer_cast(_detected_host.b()->data()),
        _detected_host.b()->size() * sizeof(char),
        _detected_host.b()->size() * sizeof(char));
    BOOST_LOG_TRIVIAL(debug) << "Calling handler";

    CUDA_ERROR_CHECK(cudaStreamSynchronize(_h2d_stream));
    _edd_raw.swap();

    // Due to the double buffering the data the output data is only
    // valid by the third pass through. Until that time the code
    // should return the continue flag.
    if (_pass < 2)
    {
        ++_pass;
        return false;
    }
    return _handler(bytes);
}

} //edd
} //effelsberg
} //psrdada_cpp

