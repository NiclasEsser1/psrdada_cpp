#include "psrdada_cpp/effelsberg/edd/GatedSpectrometer.cuh"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include <cuda.h>
#include <cuda_profiler_api.h>

#include <iostream>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {


__global__ void gating(float *G0, float *G1, const int64_t *sideChannelData,
                       size_t N, size_t heapSize, size_t bitpos,
                       size_t noOfSideChannels, size_t selectedSideChannel) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; (i < N);
       i += blockDim.x * gridDim.x) {
    const float w = G0[i];
    const int64_t sideChannelItem =
        sideChannelData[((i / heapSize) * (noOfSideChannels)) +
                        selectedSideChannel]; // Probably not optimal access as
                                              // same data is copied for several
                                              // threads, but maybe efficiently
                                              // handled by cache?

    const int bit_set = TEST_BIT(sideChannelItem, bitpos);
    G1[i] = w * bit_set;
    G0[i] = w * (!bit_set);
  }
}


__global__ void countBitSet(const int64_t *sideChannelData, size_t N, size_t
    bitpos, size_t noOfSideChannels, size_t selectedSideChannel, unsigned int
    *nBitsSet)
{
  // really not optimized reduction, but here only trivial array sizes.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int x[256];

  if (i == 0)
    nBitsSet[0] = 0;

  if (i * noOfSideChannels + selectedSideChannel < N)
    x[threadIdx.x] = TEST_BIT(sideChannelData[i * noOfSideChannels + selectedSideChannel], bitpos);
  else
    x[threadIdx.x] = 0;
  __syncthreads();

  for(int s = blockDim.x / 2; s > 0; s = s / 2)
  {
    if (threadIdx.x < s)
      x[threadIdx.x] += x[threadIdx.x + s];
    __syncthreads();
  }

  if(threadIdx.x == 0)
   atomicAdd(nBitsSet, x[threadIdx.x]);
}


template <class HandlerType>
GatedSpectrometer<HandlerType>::GatedSpectrometer(
    std::size_t buffer_bytes, std::size_t nSideChannels,
    std::size_t selectedSideChannel, std::size_t selectedBit,
    std::size_t speadHeapSize, std::size_t fft_length, std::size_t naccumulate,
    std::size_t nbits, float input_level, float output_level,
    HandlerType &handler)
    : _buffer_bytes(buffer_bytes), _nSideChannels(nSideChannels),
      _selectedSideChannel(selectedSideChannel), _selectedBit(selectedBit),
      _speadHeapSize(speadHeapSize), _fft_length(fft_length),
      _naccumulate(naccumulate), _nbits(nbits), _handler(handler), _fft_plan(0),
      _call_count(0) {
  assert(((_nbits == 12) || (_nbits == 8)));
  assert(_naccumulate > 0); // Sanity check
  BOOST_LOG_TRIVIAL(info)
      << "Creating new GatedSpectrometer instance with parameters: \n"
      << "  fft_length = " << _fft_length << "\n"
      << "  naccumulate = " << _naccumulate << "\n"
      << "  nSideChannels = " << _nSideChannels << "\n"
      << "  speadHeapSize = " << _speadHeapSize << " byte\n"
      << "  selectedSideChannel = " << _selectedSideChannel
      << "  selectedBit = " << _selectedBit;

  _sideChannelSize = nSideChannels * sizeof(int64_t);
  _totalHeapSize = _speadHeapSize + _sideChannelSize;
  _nHeaps = buffer_bytes / _totalHeapSize;
  _gapSize = (buffer_bytes - _nHeaps * _totalHeapSize);
  _dataBlockBytes = _nHeaps * _speadHeapSize;
  assert((nSideChannels == 0) ||
         (selectedSideChannel <
          nSideChannels));  // Sanity check of side channel value
  assert(selectedBit < 64); // Sanity check of selected bit
  BOOST_LOG_TRIVIAL(info) << "Resulting memory configuration: \n"
                           << "  totalSizeOfHeap: " << _totalHeapSize
                           << " byte\n"
                           << "  number of heaps per buffer: " << _nHeaps
                           << "\n"
                           << "  resulting gap: " << _gapSize << " byte\n"
                           << "  datablock size in buffer: " << _dataBlockBytes
                           << " byte\n";

  std::size_t nsamps_per_buffer = _dataBlockBytes * 8 / nbits;
  std::size_t n64bit_words = _dataBlockBytes / sizeof(uint64_t);
  _nchans = _fft_length / 2 + 1;
  int batch = nsamps_per_buffer / _fft_length;
  float dof = 2 * _naccumulate;
  float scale =
      std::pow(input_level * std::sqrt(static_cast<float>(_nchans)), 2);
  float offset = scale * dof;
  float scaling = scale * std::sqrt(2 * dof) / output_level;
  BOOST_LOG_TRIVIAL(debug)
      << "Correction factors for 8-bit conversion: offset = " << offset
      << ", scaling = " << scaling;

  BOOST_LOG_TRIVIAL(debug) << "Generating FFT plan";
  int n[] = {static_cast<int>(_fft_length)};
  CUFFT_ERROR_CHECK(cufftPlanMany(&_fft_plan, 1, n, NULL, 1, _fft_length, NULL,
                                  1, _nchans, CUFFT_R2C, batch));
  cufftSetStream(_fft_plan, _proc_stream);

  BOOST_LOG_TRIVIAL(debug) << "Allocating memory";
  _raw_voltage_db.resize(n64bit_words);
  _sideChannelData_db.resize(_sideChannelSize * _nHeaps);
  BOOST_LOG_TRIVIAL(debug) << "  Input voltages size (in 64-bit words): "
                           << _raw_voltage_db.size();
  _unpacked_voltage_G0.resize(nsamps_per_buffer);
  _unpacked_voltage_G1.resize(nsamps_per_buffer);
  BOOST_LOG_TRIVIAL(debug) << "  Unpacked voltages size (in samples): "
                           << _unpacked_voltage_G0.size();
  _channelised_voltage.resize(_nchans * batch);
  BOOST_LOG_TRIVIAL(debug) << "  Channelised voltages size: "
                           << _channelised_voltage.size();
  _power_db_G0.resize(_nchans * batch / _naccumulate);
  _power_db_G1.resize(_nchans * batch / _naccumulate);
  BOOST_LOG_TRIVIAL(debug) << "  Powers size: " << _power_db_G0.size() << ", "
                           << _power_db_G1.size();
  // on the host both power are stored in the same data buffer
  _host_power_db.resize( _power_db_G0.size() + _power_db_G1 .size());
  _noOfBitSetsInSideChannel.resize(1);

  CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_proc_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_stream));
  CUFFT_ERROR_CHECK(cufftSetStream(_fft_plan, _proc_stream));

  // Create and record process status events to signal that processing chain is clear
  CUDA_ERROR_CHECK(cudaEventCreateWithFlags(&_procA, cudaEventDisableTiming));
  CUDA_ERROR_CHECK(cudaEventRecord(_procA, _proc_stream));
  CUDA_ERROR_CHECK(cudaEventCreateWithFlags(&_procB, cudaEventDisableTiming));
  CUDA_ERROR_CHECK(cudaEventRecord(_procB, _proc_stream));

  _unpacker.reset(new Unpacker(_proc_stream));
  _detector.reset(new DetectorAccumulator(_nchans, _naccumulate, scaling,
                                          offset, _proc_stream));
} // constructor


template <class HandlerType>
GatedSpectrometer<HandlerType>::~GatedSpectrometer() {
  BOOST_LOG_TRIVIAL(debug) << "Destroying GatedSpectrometer";
  if (!_fft_plan)
    cufftDestroy(_fft_plan);
  cudaStreamDestroy(_h2d_stream);
  cudaStreamDestroy(_proc_stream);
  cudaStreamDestroy(_d2h_stream);
  cudaEventDestroy(_procA);
  cudaEventDestroy(_procB);
}


template <class HandlerType>
void GatedSpectrometer<HandlerType>::init(RawBytes &block) {
  BOOST_LOG_TRIVIAL(debug) << "GatedSpectrometer init called";
  _handler.init(block);
}


template <class HandlerType>
void GatedSpectrometer<HandlerType>::process(
    thrust::device_vector<RawVoltageType> const &digitiser_raw,
    thrust::device_vector<RawVoltageType> const &sideChannelData,
    thrust::device_vector<IntegratedPowerType> &detected_G0,
    thrust::device_vector<IntegratedPowerType> &detected_G1, thrust::device_vector<unsigned int> &noOfBitSet) {
  BOOST_LOG_TRIVIAL(debug) << "Unpacking raw voltages";
  switch (_nbits) {
  case 8:
    _unpacker->unpack<8>(digitiser_raw, _unpacked_voltage_G0);
    break;
  case 12:
    _unpacker->unpack<12>(digitiser_raw, _unpacked_voltage_G0);
    break;
  default:
    throw std::runtime_error("Unsupported number of bits");
  }
  // raw voltage buffer is free again
  CUDA_ERROR_CHECK(cudaEventRecord(_procB, _proc_stream));

  BOOST_LOG_TRIVIAL(debug) << "Perform gating";
  const int64_t *sideCD =
      (int64_t *)(thrust::raw_pointer_cast(sideChannelData.data()));
  gating<<<1024, 1024, 0, _proc_stream>>>(
      thrust::raw_pointer_cast(_unpacked_voltage_G0.data()),
      thrust::raw_pointer_cast(_unpacked_voltage_G1.data()), sideCD,
      _unpacked_voltage_G0.size(), _speadHeapSize, _selectedBit, _nSideChannels,
      _selectedSideChannel);

  countBitSet<<<(sideChannelData.size()+255)/256, 256, 0,
    _proc_stream>>>(sideCD, sideChannelData.size(), _selectedBit,
        _nSideChannels, _selectedBit,
        thrust::raw_pointer_cast(noOfBitSet.data()));

  BOOST_LOG_TRIVIAL(debug) << "Performing FFT 1";
  UnpackedVoltageType *_unpacked_voltage_ptr =
      thrust::raw_pointer_cast(_unpacked_voltage_G0.data());
  ChannelisedVoltageType *_channelised_voltage_ptr =
      thrust::raw_pointer_cast(_channelised_voltage.data());
  CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan, (cufftReal *)_unpacked_voltage_ptr,
                                 (cufftComplex *)_channelised_voltage_ptr));
  _detector->detect(_channelised_voltage, detected_G0);

  BOOST_LOG_TRIVIAL(debug) << "Performing FFT 2";
  _unpacked_voltage_ptr = thrust::raw_pointer_cast(_unpacked_voltage_G1.data());
  CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan, (cufftReal *)_unpacked_voltage_ptr,
                                 (cufftComplex *)_channelised_voltage_ptr));

//  CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
  _detector->detect(_channelised_voltage, detected_G1);
  BOOST_LOG_TRIVIAL(debug) << "Exit processing";
} // process


template <class HandlerType>
bool GatedSpectrometer<HandlerType>::operator()(RawBytes &block) {
  ++_call_count;
  BOOST_LOG_TRIVIAL(debug) << "GatedSpectrometer operator() called (count = "
                           << _call_count << ")";
  if (block.used_bytes() != _buffer_bytes) { /* Unexpected buffer size */
    BOOST_LOG_TRIVIAL(error) << "Unexpected Buffer Size - Got "
                             << block.used_bytes() << " byte, expected "
                             << _buffer_bytes << " byte)";
    cudaDeviceSynchronize();
    cudaProfilerStop();
    return true;
  }

  //CUDA_ERROR_CHECK(cudaStreamSynchronize(_h2d_stream));
  _raw_voltage_db.swap();
  _sideChannelData_db.swap();
  std::swap(_procA, _procB);

  BOOST_LOG_TRIVIAL(debug) << "   block.used_bytes() = " << block.used_bytes()
                           << ", dataBlockBytes = " << _dataBlockBytes << "\n";

  // If necessary wait until the raw data has been processed
  CUDA_ERROR_CHECK(cudaEventSynchronize(_procA));

  CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void *>(_raw_voltage_db.a_ptr()),
                                   static_cast<void *>(block.ptr()),
                                   _dataBlockBytes, cudaMemcpyHostToDevice,
                                   _h2d_stream));
  CUDA_ERROR_CHECK(cudaMemcpyAsync(
      static_cast<void *>(_sideChannelData_db.a_ptr()),
      static_cast<void *>(block.ptr() + _dataBlockBytes + _gapSize),
      _sideChannelSize * _nHeaps, cudaMemcpyHostToDevice, _h2d_stream));

  if (_call_count == 1) {
    return false;
  }

  // Synchronize all streams
  _power_db_G0.swap();
  _power_db_G1.swap();
  _noOfBitSetsInSideChannel.swap();

  process(_raw_voltage_db.b(), _sideChannelData_db.b(), _power_db_G0.a(),
          _power_db_G1.a(), _noOfBitSetsInSideChannel.a());

  // signal that data block has been processed
  //CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));

  if (_call_count == 2) {
    return false;
  }

  //CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_stream));
  _host_power_db.swap();
  CUDA_ERROR_CHECK(
      cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr()),
                      static_cast<void *>(_power_db_G0.b_ptr()),
                      _power_db_G0.size() * sizeof(IntegratedPowerType),
                      cudaMemcpyDeviceToHost, _d2h_stream));
  CUDA_ERROR_CHECK(cudaMemcpyAsync(
      static_cast<void *>(_host_power_db.a_ptr() +
                          (_power_db_G0.size() * sizeof(IntegratedPowerType))),
      static_cast<void *>(_power_db_G1.b_ptr()),
      _power_db_G1.size() * sizeof(IntegratedPowerType), cudaMemcpyDeviceToHost,
      _d2h_stream));

  int R[1];
  CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void *>(R),
        static_cast<void *>(_noOfBitSetsInSideChannel.b_ptr()),
          1 * sizeof(unsigned int),cudaMemcpyDeviceToHost, _d2h_stream));

  BOOST_LOG_TRIVIAL(info) << "NOOF BIT SET IN SIDE CHANNEL: " << R[0] << std::endl;

  if (_call_count == 3) {
    return false;
  }

  // Wrap _detected_host_previous in a RawBytes object here;
  RawBytes bytes(reinterpret_cast<char *>(_host_power_db.b_ptr()),
                 _host_power_db.size() * sizeof(IntegratedPowerType),
                 _host_power_db.size() * sizeof(IntegratedPowerType));
  BOOST_LOG_TRIVIAL(debug) << "Calling handler";
  // The handler can't do anything asynchronously without a copy here
  // as it would be unsafe (given that it does not own the memory it
  // is being passed).
  return _handler(bytes);
} // operator ()

} // edd
} // effelsberg
} // psrdada_cpp

