/*
* BufferTypes.cuh
* Author: Niclas Esser <nesser@mpifr-bonn.mpg.de>
* Description:
*/

#ifndef BUFFERTYPES_HPP_
#define BUFFERTYPES_HPP_

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/thread.hpp>

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/double_device_buffer.cuh"
#include "psrdada_cpp/double_host_buffer.cuh"
#include "psrdada_cpp/cryopaf/QueueHeader.hpp"

namespace psrdada_cpp {
namespace cryopaf{


template<class T>
class RawVoltage : public DoubleBuffer<thrust::device_vector<T>>
{
public:
    typedef T type;
public:
    RawVoltage(std::size_t size)
      : DoubleBuffer<thrust::device_vector<T>>()
    {
      this->resize(size);
      _bytes = size * sizeof(T);
    }
    ~RawVoltage()
    {

    }
    std::size_t total_bytes(){return _bytes;}
private:
    std::size_t _bytes;
};


template<class T>
class BeamOutput : public DoubleBuffer<thrust::device_vector<T>>
{

public:
    typedef T type;
    DoublePinnedHostBuffer<T> host;
public:
    BeamOutput(std::size_t size)
      : DoubleBuffer<thrust::device_vector<T>>()
    {
      this->resize(size);
      host.resize(size);
      _bytes = size * sizeof(T);
    }
    ~BeamOutput()
    {

    }
    void async_copy(cudaStream_t& stream)
    {
        CUDA_ERROR_CHECK(cudaMemcpyAsync(host.a_ptr(), this->a_ptr(), _bytes, cudaMemcpyDeviceToHost, stream));
    }

    std::size_t total_bytes(){return _bytes;}
private:
    std::size_t _bytes;
};


namespace bip = boost::interprocess;

template<class T>
class Weights : public DoubleBuffer<thrust::device_vector<T>>
{
public:
    typedef T type;
public:
    Weights(std::size_t size, std::string smem_name="SharedMemoryWeights")
      : DoubleBuffer<thrust::device_vector<T>>()
      , _smem_name(smem_name)
    {

      this->resize(size);
      _bytes = size * sizeof(T);
      t = new boost::thread(boost::bind(&Weights::run, this));
    }
    ~Weights()
    {

    }
    void run()
    {

      bip::shared_memory_object::remove("SharedMemoryWeights");
      bip::shared_memory_object smem(bip::create_only, "SharedMemoryWeights", bip::read_write);

      // Set size of shared memory including QueueHeader + payload
      BOOST_LOG_TRIVIAL(info) << "Size of shared memory for weight uploading (IPC) " << sizeof(QueueHeader) + (this->size()) * sizeof(T);
      smem.truncate(sizeof(QueueHeader) + (this->size()) * sizeof(T));

      // Map shared memory to a addressable region
      bip::mapped_region region(smem, bip::read_write);

      void* smem_addr = region.get_address(); // get it's address

      QueueHeader* qheader = static_cast<QueueHeader*>(smem_addr);  // Interpret first bytes as QueueHeader
      T *ptr = &(static_cast<T*>(smem_addr)[sizeof(QueueHeader)]); // Pointer to address of payload (behind QueueHeader)
      qheader->stop = true;
      while(qheader->stop){usleep(1000);}
      while(!qheader->stop)
      {
        bip::scoped_lock<bip::interprocess_mutex> lock(qheader->mutex);
        if(!qheader->data_in)
        {
          BOOST_LOG_TRIVIAL(debug) << "Waiting for writing weights to shared memory";
          qheader->ready_to_read.wait(lock); // Wait for read out
        }

        BOOST_LOG_TRIVIAL(debug) << "Reading new weights from shared memory";
        CUDA_ERROR_CHECK(cudaMemcpy((void*)this->b_ptr(), (void*)ptr,
            _bytes, cudaMemcpyHostToDevice));
        // Swap double buffer, so next batch is calculated with new weights
        this->swap();
        //Notify the other process that the buffer is empty
        qheader->data_in = false;
        qheader->ready_to_write.notify_all();

      }
      bip::shared_memory_object::remove("SharedMemoryWeights");
      BOOST_LOG_TRIVIAL(info) << "Closed shared memory for weights uploading";
    }

    std::size_t total_bytes(){return _bytes;}

private:
    std::size_t _bytes;
    std::string _smem_name;
    boost::thread *t;
};

}
}



#endif /* BUFFERTYPES_HPP_ */
