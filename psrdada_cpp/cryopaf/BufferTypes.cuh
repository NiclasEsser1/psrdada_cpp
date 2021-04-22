/*
* BufferTypes.cuh
* Author: Niclas Esser <nesser@mpifr-bonn.mpg.de>
* Description:
*   This file contains classes for different kinds of buffer representation.
*   All implemented classes inherit from DoubleBuffer<thrust::device_vector<T>>.
*/

#ifndef BUFFERTYPES_HPP_
#define BUFFERTYPES_HPP_

// boost::interprocess used to upload weights via POSIX shared memory
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/thread.hpp>

#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/double_device_buffer.cuh"
#include "psrdada_cpp/double_host_buffer.cuh"
#include "psrdada_cpp/cryopaf/QueueHeader.hpp"

namespace psrdada_cpp {
namespace cryopaf{

/**
* @brief  Class providing buffers for raw voltage data
*/
template<class T>
class RawVoltage : public DoubleBuffer<thrust::device_vector<T>>
{
public:
    typedef T type;
public:
    /**
    * @brief	Instantiates an object of RawVoltage
    *
    * @param	std::size_t  Number of items in buffer
    *
    * @detail Allocates twice the size in device memory as double device buffer
    */
    RawVoltage(std::size_t size)
      : DoubleBuffer<thrust::device_vector<T>>()
    {
      this->resize(size);
      _bytes = size * sizeof(T);
    }
    /**
    * @brief	Destroys an object of RawVoltage
    */
    ~RawVoltage(){}
    /**
    * @brief	Returns the number of bytes used for a single buffer
    *
    * @detail The occupied memory is twice
    */
    std::size_t total_bytes(){return _bytes;}
private:
    std::size_t _bytes;
};


/**
* @brief  Class providing buffers for beam data (is always the Output of the Pipeline)
* @detail An object of BeamOutput also contains an instance of DoublePinnedHostBuffer<T>
*         to allow an asynchronous copy to the host memory.
*/
template<class T>
class BeamOutput : public DoubleBuffer<thrust::device_vector<T>>
{

public:
    typedef T type;
    DoublePinnedHostBuffer<T> host;
public:
    /**
    * @brief	Instantiates an object of BeamOutput
    *
    * @param	std::size_t  Number of items in buffer
    *
    * @detail Allocates twice the size in device memory and in host memory as double buffers
    */
    BeamOutput(std::size_t size)
      : DoubleBuffer<thrust::device_vector<T>>()
    {
      this->resize(size);
      host.resize(size);
      _bytes = size * sizeof(T);
    }
    /**
    * @brief	Destroys an object of BeamOutput
    */
    ~BeamOutput(){}

    /**
    * @brief	Asynchronous copy to host memory
    *
    * @param	cudaStream_t& stream  Device to host stream
    */
    void async_copy(cudaStream_t& stream)
    {
        CUDA_ERROR_CHECK(cudaMemcpyAsync(host.a_ptr(), this->a_ptr(), _bytes, cudaMemcpyDeviceToHost, stream));
    }
    /**
    * @brief	Returns the number of bytes used for a single buffer
    *
    * @detail The occupied memory is twice
    */
    std::size_t total_bytes(){return _bytes;}
private:
    std::size_t _bytes;
};

// Define namespace for convinient access to boost::interprocess functionalitys, just used for weights
namespace bip = boost::interprocess;

/**
* @brief  Class providing buffers for beam weights
* @detail An object of Weights as the ability to read out a POSIX shared memory namespace
*         to load updated beam weights.
* @note   The current state is not final and will change in future. The idea for future is
*         to provide an update method which is called by a shared memory instance.
*/
template<class T>
class Weights : public DoubleBuffer<thrust::device_vector<T>>
{
public:
    typedef T type;
public:

    /**
    * @brief	Instantiates an object of BeamOutput
    *
    * @param	std::size_t  Number of items in buffer
    * @param	std::string  Name of the POSIX shared memory
    *
    * @detail Allocates twice the size in device memory as double device buffer.
    *         It also launches a boost::thread to create, read and write from shared
    *         memory.
    */
    Weights(std::size_t size, std::string smem_name="SharedMemoryWeights")
      : DoubleBuffer<thrust::device_vector<T>>()
      , _smem_name(smem_name)
    {

      this->resize(size);
      _bytes = size * sizeof(T);
      t = new boost::thread(boost::bind(&Weights::run, this));
    }
    /**
    * @brief	Destroys an object of BeamOutput
    */
    ~Weights(){}

    /**
    * @brief	Creates, read, write and removes a POSIX shared memory space
    *
    * @detail This function is a temporary solution to update beam weights on-the-fly
    *         while the pipeline is operating. In the future a clean interface will be created
    *         that provides addtional monitoring informations (e.g. power level) besides the
    *         beam weight updating mechanism.
    */
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
