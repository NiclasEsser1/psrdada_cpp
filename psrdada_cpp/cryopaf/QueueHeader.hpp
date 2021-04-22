/*
* QueueHeader.hpp
* Author: Niclas Esser <nesser@mpifr-bonn.mpg.de>
* Description:
*   This file contains the structure which is used shared memory IPC
*/
#ifndef QUEUE_HEADER_HPP_
#define QUEUE_HEADER_HPP_

#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>

namespace bip = boost::interprocess;

/**
* @brief    Header stored in POSIX shared memory to allow IPC communication
*/
struct QueueHeader
{
   QueueHeader()
      : data_in(false),
      stop(true)
   {}
   // Mutex to protect access to the queue
   bip::interprocess_mutex mutex;
   // Condition to wait when the queue is empty
   bip::interprocess_condition ready_to_read;
   // Condition to wait when the queue is full
   bip::interprocess_condition ready_to_write;
   // Is there any payload?
   bool data_in;
   // Stop flag, will force the owner to remove the shared memory
   bool stop;
};

#endif
