#ifndef QUEUE_HEADER_HPP_
#define QUEUE_HEADER_HPP_

#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>

namespace bip = boost::interprocess;

struct QueueHeader
{
   QueueHeader()
      : data_in(false)
      , stop(true)
   {
   }

   //Mutex to protect access to the queue
   bip::interprocess_mutex mutex;

   //Condition to wait when the queue is empty
   bip::interprocess_condition ready_to_read;

   //Condition to wait when the queue is full
   bip::interprocess_condition ready_to_write;

   //Is there any payload
   bool data_in;

   bool stop;
};

#endif
