#ifdef RECEIVE_HPP_

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{

template<class HandlerType>
ReceiveControl<HandlerType>::ReceiveControl(capture_conf_t& conf, MultiLog& log, HandlerType& handle)
    : config(conf), logger(log), handler(handle), _dada_client(handle.client())
{
  	_state = STARTING;

	/** Create instance to provide an TCP interface for control commands **/
  	ctrl_socket = new ControlSocket(logger, config.capture_ctrl_addr, config.capture_ctrl_port);

	config.frame_size = codif_t::size - config.offset;
	config.rbuffer_hdr_size = _dada_client.header_buffer_size();
	config.rbuffer_size = _dada_client.data_buffer_size();

	if(config.rbuffer_size%config.frame_size)
	{
		logger.write(LOG_ERR, "Failed: Ringbuffer size (%ld) is not a multiple of frame size (%ld). (File: %s line: %d)",  config.rbuffer_size, config.frame_size, __FILE__, __LINE__);
		throw std::runtime_error("Failed: Ringbuffer size mismatch, see logs for details" );
	}

	// Align raw_header vector to required size
	raw_header.resize(config.rbuffer_hdr_size );
	doubleBuffer.resize(config.rbuffer_size);
  /** Initialize DadaOutputStream **/
	// Get psrdada buffer sizes
	// Open the psrdada header file
	dada_header_file.open(config.psrdada_header_file, std::ios::in | std::ios::binary);
	if(dada_header_file.fail())
	{
		logger.write(LOG_ERR, "ifstream::open() on %s failed: %s (File: %s line: %d)\n", config.psrdada_header_file.c_str(), strerror(errno), __FILE__, __LINE__);
		throw std::runtime_error("Failed: ifstream::open(), see logs for details " );
	}

	// Readout the file
	dada_header_file.read(raw_header.data(), config.rbuffer_hdr_size );
	// Get the psrdada header from ringbuffer
	RawBytes header(raw_header.data(), config.rbuffer_hdr_size , config.rbuffer_hdr_size , false);
	// Finally initialize the handler
	handler.init(header);
}

template<class HandlerType>
ReceiveControl<HandlerType>::~ReceiveControl()
{
}

template<class HandlerType>
void ReceiveControl<HandlerType>::start()
{

    if(_state == STARTING)
    {
        _state = INITIALIZING;

        // For each port create a receiver (Worker) instance.
		int tid = 0;
        for( auto& port : config.capture_ports )
        {
            receivers.push_back( new Receiver(logger, config.capture_addr, port, tid, config) );
            tid++;
        }

        // Enviroment initialized, idle program until command received
        _state = IDLE;
        // basically listen on socket to receive control commands
        ctrl_socket->thread( new boost::thread(boost::bind(&UDPSocket::run, ctrl_socket)) );
        // register CaptureControl worker as the first thread in boost::thread_group
        thread_group.add_thread(ctrl_socket->thread());

        /** MAIN THREAD **/
        // Call watch_dog() function to observe all subthreads
        this->watch_dog();
    }else{
		BOOST_LOG_TRIVIAL(warning) << "Can not initialize when main program is in state" << state();
        logger.write(LOG_WARNING, "Can not initialize when main program is in state %s (File: %s line: %d)\n", state().c_str(), __FILE__, __LINE__);
    }
}
template<class HandlerType>
void ReceiveControl<HandlerType>::watch_dog()
{
    ctrl_socket->state(_state);
		BOOST_LOG_TRIVIAL(info) << "Current state " << state();
    while(!_quit)
    {
        // CAUTION: _state is accessed by ControlSocket thread and main thread.
        // Since main thread is just reading the memory, it should be okay to not lock/unlock the location
        if(_state != ctrl_socket->state())
        {
			BOOST_LOG_TRIVIAL(info) << "State changed from " << state() << " to " << ctrl_socket->state();
            switch(ctrl_socket->state())
            {
                case CAPTURING:
					_state = ctrl_socket->state();
                    this->launch_capture();
					ctrl_socket->state(_state);
                	break;

                case IDLE:
                	break;

                case STOPPING:
					_state = ctrl_socket->state();
                    this->stop();
					ctrl_socket->state(_state);
                	break;

                case EXIT:
					_state = ctrl_socket->state();
                    this->clean();
					ctrl_socket->state(_state);
                	break;

                case ERROR_STATE:
                	break;
            }
        }

		if( (buffer_ready()) && (_state == CAPTURING) )
		{
			lock_buffer.lock();
			doubleBuffer.swap();
			lock_buffer.unlock();

			RawBytes data((char*)doubleBuffer.b()->data(), doubleBuffer.size(), doubleBuffer.size());	// TODO: is castin from uchar to char possible???
  			handler(data);
			buffer_complete(false);
			bytes_written += data.used_bytes();
			BOOST_LOG_TRIVIAL(debug) << "Bytes written: " << bytes_written;
		}

		if( _dada_client.data_buffer_nfull() >= _dada_client.data_buffer_count() -1 )
		{
			_quit = true;
			logger.write(LOG_ERR, "All dada buffers are full, has to abort (File: %s line: %d)\n", __FILE__, __LINE__);
			BOOST_LOG_TRIVIAL(error) << "All dada buffers are full, has to abort ... ";
		}
    }
	this->clean();
	if(!ERROR_STATE)
	{
    	thread_group.join_all();
	}
}

template<class HandlerType>
void ReceiveControl<HandlerType>::stop()
{
    for(auto& obj : receivers)
    {
        if(!stop_thread(obj))
		{
			exit(EXIT_FAILURE);
        }
    }
    _state == IDLE;
}
template<class HandlerType>
void ReceiveControl<HandlerType>::clean()
{

}

template<class HandlerType>
void ReceiveControl<HandlerType>::launch_capture()
{
    if(_state  == CAPTURING)
    {
        // Iterate over all capture thread (excluded BufferControl)
        for(auto& obj : receivers)
        {
            // Start worker for all capture threads
            // Is dereferencing allowed here??? *(it)
            obj->thread( new boost::thread(boost::bind(&Receiver::run, obj)) );
            thread_group.add_thread(obj->thread());
        }
    }
	else
	{
		BOOST_LOG_TRIVIAL(warning) << "Can not capture when main program is in state " << state() << " state not changed";
        logger.write(LOG_WARNING, "Can not capture when main program is in state %s (File: %s line: %d)\n", state().c_str(), __FILE__, __LINE__);
    }
}
template<class HandlerType>
bool ReceiveControl<HandlerType>::stop_thread(UDPSocket* obj)
{
    int timeout = 0;
    while((obj->active() && timeout < 2000))
    {
        obj->stop();
        timeout++;
        usleep(100);
    }
    if(timeout >= 2000)
    {
        _state = ERROR_STATE;
        return false;
    }else{
        return true;
    }
}

template<class HandlerType>
bool ReceiveControl<HandlerType>::buffer_ready()
{
	int cnt = 0;
	for(auto& worker : receivers)
	{
		if(worker->quit())
		{
			_quit = true;
		}
		if(worker->buffer_complete())
		{
			cnt ++;
		}
	}
	if(receivers.size() == cnt){
		BOOST_LOG_TRIVIAL(info) << "Buffer completed";
		logger.write(LOG_INFO, "Buffer completed\n");
		return true;
	}
	return false;
}

template<class HandlerType>
bool ReceiveControl<HandlerType>::buffer_ready()
{
	int cnt = 0;
	for(auto& worker : receivers)
	{
		if(worker->quit())
		{
			_quit = true;
		}
		if(worker->buffer_complete())
		{
			cnt ++;
		}
	}
	if(receivers.size() == cnt){
		BOOST_LOG_TRIVIAL(info) << "Buffer completed";
		logger.write(LOG_INFO, "Buffer completed\n");
		return true;
	}
	return false;
}



template<class HandlerType>
std::string ReceiveControl<HandlerType>::state()
{
    std::string state;
    switch(_state)
    {
        case ERROR_STATE:
            state = "ERROR_STATE";
            break;
        case STARTING:
            state = "STARTING";
            break;
        case INITIALIZING:
            state = "INITIALIZING";
            break;
        case IDLE:
            state = "IDLE";
            break;
        case CAPTURING:
            state = "CAPTURING";
            break;
        case STOPPING:
            state = "STOPPING";
            break;
        case EXIT:
            state = "EXIT";
            break;
        default:
            state = "Unknown state";
            break;
    }
    return state;
}


}
}
}
}


#endif
