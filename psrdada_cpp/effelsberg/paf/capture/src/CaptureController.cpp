namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{

template<class HandlerType>
CaptureController<HandlerType>::CaptureController(capture_conf_t *conf, MultiLog& log, HandlerType& handle)
    : _conf(conf), logger(log), handler(handle), _dada_client(handle.client())
{
  _state = STARTING;
	/** Create instance to _monitor psrdada buffer **/
	_monitor = new CaptureMonitor(_conf->key, logger);

	/** Create instance to provide an TCP interface for control commands **/
  _ctrl_socket = new ControlSocket(logger, _conf->capture_ctrl_addr, _conf->capture_ctrl_port);

	_conf->frame_size = codif_t::size - _conf->offset;
	_conf->tbuffer_size = _conf->nframes_tmp_buffer * _conf->frame_size * _conf->nbeam;
	_conf->rbuffer_hdr_size = _dada_client.header_buffer_size();
	_conf->rbuffer_size = _dada_client.data_buffer_size();

	if(_conf->rbuffer_size%_conf->frame_size)
	{
		logger.write(LOG_ERR, "Failed: Ringbuffer size (%ld) is not a multiple of frame size (%ld). (File: %s line: %d)",  _conf->rbuffer_size, _conf->frame_size, __FILE__, __LINE__);
		throw std::runtime_error("Failed: Ringbuffer size mismatch, see logs for details" );
	}

	// Align raw_header vector to required size
	raw_header.resize(_conf->rbuffer_hdr_size );
	buffer.resize(_conf->rbuffer_size);
  /** Initialize DadaOutputStream **/
	// Get psrdada buffer sizes
	// Open the psrdada header file
  	input_file.open(_conf->psrdada_header_file, std::ios::in | std::ios::binary);
	if(input_file.fail())
	{
		logger.write(LOG_ERR, "ifstream::open() on %s failed: %s (File: %s line: %d)\n", _conf->psrdada_header_file.c_str(), strerror(errno), __FILE__, __LINE__);
		throw std::runtime_error("Failed: ifstream::open(), see logs for details " );
	}
	// Readout the file
	input_file.read(raw_header.data(), _conf->rbuffer_hdr_size );
	// Get the psrdada header from ringbuffer
	RawBytes header(raw_header.data(), _conf->rbuffer_hdr_size , _conf->rbuffer_hdr_size , false);
	// Finally initialize the handler
	handler.init(header);
}

template<class HandlerType>
CaptureController<HandlerType>::~CaptureController()
{
}

template<class HandlerType>
void CaptureController<HandlerType>::start()
{
    if(_state == STARTING)
    {
        _state = INITIALIZING;

        _ctrl_socket->init();
        _monitor->init();
        // For each port create a Catcher (Worker) instance.
				int tid = 0;
        for( auto& port : _conf->capture_ports )
        {
            catchers.push_back( new Catcher(_conf, logger, _conf->capture_addr, port) );
			catchers.back()->id(tid);
			catchers.back()->init();
			temp_pos_list.push_back(catchers.back()->position_of_temp_packets());
            tid++;
        }

        // Start worker BufferControl
        // Monitors capturing and write to ringbuffer
        _monitor->thread( new boost::thread(boost::bind(&AbstractThread::run, _monitor)) );
        // register BufferControl worker as the second thread in boost::thread_group
        t_grp.add_thread(_monitor->thread());
        // Enviroment initialized, idle program until command received
        _state = IDLE;
        // basically listen on socket to receive control commands
        _ctrl_socket->thread( new boost::thread(boost::bind(&AbstractThread::run, _ctrl_socket)) );
        // register CaptureControl worker as the first thread in boost::thread_group
        t_grp.add_thread(_ctrl_socket->thread());

        /** MAIN THREAD **/
        // Call watch_dog() function to observe all subthreads
        this->watch_dog();
    }else{
		BOOST_LOG_TRIVIAL(warning) << "Can not initialize when main program is in state" << state();
        logger.write(LOG_WARNING, "Can not initialize when main program is in state %s (File: %s line: %d)\n", state().c_str(), __FILE__, __LINE__);
    }
}
template<class HandlerType>
void CaptureController<HandlerType>::watch_dog()
{
    _ctrl_socket->state(_state);
		BOOST_LOG_TRIVIAL(info) << "Current state " << state();
    while(!_quit)
    {
        // CAUTION: _state is accessed by ControlSocket thread and main thread.
        // Since main thread is just reading the memory, it should be okay to not lock/unlock the location
        if(_state != _ctrl_socket->state())
        {
						BOOST_LOG_TRIVIAL(info) << "State changed from " << state() << " to " << _ctrl_socket->state();
            switch(_ctrl_socket->state())
            {
                case CAPTURING:
					_state = _ctrl_socket->state();
                    this->launch_capture();
					_ctrl_socket->state(_state);
                	break;

                case IDLE:
                	break;

                case STOPPING:
					_state = _ctrl_socket->state();
                    this->stop();
					_ctrl_socket->state(_state);
                	break;

                case EXIT:
					_state = _ctrl_socket->state();
                    this->clean();
					_ctrl_socket->state(_state);
                	break;

                case ERROR_STATE:
                	break;
            }
        }
		if( (all_thread_rdy(true)) && (_state == CAPTURING) )
		{
			// lock_buffer.lock();
			_conf->dataframe_ref = 0;
			// lock_buffer.unlock();
			// lock_buffer.lock();
			buffer.swap();
			// lock_buffer.unlock();

			RawBytes data(buffer.b()->data(), _conf->rbuffer_size, _conf->rbuffer_size);
  			handler(data);
			bytes_written += data.used_bytes();
			BOOST_LOG_TRIVIAL(debug) << "Bytes written: " << bytes_written;
			signal_to_worker(false);
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
    	t_grp.join_all();
	}
}

template<class HandlerType>
void CaptureController<HandlerType>::stop()
{
    for(auto& obj : catchers)
    {
        if(!stop_thread(obj))
		{
			exit(EXIT_FAILURE);
        }
    }
    _state == IDLE;
}
template<class HandlerType>
void CaptureController<HandlerType>::clean()
{
    this->stop();
    if(_state != ERROR_STATE)
    {
        for(auto& obj : catchers)
        {
            obj->clean();
        }
        if(!stop_thread(_monitor))
				{
	        exit(EXIT_FAILURE);
        }
        _monitor->clean();
        _ctrl_socket->clean();
    }else{
        exit(EXIT_FAILURE);
    }
}
template<class HandlerType>
void CaptureController<HandlerType>::launch_capture()
{
    if(_state  == CAPTURING)
    {
        // Iterate over all capture thread (excluded BufferControl)
        for(auto& obj : catchers)
        {
            // Start worker for all capture threads
            // Is dereferencing allowed here??? *(it)
            obj->thread( new boost::thread(boost::bind(&AbstractThread::run, obj)) );
            t_grp.add_thread(obj->thread());
        }
    }else{
		BOOST_LOG_TRIVIAL(warning) << "Can not capture when main program is in state " << state() << " state not changed";
        logger.write(LOG_WARNING, "Can not capture when main program is in state %s (File: %s line: %d)\n", state().c_str(), __FILE__, __LINE__);
    }
}
template<class HandlerType>
bool CaptureController<HandlerType>::stop_thread(AbstractThread* obj)
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
bool CaptureController<HandlerType>::all_thread_rdy(bool flag)
{
	int cnt = 0;
	for(auto& worker : catchers)
	{
		if(worker->complete() == flag)
		{
			cnt++;
		}
		if(worker->quit())
		{
			_quit = true;
		}
	}
	if(catchers.size() == cnt){
		BOOST_LOG_TRIVIAL(info) << "Buffer completed";
		logger.write(LOG_INFO, "Buffer completed\n");
		return true;
	}
	return false;
}

template<class HandlerType>
void CaptureController<HandlerType>::signal_to_worker(bool flag)
{
	for(auto& worker : catchers)
	{
		worker->complete(flag);
	}
}


template<class HandlerType>
std::string CaptureController<HandlerType>::state()
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

ControlSocket::ControlSocket(MultiLog& log, std::string addr, int port)
    : AbstractThread(log)
{
    _sock = new Socket(logger, addr, port, true, SOCK_STREAM);
    if(_sock->state() == -1)
    {
        logger.write(LOG_ERR, "CaptureController could not create server %s:%d\n", _sock->address().c_str(), _sock->port());
        exit(EXIT_FAILURE);
    }
}

void ControlSocket::init()
{
	if( !(_sock->open_connection()) )
	{
		logger.write(LOG_ERR, "open_connection() failed at %s:%d (File: %s line: %d)\n", _sock->address().c_str(), _sock->port(), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
}

void ControlSocket::run()
{
    _active = true;
    while(!_quit)
    {
		usleep(1000000);
        // Block thread until message received
        int bytes_read = _sock->reading(command, COMMAND_MSG_LENGTH);

        logger.write(LOG_INFO, "Capture control received command: %s\n", command, __FILE__, __LINE__);

        std::string str(command);

        if( str.compare(0,5, "START") == 0 ){
            BOOST_LOG_TRIVIAL(debug) << "Capturing...";
            _state = CAPTURING;
        }else if( str.compare(0,4, "STOP") == 0 ){
            BOOST_LOG_TRIVIAL(debug) << "Stopping...";
            _state = STOPPING;
        }else if( str.compare(0,4, "EXIT") == 0 ){
            BOOST_LOG_TRIVIAL(debug) << "Exit...";
            _state = EXIT;
						return;
        }else if( str.compare(0,5, "STATUS") == 0 ){
            // send_status();
        }else{
			BOOST_LOG_TRIVIAL(debug) << "Capture control received unkown command: " << command;
            logger.write(LOG_WARNING, "Capture control received unkown command: %s\n", command, __FILE__, __LINE__);
        }
    }
    _active = false;
}

void ControlSocket::stop()
{
    BOOST_LOG_TRIVIAL(info) << "Stopping controller _ctrl_socket";
}

void ControlSocket::clean()
{
    BOOST_LOG_TRIVIAL(info) << "Cleaning controller _ctrl_socket";
    _sock->close_connection();
}

}
}
}
}
