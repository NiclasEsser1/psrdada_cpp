namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{

template<class HandlerType>
CaptureController<HandlerType>::CaptureController(HandlerType& handle, capture_conf_t *conf, MultiLog& log)
    : handler(handle),
    _conf(conf),
    logger(log)
{
    _state = STARTING;
	/** Create instance to monitor psrdada buffer **/
	monitor = new CaptureMonitor(_conf->key, logger);

	/** Create instance to provide an TCP interface for control commans **/
    interface = new CaptureInterface(logger, _conf->capture_ctrl_addr, _conf->capture_ctrl_port);

	_conf->frame_size = codif_t::size - _conf->offset;
	_conf->tbuffer_size = _conf->nframes_tmp_buffer * _conf->frame_size * _conf->nbeam;
	_conf->rbuffer_hdr_size = handler.client().header_buffer_size();
	_conf->rbuffer_size = handler.client().data_buffer_size();
	if(_conf->rbuffer_size%_conf->frame_size){
		logger.write(LOG_ERR, "Failed: Ringbuffer size (%ld) is not a multiple of frame size (%ld). (File: %s line: %d)", _conf->frame_size, _conf->rbuffer_size, __FILE__, __LINE__);
		throw std::runtime_error("Failed: Ringbuffer size is not a multiple of frame size" );
	}

	// Align raw_header vector to required size
	raw_header.resize(_conf->rbuffer_hdr_size );
	block.resize(_conf->rbuffer_size);
	tbuf.resize(_conf->tbuffer_size);

    /** Initialize DadaOutputStream **/
	// Get psrdada buffer sizes
	// Open the psrdada header file
    input_file.open(_conf->psrdada_header_file, std::ios::in | std::ios::binary);
	if(input_file.fail())
	{
		logger.write(LOG_ERR, "ifstream::open() on %s failed: %s (File: %s line: %d)\n", _conf->psrdada_header_file.c_str(), strerror(errno), __FILE__, __LINE__);
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

        interface->init();
        monitor->init();
        // For each port create a Catcher (Worker) instance.
		int tid = 0;
        for( auto& port : _conf->capture_ports )
        {
            catchers.push_back( new Catcher(_conf, logger, _conf->capture_addr, port) );
			catchers.back()->id(tid);
			catchers.back()->init();
			catchers.back()->set_dptr(block.data());
			catchers.back()->set_tptr(tbuf.data());
			temp_pos_list.push_back(catchers.back()->position_of_temp_packets());
            tid++;
        }

        // Start worker BufferControl
        // Monitors capturing and write to ringbuffer
        monitor->thread( new boost::thread(boost::bind(&AbstractThread::run, monitor)) );
        // register BufferControl worker as the second thread in boost::thread_group
        t_grp.add_thread(monitor->thread());
        // Enviroment initialized, idle program until command received
        _state = IDLE;
        // basically listen on socket to receive control commands
        interface->thread( new boost::thread(boost::bind(&AbstractThread::run, interface)) );
        // register CaptureControl worker as the first thread in boost::thread_group
        t_grp.add_thread(interface->thread());

        /** MAIN THREAD **/
        // Call watch_dog() function to observe all subthreads
        this->watch_dog();
    }else{
        logger.write(LOG_WARNING, "Can not initialize when main program is in state %s (File: %s line: %d)\n", state().c_str(), __FILE__, __LINE__);
    }
}
template<class HandlerType>
void CaptureController<HandlerType>::watch_dog()
{
    interface->state(_state);
	printf("Watching state: %d\n", _state);
    while(!_quit)
    {
        // CAUTION: _state is accessed by CaptureInterface thread and main thread.
        // Since main thread is just reading the memory, it should be okay to not lock/unlock the location
        if(_state != interface->state())
        {
			printf("State changed\n");
            switch(interface->state())
            {
                case CAPTURING:
					_state = interface->state();
                    this->launch_capture();
					interface->state(_state);
                break;

                case IDLE:
                break;

                case STOPPING:
					_state = interface->state();
                    this->stop();
					interface->state(_state);
                break;

                case EXIT:
					_state = interface->state();
                    this->clean();
					interface->state(_state);
                break;

                case ERROR_STATE:
                break;
            }
        }
		// sleep(1);
		// printf("State")
		// Transfer data on complete
		if( (buffer_complete()) && (_state == CAPTURING) )
		{
			lock_buffer(false);
			RawBytes data(block.data(), _conf->rbuffer_size, _conf->rbuffer_size);
            handler(data);
			bytes_written += data.used_bytes();
			printf("Bytes written: %ld\n", bytes_written);
			lock_buffer(true);

			// Copy remaing temporary data
			// Maybe bottleneck

			for(std::size_t i = 0; i < temp_pos_list.size(); ++i)
			{
				std::vector<std::size_t> list = *temp_pos_list[i];
				for(std::size_t k = 0; k < list.size(); ++k)
				{
					if(list[k] >= 0){
						memcpy((void*)&block[list[k]], (void*)&tbuf[list[k]], _conf->frame_size);
						list[k]=-1;
					// Reached end of ringbuffer
					} else {
						break;
					}
				}
			}
			while(buffer_complete());
			lock_buffer(false);
		}
    }
	this->clean();
	if(!ERROR_STATE)
	{
    	t_grp.join_all();
	}
    printf("Leaving program...\n");
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
        if(!stop_thread(monitor))
		{
	        exit(EXIT_FAILURE);
        }
        monitor->clean();
        interface->clean();
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
bool CaptureController<HandlerType>::buffer_complete()
{
	int cnt = 0;
	for(auto& worker : catchers)
	{
		if(worker->complete())
		{
			cnt++;
		}
		if(worker->quit())
		{
			_quit = true;
		}
	}
	if(catchers.size() == cnt){
		logger.write(LOG_INFO, "Buffer completed\n");
		return true;
	}
	return false;
}
template<class HandlerType>
void CaptureController<HandlerType>::buffer_complete(bool flag)
{
	for(auto& worker : catchers)
	{
		worker->complete(flag);
	}
}
template<class HandlerType>
void CaptureController<HandlerType>::lock_buffer(bool flag)
{
	for(auto& worker : catchers)
	{
		worker->lock_buffer(flag);
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

CaptureInterface::CaptureInterface(MultiLog& log, std::string addr, int port)
    : AbstractThread(log)
{
    _sock = new Socket(logger, addr, port, true, SOCK_STREAM);
    if(_sock->state() == -1)
    {
        logger.write(LOG_ERR, "CaptureController could not create server %s:%d\n", _sock->address().c_str(), _sock->port());
        exit(EXIT_FAILURE);
    }
}

void CaptureInterface::init()
{
    if( !(_sock->open_connection()) )
    {
        logger.write(LOG_ERR, "open_connection() failed at %s:%d (File: %s line: %d)\n", _sock->address().c_str(), _sock->port(), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

void CaptureInterface::run()
{
    _active = true;
    while(!_quit)
    {
        // Block thread until message received
        int bytes_read = _sock->reading(command, COMMAND_MSG_LENGTH);

        logger.write(LOG_INFO, "Capture control received command: %s\n", command, __FILE__, __LINE__);

        std::string str(command);

        if( str.compare(0,5, "START") == 0 ){
            printf("Capturing...\n");
            _state = CAPTURING;
        }else if( str.compare(0,4, "STOP") == 0 ){
            printf("Stopping...\n");
            _state = STOPPING;
        }else if( str.compare(0,4, "EXIT") == 0 ){
            printf("Exit...\n");
            _state = EXIT;
			return;
        }else if( str.compare(0,5, "STATUS") == 0 ){
            // send_status();
        }else{
            logger.write(LOG_WARNING, "Capture control received unkown command: %s\n", command, __FILE__, __LINE__);
        }
    }
    _active = false;
}

void CaptureInterface::stop()
{
    printf("Stopping contrller interface\n");
}

void CaptureInterface::clean()
{
    printf("Cleaning controller interface\n");
    _sock->close_connection();
}

}
}
}
}
