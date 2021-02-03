#ifdef CAPTURER_HPP_

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{



Catcher::Catcher(capture_conf_t *conf, MultiLog& log, std::string addr, int port)
    : AbstractThread(log),
    config(conf)
{
	// Create endpoint address
	endpoint = new ip::udp::endpoint(ip::address::from_string(addr), port);

	// Create socket
	socket = new ip::udp::socket(io_context);

	// Establish connection to endpoint
	socket->connect(*(endpoint), ec);

	if(!ec)
	{
		BOOST_LOG_TRIVIAL(debug) << "Connect to endpoint " << addr << ":" << port;
	}
	else
	{
		BOOST_LOG_TRIVIAL(error) << "Connection to endpoint failed" << addr << ":" << port << " with error" << ec.message();
		logger.write(LOG_ERR, "Connection to endpoint %s:%d failed with error: %s (File: %s line: %d)",  addr.c_str(), port, ec.message(), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
}

Catcher::~Catcher()
{
	this->clean();
}

void Catcher::init()
{
    BOOST_LOG_TRIVIAL(debug) << "Initializing from Catcher " << _tid;
    // _socket->open_connection(0,1);
    // Vector that holds all packet positions in the temporary buffer
    tmp_pos.resize(config->nframes_tmp_buffer / config->n_catcher);
    _active = false;
}

void Catcher::run()
{
    _active = true;
    BOOST_LOG_TRIVIAL(debug) << "Running from Catcher " << _tid;

    while(!_quit && socket->is_open())
    {
		socket->receive(boost::asio::buffer(dframe.packet.buffer));
		dframe.deserialize(nullptr);
        packet_cnt++;
        // If the position is smaller 0 an old packet was received and will be neglated
        if( (cur_pos = calculate_pos()) >= 0)
        {
            memcpy((void*)&buffer.a()->data()[cur_pos], (void*)&dframe.packet.buffer[config->offset], codif_t::size);
        } else {
            lost_packet_cnt++;
        }
    }
    _active = false;
}


void Catcher::clean()
{
    this->free_mem();
    socket->close();
    BOOST_LOG_TRIVIAL(debug) << "Cleaning Catcher " << _tid;
}
void Catcher::stop()
{
    BOOST_LOG_TRIVIAL(debug) << "Stopping Catcher " << _tid;
    _quit = true;
    _active = false;
}

void Catcher::free_mem()
{
    BOOST_LOG_TRIVIAL(debug) << "Free memory Catcher " << _tid;
}


std::size_t Catcher::calculate_pos()
{
	return 0;
    // std::size_t idf = dframe.hdr.idf - config->dataframe_ref;
    // std::size_t position = (idf * config->nbeam + dframe.hdr.beam) * config->frame_size;
    // if( (position < config->rbuffer_size) && (position >= 0) )
    // {
		// 		// printf("Position: %ld_\tlocked: %d\tWorker: %d\tpacket_cnt: %ld\tidf: %ld\tbeam: %d", position, _complete.load(), _tid, packet_cnt, dframe.hdr.idf, dframe.hdr.beam);
		// 		// printf("\tWriting main buffer\n");
    // }else{
		// 		dframe.hdr.idf = config->dataframe_ref;
		// 		position -= config->rbuffer_size;
		// 		_complete.store(true);
		// 		// printf("Position: %ld_\tlocked: %d\tWorker: %d\tpacket_cnt: %ld\tidf: %ld\tbeam: %d", position, _complete.load(), _tid, packet_cnt, dframe.hdr.idf, dframe.hdr.beam);
		// 		BOOST_LOG_TRIVIAL(debug) << "Next buffer";
    // }
    // return position;
}


}
}
}
}

#endif
