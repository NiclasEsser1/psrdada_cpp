#ifdef UDPSOCKET_H_

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{

using namespace boost::asio;


UDPSocket::UDPSocket(MultiLog& log, std::string addr, int port, int tid)
    : logger(log), address(addr), _port(port), _tid(tid)
{
	// Create endpoint address
	endpoint = new ip::udp::endpoint(ip::address::from_string(address), _port);

	// Create socket
	socket = new ip::udp::socket(io_context);

	// Establish connection to endpoint
	try
	{
		socket->open(ip::udp::v4());
	}
	catch(std::exception& e)
	{
		BOOST_LOG_TRIVIAL(error) << e.what();
	}
	BOOST_LOG_TRIVIAL(debug) << "Opened connection to " << endpoint->address() << ":" << endpoint->port();

}

UDPSocket::~UDPSocket()
{
	if(socket->is_open())
	{
		socket->close();
	}
}




Transmitter::Transmitter(MultiLog& log, std::string addr, int port, int tid, int nbeams, int periods)
	: UDPSocket(log, addr, port, tid), nbeam(nbeams), period(periods)
{
	BOOST_LOG_TRIVIAL(debug) << "Instantiate Transmitter object " << _tid;
	f_beam = _tid * nbeam;
	l_beam = (_tid+1) * nbeam - 1;
}

Transmitter::~Transmitter()
{
	BOOST_LOG_TRIVIAL(debug) << "Destroying Transmitter object" << _tid;
}

void Transmitter::run()
{
	BOOST_LOG_TRIVIAL(debug) << "Running transmitter on thread " << _tid;

	int result = 0;
	_quit = false;
	dframe.sync_frame();

	while(!_quit)
	{
		dframe.serialize();
		try
		{
			result = socket->send_to(boost::asio::buffer(dframe.buffer(), codif_t::size), *endpoint);
		}
		catch (const boost::system::system_error& e)
		{
			_quit = true;
			BOOST_LOG_TRIVIAL(error) << e.what();
		}

		if (result != codif_t::size)
		{
			_quit = true;
			BOOST_LOG_TRIVIAL(error) << "Not all bytes sent " << result << "/" << codif_t::size;
		}

		packet_cnt++;

		if(frame()%249995 == 0 && beam() == f_beam)
		{
			BOOST_LOG_TRIVIAL(debug) << "Sent " << packet_cnt << " packets from Transmitter " <<_tid;
			dframe.print();
		}

		increment_dframe();
		usleep(100000);
	}
}
void Transmitter::stop()
{

}
void Transmitter::increment_dframe()
{
	if(frame() > 249999 && beam() == l_beam)
	{
		dframe.packet.hdr.beam = f_beam;
		dframe.packet.hdr.ref_idf = 0;
		dframe.packet.hdr.sec_from_epoch += 27;
		if(period_cnt == period)
		{
			_quit = true;
		}
		period_cnt++;
	}
	if(beam() < l_beam)
	{
		dframe.packet.hdr.beam++;
	}
	else
	{
		dframe.packet.hdr.beam = f_beam;
		dframe.packet.hdr.ref_idf++;
	}
}


Receiver::Receiver(MultiLog& log, std::string addr, int port, int tid, capture_conf_t& conf)
	: UDPSocket(log, addr, port, tid), config(conf)
{

	try
	{
		socket->bind(*endpoint);
	}
	catch (const boost::system::system_error& e)
	{
		_quit = true;
		BOOST_LOG_TRIVIAL(error) << e.what();
	}
	BOOST_LOG_TRIVIAL(debug) << "Instantiate Receiver object" << _tid;
}

Receiver::~Receiver()
{
	BOOST_LOG_TRIVIAL(debug) << "Destroying Receiver object" << _tid;
}

void Receiver::run()
{
	int result = 0;
	BOOST_LOG_TRIVIAL(debug) << "Running receiver on thread " << _tid;
	if(config.dataframe_ref == -1 || config.sec_ref == -1)
	{
		BOOST_LOG_TRIVIAL(warning) << "No reference time provided, receiver not synced ";
		dframe.sync_frame();
		config.dataframe_ref = dframe.hdr().ref_idf;
		config.sec_ref = dframe.hdr().sec_from_epoch;
		config.ref_epoch = dframe.hdr().ref_epoch;
	}
	while(!_quit)
	{
		try
		{
			result = socket->receive_from(boost::asio::buffer(dframe.buffer(), codif_t::size), *endpoint);
		}
		catch (const boost::system::system_error& e)
		{
			_quit = true;
			BOOST_LOG_TRIVIAL(error) << e.what();
		}
		if (result != codif_t::size)
		{
			BOOST_LOG_TRIVIAL(error) << "Not all bytes sent " << result << "/" << codif_t::size;
		}

		packet_cnt++;

		dframe.deserialize();
		calculate_position();

		if(position >= 0 && position < doubleBuffer.size())
		{
			memcpy((void*)&doubleBuffer.a()->data()[position], (void*)dframe.buffer(), codif_t::size - config.offset);
		}
		else if(position >= doubleBuffer.size() && position < 2*doubleBuffer.size())
		{
			memcpy((void*)&doubleBuffer.b()->data()[position - doubleBuffer.size()], (void*)dframe.buffer(), codif_t::size - config.offset);
			overflow_packets++;
			if(overflow_packets >= config.threshold)
			{
				BOOST_LOG_TRIVIAL(warning) << "Receiver " << _tid << " completed buffer";
				overflow_packets = 0;
				completed = true;
				config.dataframe_ref = dframe.hdr().ref_idf;
				config.sec_ref = dframe.hdr().sec_from_epoch;
				config.ref_epoch = dframe.hdr().ref_epoch;
			}
		}
		else if(position < 0)
		{
			BOOST_LOG_TRIVIAL(debug) << "Reference time not reached, waiting... ";
		}
		else
		{
			// BOOST_LOG_TRIVIAL(warning) << "Lost packet (position = " << position << ")";
			lost_packets++;
			completed = true;
			config.dataframe_ref = dframe.hdr().ref_idf;
			config.sec_ref = dframe.hdr().sec_from_epoch;
			config.ref_epoch = dframe.hdr().ref_epoch;
		}
	}
}

void Receiver::stop()
{

}

void Receiver::calculate_position()
{
	// dframe.print();
	// std::size_t epoch = dframe.hdr().ref_epoch - config.ref_epoch;

	int epoch = dframe.hdr().ref_epoch - config.ref_epoch;
	long int sec = dframe.hdr().sec_from_epoch - config.sec_ref;
    long int idf = dframe.hdr().ref_idf - config.dataframe_ref;

	if(sec < 0 && epoch <= 0)
	{
		position = -1;
		return;
	}
	position = (int64_t)(idf) + (double) sec / codif_t::frame_resolution;
    position = (position * config.nbeam + dframe.hdr().beam) * (codif_t::size - config.offset);
	if(_tid == 0)
	{
		std::cout << "SEC: Diff " << sec << " cur " << dframe.hdr().sec_from_epoch << " ref " << config.sec_ref << std::endl;
		std::cout << "IDF: Diff " << idf << " cur " << dframe.hdr().ref_idf << " ref " << config.dataframe_ref << std::endl;
		std::cout << "Position:  " << position << std::endl;
	}
}


ControlSocket::ControlSocket(MultiLog& log, std::string addr, int port)
    : UDPSocket(log, addr, port)
{
	try
	{
		socket->bind(*endpoint);
	}
	catch (const boost::system::system_error& e)
	{
		_quit = true;
		BOOST_LOG_TRIVIAL(error) << e.what();
	}
	BOOST_LOG_TRIVIAL(debug) << "Instantiated ControlSocket on " << addr << ":" << _port;
}

ControlSocket::~ControlSocket()
{
	BOOST_LOG_TRIVIAL(debug) << "Destroying ControlSocket object" << _tid;

}

void ControlSocket::run()
{
    _active = true;
	int result = 0;
    while(!_quit)
    {
        // Block thread until message received
		try
		{
			result = socket->receive_from(boost::asio::buffer(&command[0], COMMAND_MSG_LENGTH), *endpoint);
		}
		catch (const boost::system::system_error& e)
		{
			_quit = true;
			BOOST_LOG_TRIVIAL(error) << e.what();
		}

        logger.write(LOG_INFO, "Capture control received command: %s\n", command, __FILE__, __LINE__);

        std::string str(command);
        BOOST_LOG_TRIVIAL(debug) << "Received command";

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
		usleep(10000);
		// break;
    }
    _active = false;
}

void ControlSocket::stop()
{
    BOOST_LOG_TRIVIAL(info) << "Stopping controller ctrl_server";
}


} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test


#endif
