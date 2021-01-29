#ifdef TRANSMITTER_TESTER_H_

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{
namespace test{

Transmitter::Transmitter(capture_conf_t& conf, MultiLog& log, int port)
    : AbstractThread(log), _conf(conf)
{
		_sock = new Socket(logger, _conf.capture_addr, port, true);
}

Transmitter::~Transmitter()
{

}

void Transmitter::init()
{
    BOOST_LOG_TRIVIAL(debug) << "Initializing Transmitter " << _tid;
	_beams_per_thread = _conf.nbeam / _conf.n_catcher;
    _active = false;
	_seconds = 0;
	_frame_idx = 0;
	_epoch = 42;
	_freq_idx = 1284;
	_beam_idx = _tid * _beams_per_thread;
}

void Transmitter::run()
{
	int cnt = 0;
    _active = true;
    BOOST_LOG_TRIVIAL(debug) << "Running from Transmitter " << _tid;
	dframe.packet.create(_seconds, _frame_idx, _epoch, _freq_idx, _beam_idx);
    while(!_quit)
    {
		if(!_sock->transmit(dframe.serialize(), codif_t::size, 0, _sock->sock_conf(), _sock->addrlen()))
		{
			_quit = true;
			break;
		}
		_beam_idx++;
		if( _seconds > dframe.packet.hdr.period)
		{
			BOOST_LOG_TRIVIAL(debug) << "Closing Transmitter " << _tid << ", " << packet_cnt << " packets send";
			_quit = true;
		}
		if( _frame_idx == dframe.packet.hdr.totalsamples)
		{
			_frame_idx = 0;
			_seconds += dframe.packet.hdr.period;
		}
		if( _beam_idx == (_tid + 1) * _beams_per_thread )
		{
			_beam_idx = _tid * _beams_per_thread;
			_frame_idx++;
		}
	  	packet_cnt++;
    }

    _active = false;
}

void Transmitter::clean()
{
    _sock->close_connection();
    BOOST_LOG_TRIVIAL(debug) << "Cleaning Transmitter " << _tid;
}

void Transmitter::stop()
{
    BOOST_LOG_TRIVIAL(debug) << "Stopping Transmitter " << _tid;
    _quit = true;
    _active = false;
}

}
} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test


#endif
