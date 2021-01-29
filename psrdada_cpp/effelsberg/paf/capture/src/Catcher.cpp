#include "psrdada_cpp/effelsberg/paf/capture/Catcher.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{



Catcher::Catcher(capture_conf_t *conf, MultiLog& log, std::string addr, int port)
    : AbstractThread(log),
    _conf(conf)
{
    _sock = new Socket(logger, addr, port, false);
	_complete.store(false);
}

Catcher::~Catcher()
{

}

void Catcher::init()
{
    BOOST_LOG_TRIVIAL(debug) << "Initializing from Catcher " << _tid;
    // _socket->open_connection(0,1);
    // Vector that holds all packet positions in the temporary buffer
    tmp_pos.resize(_conf->nframes_tmp_buffer / _conf->n_catcher);
    _active = false;
}

void Catcher::run()
{
    _active = true;
    BOOST_LOG_TRIVIAL(debug) << "Running from Catcher " << _tid;

    // if(_conf->dataframe_ref == -1 || _conf->sec_ref == -1)
    // {
    //     // TODO: Sniff a packets and evaluate the reference
    //     _conf->dataframe_ref = 0;
    //     _conf->sec_ref = 0;
    // }
    // TEST
    // int beam_per_thread = _conf->nbeam/_conf->n_catcher;
    // dframe.hdr.beam = beam_per_thread * _tid;
    // dframe.hdr.idf = 0;

    while(!_quit)
    {
		// if (!_sock->receive(dframe.packet.buffer, codif_t::size)){
		// 	_quit = true;
		// }
		// dframe.deserialize(nullptr);
        // packet_cnt++;
        // // If the position is smaller 0 an old packet was received and will be neglated
        // if( (cur_pos = calculate_pos()) >= 0)
        // {
        //     memcpy((void*)&buffer.a()->data()[cur_pos], (void*)&dframe.packet.buffer[_conf->offset], codif_t::size);
        // } else {
        //     lost_packet_cnt++;
        // }
    }
    _active = false;
}


void Catcher::clean()
{
    this->free_mem();
    _sock->close_connection();
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
    // std::size_t idf = dframe.hdr.idf - _conf->dataframe_ref;
    // std::size_t position = (idf * _conf->nbeam + dframe.hdr.beam) * _conf->frame_size;
    // if( (position < _conf->rbuffer_size) && (position >= 0) )
    // {
		// 		// printf("Position: %ld_\tlocked: %d\tWorker: %d\tpacket_cnt: %ld\tidf: %ld\tbeam: %d", position, _complete.load(), _tid, packet_cnt, dframe.hdr.idf, dframe.hdr.beam);
		// 		// printf("\tWriting main buffer\n");
    // }else{
		// 		dframe.hdr.idf = _conf->dataframe_ref;
		// 		position -= _conf->rbuffer_size;
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
