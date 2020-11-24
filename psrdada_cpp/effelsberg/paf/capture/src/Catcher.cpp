#include "psrdada_cpp/effelsberg/paf/capture/Catcher.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{


Catcher::Catcher(capture_conf_t *conf, MultiLog& log, std::string addr, int port)
    : AbstractThread(log),
    _conf(conf)
{
    create_socket(addr, port);
	lock.store(false);
	_complete.store(false);
}

Catcher::~Catcher()
{

}

void Catcher::init()
{
    printf("Initializing from Catcher %d\n", _tid);
    // _socket->open_connection(0,1);
    // Vector that holds all packet positions in the temporary buffer
    tmp_pos.resize(_conf->nframes_tmp_buffer / _conf->n_catcher);
    _active = false;
}
void Catcher::run()
{
    _active = true;
    if(!rbuf)
    {
        _quit = true;
    }
    printf("Running from Catcher %d\n", _tid);

    if(_conf->dataframe_ref == -1 || _conf->sec_ref == -1)
    {
        // TODO: Sniff a packets and evaluate the reference
        _conf->dataframe_ref = 0;
        _conf->sec_ref = 0;
    }
    // TEST
    int beam_per_thread = _conf->nbeam/_conf->n_catcher;
    dframe.hdr.beam = beam_per_thread * _tid;
    dframe.hdr.idf = 0;

    while(!_quit)
    {
        // dframe.generate((char*)header_bytes); // Replaced by _sock->receive()
        // dframe.update();
        packet_cnt++;
        dframe.hdr.beam++;
        // TEST
        if(dframe.hdr.beam == beam_per_thread * (_tid+1))
        {
            dframe.hdr.beam = beam_per_thread * _tid;
            dframe.hdr.idf++;
            // printf("Thread %d, hdr.beam: %d; hdr.idf: %lu; buf", _tid, dframe.hdr.beam,dframe.hdr.idf );
            // printf("\tCur pos: %ld\n", cur_pos);
        }
        // If the position is smaller 0 an old packet was received and will be neglated
        if( (cur_pos = calculate_pos()) > 0)
        {
            // Copy dataframe to "correct" buffer
            if(tmp_cnt == 0)
            {
                memcpy((void*)&rbuf[cur_pos], (void*)&dframe.packet[_conf->offset], _conf->frame_size);
                // printf("\t Copied to main buffer\n");
            // Copy dataframe to temp buffer
            }else{
                memcpy((void*)&tbuf[cur_pos], (void*)&dframe.packet[_conf->offset], _conf->frame_size);
                // printf("\t Copied to temp buffer\n");

            }
        } else {
            lost_packet_cnt++;
        }
        usleep(100000);

    }
    _active = false;
}
void Catcher::clean()
{
    this->free_mem();
    _sock->close_connection();
    printf("Cleaning Catcher %d\n", _tid);
}
void Catcher::stop()
{
    printf("Stopping Catcher %d\n", _tid);
    _quit = true;
    _active = false;
}
bool Catcher::create_socket(std::string addr, int port)
{
    _sock = new Socket(logger, addr, port, false);
}

void Catcher::free_mem()
{
    printf("Free memory Catcher %d\n", _tid);
}


std::size_t Catcher::calculate_pos()
{
    std::size_t idf = dframe.hdr.idf - _conf->dataframe_ref;
    std::size_t position = (idf * _conf->nbeam + dframe.hdr.beam) * _conf->frame_size;
    if( (position < _conf->rbuffer_size +1) && (position >= 0) )
    {
		printf("Position: %ld_\tlocked: %d\tWorker: %d", position, lock.load(), _tid);
		printf("\tWriting main buffer\n");
        return position;
    }else if( (position > _conf->rbuffer_size) && !(lock.load()) ){
        position -= _conf->rbuffer_size;
        tmp_cnt++;
        tmp_pos[tmp_cnt] = position;

		printf("Position: %ld_\tlocked: %d\tWorker: %d", position, lock.load(), _tid);
		printf("\tWriting temp buffer");
        if(tmp_cnt >= _conf->nframes_tmp_buffer * _conf->nbeam / _conf->n_catcher)
        {
            printf("Overflow in temporary buffer on Worker %d\n",_tid);

            logger.write(LOG_ERR, "Overflow in temporary buffer (File: %s line: %d)\n", __FILE__, __LINE__);
            _quit = true;
            return -1;
        // If threshold of collected dataframes is reached, set the thread to complete
        }else if(tmp_cnt >= _conf->threshold){
			printf("\tThreshold reach\n");
            _complete.store(true);
        }



	// If data transfered to ringbuffer
	}else if( (lock.load()) && (_complete.load()) ){
        tmp_cnt = 0;
		dframe.hdr.idf = 0;
		position -= _conf->rbuffer_size;
		// printf("Next buffer; position: %ld hdr.idf: %ld tmp_cnt: %ld\n", position, dframe.hdr.idf, tmp_cnt);
		_complete.store(false);
        // Set new reference information for next ringbuffer
		printf("Position: %ld_\tlocked: %d\tWorker: %d", position, lock.load(), _tid);
		printf("\tNext buffer\n");
    }else{
		printf("Position: %ld_\tlocked: %d\tWorker: %d", position, lock.load(), _tid);

		printf("\tUnhandled lost packets\n");

        logger.write(LOG_WARNING, "Lost packets while copying to ringbuffer (File: %s line: %d)\n", __FILE__, __LINE__);
        return -1;
    }
    return position;
}


}
}
}
}
