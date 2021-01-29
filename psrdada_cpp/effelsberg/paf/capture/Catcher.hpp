#ifndef CAPTURER_HPP_
#define CAPTURER_HPP_

#include <atomic>

#include "psrdada_cpp/effelsberg/paf/capture/Threading.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Socket.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Types.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/double_buffer.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{

extern boost::mutex lock_buffer;
extern DoubleBuffer<std::vector<char>> buffer;

class Catcher : public AbstractThread
{
public:
    Catcher(capture_conf_t *conf, MultiLog& log, std::string addr, int port);
    ~Catcher();
    void init();
// protected:
    void run();
    void clean();
    void stop();
    void complete(bool val){_complete.store(val);}
    bool complete(){return _complete.load();}

    std::vector<std::size_t>* position_of_temp_packets(){return &tmp_pos;}
private:
    bool create_socket(std::string addr, int port);
    void free_mem();
    std::size_t calculate_pos();


private:
    capture_conf_t *_conf;

    char* rbuf = nullptr;
    char* tbuf = nullptr;
    //
    std::size_t tmp_cnt = 0;
    std::size_t packet_cnt = 0;
    std::size_t lost_packet_cnt = 0;
    std::size_t cur_pos = 0;
    std::vector<std::size_t> tmp_pos;

    Socket *_sock = nullptr;
    DataFrame<codif_t> dframe;

    bool referencer = false;
    std::atomic<bool> _complete;
};

}
}
}
}

#endif
