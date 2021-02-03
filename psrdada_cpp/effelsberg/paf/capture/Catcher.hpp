#ifndef CAPTURER_HPP_
#define CAPTURER_HPP_

#include <atomic>
#include <boost/asio.hpp>
#include <boost/asio/buffer.hpp>


#include "psrdada_cpp/effelsberg/paf/capture/Threading.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Types.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/double_buffer.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{

extern boost::mutex lock_buffer;
extern DoubleBuffer<std::vector<char>> buffer;

using namespace boost::asio;

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

    std::vector<std::size_t>* position_of_temp_packets(){return &tmp_pos;}
private:
    void free_mem();
    std::size_t calculate_pos();


private:
    boost::system::error_code ec;
    boost::asio::io_service io_context;
    ip::udp::endpoint* endpoint;
    ip::udp::socket* socket;

    capture_conf_t *config;

    char* rbuf = nullptr;
    char* tbuf = nullptr;

    std::size_t tmp_cnt = 0;
    std::size_t packet_cnt = 0;
    std::size_t lost_packet_cnt = 0;
    std::size_t cur_pos = 0;
    std::vector<std::size_t> tmp_pos;

    DataFrame<codif_t> dframe;
};

}
}
}
}


#include "psrdada_cpp/effelsberg/paf/capture/details/Catcher.cpp"


#endif
