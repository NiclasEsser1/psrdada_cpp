#ifndef RECEIVE_HPP_
#define RECEIVE_HPP_

#include <string>
#include <vector>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <atomic>
#include <functional>

#include <boost/asio.hpp>
#include <boost/asio/buffer.hpp>

#include "psrdada_cpp/dada_client_base.hpp"
#include "psrdada_cpp/double_buffer.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"

#include "psrdada_cpp/effelsberg/paf/capture/UDPSocket.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Types.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{


using namespace boost::asio;

extern boost::mutex lock_buffer;
extern DoubleBuffer<std::vector<unsigned char>> doubleBuffer; // Global double buffer


template<class HandlerType>
class ReceiveControl
{
public:
    ReceiveControl(capture_conf_t& conf, MultiLog& log, HandlerType& handle);
    ~ReceiveControl();
    void start();

    void watch_dog();
    void launch_capture();
    void stop();
    void clean();
    bool stop_thread(UDPSocket* obj);
    bool buffer_ready();
    void signal_to_worker(bool flag);
private:
    std::string state();

private:
    bool _quit = false;
    state_t _state;
    capture_conf_t& config;

    std::size_t bytes_written = 0;
    std::size_t total_bytes;

    std::ifstream dada_header_file;
    std::vector<char> raw_header;
    std::vector<Receiver*> receivers;

    boost::thread_group thread_group;

    ControlSocket *ctrl_socket;
    const DadaWriteClient &_dada_client;
    HandlerType& handler;
    MultiLog& logger;
};




}
}
}
}

#include "psrdada_cpp/effelsberg/paf/capture/details/Receive.cpp"

#endif
