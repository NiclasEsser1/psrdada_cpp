#ifndef CAPTURE_CONTROLLER_HPP_
#define CAPTURE_CONTROLLER_HPP_

#include <string>
#include <vector>
#include <unistd.h>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <atomic>
#include <functional>
#include "psrdada_cpp/dada_client_base.hpp"
#include "psrdada_cpp/double_buffer.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Threading.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Socket.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Types.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Catcher.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/CaptureMonitor.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{

boost::mutex lock_buffer;
DoubleBuffer<std::vector<char>> buffer; // Global double buffer

enum capture_state_t{ERROR_STATE = -1, STARTING, INITIALIZING, IDLE, CAPTURING, STOPPING, EXIT};
const size_t COMMAND_MSG_LENGTH = 16; // Message length in bytes received by capture control socket


class ControlSocket : public AbstractThread
{
public:
    ControlSocket(MultiLog& log, std::string addr, int port);
    ~ControlSocket();
    void init();
    void run();
    void clean();
    void stop();
    void state(capture_state_t val){_state = val;}
    capture_state_t state(){return _state;}
    Socket* socket(){return _sock;}
private:
    char command[COMMAND_MSG_LENGTH];
    Socket *_sock = nullptr;
    capture_state_t _state;
};


template<class HandlerType>
class CaptureController
{
public:
    CaptureController(capture_conf_t* conf, MultiLog& log, HandlerType& handle);
    ~CaptureController();
    void start();

    void watch_dog();
    void launch_capture();
    void stop();
    void clean();
    bool stop_thread(AbstractThread* obj);
    bool all_thread_rdy(bool flag);
    void signal_to_worker(bool flag);
private:
    std::string state();

private:
    bool _quit = false;
    capture_state_t _state;
    capture_conf_t *_conf = nullptr;

    std::ifstream input_file;
    std::vector<char> raw_header;
    std::vector<std::vector<std::size_t>*> temp_pos_list;
    std::size_t bytes_written = 0;
    std::size_t total_bytes;

    CaptureMonitor *_monitor;
    ControlSocket *_ctrl_socket;
    std::vector<Catcher*> catchers;
    boost::thread_group t_grp;


    const DadaWriteClient &_dada_client;
    HandlerType& handler;
    MultiLog& logger;
};

}
}
}
}

#include "psrdada_cpp/effelsberg/paf/capture/src/CaptureController.cpp"

#endif
