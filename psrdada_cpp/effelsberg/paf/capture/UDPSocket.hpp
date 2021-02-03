#ifndef UDPSOCKET_H_
#define UDPSOCKET_H_

#include <iostream>
#include <atomic>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/asio/buffer.hpp>

#include "psrdada_cpp/double_buffer.hpp"
#include "psrdada_cpp/multilog.hpp"

#include "psrdada_cpp/effelsberg/paf/capture/Types.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{

boost::mutex lock_buffer;
DoubleBuffer<std::vector<unsigned char>> doubleBuffer;

using namespace boost::asio;

enum state_t{ERROR_STATE = -1, STARTING, INITIALIZING, IDLE, CAPTURING, STOPPING, EXIT};
const size_t COMMAND_MSG_LENGTH = 16; // Message length in bytes received by capture control socket

class UDPSocket
{
public:
    UDPSocket(MultiLog& log, std::string addr, int port, int tid = 0);
    ~UDPSocket();

    virtual void run() = 0;
    virtual void stop() = 0;

    void id(int id){_tid = id;}
    int id(){return _tid;}
    boost::thread* thread(){return t;}
    void thread(boost::thread* td){t = td;};
    void delete_thread(){delete t;}
    bool active(){return _active;}
    bool quit(){return _quit;}
// protected:
protected:
    boost::thread *t;
    io_service io_context;
    ip::udp::endpoint* endpoint;
    ip::udp::socket* socket;
    std::string address;
    int _port;

    MultiLog& logger;
    int _tid;
    bool _quit = false;
    bool _active = false;
    std::size_t packet_cnt = 0;
};

class Transmitter : public UDPSocket{
public:
  Transmitter(MultiLog& log, std::string addr, int port, int tid, int nbeams = 36, int periods = 5);
  ~Transmitter();
  void run();
  void clean();
  void stop();

  void beam(uint32_t val){dframe.packet.hdr.beam = val;}

  uint32_t beam(){return dframe.packet.hdr.beam;}
  uint32_t frame(){return dframe.packet.hdr.ref_idf;}


private:
  void increment_dframe();

private:
  DataFrame<codif_t> dframe;
  int period;
  int period_cnt = 0;
  int nbeam;
  int f_beam;
  int l_beam;
};

class Receiver : public UDPSocket{
public:
  Receiver(MultiLog& log, std::string addr, int port, int tid, capture_conf_t& conf);
  ~Receiver();

  void run();
  void stop();

  bool buffer_complete(){return completed;}

private:
  void calculate_position();

private:
  DataFrame<codif_t> dframe;
  long int position;
  std::size_t lost_packets = 0;
  std::size_t overflow_packets = 0;
  capture_conf_t& config;

  bool completed = false;
};


class ControlSocket : public UDPSocket
{
public:
    ControlSocket(MultiLog& log, std::string addr, int port);
    ~ControlSocket();

    void run();
    void stop();

    void state(state_t val){_state = val;}
    state_t state(){return _state;}
private:
  void handle_write(const boost::system::error_code&, size_t);

private:
    char command[COMMAND_MSG_LENGTH];
    state_t _state;
};


} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test

#include "psrdada_cpp/effelsberg/paf/capture/details/UDPSocket.cpp"

#endif //UDPSOCKET_H_
