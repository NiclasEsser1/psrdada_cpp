#ifndef Sockets_HPP_
#define Sockets_HPP_

#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

// error messages
#include <errno.h>
#include <string.h>

#include "psrdada_cpp/multilog.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{

// POSIX socket
class Socket
{
public:
    Socket(MultiLog &log, std::string addr, int port, bool role = true, int type = SOCK_DGRAM, int family = AF_INET, int protocol = 0);

    ~Socket();

    bool open_connection(int nof_clients = 1, int reuse = 1);

    bool close_connection();

    int reading(char* ptr, std::size_t size);

    // template<typename T>
    int receive(char* ptr, std::size_t size, int flag = 0, struct sockaddr* _from_sock = nullptr, socklen_t* _fromlen = nullptr);

    bool transmit(const char* msg, std::size_t size, int flag = 0,  const struct sockaddr* dest_addr = NULL, socklen_t addrlen = 0);

    bool set_timeout(struct timeval tout);

    std::string address(){return _addr;}
    int port(){return _port;}
    int state(){return _state;}
    struct sockaddr_in *sockaddr_in(){return &_sock_addr_in;}
    struct sockaddr *sockaddr(){return &_sock_addr_in;}
    socklen_t addrlen(){return _addrlen;}
private:
    int _state;
    int _sock;
    int _type;
    int _family;
    int _protocol;
    int _port;
    bool _role; // Server (=1) or client (=0) role
    bool _active = false;
    std::string _addr;
    struct sockaddr_in _sock_addr_in; // Struct for creating POSIX socket
    struct sockaddr _sock_addr; // Struct for creating POSIX socket
    socklen_t _addrlen;
    MultiLog& logger;
    int _client = 0;
};


}
}
}
}

#endif
