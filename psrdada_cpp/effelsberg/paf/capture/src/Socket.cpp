#include "psrdada_cpp/effelsberg/paf/capture/Socket.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{

Socket::Socket(MultiLog &log, std::string addr, int port, bool role, int type, int family, int protocol)
    :
    logger(log),
    _addr(addr),
	_type(type),
    _port(port),
    _family(family),
    _protocol(protocol),
    _role(role)
{
    _state = 0;
    _sock_conf.sin_family = _family;
    _sock_conf.sin_port = htons(_port);
    inet_pton(_sock_conf.sin_family, _addr.c_str(), &_sock_conf.sin_addr);
	// Get socket stream
	_sock = socket(_family, _type, _protocol);
	if(_sock < 0)
	{
		logger.write(LOG_ERR, "socket(%d, %d, %d) failed: %s (File: %s line: %d)\n", _family, _type,_protocol, strerror(errno), __FILE__, __LINE__);
		_state = -1;
	}
}
Socket::~Socket()
{
    close_connection();
}
bool Socket::open_connection(int nof_clients, int reuse)
{
	// Allow socket address to be reused
	// if( setsockopt(_sock, SOL_SOCKET, SO_REUSEADDR, (void*)&reuse, sizeof(reuse)) < 0 )
	// {
	//     logger.write(LOG_ERR, "setsockopt() failed: %s (File: %s line: %d)\n", strerror(errno), __FILE__, __LINE__);
	// }

	// If socket should be server
    if(_role)
    {
		if( bind(_sock, (struct sockaddr *)&_sock_conf, sizeof(_sock_conf)) < 0 )
		{
			logger.write(LOG_ERR, "bind() at %s:%d failed: %s (File: %s line: %d\n)", _addr.c_str(), _port, strerror(errno), __FILE__, __LINE__);
			_state = -1;
			return false;
		}
		if( listen(_sock, nof_clients) < 0 )
		{
		   	logger.write(LOG_ERR, "listen() at %s:%d failed: %s (File: %s line: %d\n)", strerror(errno), _addr.c_str(), _port, __FILE__, __LINE__);
			_state = -1;
			return false;
		}

		BOOST_LOG_TRIVIAL(info) << "Waiting for client to connect ...";

		logger.write(LOG_INFO, "Waiting for controlling client request %s:%d ...\n)", _addr.c_str(), _port);

		if( (_client = accept(_sock, (struct sockaddr *)&_sock_conf, &_addrlen)) < 0 )
		{
		   	logger.write(LOG_ERR, "accept() at %s:%d failed: %s (File: %s line: %d\n)", _addr.c_str(), _port, strerror(errno), __FILE__, __LINE__);
			_state = -1;
			return false;
		}
		logger.write(LOG_INFO, "Controlling client connected\n)");

		BOOST_LOG_TRIVIAL(info) << "Client connected on " << _addr << ":" << _port;
	// Is socket plays client role
    }else{
	    if( connect(_sock, (struct sockaddr *)&_sock_conf, sizeof(_sock_conf)) < 0 )
	    {
	        logger.write(LOG_ERR, "connect() at %s:%d failed: %s (File: %s line: %d\n)", strerror(errno), _addr.c_str(), _port, __FILE__, __LINE__);
	        _state = -1;
	        return false;
	    }
	}
    _state = 1;
    return true;
}
bool Socket::close_connection()
{
	BOOST_LOG_TRIVIAL(info) << "Closing socket connection on " << _addr << ":" << _port;
    shutdown(_sock, 2);
	close(_sock);
}

// template<typename T>
int Socket::receive(char* ptr, std::size_t size, int flag, struct sockaddr* _from_sock, socklen_t* _fromlen)
{
	if( recvfrom(_sock, (void *)ptr, size, flag, _from_sock, _fromlen) < 0 )
	{
		logger.write(LOG_INFO, "No message received %s:%d (File: %s line: %d)", _addr, _port, __FILE__, __LINE__);
		return false;
	}
	return true;
}
int Socket::reading(char* ptr, std::size_t size)
{
	if(_role)
	{
		return read(_client, ptr, size);
	}else{
		return read(_sock, ptr, size);
	}
}

bool Socket::transmit(const char* msg, std::size_t size, int flag, const struct sockaddr *dest_addr, socklen_t addrlen)
{
	int result = 0;
	if ( (result = sendto(_sock, msg, size, flag, dest_addr, addrlen)) == -1)
	{
		BOOST_LOG_TRIVIAL(error) << "Error when transmitting to " << _addr << ":" << _port;
		logger.write(LOG_ERR, "send() at %s:%d failed: %s (File: %s line: %d\n)", strerror(errno), _addr.c_str(), _port, __FILE__, __LINE__);
		_state = -1;
		return false;
	}
	return true;
}

// bool Socket::transmit(const char* msg, std::size_t size)
// {
// 	if ( write(_sock, msg, size) != size)
// 	{
// 		BOOST_LOG_TRIVIAL(error) << "Error when transmitting to " << _addr << ":" << _port;
// 		logger.write(LOG_ERR, "send() at %s:%d failed: %s (File: %s line: %d\n)", strerror(errno), _addr.c_str(), _port, __FILE__, __LINE__);
// 		_state = -1;
// 		return false;
// 	}
// 	return true;
// }


bool Socket::set_timeout(struct timeval tout)
{
	if( setsockopt(_sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tout, sizeof(tout)) < 0)
	{
		logger.write(LOG_ERR, "Could not set timeout for socket %s:%d (File: %s line: %d\n)", _addr, _port, __FILE__, __LINE__);
		return false;
	}
	return true;
}

}
}
}
}
