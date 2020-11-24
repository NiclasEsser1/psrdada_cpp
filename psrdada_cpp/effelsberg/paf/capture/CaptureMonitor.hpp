#ifndef BUFFER_CONTROLLER_HPP_
#define BUFFER_CONTROLLER_HPP_

#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/dada_write_client.hpp"
#include "psrdada_cpp/raw_bytes.hpp"

#include "psrdada_cpp/effelsberg/paf/capture/Threading.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Socket.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Types.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{


class CaptureMonitor : public AbstractThread
{
public:


public:
    CaptureMonitor(key_t key, MultiLog& log);
    ~CaptureMonitor();
    void init();
// protected:
    void run();
    void clean();
    void stop();
    void disconnect_psrdada();

private:
    key_t _key;
};



}
}
}
}

#endif
