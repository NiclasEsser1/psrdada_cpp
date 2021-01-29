#include "psrdada_cpp/effelsberg/paf/capture/CaptureMonitor.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{

CaptureMonitor::CaptureMonitor(key_t key, MultiLog& log)
    : AbstractThread(log),
    _key(key)
{
}
CaptureMonitor::~CaptureMonitor()
{
}
void CaptureMonitor::init()
{
    BOOST_LOG_TRIVIAL(debug) << "Initializing CaptureMonitor " << _tid;
}
void CaptureMonitor::run()
{
    _active = true;
    BOOST_LOG_TRIVIAL(debug) << "Running worker from CaptureMonitor " << _tid;
    while(!_quit)
    {
        sleep(1);
    }
    _active = false;
}
void CaptureMonitor::clean()
{
    BOOST_LOG_TRIVIAL(debug) << "Cleaning CaptureMonitor " << _tid;
}
void CaptureMonitor::stop()
{
    // _writer->reset();
    BOOST_LOG_TRIVIAL(debug) << "Stopping CaptureMonitor " << _tid;
    _active = false;
    _quit = true;
}

}
}
}
}
