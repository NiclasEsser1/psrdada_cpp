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
    printf("Initializing CaptureMonitor %d\n", _tid);
}
void CaptureMonitor::run()
{
    _active = true;
    printf("Running worker from CaptureMonitor %d\n", _tid);
    while(!_quit)
    {
        sleep(1);
    }
    _active = false;
}
void CaptureMonitor::clean()
{
    printf("Cleaning CaptureMonitor %d\n", _tid);
}
void CaptureMonitor::stop()
{
    // _writer->reset();
    printf("Stopping CaptureMonitor %d\n", _tid);
    _active = false;
    _quit = true;
}

}
}
}
}
