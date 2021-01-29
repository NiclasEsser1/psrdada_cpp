#ifndef THREADPOOL_HPP_
#define THREADPOOL_HPP_

#include "psrdada_cpp/multilog.hpp"
#include <boost/thread.hpp>


namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{

class AbstractThread
{
public:
    AbstractThread(MultiLog& log) : logger(log){}
    // abstract methods
    virtual void init() = 0;
    virtual void run() = 0;
    virtual void stop() = 0;
    virtual void clean() = 0;
    
    void id(int id){_tid = id;}
    int id(){return _tid;}
    boost::thread* thread(){return t;}
    void thread(boost::thread* td){t = td;};
    bool active(){return _active;}
    bool quit(){return _quit;}
// protected:
protected:
    boost::thread *t;
    MultiLog& logger;
    int _tid;
    bool _quit = false;
    bool _active = false;
};


}
}
}
}

#endif
