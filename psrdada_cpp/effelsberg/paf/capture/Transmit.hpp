#ifndef TRANSMIT_HPP_
#define TRANSMIT_HPP_


#include "psrdada_cpp/effelsberg/paf/capture/UDPSocket.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Types.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{




class TransmitController
{
public:
    TransmitController(MultiLog&, int, std::string, std::vector<int>&);
    ~TransmitController();
    void start(int);

    uint32_t* sync_epoch();

private:
    int beams;
    MultiLog& logger;
    std::string dest_address;
    std::vector<int>& port_list;
    std::vector<Transmitter*> transmitters;
    boost::thread_group* thread_grp;
};

}
}
}
}

#include "psrdada_cpp/effelsberg/paf/capture/details/Transmit.cpp"

#endif
