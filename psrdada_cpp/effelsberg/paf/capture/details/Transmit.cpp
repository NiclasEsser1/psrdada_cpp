#ifdef TRANSMIT_HPP_

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{


TransmitController::TransmitController(MultiLog& log, int nbeams, std::string addr, std::vector<int>& ports)
    : logger(log), beams(nbeams), dest_address(addr), port_list(ports)
{
    int tid = 0;
    int beams_per_node = (int)nbeams/ports.size();
    for( auto& port : ports )
    {
        transmitters.push_back( new Transmitter(logger, dest_address, port, tid, beams_per_node) );
        tid++;
    }

}

TransmitController::~TransmitController()
{

}

void TransmitController::start(int n_periods = 5)
{
    // Start worker for all capture threads
    thread_grp = new boost::thread_group();
    for(auto& obj : transmitters)
    {
        obj->thread( new boost::thread(boost::bind(&Transmitter::run, obj)) );
        thread_grp->add_thread(obj->thread());
    }

    thread_grp->join_all();

    for(auto& obj : transmitters)
    {
        thread_grp->remove_thread(obj->thread());
    }
    BOOST_LOG_TRIVIAL(debug) << "Transmitters have finshed their jobs ";

}

} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test


#endif
