#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cryopaf/PipelineInterface.cuh"


const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;

using namespace psrdada_cpp;
using namespace psrdada_cpp::cryopaf;

template<typename T>
void launch(PipelineInterfaceConfig& conf)
{
    MultiLog log(conf.logname);
    if (conf.mode == "bypass" || conf.mode == "random")
    {
      SimpleWeightGenerator<T> handle(conf, log);
      handle.run();
    }else if (conf.mode == "tcp"){
      throw std::runtime_error("Not implemented yet.");
    }else if (conf.mode == "file"){
      throw std::runtime_error("Not implemented yet.");
    }else{
      throw std::runtime_error("Not implemented yet.");
    }
}

int main(int argc, char** argv)
{
    try
    {
        // Variables to store command line options
        PipelineInterfaceConfig conf;
        std::string precision;

        // Parse command line
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
        ("help,h", "Print help messages")
        ("channels", po::value<std::size_t>(&conf.n_channel)->default_value(7), "Number of channels")
        ("elements", po::value<std::size_t>(&conf.n_elements)->default_value(32), "Number of antennas")
        ("pol", po::value<std::size_t>(&conf.n_pol)->default_value(2), "Polarisation")
        ("beams", po::value<std::size_t>(&conf.n_beam)->default_value(32), "Number of beams")
        ("mode", po::value<std::string>(&conf.mode)->default_value("bypass"), "Power or voltage BF")
        ("precision", po::value<std::string>(&precision)->default_value("single"), "Supported types: half, single, double")
        ("log", po::value<std::string>(&conf.logname)->default_value("cryo_beamform.log"), "Store profile data to csv file");

        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if ( vm.count("help")  )
            {
                std::cout << "Cryopaf -- The CryoPAF Controller implementations" << std::endl
                << desc << std::endl;
                return SUCCESS;
            }
            po::notify(vm);
        }
        catch(po::error& e)
        {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << desc << std::endl;
            return ERROR_IN_COMMAND_LINE;
        }
        if(precision == "half")
        {
          std::cout << "Not implemented yet" << std::endl;
        }
        else if(precision == "single")
        {
          launch<float2>(conf);
        }
        else if(precision == "double")
        {
          std::cout << "Not implemented yet" << std::endl;
        }
        else
        {
          std::cout << "Type not known" << std::endl;
        }
    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached the top of main: "
            << e.what() << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;
    }
    return SUCCESS;
}
