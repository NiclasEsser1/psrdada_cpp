#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/complex.h>
#include <cuda.h>
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>
#include <unordered_map>


#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/multilog.hpp"

#include "psrdada_cpp/cryopaf/Pipeline.cuh"


const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;

using namespace psrdada_cpp;
using namespace psrdada_cpp::cryopaf;

template<typename T>
void launch(PipelineConfig& conf)
{
    MultiLog log(conf.logname);
    DadaOutputStream output (conf.out_key, log);
    if (conf.mode == "voltage")
    {
      Pipeline<decltype(output), T, T> pipeline(conf, log, output);
      DadaInputStream<decltype(pipeline)> input(conf.in_key, log, pipeline);
      input.start();
    }else if (conf.mode == "power"){
      Pipeline<decltype(output), T, decltype(T::x)> pipeline(conf, log, output);
      DadaInputStream<decltype(pipeline)> input(conf.in_key, log, pipeline);
      input.start();
    }else{
      throw std::runtime_error("Not implemented yet.");
    }
}

int main(int argc, char** argv)
{
    try
    {
        // Variables to store command line options
        PipelineConfig conf;
        std::string precision;
        std::string kind;

        // Parse command line
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
        ("help,h", "Print help messages")
        ("in_key", po::value<std::string>()->required()
          ->notifier([&conf](std::string key)
              {
                  conf.in_key = string_to_key(key);
              }), "Input dada key")
        ("out_key", po::value<std::string>()->required()
          ->notifier([&conf](std::string key)
              {
                  conf.out_key = string_to_key(key);
              }), "Output dada key")
        ("samples", po::value<std::size_t>(&conf.n_samples)->default_value(262144), "Number of samples within one heap")
        ("channels", po::value<std::size_t>(&conf.n_channel)->default_value(7), "Number of channels")
        ("elements", po::value<std::size_t>(&conf.n_elements)->default_value(36), "Number of antennas")
        ("pol", po::value<std::size_t>(&conf.n_pol)->default_value(2), "Polarisation")
        ("beams", po::value<std::size_t>(&conf.n_beam)->default_value(36), "Number of beams")
        ("integration", po::value<std::size_t>(&conf.integration)->default_value(1), "Beamform type:")
        ("device", po::value<int>(&conf.device_id)->default_value(0), "Device ID of GPU")
        ("mode", po::value<std::string>(&conf.mode)->default_value("power"), "Power or voltage BF")
        ("precision", po::value<std::string>(&precision)->default_value("single"), "supported arguments: half, single, double")
        ("protocol", po::value<std::string>(&conf.protocol)->default_value("codif"), "Input protocol")
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
          launch<__half2>(conf);
        }
        else if(precision == "single")
        {
          launch<float2>(conf);
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
