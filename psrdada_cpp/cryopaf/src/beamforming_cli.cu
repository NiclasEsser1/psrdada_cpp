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
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/multilog.hpp"

// #include "psrdada_cpp/cryopaf/Types.cuh"
#include "psrdada_cpp/cryopaf/VoltageBeamformer.cuh"
#include "psrdada_cpp/cryopaf/PowerBeamformer.cuh"
#include "psrdada_cpp/cryopaf/Unpacker.cuh"


const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;

using namespace psrdada_cpp;
using namespace psrdada_cpp::cryopaf;

void launch(bf_config_t& conf)
{
    MultiLog log(conf.logname);

    DadaOutputStream output (conf.out_key, log);

    if (conf.kind == "voltage")
    {
      VoltageBeamformer<decltype(output), RawVoltage<float2>, Weights<float2>, VoltageBeam<float2>> beamformer (conf, log, output);
      Unpacker<decltype(beamformer), RawVoltage<uint64_t>, RawVoltage<float2>> unpacker (conf, log, beamformer);
      DadaInputStream<decltype(unpacker)> input (conf.in_key, log, unpacker);
      input.start();
    }else if (conf.kind == "power"){
      PowerBeamformer<decltype(output), RawVoltage<float2>, Weights<float2>, PowerBeam<float>> beamformer (conf, log, output);
      Unpacker<decltype(beamformer), RawVoltage<uint64_t>, RawVoltage<float2>> unpacker (conf, log, beamformer);
      DadaInputStream<decltype(unpacker)> input (conf.in_key, log, unpacker);
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
        bf_config_t conf;

        int precision;
        std::string kind;
        std::string filename;

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
        ("antennas", po::value<std::size_t>(&conf.n_antenna)->default_value(32), "Number of antennas")
        ("pol", po::value<std::size_t>(&conf.n_pol)->default_value(2), "Polarisation")
        ("beams", po::value<std::size_t>(&conf.n_beam)->default_value(32), "Number of beams")
        ("interval", po::value<std::size_t>(&conf.interval)->default_value(1), "Beamform type:")
        ("device", po::value<int>(&conf.device_id)->default_value(0), "Device ID of GPU")
        ("kind", po::value<std::string>(&conf.kind)->default_value("power"), "Power or voltage BF")
        ("precision", po::value<int>(&precision)->default_value(1), "0 = half; 1 = single")
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
        CUDA_ERROR_CHECK( cudaSetDevice(conf.device_id) );
        launch(conf);
    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached the top of main: "
            << e.what() << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;
    }
    return SUCCESS;
}
