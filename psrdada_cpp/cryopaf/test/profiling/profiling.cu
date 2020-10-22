#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/complex.h>
#include <cuda.h>
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>

#include "psrdada_cpp/cryopaf/PowerBeamformer.cuh"
#include "psrdada_cpp/cryopaf/VoltageBeamformer.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/cryopaf/types.cuh"

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

using namespace psrdada_cpp;
using namespace psrdada_cpp::cryopaf;
using namespace std::chrono;

namespace
{
  const size_t ERROR_IN_COMMAND_LINE = 1;
  const size_t SUCCESS = 0;
  const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace


template<typename T, typename U>
void profile(bf_config_t conf, std::size_t iter, int dev_id, int kind, std::string filename)
{
    std::ofstream fid;
    high_resolution_clock::time_point start, stop;
    double ms, ms_avg;


    CUDA_ERROR_CHECK(cudaSetDevice(dev_id));

    // Calulate memory size for input, weights and output
    std::size_t input_size = conf.n_samples * conf.n_antenna * conf.n_channel * conf.n_pol;
    std::size_t weights_size =  conf.n_beam * conf.n_antenna * conf.n_channel * conf.n_pol;
    std::size_t output_size = conf.n_samples * conf.n_beam * conf.n_channel;
    std::size_t required_mem = input_size * sizeof(T)
        + weights_size * sizeof(T)
        + output_size * sizeof(T) * conf.n_pol;

    std::cout << "Required device memory: " << std::to_string(required_mem/(1024*1024)) << "MiB" << std::endl;
    std::cout << "Required host memory: " << std::to_string(2*required_mem/(1024*1024)) << "MiB" << std::endl;

    // If desired, create CSV file for benchmark
    if(!filename.empty())
    {
        bool exists =  boost::filesystem::exists(filename); // If file exists head line is not needed
        fid.open(filename, std::fstream::app);
        if(!exists)
            fid << "#timestamps,#Nchans,#Nants,#Npol,#Nelements,#Nbeams,#Naccumulate,bf_type,input_mb,weights_mb,output_mb,transfer_h2d,bw_h2d,upload_time,Benchmark(s),Performance(Tops/s)\n";
        fid << std::to_string(conf.n_samples) << "," << std::to_string(conf.n_channel) << ",";
        fid << std::to_string(conf.n_antenna) << "," << std::to_string(conf.n_pol) << ",";
        fid << std::to_string(conf.n_antenna*conf.n_pol) << "," << std::to_string(conf.n_beam) << ",";
        fid << std::to_string(conf.interval) << "," << std::to_string(conf.bf_type) << ",";
        fid << std::to_string(input_size*sizeof(T)/(1024*1024)) << "," << std::to_string(weights_size*sizeof(T)/(1024*1024)) << ",";
        fid << std::to_string(output_size*sizeof(U)/(1024*1024)) << ",";
    }

    // Allocate host vectors
    thrust::host_vector<T> host_input(input_size);
    thrust::host_vector<T> host_weights(weights_size);
    thrust::host_vector<U> host_output(1);
    thrust::device_vector<U> dev_output(1);


    std::cout << "Generating test samples... " << std::endl;
    // Generate test samples / normal distributed noise for input signal
    for (size_t i = 0; i < host_input.size(); i++)
    {
        host_input[i].x = .1;
        host_input[i].y = .1;
    }
    // Build complex weight as C * exp(i * theta).
    for (size_t i = 0; i < host_weights.size(); i++)
    {
        host_weights[i].x = .1;
        host_weights[i].y = .1;
    }

    // Allocate device memory & assign test samples
    // Input and weights are equal for host and device vector
    start = high_resolution_clock::now();
    thrust::device_vector<T> dev_input = host_input;
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    stop = high_resolution_clock::now();
    ms = ((double)duration_cast<microseconds>(stop - start).count())/1000;
    if(fid) fid << std::to_string(ms) << "," << std::to_string(input_size/ms) << ",";

    thrust::device_vector<T> dev_weights = host_weights;


    /** Voltage Beamformer **/
    if constexpr (std::is_same<T, U>::value)
    {
        host_output.resize(output_size * conf.n_pol);
        dev_output.resize(output_size * conf.n_pol);
        beamforming::VoltageBeamformer<T> bf(&conf, dev_id);
        for(int i = 0; i < iter; i++)
        {
            start = high_resolution_clock::now();
            bf.process(dev_input, dev_output, dev_weights);    // launches CUDA kernel
            CUDA_ERROR_CHECK(cudaDeviceSynchronize());
            stop = high_resolution_clock::now();
            ms = ((double)duration_cast<microseconds>(stop - start).count())/1000;
            // std::cout << "Elpased time (Power): " << std::to_string(ms) << " ms" << std::endl;
            ms_avg += ms;
        }
    /** Power Beamformer **/
    }else{
        host_output.resize(output_size / conf.interval);
        dev_output.resize(output_size / conf.interval);
        beamforming::PowerBeamformer<T, U> bf(&conf, dev_id);
        for(int i = 0; i < iter; i++)
        {
            start = high_resolution_clock::now();
            bf.process(dev_input, dev_output, dev_weights);    // launches CUDA kernel
            CUDA_ERROR_CHECK(cudaDeviceSynchronize());
            stop = high_resolution_clock::now();
            ms = ((double)duration_cast<microseconds>(stop - start).count())/1000;
            // std::cout << "Elpased time (Power): " << std::to_string(ms) << " ms" << std::endl;
            ms_avg += ms;
        }
    }



    if(fid) fid << std::to_string(ms_avg/iter) << "," << input_size*sizeof(T)/(ms_avg/iter) << "\n";
    if(fid) fid.close();
}


int main(int argc, char** argv)
{
    try
    {
        // Variables to store command line options
        cryopaf::bf_config_t conf;
        int iter;
        int dev_id;
        int kind;
        int precision;
        std::string filename;

        // Parse command line
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
        ("help,h", "Print help messages")
        ("samples", po::value<std::size_t>(&conf.n_samples)->default_value(1024), "Number of samples within one heap")
        ("channels", po::value<std::size_t>(&conf.n_channel)->default_value(256), "Number of channels")
        ("antennas", po::value<std::size_t>(&conf.n_antenna)->default_value(WARP_SIZE*4), "Number of antennas")
        ("pol", po::value<std::size_t>(&conf.n_pol)->default_value(2), "Polarisation")
        ("beams", po::value<std::size_t>(&conf.n_beam)->default_value(64), "Number of beams")
        ("interval", po::value<std::size_t>(&conf.interval)->default_value(64), "Beamform type:")
        ("type", po::value<std::size_t>(&conf.bf_type)->default_value(0), "Beamform type:")
        ("kind", po::value<int>(&kind)->default_value(0), "Power (0) or voltage (1) BF")
        ("id", po::value<int>(&dev_id)->default_value(0), "Beamform type:")
        ("iteration", po::value<int>(&iter)->default_value(1), "Beamform type:")
        ("precision", po::value<int>(&precision)->default_value(1), "0 = half; 1 = single")
        ("outputfile", po::value<std::string>(&filename)->default_value(""), "Store profile data to csv file");

        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if ( vm.count("help")  )
            {
                std::cout << "Cryopaf -- The CryoPAF beamformer implementations" << std::endl
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


        if(kind == 0)
        {
            if(precision == 0)
                profile<__half2, __half>(conf, iter, dev_id, kind, filename);
            else if(precision == 1)
                profile<float2, float>(conf, iter, dev_id, kind, filename);
        }
        else if(kind == 1)
        {
            if(precision == 0)
                profile<__half2, __half2>(conf, iter, dev_id, kind, filename);
            else if(precision == 1)
                profile<float2, float2>(conf, iter, dev_id, kind, filename);
        }



    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached the top of main: "
            << e.what() << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;
    }
  return 0;
}
