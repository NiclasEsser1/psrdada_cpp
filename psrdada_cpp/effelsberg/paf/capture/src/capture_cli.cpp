#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/dada_write_client.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"

#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"

#include "multilog.h" // from psrdada

#include <sys/types.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include "psrdada_cpp/effelsberg/paf/capture/Types.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/CaptureController.hpp"

const std::string program_name = "capture_cli";
const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;

using namespace psrdada_cpp;
using namespace psrdada_cpp::effelsberg::paf::capture;

std::vector<std::size_t> seperate_by(std::string str, std::string expr)
{
    std::vector<std::size_t> res;
    if(str.find(expr) != std::string::npos)
    {

        std::vector<std::string> split;
        boost::split(split, str, boost::is_any_of(expr));
        for(auto& it : split)
        {
            res.push_back(std::stoi(it));
        }
    }else{
        res.push_back(std::stoi(str));
        res.push_back(-1);
    }
    return res;
}

int main(int argc, char** argv)
{

    // Variables to store command line options
    capture_conf_t conf;
    try
    {

        // Parse command line
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
        ("help,h", "Print help messages")
        ("key,k", po::value<std::string>()
            ->default_value("dada")
            ->notifier([&conf](std::string key)
            {
                conf.key = string_to_key(key);
            }),
            "The shared memory (hex string) for the dada buffer for capturing"
        )
        ("addr,a", po::value<std::string>(&conf.capture_addr)
            ->default_value("127.0.0.1"),
            "The address to which sockets are listening. It is assumed each socket listen to the same address"
        )
        ("log,l", po::value<std::string>()
            ->default_value("/home/psrdada_cpp/log")
            ->notifier([&conf](std::string val)
            {
                // 0x5c == "/" (UNIX ASCII)
                if( val.back() != 0x5C )
                    val += "/";
                conf.log = val + program_name + ".log";
            }),
            "Directory for logging. NOTE: don't provide the logging name, will be automatically generated."
        )
        ("header_file,f", po::value<std::string>(&conf.psrdada_header_file)
            ->default_value("/home/psrdada_cpp/log/header_data.txt"),
            "Header file containing psrdada header data"
        )
        ("catch_ports,p", po::value<std::string>()
            ->default_value("17100:2,17101:3")
            ->notifier([&conf](std::string val)
            {
                std::vector<std::string> split;
                boost::split(split, val, boost::is_any_of(","));
                for(int i = 0; i < split.size(); ++i)
                {
                    std::vector<std::size_t> result = seperate_by(split.at(i), ":");
                    conf.capture_ports.push_back(result[0]);
                    conf.capture_cpu_bind.push_back(result[1]);
                    conf.n_catcher = i+1;
                }
            }),
            "The ports for capturing UDP packets. Each passed port starts a thread. To bind the thread to a cpu core seperate the port with : (e.g. 17100:1 would bind a thread to CPU1 and listen on port 17100)"
        )
        ("control_port,cp", po::value<std::string>()
            ->default_value("17099:1")
            ->notifier([&conf](std::string val)
            {
                std::vector<std::size_t> result = seperate_by(val, ":");
                conf.capture_ctrl_port = result.at(0);
                conf.capture_ctrl_cpu_bind = result.at(1);
            }),
            "The addres for control socket. Allows to receive commands during capturing."
        )
        ("control_addr,cp", po::value<std::string>(&conf.capture_ctrl_addr)
            ->default_value("127.0.0.1"),
            "The addres for control socket. Allows to receive commands during capturing."
        )
        ("buf_bind,bb", po::value<std::size_t>(&conf.buffer_ctrl_cpu_bind)
            ->default_value(-1),
            "The port for control socket. Allows to receive commands during capturing."
        )
        ("nbeam,b", po::value<std::size_t>(&conf.nbeam)
            ->default_value(36),
            "Number of beams/elements (polarization excluded)"
        )
        ("offset,o", po::value<std::size_t>(&conf.offset)
            ->default_value(0),
            "Skip bytes in data packets (reduces the datarate, but will skip header data)"
        )
        ("threshold,t", po::value<std::size_t>(&conf.threshold)
            ->default_value(0),
            "Threshold value to define lost data packets"
        )
        ("temp_size,s", po::value<std::size_t>(&conf.nframes_tmp_buffer)
            ->default_value(128),
            "Maximum number of dataframe within the temporary buffer"
        )
        ("reference,r", po::value<std::string>()
            ->default_value("-1:-1")
            ->notifier([&conf](std::string val)
            {
                std::vector<std::size_t> result = seperate_by(val, ":");
                conf.dataframe_ref = result.at(0);
                conf.sec_ref = result.at(1);
            }),
            "Reference information dataframe ID and reference seconds (e.g. ). If not passed, the first received packet is used as reference."
        );

        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if ( vm.count("help")  )
            {
                  std::cout << "CAPTURING -- The BMF capturing program for bypassing the beamformer" << std::endl
                  << desc << std::endl;
                  return SUCCESS;
            }
            po::notify(vm);

            /**
            --------------------------
            Application code is below
            --------------------------
            **/

            // Multilog offers no possibilty to add a file stream to the underlying multilog_t object. Therefore get the underlying object and set it.
            MultiLog logger(program_name);
            FILE* fid; // Create file stream
            if( (fid = fopen(conf.log.c_str(), "a")) == NULL) // open and check if valid. If file already exists logging will appended
            {
                throw std::runtime_error("IOError: Not able to open log file\n");
            }
            multilog_t* ulogger = logger.native_handle();
            multilog_add(ulogger, fid);
            conf.print();
            DadaOutputStream ostream(conf.key, logger);
            CaptureController<decltype(ostream)> ctrl(ostream, &conf, logger);
            ctrl.start();
        }
        catch(po::error& e)
        {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << desc << std::endl;
            return ERROR_IN_COMMAND_LINE;
        }

      }
      catch(std::exception& e)
      {
          std::cerr << "Unhandled Exception reached the top of main: "
              << e.what() << ", application will now exit. (See " << conf.log << " for more informations)" << std::endl;
          return ERROR_UNHANDLED_EXCEPTION;
      }

     return 0;
}
