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
#include "psrdada_cpp/effelsberg/paf/capture/Transmit.hpp"

const std::string program_name = "codifsend_cli";
const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;

using namespace psrdada_cpp;
using namespace psrdada_cpp::effelsberg::paf::capture;


int main(int argc, char** argv)
{
    // Variables to store command line options
    std::vector<int> ports;
    std::string dest_addr;
    std::string fname_log = "/home/psrdada_cpp/log/" + program_name + ".log";
    int nbeam;

    try
    {
        // Parse command line
        namespace po = boost::program_options;
        po::options_description desc("Options");

        desc.add_options()
        ("help,h", "Print help messages")
        ("addr,a", po::value<std::string>(&dest_addr)
            ->default_value("127.0.0.1"),
            "The address to send codif packets."
        )
        ("port,p", po::value<std::string>()
            ->default_value("17100,17101")
            ->notifier([&ports](std::string val)
            {
                std::vector<std::string> split;
                boost::split(split, val, boost::is_any_of(","));
                for(int i = 0; i < split.size(); ++i)
                {
                    ports.push_back(std::stoi(split.at(i)));
                }
            }),
            "Destination ports of codif packets."
        )
        ("nbeam,b", po::value<int>(&nbeam)
            ->default_value(36),
            "Number of beams/elements"
        );

        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if ( vm.count("help")  )
            {
                  std::cout << "CAPTURING -- The PAF capture programm" << std::endl
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
            if( (fid = fopen(fname_log.c_str(), "a")) == NULL) // open and check if valid. If file already exists logging will appended
            {
                throw std::runtime_error("IOError: Not able to open log file\n");
            }
            multilog_t* ulogger = logger.native_handle();
            multilog_add(ulogger, fid);

            TransmitController ctrl(logger, nbeam, dest_addr, ports);
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
              << e.what() << ", application will now exit. (See " << fname_log << " for more informations)" << std::endl;
          return ERROR_UNHANDLED_EXCEPTION;
      }

     return 0;
}
