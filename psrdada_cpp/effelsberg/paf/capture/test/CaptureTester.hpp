#ifndef CAPTURE_TESTER_H_
#define CAPTURE_TESTER_H_

#include <string>
#include <vector>
#include <unistd.h>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <atomic>
#include <functional>
#include <boost/asio.hpp>

#include "psrdada_cpp/dada_client_base.hpp"
#include "psrdada_cpp/double_buffer.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/cli_utils.hpp"

#include "psrdada_cpp/effelsberg/paf/capture/Threading.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Types.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Catcher.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/CaptureMonitor.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/CaptureController.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Transmitter.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{
namespace test{


const std::string program_name = "capture_test";

class CaptureTester : public ::testing::Test
{
public:
  CaptureTester();
  ~CaptureTester();

  void test();
  void receive();
  void transmit();
  void compare();


protected:
  void SetUp() override;
  void TearDown() override;

private:
  capture_conf_t conf;
};


}
} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test

#include "psrdada_cpp/effelsberg/paf/capture/test/src/CaptureTester.cpp"


#endif //CAPTURE_TESTER_H_
