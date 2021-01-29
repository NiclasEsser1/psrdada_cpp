#ifndef PACKET_TESTER_H_
#define PACKET_TESTER_H_

#include <string>
#include <vector>
#include <unistd.h>
#include <iostream>     // std::cout
#include "psrdada_cpp/effelsberg/paf/capture/Types.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{
namespace test{

#define RUNS 10

class PacketTester : public ::testing::Test
{
public:
  PacketTester();
  ~PacketTester();

  void test();
  void compare_header(DataFrame<codif_t>, DataFrame<codif_t>);
protected:
  void SetUp() override;
  void TearDown() override;

};


}
} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test

#include "psrdada_cpp/effelsberg/paf/capture/test/src/PacketTester.cpp"


#endif //PACKET_TESTER_H_
