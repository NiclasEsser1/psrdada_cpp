#include "gtest/gtest.h"
// #include "psrdada_cpp/cryopaf/beamforming/test/VoltageBeamformTester.cuh"
#include "psrdada_cpp/cryopaf/beamforming/test/PowerBeamformTester.cuh"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
