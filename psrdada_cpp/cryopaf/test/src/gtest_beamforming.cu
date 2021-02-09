#include <gtest/gtest.h>
#include "psrdada_cpp/cryopaf/test/VoltageBeamformTester.cuh"
// #include "psrdada_cpp/cryopaf/test/PowerBeamformTester.cuh"
#include "psrdada_cpp/cryopaf/test/UnpackerTester.cuh"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
