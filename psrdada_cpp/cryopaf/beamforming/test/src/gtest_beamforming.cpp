#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuComplex.h>
#include "psrdada_cpp/cryopaf/beamforming/test/beamformer_tester.cuh"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
