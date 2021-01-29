#include "gtest/gtest.h"
#include "psrdada_cpp/effelsberg/paf/capture/test/CaptureTester.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/test/PacketTester.hpp"
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
