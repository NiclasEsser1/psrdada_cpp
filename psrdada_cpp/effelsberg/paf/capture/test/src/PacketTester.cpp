#ifdef PACKET_TESTER_H_

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{
namespace test{

PacketTester::PacketTester()
    : ::testing::Test()
{
}

PacketTester::~PacketTester()
{

}

void PacketTester::SetUp()
{
  std::cout << std::endl
    << "-------------------------------------------------------------" << std::endl
    << " 	Testing packing and unpacking " << std::endl
    << "-------------------------------------------------------------" << std::endl << std::endl;
}
void PacketTester::TearDown()
{

}
void PacketTester::test()
{
    uint32_t seconds = 1;
    uint32_t frame_idx = 0;
    uint32_t epoch = 12;
    uint32_t freq_idx = 1;
    uint32_t beam_idx = 1;
    char* msg;
    DataFrame<codif_t> df_to_send;
    DataFrame<codif_t> df_to_recv;

    msg = df_to_send.serialize();
    df_to_recv.deserialize(msg);
    compare_header(df_to_send, df_to_recv);
}
void PacketTester::compare_header(DataFrame<codif_t> send, DataFrame<codif_t> recv)
{
    ASSERT_TRUE(send.packet == recv.packet) << "Header of both frames are unqeual: " << std::endl;
}


/**
* Testing with Google Test Framework
*/

TEST_F(PacketTester, PacketTesterCodif){
  test();
}


}
} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test

#endif
