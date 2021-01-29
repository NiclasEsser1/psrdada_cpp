#ifdef CAPTURE_TESTER_H_

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{
namespace test{

CaptureTester::CaptureTester()
    : ::testing::Test()
{

}

CaptureTester::~CaptureTester()
{

}

void CaptureTester::SetUp()
{
	int result;

	conf.key = 0xdada;
	conf.capture_addr = "127.0.0.1";
	conf.capture_ctrl_addr = "127.0.0.1";
	conf.psrdada_header_file = "/home/psrdada_cpp/log/header_data.txt";
	conf.log = "/home/psrdada_cpp/log/" + program_name + ".log";
	conf.capture_ctrl_port = 17099;
	conf.capture_ctrl_cpu_bind = 1;
	conf.buffer_ctrl_cpu_bind = 2;
	conf.nbeam=36;
	conf.capture_ports = {17100};//, 17101};
	conf.capture_cpu_bind = {3};//, 4};
	conf.frame_size = 7232;
	conf.offset = 0;
	conf.n_buffers = 8;
	conf.n_catcher = 1;  // number of catching threads
	conf.rbuffer_size = 266600448;

	std::stringstream cmd;
	cmd << "dada_db -d -k " << std::hex << conf.key;
	BOOST_LOG_TRIVIAL(info) << "Executing command: " << cmd.str();
	result = system(cmd.str().c_str());
	cmd.str("");
	cmd << "dada_db"
		<< " -b " << std::dec << conf.rbuffer_size
		<< " -n " << conf.n_buffers
		<< " -k " << std::hex << conf.key;
	BOOST_LOG_TRIVIAL(info) << "Executing command: " << cmd.str();
	result = system(cmd.str().c_str());
}
void CaptureTester::TearDown()
{

}
void CaptureTester::test()
{
	MultiLog logger(program_name);
	FILE* fid; // Create file stream
	if( (fid = fopen(conf.log.c_str(), "a")) == NULL) // open and check if valid. If file already exists logging will appended
	{
		throw std::runtime_error("IOError: Not able to open log file\n");
	}
	multilog_t* ulogger = logger.native_handle();
	multilog_add(ulogger, fid);

	std::vector<Transmitter*> transmitters;
    boost::thread_group thread_group;

	DadaOutputStream ostream(conf.key, logger);

	CaptureController<decltype(ostream)> ctrl(&conf, logger, ostream);

	Socket transmit_ctrl(logger, conf.capture_ctrl_addr, conf.capture_ctrl_port, false, SOCK_STREAM);

	int tid = 0;
    for( auto& port : conf.capture_ports )
    {
        transmitters.push_back(new Transmitter(conf, logger, port));
        transmitters.back()->id(tid);
        transmitters.back()->init();
		transmitters.back()->thread( new boost::thread(boost::bind(&AbstractThread::run, transmitters.back())) );
        tid++;
		thread_group.add_thread(transmitters.back()->thread());
    }

	boost::thread *ctrl_thread = new boost::thread(boost::bind(&CaptureController<decltype(ostream)>::start, &ctrl));
	thread_group.add_thread(ctrl_thread);
	usleep(1000000);
	transmit_ctrl.open_connection();
	usleep(1000000);
	if(transmit_ctrl.transmit("START", 5)){
		BOOST_LOG_TRIVIAL(info) << "Succesful send START command";
	}else{
		BOOST_LOG_TRIVIAL(info) << "Unsuccesful send START command";
	}

	// usleep(1000000);
	usleep(1000);
	thread_group.join_all();
	transmit_ctrl.close_connection();
	BOOST_LOG_TRIVIAL(info) << "Cleaning all\n";
}


/**
* Testing with Google Test Framework
*/

TEST_F(CaptureTester, CaptureTesterCodif){
  std::cout << std::endl
    << "-------------------------------------------------------------" << std::endl
    << " Testing capturing with CODIF " 					<< std::endl
    << "-------------------------------------------------------------" << std::endl << std::endl;
  test();
}



}
} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test

#endif
