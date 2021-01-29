#ifndef TRANSMITTER_TESTER_H_
#define TRANSMITTER_TESTER_H_

#include <string>
#include <vector>
#include <unistd.h>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <atomic>
#include <functional>

#include "psrdada_cpp/effelsberg/paf/capture/Threading.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Socket.hpp"
#include "psrdada_cpp/effelsberg/paf/capture/Types.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{
namespace test{

class Transmitter : public AbstractThread{
public:
  Transmitter(capture_conf_t& conf, MultiLog& log, int port);
  ~Transmitter();
  void init();
  void run();
  void clean();
  void stop();

  void seconds(uint32_t val){_seconds = val;}
  void frame(uint32_t val){_frame_idx = val;}
  void epoch(uint32_t val){_epoch = val;}
  void freq(uint32_t val){_freq_idx = val;}
  void beam(uint32_t val){_beam_idx = val;}

private:
  bool _quit = false;
  Socket *_sock = nullptr;
  std::size_t packet_cnt = 0;
  DataFrame<codif_t> dframe;
  capture_conf_t& _conf;

  int _beams_per_thread;

  uint32_t _seconds;
  uint32_t _frame_idx;
  uint32_t _epoch;
  uint32_t _freq_idx;
  uint32_t _beam_idx;
};


}
} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test

#include "psrdada_cpp/effelsberg/paf/capture/test/src/Transmitter.cpp"

#endif //CAPTURE_TESTER_H_
