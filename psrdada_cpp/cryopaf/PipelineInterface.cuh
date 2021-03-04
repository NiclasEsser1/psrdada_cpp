#ifndef PIPELINE_INTERFACE_HPP
#define PIPELINE_INTERFACE_HPP

#include <vector>
#include <string>
#include <unistd.h>
#include <random>
#include <cmath>
#include <complex>

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/thread.hpp>

#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cryopaf/QueueHeader.hpp"

namespace psrdada_cpp{
namespace cryopaf{

namespace bip = boost::interprocess;

struct PipelineInterfaceConfig{
   std::string logname;
   std::size_t n_channel;
   std::size_t n_elements;
   std::size_t n_pol;
   std::size_t n_beam;
   std::string mode;
   void print()
   {
     std::cout << "Pipeline interface configuration" << std::endl;
     std::cout << "logname: " << logname << std::endl;
     std::cout << "n_channel: " << n_channel << std::endl;
     std::cout << "n_elements: " << n_elements << std::endl;
     std::cout << "n_pol: " << n_pol << std::endl;
     std::cout << "n_beam: " << n_beam << std::endl;
     std::cout << "mode: " << mode << std::endl;
   }
};

template<class T>
class PipelineInterface
{
public:
  PipelineInterface(PipelineInterfaceConfig& config, MultiLog& logger);
  ~PipelineInterface();
  void run();
  virtual void update() = 0;

private:
  std::string smem_name = "SharedMemoryWeights";

  bip::shared_memory_object smem;
  bip::mapped_region region;
  void* smem_addr = nullptr;
  T* smem_weights = nullptr;
  QueueHeader *qheader;

protected:
  PipelineInterfaceConfig& conf;
  MultiLog& log;
  std::vector<T> vect_weights;
  bool quit = false;
  std::size_t update_cnt = 0;
};


template<class T>
class SimpleWeightGenerator : public PipelineInterface<T>
{
public:
  SimpleWeightGenerator(PipelineInterfaceConfig& config, MultiLog& logger);
  ~SimpleWeightGenerator();
  void update();
private:
  void bypass();
  void random();
};

}
}
#include "psrdada_cpp/cryopaf/src/PipelineInterface.cu"
#endif // end PIPELINE_INTERFACE_HPP
