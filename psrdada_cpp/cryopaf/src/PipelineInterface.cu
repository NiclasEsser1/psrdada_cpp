#ifdef PIPELINE_INTERFACE_HPP

namespace psrdada_cpp{
namespace cryopaf{

namespace bip = boost::interprocess;

template<class T>
PipelineInterface<T>::PipelineInterface(PipelineInterfaceConfig& config, MultiLog& logger)
  : conf(config), log(logger)
{
  smem = bip::shared_memory_object(bip::open_only, "SharedMemoryWeights", bip::read_write);
  try
  {
    //Map the whole shared memory in this process
    region = bip::mapped_region(smem, bip::read_write);
    //Get the address of the mapped region
    smem_addr = region.get_address();
  }
  catch(bip::interprocess_exception &ex)
  {
     std::cout << ex.what() << std::endl;
     exit(1);
  }
  qheader = new (smem_addr) QueueHeader();
  smem_weights = &(static_cast<T*>(smem_addr)[sizeof(QueueHeader)]);
  vect_weights = std::vector<T>(conf.n_beam * conf.n_channel * conf.n_elements * conf.n_pol);
}

template<class T>
PipelineInterface<T>::~PipelineInterface()
{

}

template<class T>
void PipelineInterface<T>::run()
{
  while(!quit)
  {
    bip::scoped_lock<bip::interprocess_mutex> lock(qheader->mutex);

    if(qheader->data_in)
    {
       std::cout << "Waiting for reading weights from shared memory" << std::endl;
       qheader->ready_to_write.wait(lock);
    }
    this->update();
    std::cout << "Writing new weights to shared memory " << std::endl;
    memcpy(smem_weights, (void*)&vect_weights[0], vect_weights.size()*sizeof(T));
    qheader->data_in = true;
    qheader->stop = false;
    qheader->ready_to_read.notify_all();
  }
  qheader->data_in = true;
  qheader->stop = true;
}

template<class T>
SimpleWeightGenerator<T>::SimpleWeightGenerator(PipelineInterfaceConfig& config, MultiLog& logger)
  : PipelineInterface<T>(config, logger)
{

}

template<class T>
SimpleWeightGenerator<T>::~SimpleWeightGenerator()
{

}

template<class T>
void SimpleWeightGenerator<T>::update()
{
  if(this->conf.mode == "random")
  {
      random();
  }
  else if(this->conf.mode == "bypass")
  {

      if(this->update_cnt > 0)
      {
        this->quit = true;
      }
      bypass();
  }
  else
  {
      throw std::runtime_error("Not implemented yet.");
  }
  this->update_cnt += 1;
}

template<typename T>
void SimpleWeightGenerator<T>::bypass()
{
  std::size_t pos_x, pos_y;
  int be = this->conf.n_beam * this->conf.n_elements;
  int pbe = this->conf.n_pol * be;
  for(int f = 0; f < this->conf.n_channel; f++)
  {
    for(int b = 0; b < this->conf.n_beam; b++)
    {
      for(int e = 0; e < this->conf.n_elements; e++)
      {
        pos_x = f * pbe + 0 * be + b * this->conf.n_elements + e;
        pos_y = f * pbe + 1 * be + b * this->conf.n_elements + e;
        if(b == e)
        {
          this->vect_weights[pos_x] = {1,0}; // Set X pol
          this->vect_weights[pos_y] = {1,0}; // Set Y pol
        }
        else
        {
          this->vect_weights[pos_x] = {0,0};
          this->vect_weights[pos_y] = {0,0};
        }
      }
    }
  }
}

template<typename T>
void SimpleWeightGenerator<T>::random()
{
  std::default_random_engine generator;
  std::normal_distribution<float> normal_dist(0.0, 1.0);
  for (size_t i = 0; i < this->vect_weights.size(); i++)
  {
			this->vect_weights[i] = {normal_dist(generator), normal_dist(generator)};
  }
}

}
}
#endif
