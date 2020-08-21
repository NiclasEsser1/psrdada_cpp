#ifndef CUDACOLORMAP_CUH_
#define CUDACOLORMAP_CUH_


#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <string>

#include "cu_kernels.cuh"
#include "bitmap_io.hpp"


namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{

enum{JET, VIRIDIS, ACCENT, MAGMA, INFERNO, BLUE};

class CudaColormap{

public:
  CudaColormap(std::size_t width, std::size_t height, int type=JET);
  virtual ~CudaColormap();


  void save(std::string dir, const thrust::device_vector<float>& idata);
  void load(std::string dir, thrust::host_vector<unsigned char>& data);
  void transform(const thrust::device_vector<float>& idata, thrust::device_vector<unsigned char>& odata);

private:
  std::size_t _x;
  std::size_t _y;
  int _type;
  Bitmap_IO *bmp;
};


}
}
}

#endif /* CUDACOLORMAP_CUH_*/
