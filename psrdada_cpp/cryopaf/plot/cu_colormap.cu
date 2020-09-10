#include "psrdada_cpp/cryopaf/beamforming/cu_colormap.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{



CudaColormap::CudaColormap(std::size_t width, std::size_t height, int type)
  : _x(width),_y(height),_type(type)
{
  bmp = new Bitmap_IO(_x, _y);
}

CudaColormap::~CudaColormap(){

}
void CudaColormap::transform(const thrust::device_vector<float>& idata, thrust::device_vector<unsigned char>& odata)
{
  int tx = 32;
  int ty = 32;
  int bx = _x/tx+1;
  int by = _y/ty+1;

  const float *raw_idata = thrust::raw_pointer_cast(idata.data());
  unsigned char *raw_odata = thrust::raw_pointer_cast(odata.data());
  thrust::device_vector<float>::iterator iter_max = thrust::max_element(idata.begin(),idata.end(), compare());
  thrust::device_vector<float>::iterator iter_min = thrust::min_element(idata.begin(),idata.end(), compare());
  float max = idata[(unsigned int)*(iter_max)];
  float min = idata[(unsigned int)*(iter_min)];

  std::cout << max << min << std::endl;

  dim3 blockSize(tx,ty);
  dim3 gridSize(bx,by);
  switch(_type)
  {
    case JET:
      colormapJet<<<gridSize,blockSize>>>(raw_idata, raw_odata, max, min, _x, _y);
      break;
    case VIRIDIS:
      colormapViridis<<<gridSize,blockSize>>>(raw_idata, raw_odata, max, min, _x, _y);
      break;
    case ACCENT:
      colormapAccent<<<gridSize,blockSize>>>(raw_idata, raw_odata, max, min, _x, _y);
      break;
    case MAGMA:
      colormapMagma<<<gridSize,blockSize>>>(raw_idata, raw_odata, max, min, _x, _y);
      break;
    case INFERNO:
      colormapInferno<<<gridSize,blockSize>>>(raw_idata, raw_odata, max, min, _x, _y);
      break;
    case BLUE:
      colormapBlue<<<gridSize,blockSize>>>(raw_idata, raw_odata, max, min, _x, _y);
      break;
    default:
      std::cout << "This colormapping is not provided" << std::endl;
  }
}

void CudaColormap::save(std::string dir, const thrust::device_vector<float>& idata)
{
  thrust::device_vector<unsigned char> dev_output(idata.size()*3);
  transform(idata, dev_output);
  thrust::host_vector<unsigned char> host_output = dev_output;

  bmp->setImagePtr(thrust::raw_pointer_cast((char*)host_output.data()));
  bmp->save(dir);
}

void CudaColormap::load(std::string dir, thrust::host_vector<unsigned char>& idata){
  std::cout << "Not implemented " << std::endl;
}

}
}
}
