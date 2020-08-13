#include "psrdada_cpp/cryopaf/beamforming/cu_kernels.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{


template<typename T> __device__
T complex_mul(const T a, const T b)
{
		cuComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}
template __device__ cuComplex complex_mul<cuComplex>
	(cuComplex, cuComplex);
//int2 complex_mul<cuComplex>(int2, int2);



template<typename T, typename U>__global__
void simple_bf_tafpt_stokes_I(const T *idata, U *odata, const T *weights, const bf_config_t *conf)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;	// Time dimension
	int tidy = blockIdx.y * blockDim.y;								// Beam dimension
	int tidz = blockIdx.z * blockDim.z;								// Antenna dimension
	cuComplex tmp;
	float real, imag;

	int in_offset = tidx  * conf->n_antenna * conf->n_channel * conf->n_pol + tidz * conf->n_channel * conf->n_pol;
	int out_offset = tidy * conf->n_samples * conf->n_channel + tidx * conf->n_channel;
	int weight_offset = tidy * conf->n_antenna * conf->n_channel * conf->n_pol + tidz * conf->n_channel * conf->n_pol;

	if(tidx < conf->n_samples && tidy < conf->n_beam && tidz < conf->n_antenna)
	{
		for(int i = 0; i < conf->n_channel; i++)
		{
			real = 0; imag = 0;
			for(int k = 0; k < conf->n_pol; k++)
			{
				tmp = complex_mul(idata[in_offset + i * conf->n_pol + k],
					  weights[weight_offset + i * conf->n_pol + k]);
				real += tmp.x;
				imag += tmp.y;
			}
			odata[out_offset + i] += real*real + imag*imag;
		}
	}
}
template __global__ void simple_bf_tafpt_stokes_I<cuComplex, float>
	(const cuComplex*, float*, const cuComplex*, const bf_config_t*);



template<typename T>__global__
void simple_bf_tafpt(const T *idata, T *odata, const T *weights, const bf_config_t *conf)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;		// Time dimension
	int tidy = blockIdx.y * blockDim.y;									// Beam dimension
	int tidz = blockIdx.z * blockDim.z;									// Antenna dimension

	int in_offset = tidx  * conf->n_antenna * conf->n_channel * conf->n_pol + tidz * conf->n_channel * conf->n_pol;
	int out_offset = tidy * conf->n_samples * conf->n_channel * conf->n_pol + tidx * conf->n_channel * conf->n_pol;
	int weight_offset = tidy * conf->n_antenna * conf->n_channel * conf->n_pol + tidz * conf->n_channel * conf->n_pol;

	cuComplex tmp;

	if(tidx < conf->n_samples && tidy < conf->n_beam && tidz < conf->n_antenna)
	{
		for(int i = 0; i < conf->n_channel; i++)
		{
			for(int k = 0; k < conf->n_pol; k++)
			{
				tmp = complex_mul(idata[in_offset + i * conf->n_pol + k], weights[weight_offset + i * conf->n_pol + k]);
				odata[out_offset + i * conf->n_pol + k].x += tmp.x;
				odata[out_offset + i * conf->n_pol + k].y += tmp.y;
			}
		}
	}
}
template __global__ void simple_bf_tafpt<cuComplex>(const cuComplex*, cuComplex*, const cuComplex*, const bf_config_t*);
//template __global__ void simple_bf_tafpt_stokes_I<int2, int>(const int2*, int*, const int2*, const bf_config_t*);


} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp
