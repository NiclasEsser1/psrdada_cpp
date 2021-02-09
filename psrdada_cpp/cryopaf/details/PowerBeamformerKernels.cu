#ifdef POWERBEAMFORMER_CUH_

namespace psrdada_cpp{
namespace cryopaf{


template<typename T, typename U>
__device__ void warp_reduce_v2p(T *s_odata, U *s_idata, int warp_idx){
    if(warp_idx < 16)
    {
			s_idata[warp_idx] = cadd(s_idata[warp_idx], s_idata[warp_idx + 16]);
		}
__syncthreads();
    if(warp_idx < 8)
		{
			s_idata[warp_idx] = cadd(s_idata[warp_idx], s_idata[warp_idx + 8]);
		}
__syncthreads();
		if(warp_idx < 4)
		{
			s_idata[warp_idx] = cadd(s_idata[warp_idx], s_idata[warp_idx + 4]);
		}
__syncthreads();
    if(warp_idx < 2)
		{
			s_idata[warp_idx] = cadd(s_idata[warp_idx], s_idata[warp_idx + 2]);
		}
__syncthreads();
    if(warp_idx < 1)
    {
			T x_power = s_idata[warp_idx].x * s_idata[warp_idx].x + s_idata[warp_idx].y * s_idata[warp_idx].y;
			T y_power = s_idata[warp_idx + 1].x * s_idata[warp_idx + 1].x + s_idata[warp_idx + 1].y * s_idata[warp_idx + 1].y;
    	s_odata[0] += x_power + y_power;
    }
	__syncthreads();

}


template<typename T=float2, typename U=float>__global__
void bf_tfap_power(const float2 *idata, float *odata, const float2* weights, const bf_config_t conf)
{
	int tidx = threadIdx.x;
	int bidx = blockIdx.x;      // Time dimension (T)
	int bidy = blockIdx.y;      // Channel dimension (F)

	const int n_elements = conf.n_antenna * conf.n_pol;
	const int n_timestamp = SHARED_IDATA / n_elements;	// Number of timestamps loaded into shared memory
	const int n_timestamp_iter = NTHREAD / n_elements;	// Number of timestamps loaded in one iteration by all active threads

	// WARP grouping
	const int warp_idx = tidx % WARP_SIZE;
	const int warp = tidx / WARP_SIZE;
	const int n_warps_per_grp = WARP_SIZE / n_timestamp_iter;
	const int warp_grp = warp / n_warps_per_grp;	// Devide all warps into groups to load one timestamp by each group
	const int warp_grp_idx = warp_idx + (warp - n_warps_per_grp * warp_grp) * WARP_SIZE; // Index of thread within a warp group

	const int idata_offset = bidx * conf.interval * conf.n_channel * n_elements + bidy * n_elements;
	const int odata_offset = bidy * conf.n_samples / conf.interval + bidx;		//

	int idata_glob_idx;
	T voltage, weight;


	__shared__ float2 s_idata[SHARED_IDATA]; // Shared memory for input data

	extern __shared__ unsigned char s_mem[];
	U *s_odata = reinterpret_cast<U*>(&s_mem[0]); // Shared memory for power data
	T *s_intermediate = reinterpret_cast<T*>(&s_mem[conf.n_beam * sizeof(U)]); // Shared memory for intermediate results
	// T *s_weights = reinterpret_cast<T*>(&s_mem[conf.n_beam * sizeof(U) + NTHREAD*sizeof(T)]);


	/* IMPORTANT: s_odata has to be initialized to zero for each element in the array*/
	for(int b = tidx; b < conf.n_beam; b += NTHREAD)
		if(b < conf.n_beam) s_odata[b] = 0;


	for(int t = 0; t < conf.interval; t += n_timestamp)
	{
		for(int i = 0; i < n_timestamp; i+=n_timestamp_iter)
		{
			idata_glob_idx = (t + warp_grp + i) * conf.n_channel * n_elements + warp_grp_idx;
			s_idata[i/n_timestamp_iter * NTHREAD + tidx] = idata[idata_offset + idata_glob_idx];
		}

		__syncthreads();

		for(int b = warp; b < conf.n_beam; b += WARPS)
		{
			for(int i = 0; i < n_timestamp; i++)
			{
				voltage = {0,0};
				for(int a = warp_idx; a < n_elements; a += WARP_SIZE)
				{
					weight = weights[b * conf.n_channel * n_elements + bidy * n_elements + a];
					// Complex multiplication: raw voltage * weight = voltage
					voltage = cmadd(s_idata[i * n_elements + a], weight, voltage);
				}
				// Every thread accumulated n_elements/WARP_SIZE
				// Load accumulated result to shared memory; Every thread has its own field, otherwise race condition may occur
				s_intermediate[tidx] = voltage;
				// Reduction
				warp_reduce_v2p(&s_odata[b], &s_intermediate[warp * WARP_SIZE], warp_idx);
			}
		}
	}

	for(int b = tidx; b < conf.n_beam; b += NTHREAD)
	{
		if(b < conf.n_beam)
		{
			const int odata_glob_idx = b * conf.n_channel * conf.n_samples / conf.interval;
			const float power = s_odata[b] / conf.interval;
			odata[odata_offset + odata_glob_idx] = s_odata[b] / conf.interval;
		}
	}
}

/********************/
/** Naive approach **/
/********************/

template<typename T=float2, typename U=float>__global__
void simple_bf_tfap_power(const float2 *idata, float *odata, const float2 *weights, const bf_config_t conf)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;	// Time dimension
	int tidy = blockIdx.y * blockDim.y;								// Beam dimension
	int tidz = blockIdx.z * blockDim.z;								// Channel dimension
	float2 acc{.0,.0};

	int in_offset = tidx * conf.n_antenna * conf.n_channel * conf.n_pol + tidz * conf.n_pol;
	int out_offset = tidy * conf.n_samples * conf.n_channel + tidx * conf.n_channel;
	int weight_offset = tidy * conf.n_antenna * conf.n_channel * conf.n_pol + tidz * conf.n_pol;

	if(tidx < conf.n_samples && tidy < conf.n_beam && tidz < conf.n_channel)
	{
		for(int i = 0; i < conf.n_antenna; i++)
		{
			acc.x = 0; acc.y = 0;
			for(int k = 0; k < conf.n_pol; k++)
			{
				acc = cuCaddf(acc, cuCmulf(idata[in_offset + i * conf.n_channel * conf.n_pol + k],
					  weights[weight_offset + i * conf.n_channel * conf.n_pol + k]));
			}
			odata[out_offset + tidz] += (acc.x*acc.x + acc.y*acc.y);
		}
	}
}

template<typename T=__half2, typename U=__half>__global__
void simple_bf_tfap_power(const __half2 *idata, __half *odata, const __half2 *weights, const bf_config_t conf)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;	// Time dimension
	int tidy = blockIdx.y * blockDim.y;								// Beam dimension
	int tidz = blockIdx.z * blockDim.z;								// Channel dimension
	__half2 acc(0,0);

	int in_offset = tidx * conf.n_antenna * conf.n_channel * conf.n_pol + tidz * conf.n_pol;
	int out_offset = tidy * conf.n_samples * conf.n_channel + tidx * conf.n_channel;
	int weight_offset = tidy * conf.n_antenna * conf.n_channel * conf.n_pol + tidz * conf.n_pol;

	if(tidx < conf.n_samples && tidy < conf.n_beam && tidz < conf.n_channel)
	{
		for(int i = 0; i < conf.n_antenna; i++)
		{
			acc.x = 0; acc.y = 0;
			for(int k = 0; k < conf.n_pol; k++)
			{
				acc = __hadd2(acc, __hCmul2(idata[in_offset + i * conf.n_channel * conf.n_pol + k],
					  weights[weight_offset + i * conf.n_channel * conf.n_pol + k]));
			}
			odata[out_offset + tidz] += (acc.x*acc.x + acc.y*acc.y);
		}
	}
}


}
}

#endif
