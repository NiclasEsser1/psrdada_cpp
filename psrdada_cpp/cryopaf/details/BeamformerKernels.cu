#ifdef BEAMFORMER_CUH_

namespace psrdada_cpp{
namespace cryopaf{

// Function description in Beamformer.cuh
template<typename T>__global__
void beamformer_voltage_fpte_fpbe_fptb(
  const T *idata,
  const T* wdata,
  T *odata,
  int time, int elem, int beam)
{
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int beam_axis = blockDim.x * blockIdx.x + tidx;
  const int time_axis = blockDim.y * blockIdx.y + tidy;
  const int freq = blockIdx.z;

  const int size_idata = elem * time;
  const int size_wdata = elem * beam;
  const int size_odata = time * beam;

  const int idata_offset = freq * 2 * size_idata;
  const int wdata_offset = freq * 2 * size_wdata;
  const int odata_offset = freq * 2 * size_odata;

  T voltage_x = {0,0};
  T voltage_y = {0,0};

  __shared__ T s_idata_x[TILE_SIZE][TILE_SIZE];
  __shared__ T s_idata_y[TILE_SIZE][TILE_SIZE];
  __shared__ T s_wdata_x[TILE_SIZE][TILE_SIZE];
  __shared__ T s_wdata_y[TILE_SIZE][TILE_SIZE];

  s_idata_x[tidy][tidx] = {0,0};
  s_idata_y[tidy][tidx] = {0,0};
  s_wdata_x[tidy][tidx] = {0,0};
  s_wdata_y[tidy][tidx] = {0,0};

  for( int k = 0; k < elem; k+=TILE_SIZE )
  {
    if(time_axis < time && tidx + k < elem)
    {
      s_idata_x[tidy][tidx] = idata[idata_offset + time_axis * elem + tidx + k];
      s_idata_y[tidy][tidx] = idata[idata_offset + size_idata + time_axis * elem + tidx + k];
    }
    else
    {
      s_idata_x[tidy][tidx] = {0,0};
      s_idata_y[tidy][tidx] = {0,0};
    }

__syncthreads();

    if(beam_axis < beam && tidy + k < elem)
    {
      s_wdata_x[tidy][tidx] = wdata[wdata_offset + beam_axis * elem + tidy + k];
      s_wdata_y[tidy][tidx] = wdata[wdata_offset + size_wdata + beam_axis * elem + tidy + k];
    }
    else
    {
      s_wdata_x[tidy][tidx] = {0,0};
      s_wdata_y[tidy][tidx] = {0,0};
    }

__syncthreads();

    for(int j = 0; j < TILE_SIZE; j++)
    {
      if(j + k == elem)
      {
        break;
      }
      voltage_x = cmadd(s_idata_x[tidy][j], s_wdata_x[j][tidx], voltage_x);
      voltage_y = cmadd(s_idata_y[tidy][j], s_wdata_y[j][tidx], voltage_y);
    }
  }

__syncthreads();

  if(beam_axis < beam && time_axis < time)
  {
    const int out_idx_x = odata_offset + (((int)blockIdx.y * TILE_SIZE) + tidy) * beam + beam_axis;
    const int out_idx_y = odata_offset + time * beam + (((int)blockIdx.y * TILE_SIZE) + tidy) * beam + beam_axis;
    odata[out_idx_x] = voltage_x;
    odata[out_idx_y] = voltage_y;
  }
}

/**
* @brief Device function to reduce / integrate the Stokes I power samples
*
* @param T data[32][32] Two dimensional array containing unintegrated power data (has to be in shared memory)
* @param int tidy       Thread index in y-dimension
* @param int tidy       Thread index in y-dimension
* @param int integrate  Integration time / samples to accumulat
*/
template<typename T>
__device__ void warp_reduce_tile(T data[TILE_SIZE][TILE_SIZE], const int tidy, const int tidx, const int integrate)
{
  if(integrate < 2) return;
  if(tidy < 16)
  {
    data[tidy * 2][tidx] += data[tidy * 2 + 1][tidx];
  }
__syncthreads();
  if(integrate < 4) return;
  if(tidy < 8)
  {
    data[tidy * 4][tidx] += data[tidy * 4 + 2][tidx];
  }
__syncthreads();
  if(integrate < 8) return;
  if(tidy < 4)
  {
    data[tidy * 8][tidx] += data[tidy * 8 + 4][tidx];
  }
__syncthreads();
  if(integrate < 16) return;
  if(tidy < 2)
  {
    data[tidy * 16][tidx] += data[tidy * 16 + 8][tidx];
  }
__syncthreads();
  if(integrate < 32) return;
  if(tidy < 1)
  {
    data[tidy][tidx] += data[tidy + 16][tidx];
  }
__syncthreads();
}

// Function description in Beamformer.cuh
template<typename T=float2, typename U=float>__global__
void beamformer_power_fpte_fpbe_ftb(
  const float2 *idata,
  const float2* wdata,
  float *odata,
  int time, int elem, int beam, int integrate)
{
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int beam_axis = blockDim.x * blockIdx.x + tidx;
  const int time_axis = blockDim.y * blockIdx.y + tidy;
  const int freq = blockIdx.z;

  const int size_idata = elem * time;
  const int size_wdata = elem * beam;
  const int size_odata = time / integrate * beam;

  const int idata_offset = freq * 2 * size_idata;
  const int wdata_offset = freq * 2 * size_wdata;
  const int odata_offset = freq * size_odata;

  float2 voltage_x = {0,0};
  float2 voltage_y = {0,0};

  __shared__ float2 s_idata_x[TILE_SIZE][TILE_SIZE];
  __shared__ float2 s_idata_y[TILE_SIZE][TILE_SIZE];
  __shared__ float2 s_wdata_x[TILE_SIZE][TILE_SIZE];
  __shared__ float2 s_wdata_y[TILE_SIZE][TILE_SIZE];
  __shared__ float s_odata[TILE_SIZE][TILE_SIZE];

  s_idata_x[tidy][tidx] = {0,0};
  s_idata_y[tidy][tidx] = {0,0};
  s_wdata_x[tidy][tidx] = {0,0};
  s_wdata_y[tidy][tidx] = {0,0};
  s_odata[tidy][tidx] = {0};

  for( int k = 0; k < elem; k+=TILE_SIZE )
  {
    if(time_axis < time && tidx + k < elem)
    {
      s_idata_x[tidy][tidx] = idata[idata_offset + time_axis * elem + tidx + k];
      s_idata_y[tidy][tidx] = idata[idata_offset + size_idata + time_axis * elem + tidx + k];
    }
    else
    {
      s_idata_x[tidy][tidx] = {0,0};
      s_idata_y[tidy][tidx] = {0,0};
    }

__syncthreads();

    if(beam_axis < beam && tidy + k < elem)
    {
      s_wdata_x[tidy][tidx] = wdata[wdata_offset + beam_axis * elem + tidy + k];
      s_wdata_y[tidy][tidx] = wdata[wdata_offset + size_wdata + beam_axis * elem + tidy + k];
    }
    else
    {
      s_wdata_x[tidy][tidx] = {0,0};
      s_wdata_y[tidy][tidx] = {0,0};
    }

__syncthreads();

    for(int j = 0; j < TILE_SIZE; j++)
    {
      if(j + k == elem)
      {
        break;
      }
      voltage_x = cmadd(s_idata_x[tidy][j], s_wdata_x[j][tidx], voltage_x);
      voltage_y = cmadd(s_idata_y[tidy][j], s_wdata_y[j][tidx], voltage_y);
    }
  }
  s_odata[tidy][tidx] = voltage_x.x * voltage_x.x + voltage_x.y * voltage_x.y;
  s_odata[tidy][tidx] += voltage_y.x * voltage_y.x + voltage_y.y * voltage_y.y;

__syncthreads();

  warp_reduce_tile(s_odata, tidy, tidx, integrate);

  if(beam_axis < beam && tidy < ceil((float)TILE_SIZE / integrate))
  {
    const int out_idx = odata_offset + (((int)blockIdx.y * TILE_SIZE / integrate) + tidy) * beam + beam_axis;
    odata[out_idx] = s_odata[tidy * integrate][tidx] / integrate;
  }
}

// Function description in Beamformer.cuh
__global__ void beamformer_power_fpte_fpbe_ftb(
  const __half2 *idata,
  const __half2 *wdata,
  __half *odata,
  int time, int elem, int beam, int integrate)
{
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int beam_axis = blockDim.x * blockIdx.x + tidx;
  const int time_axis = blockDim.y * blockIdx.y + tidy;
  const int freq = blockIdx.z;

  const int size_idata = elem * time;
  const int size_wdata = elem * beam;
  const int size_odata = time / integrate * beam;

  const int idata_offset = freq * 2 * size_idata;
  const int wdata_offset = freq * 2 * size_wdata;
  const int odata_offset = freq * size_odata;

  __half2 voltage_x = {0,0};
  __half2 voltage_y = {0,0};

  __shared__ __half2 s_idata_x[TILE_SIZE][TILE_SIZE];
  __shared__ __half2 s_idata_y[TILE_SIZE][TILE_SIZE];
  __shared__ __half2 s_wdata_x[TILE_SIZE][TILE_SIZE];
  __shared__ __half2 s_wdata_y[TILE_SIZE][TILE_SIZE];
  __shared__ __half s_odata[TILE_SIZE][TILE_SIZE];

  s_idata_x[tidy][tidx] = {0,0};
  s_idata_y[tidy][tidx] = {0,0};
  s_wdata_x[tidy][tidx] = {0,0};
  s_wdata_y[tidy][tidx] = {0,0};
  s_odata[tidy][tidx] = {0};

  for( int k = 0; k < elem; k+=TILE_SIZE )
  {
    if(time_axis < time && tidx + k < elem)
    {
      s_idata_x[tidy][tidx] = idata[idata_offset + time_axis * elem + tidx + k];
      s_idata_y[tidy][tidx] = idata[idata_offset + size_idata + time_axis * elem + tidx + k];
    }
    else
    {
      s_idata_x[tidy][tidx] = {0,0};
      s_idata_y[tidy][tidx] = {0,0};
    }

__syncthreads();

    if(beam_axis < beam && tidy + k < elem)
    {
      s_wdata_x[tidy][tidx] = wdata[wdata_offset + beam_axis * elem + tidy + k];
      s_wdata_y[tidy][tidx] = wdata[wdata_offset + size_wdata + beam_axis * elem + tidy + k];
    }
    else
    {
      s_wdata_x[tidy][tidx] = {0,0};
      s_wdata_y[tidy][tidx] = {0,0};
    }

__syncthreads();

    for(int j = 0; j < TILE_SIZE; j++)
    {
      if(j + k == elem)
      {
        break;
      }
      voltage_x = cmadd(s_idata_x[tidy][j], s_wdata_x[j][tidx], voltage_x);
      voltage_y = cmadd(s_idata_y[tidy][j], s_wdata_y[j][tidx], voltage_y);
    }
  }
  s_odata[tidy][tidx] = voltage_x.x * voltage_x.x + voltage_x.y * voltage_x.y;
  s_odata[tidy][tidx] += voltage_y.x * voltage_y.x + voltage_y.y * voltage_y.y;

__syncthreads();

  warp_reduce_tile(s_odata, tidy, tidx, integrate);

  if(beam_axis < beam && tidy < ceil((float)TILE_SIZE / integrate))
  {
    const int out_idx = odata_offset + (((int)blockIdx.y * TILE_SIZE / integrate) + tidy) * beam + beam_axis;
    odata[out_idx] = __hdiv(s_odata[tidy * integrate][tidx], __float2half((float)integrate));
  }
}


// ######################################################
// NOTE: Kernels above are deprecated and not longer used
// ######################################################


// ##########################################
//  Optimized Voltage Beamformer (deprecated)
// ##########################################

/*

template<typename T=float2>__global__
void beamformer_voltage_tfap_bftp_bfap(const float2 *idata, float2 *odata, const float2 *weights, const bf_config_t conf)
{
	// Grid layout: x = time; y = beam; z = channel
	// Block layout: A block consist of NTHREADS and WARPS. Every warp
	// calculates both polarisations of one beam for one channel at one given time step.
	// Within a block 32 adjacent time steps for one beam are calculated (same channel).
	// Data products are as follow (glob mem):
	// 		idata: TFAP(t)
	//		odata: BFT
	//		weights: BFAP
	// constraints:
	//	- n_elements must be a multiple of WARP_SIZE 32
	//	- n_samples must be a multiple of WARP_SIZE 32

	int tidx = threadIdx.x;
	int bidx = blockIdx.x;      // Time dimension (T)
	int bidy = blockIdx.y;      // Beam dimension (B)
	int bidz = blockIdx.z;      // Channel dimension (F)

	int n_elements = conf.n_elements * conf.n_pol; // Number of elements, product of antenna (A) and polarisation (P)
	int warp = tidx / WARP_SIZE; // Calculate the current warp
	int warp_idx = tidx % WARP_SIZE;    // thread index -> warp index

	// Each thread has its own indices for accessing the global memory (idata, odata, weights).
	int idata_glob_idx = warp_idx + n_elements * (bidx * WARPS * conf.n_channel + warp * conf.n_channel + bidz);
	int weights_glob_idx = warp_idx + n_elements * (bidy * conf.n_channel + bidz);
	int output_glob_idx = (bidy * conf.n_samples * conf.n_channel
			+ bidz * conf.n_samples + bidx * WARPS + warp)*conf.n_pol + warp_idx; // multiplied by two, since two polarisations

	// To enable higher throughput and more efficient data transfer, shared memory
	// is required. The size in bytes of shared memory is calculated as follows:
	//	shared_mem_bytes = sizeof(T) * (A * P * (WARPS + 1) + NTHREADS)
	//	idata  = sizeof(T) * A * P * WARPS	<- every warp loads data of all elements at one time step
	// 	weight = sizeof(T) * A * P							<- weights are the same for all warps
	//  odata  = sizeof(T) * NTHREADS						<- Every thread calculates one output sample
	extern __shared__ float2 shared_mem_fp32[];	// dynamically allocated
	float2* shared_idata = (&shared_mem_fp32[0]);	// idata space comes first
	float2* shared_weights = (&shared_mem_fp32[n_elements * WARPS]);	// weight space with idata space as offset

	// To prevent overflows when using integer values, the datatype of shared_odata has to be float2 (cuComplex).
	float2 __shared__ shared_odata[NTHREAD];

	shared_odata[tidx] = {0,0};	// intialize output with zeros

	float2 acc = {0,0}; // local register for storing intermediate results

	// Load idata and weights into shared memory for every warp
#pragma unroll
	for(int i = 0; i < n_elements; i+=WARP_SIZE)
	{
		// It is important to access 32 adjacent samples, to increase the hit rate of cached memory
		// Here each thread within a warp accesses adjacent samples!
		shared_idata[warp * n_elements + i + warp_idx] = idata[idata_glob_idx + i];

		// Since all warps within a block are using the same weights, only one warp needs to load the weights.
		// This may not be the most efficient way, since all other 31 warps are idled until the weights are loaded.
		// However, this approach prevents race conditions.
		if(warp == 0)
			shared_weights[i + warp_idx] = weights[weights_glob_idx + i];
	}

	__syncthreads();	// Synchronize all threads within a block, to ensure all data is loaded.

	// Iterate across all elements.
	// Each thread within a warp performs n complex multiplications and 2*n additions (n = n_elements/WARP_SIZE).
	// 		FLOP/thread = n * (6+2)
#pragma unroll
	for(int i = 0; i < n_elements; i+=WARP_SIZE)
	{
		acc = cuCaddf(acc, cuCmulf(shared_idata[warp * n_elements + i + warp_idx], shared_weights[i + warp_idx]));
	}
	shared_odata[tidx] = acc;

	__syncthreads(); // Synchronize all threads within a block, to ensure all computitations are done.

	// Since odata contains NTHREAD samples which have not been combined to WARPS time steps a reduction is required.
	int i = WARP_SIZE / 2;
	// This reduction may no be very efficient since many threads within a warp are idled
#pragma unroll
	while(i != conf.n_pol - 1)
	{
		if(warp_idx < i)
			shared_odata[tidx] = cuCaddf(shared_odata[tidx], shared_odata[tidx + i]);
		__syncthreads();
		i /= 2;
	}
	// After reduction the first two samples in shared_odata with warp offset contains both polarisations.
	// So, if warp_idx is 0 or 1, assign the samples to the global output buffer. In total 64
	// samples are transfered back to the global memory for each block.
	if(warp_idx < conf.n_pol)
	{
		// TODO: In case of integer inputs, conversion is implemented here!!!
		odata[output_glob_idx] = shared_odata[tidx];	// Polarisation 0 and 1
	}

}


template<typename T=__half2>__global__
void beamformer_voltage_tfep_bfep_bftp(const __half2 *idata, __half2 *odata, const __half2 *weights, const bf_config_t conf)
{
	// Grid layout: x = time; y = beam; z = channel
	// Block layout: A block consist of 1024 threads and 32 warps. Every warp
	// calculates both polarisations of one beam for one channel at one given time step.
	// Within a block 32 adjacent time steps for one beam are calculated (same channel).
	// Data products are as follow (glob mem):
	// 		idata: TFEP(t)
	//		odata: BFTP
	//		weights: BFEP
	// constraints:
	//	- n_elements must be a multiple of WARP_SIZE 32
	//	- n_samples must be a multiple of WARP_SIZE 32

	const int tidx = threadIdx.x;
	const int bidx = blockIdx.x;      // Time dimension (T)
	const int bidy = blockIdx.y;      // Beam dimension (B)
	const int bidz = blockIdx.z;      // Channel dimension (F)
#if __CUDA_ARCH__ >= 530
	const int n_elements = conf.n_elements * conf.n_pol; // Number of elements, product of antenna (A) and polarisation (P)
	const int warp = tidx / WARP_SIZE; // Calculate the current warp
	const int warp_idx = tidx % WARP_SIZE;    // thread index -> warp index

	// Each thread has its own indices for accessing the global memory (idata, odata, weights).
	const int idata_glob_idx = warp_idx + n_elements * (bidx * WARPS * conf.n_channel + warp * conf.n_channel + bidz);
	const int weights_glob_idx = warp_idx + n_elements * (bidy * conf.n_channel + bidz);
	const int output_glob_idx = (bidy * conf.n_samples * conf.n_channel
			+ bidz * conf.n_samples + bidx * WARPS + warp)*conf.n_pol + warp_idx; // multiplied by two, since two polarisations

	// To enable higher throughput and more efficient data transfer, shared memory
	// is required. The size in bytes of shared memory is calculated as follows:
	//	shared_mem_bytes = sizeof(T) * (A * P * (WARPS + 1) + NTHREADS)
	//	idata  = sizeof(T) * A * P * WARPS	<- every warp loads data of all elements at one time step
	// 	weight = sizeof(T) * A * P							<- weights are the same for all warps
	//  odata  = sizeof(T) * NTHREADS						<- Every thread calculates one output sample
	extern __shared__ __half2 shared_mem_fp16[];	// dynamically allocated
	__half2* shared_idata = (&shared_mem_fp16[0]);	// idata space comes first
	__half2* shared_weights = (&shared_mem_fp16[n_elements * WARPS]);	// weight space with idata space as offset

	// To prevent overflows when using integer values, the datatype of shared_odata has to be float2 (cuCOmplex).
	__half2 __shared__ shared_odata[NTHREAD];

	shared_odata[tidx] = {0,0};	// intialize output with zeros

	__half2 acc = {0,0}; // local register for storing intermediate results

	// Load idata and weights into shared memory for every warp
#pragma unroll
	for(int i = 0; i < n_elements; i+=WARP_SIZE)
	{
		// It is important to access 32 adjacent samples, to increase the hit rate of cached memory
		// Here each thread within a warp accesses adjacent samples!
		shared_idata[warp * n_elements + i + warp_idx] = idata[idata_glob_idx + i];

		// Since all warps within a block are using the same weights, only one warp needs to load the weights.
		// This may not be the most efficient way, since all other 31 warps are idled until the weights are loaded.
		// However, this approach prevents race conditions.
		if(warp == 0)
			shared_weights[i + warp_idx] = weights[weights_glob_idx + i];
	}

	__syncthreads();	// Synchronize all threads within a block, to ensure all data is loaded.

	// Iterate across all elements.
	// Each thread within a warp performs n complex multiplications and 2*n additions (n = n_elements/WARP_SIZE).
	// 		FLOP/thread = n * (6+2)
#pragma unroll
	for(int i = 0; i < n_elements; i+=WARP_SIZE)
	{
		acc = __hadd2(acc, (__hCmul2(shared_idata[warp * n_elements + i + warp_idx], shared_weights[i + warp_idx])));
	}
	shared_odata[tidx] = (acc);

	__syncthreads(); // Synchronize all threads within a block, to ensure all computitations are done.

	// Since odata contains 1024 samples which have not been combined to 32 time steps a reduction is required.
	int i = WARP_SIZE / 2;
	// This reduction may no be very efficient since many threads within a warp are idled
#pragma unroll
	while(i != conf.n_pol - 1)
	{
		if(warp_idx < i)
			shared_odata[tidx] = __hadd2(shared_odata[tidx], shared_odata[tidx + i]);
		__syncthreads();
		i /= 2;
	}

	// After reduction the first two samples in shared_odata with warp offset contains both polarisations.
	// So, if warp_idx is 0 or 1, assign the samples to the global output buffer. In total 64
	// samples are transfered back to the global memory for each block.
	if(warp_idx < conf.n_pol)
	{
		// TODO: In case of integer inputs, conversion is implemented here!!!
		odata[output_glob_idx] = (shared_odata[tidx]);	// Polarisation 0 and 1
	}

#else
if(tidx == 0 && bidx==0 && bidy == 0 && bidz == 0)
	printf("Warning: CUDA architecture does not support half precisison. Beamforming not executed...\n");
#endif

}

// ##########################################
//  Optimized Power Beamformer (deprecated)
// ##########################################

// Function to reduce voltage values in shared memory to power (called by beamformer_power_tfep_bfep_bft)
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
void beamformer_power_tfep_bfep_bft(const float2 *idata, const float2* weights, float *odata, const bf_config_t conf)
{
	int tidx = threadIdx.x;
	int bidx = blockIdx.x;      // Time dimension (T)
	int bidy = blockIdx.y;      // Channel dimension (F)

	const int n_elements = conf.n_elements * conf.n_pol;
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


	// IMPORTANT: s_odata has to be initialized to zero for each element in the array
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
			odata[odata_offset + odata_glob_idx] = s_odata[b] / conf.interval;
		}
	}
}

//####################
//## Naive approach ##
//####################

template<typename T=float2, typename U=float>__global__
void simple_bf_tfap_power(const float2 *idata, float *odata, const float2 *weights, const bf_config_t conf)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;	// Time dimension
	int tidy = blockIdx.y * blockDim.y;								// Beam dimension
	int tidz = blockIdx.z * blockDim.z;								// Channel dimension
	float2 acc{.0,.0};

	int in_offset = tidx * conf.n_elements * conf.n_channel * conf.n_pol + tidz * conf.n_pol;
	int out_offset = tidy * conf.n_samples * conf.n_channel + tidx * conf.n_channel;
	int weight_offset = tidy * conf.n_elements * conf.n_channel * conf.n_pol + tidz * conf.n_pol;

	if(tidx < conf.n_samples && tidy < conf.n_beam && tidz < conf.n_channel)
	{
		for(int i = 0; i < conf.n_elements; i++)
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

	int in_offset = tidx * conf.n_elements * conf.n_channel * conf.n_pol + tidz * conf.n_pol;
	int out_offset = tidy * conf.n_samples * conf.n_channel + tidx * conf.n_channel;
	int weight_offset = tidy * conf.n_elements * conf.n_channel * conf.n_pol + tidz * conf.n_pol;

	if(tidx < conf.n_samples && tidy < conf.n_beam && tidz < conf.n_channel)
	{
		for(int i = 0; i < conf.n_elements; i++)
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



template<typename T=float2>__global__
void simple_bf_tafp_voltage(const float2 *idata, float2 *odata, const float2 *weights, const bf_config_t conf)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;		// Time dimension
	int tidy = blockIdx.y * blockDim.y;									// Beam dimension
	int tidz = blockIdx.z * blockDim.z;									// Channel dimension

	int in_offset = tidx * conf.n_elements * conf.n_channel * conf.n_pol + tidz * conf.n_pol;
	int out_offset = tidy * conf.n_samples * conf.n_channel * conf.n_pol + tidx * conf.n_channel * conf.n_pol + tidz * conf.n_pol ;
	int weight_offset = tidy * conf.n_elements * conf.n_channel * conf.n_pol + tidz * conf.n_pol;

	float2 acc;

	if(tidx < conf.n_samples && tidy < conf.n_beam && tidz < conf.n_channel)
	{
		for(int k = 0; k < conf.n_pol; k++)
		{
			acc = {0,0};
			for(int i = 0; i < conf.n_elements; i++)
			{
				acc = cuCaddf(acc, cuCmulf(idata[in_offset + i * conf.n_channel * conf.n_pol + k],
					  weights[weight_offset + i * conf.n_channel * conf.n_pol + k]));
			}
			odata[out_offset + k] = acc;
		}
	}
}

template<typename T=__half2>__global__
void simple_bf_tafp_voltage(const __half2 *idata, __half2 *odata, const __half2 *weights, const bf_config_t conf)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;		// Time dimension
	int tidy = blockIdx.y * blockDim.y;									// Beam dimension
	int tidz = blockIdx.z * blockDim.z;									// Channel dimension

	int in_offset = tidx * conf.n_elements * conf.n_channel * conf.n_pol + tidz * conf.n_pol;
	int out_offset = tidy * conf.n_samples * conf.n_channel * conf.n_pol + tidx * conf.n_channel * conf.n_pol + tidz * conf.n_pol ;
	int weight_offset = tidy * conf.n_elements * conf.n_channel * conf.n_pol + tidz * conf.n_pol;

	__half2 acc;

	if(tidx < conf.n_samples && tidy < conf.n_beam && tidz < conf.n_channel)
	{
		for(int k = 0; k < conf.n_pol; k++)
		{
			acc = {0,0};
			for(int i = 0; i < conf.n_elements; i++)
			{
				acc = __hadd2(acc, __hCmul2(idata[in_offset + i * conf.n_channel * conf.n_pol + k],
						weights[weight_offset + i * conf.n_channel * conf.n_pol + k]));
			}
			odata[out_offset + k] = acc;
		}
	}
}

*/

}
}

#endif
