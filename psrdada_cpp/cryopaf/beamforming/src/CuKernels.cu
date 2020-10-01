#ifdef CUKERNELS_CUH_


/** UTILS **/
__device__ __half2 __hCmul2(__half2 a, __half2 b)
{
		const __half r = a.x * b.x - a.y * b.y;
		const __half i = a.x * b.y + a.y * b.x;

		__half2 val; val.x = r; val.y = i;
		return val;
}


/** STOKES I Beamformer **/
template<typename T=float2, typename U=float>__global__
void simple_bf_tafpt_power(const float2 *idata, float *odata, const float2 *weights, const bf_config_t conf)
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
void simple_bf_tafpt_power(const __half2 *idata, __half *odata, const __half2 *weights, const bf_config_t conf)
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


template<typename T=float2, typename U=float>__global__
void bf_tafpt_power(const float2 *idata, float *odata, const cudaTextureObject_t weights, const bf_config_t conf)
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
	float2 voltage, weight;
	float power;


	__shared__ float2 s_idata[SHARED_IDATA]; // Shared memory for input data

	extern __shared__ float s_mem[];
	float *s_odata = &s_mem[0]; // Shared memory for output data
	float *s_intermediate = &s_mem[conf.n_beam]; // Shared memory for intermediate results


	/* IMPORTANT: s_odata has to be initialized to zero for each element in the array*/
	for(int b = 0; b < conf.n_beam; b += NTHREAD)
	{
		if(b + tidx < conf.n_beam)
			s_odata[b + tidx] = 0;
	}


	for(int t = 0; t < conf.interval; t += n_timestamp)
	{
		for(int i = 0; i < n_timestamp; i+=n_timestamp_iter)
		{
			idata_glob_idx = (t + warp_grp + i) * conf.n_channel * n_elements + warp_grp_idx;
			s_idata[i/n_timestamp_iter * NTHREAD + tidx] = idata[idata_offset + idata_glob_idx];
		}

		__syncthreads();

		for(int b = 0; b < conf.n_beam; b += WARPS)
		{
			power = 0;
			for(int a = 0; a < n_elements; a += WARP_SIZE)
			{
					weight = tex3D<float2>(weights, (a + warp_idx), bidy, warp + b);

					for(int i = 0; i < n_timestamp; i++)
					{
						// Complex multiplication: raw voltage * weight = voltage
						voltage = cuCmulf(s_idata[i * n_elements + (a + warp_idx)], weight);

						// Cacluate (real) power; Square and root cancel each other out
						power += voltage.x * voltage.x + voltage.y * voltage.y;
					}

			}
			// Every thread accumulated polarizations + n_antenna/WARP_SIZE
			// Load accumulated result to shared memory; Every thread has its own field, otherwise race condition may occur
			s_intermediate[tidx] = power;

			// Reduction
			int i = WARP_SIZE/2;
			while(i != 0)
			{
				if(warp_idx < i)
					s_intermediate[tidx] += s_intermediate[tidx + i];
				i /= 2;
			}


			// After reduction the first warp adds intermediate results to dedicated shared output memory
			// 31/32 are idled :-(
			__syncthreads();
			if(warp_idx == 0)
				s_odata[b + warp] += s_intermediate[tidx];
		}
	}

	for(int b = 0; b < conf.n_beam; b += NTHREAD)
	{
		if(b + tidx < conf.n_beam)
		{
			const int odata_glob_idx = (b + tidx) * conf.n_channel * conf.n_samples / conf.interval;
			odata[odata_offset + odata_glob_idx] = s_odata[b + tidx] / conf.interval;
		}
	}
}


template<typename T=__half2, typename U=__half>__global__
void bf_tafpt_power(const __half2 *idata, __half *odata, const cudaTextureObject_t weights, const bf_config_t conf)
{
	// int tidx = threadIdx.x + blockIdx.x * blockDim.x;	// Time dimension
	// int tidy = blockIdx.y * blockDim.y;								// Beam dimension
	// int tidz = blockIdx.z * blockDim.z;								// Antenna dimension



}


/** Voltage Beamformer **/


template<typename T=float2>__global__
void simple_bf_tafpt_voltage(const float2 *idata, float2 *odata, const float2 *weights, const bf_config_t conf)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;		// Time dimension
	int tidy = blockIdx.y * blockDim.y;									// Beam dimension
	int tidz = blockIdx.z * blockDim.z;									// Channel dimension

	int in_offset = tidx * conf.n_antenna * conf.n_channel * conf.n_pol + tidz * conf.n_pol;
	int out_offset = tidy * conf.n_samples * conf.n_channel * conf.n_pol + tidx * conf.n_channel * conf.n_pol + tidz * conf.n_pol ;
	int weight_offset = tidy * conf.n_antenna * conf.n_channel * conf.n_pol + tidz * conf.n_pol;

	float2 acc;

	if(tidx < conf.n_samples && tidy < conf.n_beam && tidz < conf.n_channel)
	{
		for(int k = 0; k < conf.n_pol; k++)
		{
			acc = {0,0};
			for(int i = 0; i < conf.n_antenna; i++)
			{
				acc = cuCaddf(acc, cuCmulf(idata[in_offset + i * conf.n_channel * conf.n_pol + k],
					  weights[weight_offset + i * conf.n_channel * conf.n_pol + k]));
			}
			odata[out_offset + k] = acc;
		}
	}
}

template<typename T=__half2>__global__
void simple_bf_tafpt_voltage(const __half2 *idata, __half2 *odata, const __half2 *weights, const bf_config_t conf)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;		// Time dimension
	int tidy = blockIdx.y * blockDim.y;									// Beam dimension
	int tidz = blockIdx.z * blockDim.z;									// Channel dimension

	int in_offset = tidx * conf.n_antenna * conf.n_channel * conf.n_pol + tidz * conf.n_pol;
	int out_offset = tidy * conf.n_samples * conf.n_channel * conf.n_pol + tidx * conf.n_channel * conf.n_pol + tidz * conf.n_pol ;
	int weight_offset = tidy * conf.n_antenna * conf.n_channel * conf.n_pol + tidz * conf.n_pol;

	__half2 acc;

	if(tidx < conf.n_samples && tidy < conf.n_beam && tidz < conf.n_channel)
	{
		for(int k = 0; k < conf.n_pol; k++)
		{
			acc = {0,0};
			for(int i = 0; i < conf.n_antenna; i++)
			{
				acc = __hadd2(acc, __hCmul2(idata[in_offset + i * conf.n_channel * conf.n_pol + k],
						weights[weight_offset + i * conf.n_channel * conf.n_pol + k]));
			}
			odata[out_offset + k] = acc;
		}
	}
}


template<typename T=float2>__global__
void bf_tfpat_voltage(const float2 *idata, float2 *odata, const float2 *weights, const bf_config_t conf)
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
	//	- n_antenna must be a multiple of WARP_SIZE 32
	//	- n_samples must be a multiple of WARP_SIZE 32

	int tidx = threadIdx.x;
	int bidx = blockIdx.x;      // Time dimension (T)
	int bidy = blockIdx.y;      // Beam dimension (B)
	int bidz = blockIdx.z;      // Channel dimension (F)

	int n_elements = conf.n_antenna * conf.n_pol; // Number of elements, product of antenna (A) and polarisation (P)
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
void bf_tfpat_voltage(const __half2 *idata, __half2 *odata, const __half2 *weights, const bf_config_t conf)
{

	// Grid layout: x = time; y = beam; z = channel
	// Block layout: A block consist of 1024 threads and 32 warps. Every warp
	// calculates both polarisations of one beam for one channel at one given time step.
	// Within a block 32 adjacent time steps for one beam are calculated (same channel).
	// Data products are as follow (glob mem):
	// 		idata: TFAP(t)
	//		odata: BFT
	//		weights: BFAP
	// constraints:
	//	- n_antenna must be a multiple of WARP_SIZE 32
	//	- n_samples must be a multiple of WARP_SIZE 32

	const int tidx = threadIdx.x;
	const int bidx = blockIdx.x;      // Time dimension (T)
	const int bidy = blockIdx.y;      // Beam dimension (B)
	const int bidz = blockIdx.z;      // Channel dimension (F)
#if __CUDA_ARCH__ >= 530
	const int n_elements = conf.n_antenna * conf.n_pol; // Number of elements, product of antenna (A) and polarisation (P)
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

#endif /* CUKERNELS_CUH_ */
