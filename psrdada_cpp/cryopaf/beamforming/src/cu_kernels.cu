#include "psrdada_cpp/cryopaf/beamforming/cu_kernels.cuh"

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{




template<typename T>__global__
void simple_bf_tafpt_power(const thrust::complex<T> *idata, T *odata, const thrust::complex<T> *weights, const bf_config_t *_conf)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;	// Time dimension
	int tidy = blockIdx.y * blockDim.y;								// Beam dimension
	int tidz = blockIdx.z * blockDim.z;								// Antenna dimension
	thrust::complex<double> acc(0,0);

	int in_offset = tidx * _conf->n_antenna * _conf->n_channel * _conf->n_pol + tidz * _conf->n_pol;
	int out_offset = tidy * _conf->n_samples * _conf->n_channel + tidx * _conf->n_channel;
	int weight_offset = tidy * _conf->n_antenna * _conf->n_channel * _conf->n_pol + tidz * _conf->n_pol;

	if(tidx < _conf->n_samples && tidy < _conf->n_beam && tidz < _conf->n_channel)
	{
		for(int i = 0; i < _conf->n_antenna; i++)
		{
			acc.real(0); acc.imag(0);
			for(int k = 0; k < _conf->n_pol; k++)
			{
				acc += idata[in_offset + i * _conf->n_channel * _conf->n_pol + k]
					  * weights[weight_offset + i * _conf->n_channel * _conf->n_pol + k];
			}
			odata[out_offset + tidz] += (T)(acc.real()*acc.real() + acc.imag()*acc.imag());
		}
	}
}
// template __global__ void simple_bf_tafpt_power<short>
// 	(const thrust::complex<short>*, short*, const thrust::complex<short>*, const bf_config_t*);
template __global__ void simple_bf_tafpt_power<float>
	(const thrust::complex<float>*, float*, const thrust::complex<float>*, const bf_config_t*);
template __global__ void simple_bf_tafpt_power<double>
	(const thrust::complex<double>*, double*, const thrust::complex<double>*, const bf_config_t*);



template<typename T>__global__
void bf_tafpt_power(const thrust::complex<T> *idata, T *odata, const thrust::complex<T> *weights, const bf_config_t *_conf)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;	// Time dimension
	int tidy = blockIdx.y * blockDim.y;								// Beam dimension
	int tidz = blockIdx.z * blockDim.z;								// Antenna dimension
}
// template __global__ void bf_tafpt_power<short>
// 	(const thrust::complex<short>*, short*, const thrust::complex<short>*, const bf_config_t*);
template __global__ void bf_tafpt_power<float>
	(const thrust::complex<float>*, float*, const thrust::complex<float>*, const bf_config_t*);
template __global__ void bf_tafpt_power<double>
	(const thrust::complex<double>*, double*, const thrust::complex<double>*, const bf_config_t*);



template<typename T>__global__
void simple_bf_tafpt_voltage(const thrust::complex<T> *idata, thrust::complex<T> *odata, const thrust::complex<T> *weights, const bf_config_t *conf)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;		// Time dimension
	int tidy = blockIdx.y * blockDim.y;									// Beam dimension
	int tidz = blockIdx.z * blockDim.z;									// Channel dimension

	int in_offset = tidx * conf->n_antenna * conf->n_channel * conf->n_pol + tidz * conf->n_pol;
	int out_offset = tidy * conf->n_samples * conf->n_channel * conf->n_pol + tidx * conf->n_channel * conf->n_pol + tidz * conf->n_pol ;
	int weight_offset = tidy * conf->n_antenna * conf->n_channel * conf->n_pol + tidz * conf->n_pol;

	thrust::complex<double> acc;

	if(tidx < conf->n_samples && tidy < conf->n_beam && tidz < conf->n_channel)
	{
		for(int k = 0; k < conf->n_pol; k++)
		{
			acc.real(0); acc.imag(0);
			for(int i = 0; i < conf->n_antenna; i++)
			{
				acc += idata[in_offset + i * conf->n_channel * conf->n_pol + k]
					* weights[weight_offset + i * conf->n_channel * conf->n_pol + k];
			}
			odata[out_offset + k] = (thrust::complex<T>)acc;
		}
	}
}
// template __global__ void simple_bf_tafpt_voltage<short>
// 	(const thrust::complex<short>*, thrust::complex<short>*, const thrust::complex<short>*, const bf_config_t*);
template __global__ void simple_bf_tafpt_voltage<float>
	(const thrust::complex<float>*, thrust::complex<float>*, const thrust::complex<float>*, const bf_config_t*);
template __global__ void simple_bf_tafpt_voltage<double>
	(const thrust::complex<double>*, thrust::complex<double>*, const thrust::complex<double>*, const bf_config_t*);




template<typename T>__global__
void bf_tfpat_voltage(const thrust::complex<T> *idata, thrust::complex<T> *odata, const thrust::complex<T> *weights, const bf_config_t *conf)
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

	int tidx = threadIdx.x;
	int bidx = blockIdx.x;      // Time dimension (T)
	int bidy = blockIdx.y;      // Beam dimension (B)
	int bidz = blockIdx.z;      // Channel dimension (F)

	int n_elements = conf->n_antenna * conf->n_pol; // Number of elements, product of antenna (A) and polarisation (P)
	int warp = tidx / WARP_SIZE; // Calculate the current warp
	int warp_idx = tidx % WARP_SIZE;    // thread index -> warp index

	// Each thread has its own indices for accessing the global memory (idata, odata, weights).
	int idata_glob_idx = warp_idx + n_elements * (bidx * WARP_SIZE * conf->n_channel + warp * conf->n_channel + bidz);
	int weights_glob_idx = warp_idx + n_elements * (bidy * conf->n_channel + bidz);
	int output_glob_idx = (bidy * conf->n_samples * conf->n_channel
			+ bidz * conf->n_samples + bidx * WARP_SIZE + warp)*conf->n_pol + warp_idx; // multiplied by two, since two polarisations

	// To enable higher throughput and more efficient data transfer, shared memory
	// is required. The size in bytes of shared memory is calculated as follows:
	//	shared_mem_bytes = sizeof(T) * (A * P * (WARP_SIZE + 1) + NTHREADS)
	//	idata  = sizeof(T) * A * P * WARP_SIZE	<- every warp loads data of all elements at one time step
	// 	weight = sizeof(T) * A * P							<- weights are the same for all warps
	//  odata  = sizeof(T) * NTHREADS						<- Every thread calculates one output sample
	extern __shared__ __align__(sizeof(thrust::complex<T>)) unsigned char shared_mem[];	// dynamically allocated
	thrust::complex<T>* shared_idata = reinterpret_cast<thrust::complex<T>*>(&shared_mem[0]);	// idata space comes first
	thrust::complex<T>* shared_weights = reinterpret_cast<thrust::complex<T>*>(&shared_mem[n_elements * WARP_SIZE * sizeof(thrust::complex<T>)]);	// weight space with idata space as offset

	// To prevent overflows when using integer values, the datatype of shared_odata has to be float2 (cuCOmplex).
	thrust::complex<T>* shared_odata = reinterpret_cast<thrust::complex<T>*>(&shared_mem[n_elements * (WARP_SIZE + 1) * sizeof(thrust::complex<T>)]);

	shared_odata[warp * WARP_SIZE + warp_idx] = thrust::complex<T>(0,0);	// intialize output with zeros

	thrust::complex<double> acc(0,0); // local register for storing intermediate results

	// Load idata and weights into shared memory for every warp
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
	for(int i = 0; i < n_elements; i+=WARP_SIZE)
	{
		acc += shared_idata[warp * n_elements + i + warp_idx] * shared_weights[i + warp_idx];
	}
	shared_odata[tidx] = (thrust::complex<T>)acc;

	__syncthreads(); // Synchronize all threads within a block, to ensure all computitations are done.

	// Since odata contains 1024 samples which have not been combined to 32 time steps a reduction is required.
	int i = WARP_SIZE / 2;
	// This reduction may no be very efficient since many threads within a warp are idled
	while(i != conf->n_pol - 1)
	{
		if(warp_idx < i)
			shared_odata[tidx] += shared_odata[tidx + i];
		__syncthreads();
		i /= 2;
	}
	// After reduction the first two samples in shared_odata with warp offset contains both polarisations.
	// So, if warp_idx is 0 or 1, assign the samples to the global output buffer. In total 64
	// samples are transfered back to the global memory for each block.
	if(warp_idx < conf->n_pol)
	{
		// TODO: In case of integer inputs, conversion is implemented here!!!
		odata[output_glob_idx] = shared_odata[warp * WARP_SIZE + warp_idx];	// Polarisation 0 and 1
	}

}
// template __global__ void bf_tfpat_voltage<short>
// 	(const thrust::complex<short>*, thrust::complex<short>*, const thrust::complex<short>*, const bf_config_t*);
template __global__ void bf_tfpat_voltage<float>
	(const thrust::complex<float>*, thrust::complex<float>*, const thrust::complex<float>*, const bf_config_t*);
template __global__ void bf_tfpat_voltage<double>
	(const thrust::complex<double>*, thrust::complex<double>*, const thrust::complex<double>*, const bf_config_t*);

} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp
