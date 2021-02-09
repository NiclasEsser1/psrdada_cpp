#ifndef UTILS_CU
#define UTILS_CU

/** UTILS **/
__device__ __half2 __hCmul2(__half2 a, __half2 b)
{
		const __half r = a.x * b.x - a.y * b.y;
		const __half i = a.x * b.y + a.y * b.x;

		__half2 val; val.x = r; val.y = i;
		return val;
}

template<typename T>
__host__ __device__ T cmadd(T a, T b, T c)
{
	T val;
	val.x = a.x * b.x - a.y * b.y + c.x;
	val.y = a.x * b.y + a.y * b.x + c.y;
	return val;
}

template<typename T>
__host__ __device__ T cadd(T a, T b)
{
	T val;
	val.x = a.x + b.x;
	val.y = a.y + b.y;
	return val;
}

template<typename T>
__host__ __device__ T csub(T a, T b)
{
	T val;
	val.x = a.x - b.x;
	val.y = a.y - b.y;
	return val;
}

template<typename T>
__host__ __device__ double cabs(T a)
{
	return (double)sqrt((double)(a.x * a.x + a.y * a.y));
}

#endif
