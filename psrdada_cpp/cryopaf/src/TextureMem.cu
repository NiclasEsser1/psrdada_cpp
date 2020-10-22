#ifdef TEXTUREMEM_H_

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{


template<class T>
TextureMem<T>::TextureMem(std::size_t width, std::size_t height, std::size_t depth, int dev_id)
	: x(width), y(height), z(depth), id(dev_id)
{
	// Set device to use
	CUDA_ERROR_CHECK(cudaSetDevice(id));
	// Retrieve device properties
	CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, id));

	vol = make_cudaExtent(x, y, z);
	byte_size = x*y*z*sizeof(T);
	if(x * sizeof(T) > prop.maxTexture3D[0] || y * sizeof(T) > prop.maxTexture3D[0] || z * sizeof(T) > prop.maxTexture3D[0] )
	{
		std::cout << "The requested size for texture memory exceeds the size provided by device " << std::endl;
	}
	print_layout();
	memset(&res_desc, 0, sizeof(cudaResourceDesc));
	memset(&tex_desc, 0, sizeof(cudaTextureDesc));

	if constexpr (std::is_same<T, __half2>::value)
		chan_desc = cudaCreateChannelDescHalf();
	else
		chan_desc = cudaCreateChannelDesc<T>();

	CUDA_ERROR_CHECK(cudaMalloc3DArray(&array, &chan_desc, vol))

}

template<class T>
TextureMem<T>::TextureMem(cudaExtent volume)
	: vol(volume)
{
	x = vol.width;
	y = vol.height;
	z = vol.depth;
	byte_size = x*y*z*sizeof(T);
	memset(&tex_desc, 0, sizeof(cudaTextureDesc));
	memset(&res_desc, 0, sizeof(cudaResourceDesc));

}

template<class T>
TextureMem<T>::~TextureMem()
{
	cudaDestroyTextureObject(tex);
}

template<class T>
void TextureMem<T>::set(thrust::device_vector<T> vec)
{
	T* p = thrust::raw_pointer_cast(vec.data()); // Pointer to global memory
	pitched_ptr = make_cudaPitchedPtr((void *)p, vol.width*sizeof(T), vol.width, vol.height);
	copy_params.srcPtr = pitched_ptr;
	copy_params.srcPos = {0,0,0};
	copy_params.dstArray = array;
	copy_params.dstPos = {0,0,0};
	copy_params.extent = vol;
	copy_params.kind = cudaMemcpyDeviceToDevice;

	CUDA_ERROR_CHECK(cudaMemcpy3D(&copy_params));

	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = array;

	tex_desc.normalizedCoords = false; // access with normalized texture coordinates
	tex_desc.filterMode = cudaFilterModePoint; // linear interpolation
	for(int i = 0; i < dim; i++)
		tex_desc.addressMode[i] = cudaAddressModeWrap;

	tex_desc.readMode = cudaReadModeElementType;

	CUDA_ERROR_CHECK(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, NULL));
}

template<class T>
void TextureMem<T>::set(thrust::host_vector<T> vec)
{
	T* p = thrust::raw_pointer_cast(vec.data()); // Pointer to global memory
	pitched_ptr = make_cudaPitchedPtr((void *)p, vol.width*sizeof(T), vol.width, vol.height);
	copy_params.srcPtr = pitched_ptr;
	copy_params.srcPos = {0,0,0};
	copy_params.dstArray = array;
	copy_params.dstPos = {0,0,0};
	copy_params.extent = vol;
	copy_params.kind = cudaMemcpyHostToDevice;

	CUDA_ERROR_CHECK(cudaMemcpy3D(&copy_params));
	// printf("%f + i %f\n",(float)array[0][0][0].x, (float)array[0][0][0].y);

	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = array;

	tex_desc.normalizedCoords = false; // access with normalized texture coordinates
	tex_desc.filterMode = cudaFilterModePoint; // linear interpolation
	for(int i = 0; i < dim; i++)
		tex_desc.addressMode[i] = cudaAddressModeWrap;

	tex_desc.readMode = cudaReadModeElementType;

	CUDA_ERROR_CHECK(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, NULL));
}


template<class T>
void TextureMem<T>::print_layout()
{
	std::cout << "Texture alignment: (x = " << std::to_string(x) << "; y = " << std::to_string(y) << "; z = " << std::to_string(z) << ")" << std::endl;
}

template<class T>
void TextureMem<T>::resize(std::size_t width, std::size_t height, std::size_t depth)
{
}

template<class T>
void TextureMem<T>::free_texture()
{
}

} // namespace beamforming
} // namespace cryopaf
} // namespace psrdada_cpp

#endif
