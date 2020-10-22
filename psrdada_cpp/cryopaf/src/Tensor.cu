#ifdef TENSOR_CUH_

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{

template<class T>
Tensor<T>::Tensor(T* mem_ptr, std::vector<int> mode, std::unordered_map<int, int64_t> extent)
    : _mode(mode), _mem_ptr(mem_ptr)
{
    for(auto key : _mode)
    {
        _extent.push_back(extent[key]);
        _elements *= extent[key];
        // printf("_extent[%c] = %d; extent[%c] = %d\n", key, _elements, key, extent[key]);
    }
    _size = sizeof(T) * _elements;
    parse_type();
}


template<class T>
Tensor<T>::~Tensor()
{

}


template<class T>
void Tensor<T>::parse_type()
{
    if constexpr (std::is_same<T, __half>::value)
        _type = CUDA_R_16F;
    else if constexpr (std::is_same<T, __half2>::value)
        _type = CUDA_C_16F;
    else if constexpr (std::is_same<T, float>::value)
        _type = CUDA_R_32F;
    else if constexpr (std::is_same<T, float2>::value)
        _type = CUDA_C_32F;
    else if constexpr (std::is_same<T, double>::value)
        _type = CUDA_R_64F;
    else if constexpr (std::is_same<T, double2>::value)
        _type = CUDA_C_64F;
    else if constexpr (std::is_same<T, char>::value)
        _type = CUDA_R_8I;
    else if constexpr (std::is_same<T, char2>::value)
        _type = CUDA_C_8I;
    else if constexpr (std::is_same<T, unsigned char>::value)
        _type = CUDA_R_8U;
    // else if constexpr (std::is_same<T, unsigned char2>::value)
    //     _type = CUDA_C_8U;
    else if constexpr (std::is_same<T, int>::value)
        _type = CUDA_R_32I;
    else if constexpr (std::is_same<T, int2>::value)
        _type = CUDA_C_32I;
    else if constexpr (std::is_same<T, unsigned int>::value)
        _type = CUDA_R_32U;
    // else if constexpr (std::is_same<T, unsigned int>::value)
    //     _type = CUDA_C_32U;
}

template<class T>
void Tensor<T>::init_desc(cutensorHandle_t *handle)
{
    // if(_type == CUDA_R_32F)
        printf("cudtype: %d\n",_type);
    CUTENSOR_ERROR_CHECK(cutensorInitTensorDescriptor(handle,
        &_desc, _mode.size(), _extent.data(), NULL, _type, _operator));
}

template<class T>
void Tensor<T>::init_alignment(cutensorHandle_t *handle)
{
    CUTENSOR_ERROR_CHECK(cutensorGetAlignmentRequirement(handle,
        (void*)_mem_ptr, &_desc, &_alignment));
    printf("Alignment %u \n", _alignment);
}




template<class TypeA, class TypeB, class TypeC>
TensorGETT<TypeA, TypeB, TypeC>::TensorGETT(Tensor<TypeA> a, Tensor<TypeB> b, Tensor<TypeC> c)
    : _a(a), _b(b), _c(c)
{
}

template<class TypeA, class TypeB, class TypeC>
TensorGETT<TypeA, TypeB, TypeC>::~TensorGETT()
{

}

template<class TypeA, class TypeB, class TypeC>
void TensorGETT<TypeA, TypeB, TypeC>::init()
{
    CUTENSOR_ERROR_CHECK(cutensorInit(&_handle));

    _a.init_desc(&_handle);
    _b.init_desc(&_handle);
    _c.init_desc(&_handle);
    _a.init_alignment(&_handle);
    _b.init_alignment(&_handle);
    _c.init_alignment(&_handle);

    CUTENSOR_ERROR_CHECK(cutensorInitContractionDescriptor(&_handle, &_desc,
        &_a._desc, _a._mode.data(), _a._alignment,
        &_b._desc, _b._mode.data(), _b._alignment,
        &_c._desc, _c._mode.data(), _c._alignment,
        &_c._desc, _c._mode.data(), _c._alignment,
        _type));
    CUTENSOR_ERROR_CHECK(cutensorInitContractionFind(&_handle, &_find, _algo));

    CUTENSOR_ERROR_CHECK(cutensorContractionGetWorkspace(&_handle, &_desc,
        &_find, _work_preference, &_worksize));

    if(_worksize > 0) {CUDA_ERROR_CHECK(cudaMalloc(&_work, _worksize));}

    CUTENSOR_ERROR_CHECK(cutensorInitContractionPlan(&_handle, &_plan, &_desc, &_find, _worksize));
}

template<class TypeA, class TypeB, class TypeC>
void TensorGETT<TypeA, TypeB, TypeC>::process(float alpha, float beta)
{
    CUTENSOR_ERROR_CHECK(cutensorContraction(&_handle, &_plan, (void*)&alpha, _a._mem_ptr, _b._mem_ptr,
        (void*)&beta, _c._mem_ptr, _c._mem_ptr, _work, _worksize, _stream));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

}
}
}

#endif
