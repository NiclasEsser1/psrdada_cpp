#ifndef TENSOR_CUH_
#define TENSOR_CUH

#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cutensor.h>
#include <unordered_map>
#include <boost/any.hpp>
#include <unordered_map>

#include "psrdada_cpp/cuda_utils.hpp"


namespace psrdada_cpp {
namespace cryopaf {
namespace beamforming {

template<class T>
struct Tensor
{
    Tensor(T* mem_ptr, std::vector<int> mode, std::unordered_map<int, int64_t> extent);
    ~Tensor();
    void parse_type();
    void init_desc(cutensorHandle_t *handle);
    void init_alignment(cutensorHandle_t *handle);


    T* _mem_ptr;
    cudaDataType_t _type;
    cutensorTensorDescriptor_t _desc;
    cutensorOperator_t _operator = CUTENSOR_OP_IDENTITY;
    std::vector<int> _mode;
    std::vector<int64_t> _extent;
    // std::vector<int64_t> _stride;
    int _size;
    int _elements = 1;
    uint32_t _alignment;
};
// template struct Tensor<float>;

template<class TypeA, class TypeB, class TypeC>
class TensorGETT
{
public:
    TensorGETT(Tensor<TypeA> a, Tensor<TypeB> b, Tensor<TypeC> c);
    ~TensorGETT();

    void init();
    void process(float alpha = 1, float beta = 0);

    // void set(cutensorAlgo_t algo){_algo = algo;}
    // void set(cutensorOperator_t operator){_operator = operator;}

private:
    cutensorHandle_t _handle;
    cutensorContractionDescriptor_t _desc;
    cutensorContractionFind_t _find;
    cutensorContractionPlan_t _plan;
    cutensorComputeType_t _type = CUTENSOR_COMPUTE_32F;
    cutensorAlgo_t _algo = CUTENSOR_ALGO_DEFAULT;
    cutensorOperator_t _operator = CUTENSOR_OP_IDENTITY;
    cutensorWorksizePreference_t _work_preference = CUTENSOR_WORKSPACE_RECOMMENDED;
    uint64_t _worksize = 0;
    void* _work = nullptr;

    Tensor<TypeA> _a;
    Tensor<TypeB> _b;
    Tensor<TypeC> _c;

    cudaStream_t _stream = 0;
    //std::tuple<Tensor<TypeA>, Tensor<TypeB>, Tensor<TypeC>> _tensors
};

#include "Tensor.cu"
}
}
}

#include "psrdada_cpp/cryopaf/src/Tensor.cu"

#endif /** TENSOR_CUH_ **/
