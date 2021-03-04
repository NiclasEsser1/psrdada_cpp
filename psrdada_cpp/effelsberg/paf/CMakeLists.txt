if(ENABLE_CUDA)
set(CMAKE_CXX_STANDARD 17)
set(PSRDADA_CPP_EFFELSBERG_PAF_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_effelsberg_paf
    ${DEPENDENCY_LIBRARIES})

set(psrdada_cpp_effelsberg_paf_src
    src/Unpacker.cu)

cuda_add_library(${CMAKE_PROJECT_NAME}_effelsberg_paf ${psrdada_cpp_effelsberg_paf_src})


add_subdirectory(test)
endif(ENABLE_CUDA)
