set(profiling_src
    KernelStatistics.cuh
    profiling.cu
)
cuda_add_executable(profiling ${profiling_src} )
target_link_libraries(profiling ${PSRDADA_CPP_CRYOPAF_LIBRARIES})
