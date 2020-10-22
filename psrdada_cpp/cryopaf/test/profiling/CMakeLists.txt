set(profiling_src
    profiling.cu
)
cuda_add_executable(profiling ${profiling_src} )
target_link_libraries(profiling ${PSRDADA_CPP_CRYOPAF_BEAMFORMING_LIBRARIES})
