set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")

if(ENABLE_CUDA)
set(PSRDADA_CPP_CRYOPAF_BEAMFORMING_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_cryopaf_beamforming
    ${DEPENDENCY_LIBRARIES})

set(psrdada_cpp_cryopaf_beamforming_src
    src/cu_beamformer.cu
    src/cu_kernels.cu)

cuda_add_library(${CMAKE_PROJECT_NAME}_cryopaf_beamforming ${psrdada_cpp_cryopaf_beamforming_src})

cuda_add_executable(beamforming src/beamforming_cli.cu)
target_link_libraries(beamforming ${PSRDADA_CPP_CRYOPAF_BEAMFORMING_LIBRARIES})
install(TARGETS beamforming DESTINATION bin)

add_subdirectory(test)

endif(ENABLE_CUDA)
