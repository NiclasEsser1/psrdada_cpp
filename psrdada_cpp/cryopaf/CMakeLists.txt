if(ENABLE_CUDA)
set(PSRDADA_CPP_CRYOPAF_BEAMFORMING_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_cryopaf_beamforming
    ${DEPENDENCY_LIBRARIES}
    -lcublas
    -lcutensor)

set(psrdada_cpp_cryopaf_beamforming_src
    TextureMem.cuh
    DeviceFunc.cuh
    PowerBeamformer.cuh
    VoltageBeamformer.cuh
    Tensor.cuh)

cuda_add_library(${CMAKE_PROJECT_NAME}_cryopaf_beamforming ${psrdada_cpp_cryopaf_beamforming_src})

cuda_add_executable(beamforming src/beamforming_cli.cu)

target_link_libraries(beamforming ${PSRDADA_CPP_CRYOPAF_BEAMFORMING_LIBRARIES})
install(TARGETS beamforming DESTINATION bin)

add_subdirectory(test)
add_subdirectory(tools)

endif(ENABLE_CUDA)
