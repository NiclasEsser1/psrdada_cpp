if(ENABLE_CUDA)
set(PSRDADA_CPP_CRYOPAF_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_cryopaf
    ${DEPENDENCY_LIBRARIES}
    -lboost_system
    -lpthread)

set(psrdada_cpp_cryopaf_src
    Unpacker.cuh
    Beamformer.cuh
    BufferTypes.cuh
    Pipeline.cuh
    PipelineInterface.cuh)

cuda_add_library(${CMAKE_PROJECT_NAME}_cryopaf ${psrdada_cpp_cryopaf_src})

cuda_add_executable(beamforming src/beamforming_cli.cu)
target_link_libraries(beamforming ${PSRDADA_CPP_CRYOPAF_LIBRARIES})
install(TARGETS beamforming DESTINATION bin)

cuda_add_executable(weightupdater src/weightupdater_cli.cu)
target_link_libraries(weightupdater ${PSRDADA_CPP_CRYOPAF_LIBRARIES})
install(TARGETS weightupdater DESTINATION bin)

add_subdirectory(profiling)
add_subdirectory(test)

endif(ENABLE_CUDA)
