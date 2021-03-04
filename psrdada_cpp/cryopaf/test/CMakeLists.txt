include_directories(${GTEST_INCLUDE_DIR})

link_directories(${GTEST_LIBRARY_DIR})
set(CMAKE_CXX_STANDARD 17)
set(gtest_beamforming_src
    src/BeamformTester.cu
    src/UnpackerTester.cu
    src/gtest_beamforming.cu
)
cuda_add_executable(gtest_beamforming ${gtest_beamforming_src} )
target_link_libraries(gtest_beamforming ${PSRDADA_CPP_CRYOPAF_LIBRARIES})
add_test(gtest_beamforming gtest_beamforming --test_data "${CMAKE_CURRENT_LIST_DIR}/data")

add_subdirectory(profiling)
