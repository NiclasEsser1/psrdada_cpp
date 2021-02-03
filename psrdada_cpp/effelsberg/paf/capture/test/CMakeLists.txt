include_directories(${GTEST_INCLUDE_DIR})

link_directories(${GTEST_LIBRARY_DIR})

set(gtest_capture_src
    CaptureTester.hpp
    PacketTester.hpp
    src/gtest_capture.cpp
)
add_executable(gtest_capture ${gtest_capture_src} )
target_link_libraries(gtest_capture ${PSRDADA_CPP_EFFELSBERG_PAF_CAPTURING_LIBRARIES})
add_test(gtest_capture gtest_capture --test_data "${CMAKE_CURRENT_LIST_DIR}/data")
