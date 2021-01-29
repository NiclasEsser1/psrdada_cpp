if(ENABLE_CUDA)

set(PSRDADA_CPP_EFFELSBERG_PAF_CAPTURING_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_effelsberg_paf_capture
    ${DEPENDENCY_LIBRARIES}
    #-lpsrdada
    -lpsrdada_cpp
    -lboost_system)

set(psrdada_cpp_effelsberg_paf_capture_src
    src/Socket.cpp
    src/CaptureMonitor.cpp
    src/Catcher.cpp
    CaptureController.hpp
    )

cuda_add_library(${CMAKE_PROJECT_NAME}_effelsberg_paf_capture ${psrdada_cpp_effelsberg_paf_capture_src})

cuda_add_executable(capture src/capture_cli.cpp)
target_link_libraries(capture ${PSRDADA_CPP_EFFELSBERG_PAF_CAPTURING_LIBRARIES})
install(TARGETS capture DESTINATION bin)

add_subdirectory(test)
endif(ENABLE_CUDA)
