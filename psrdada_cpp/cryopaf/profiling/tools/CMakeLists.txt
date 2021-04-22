set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")

if(ENABLE_CUDA)
set(PSRDADA_CPP_CRYOPAF_TOOLS_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_cryopaf_tools
    ${DEPENDENCY_LIBRARIES})

set(psrdada_cpp_cryopaf_tools_src
    device_query.cpp)

cuda_add_library(${CMAKE_PROJECT_NAME}_cryopaf_tools ${psrdada_cpp_cryopaf_tools_src})

cuda_add_executable(device_query device_query.cpp)

endif(ENABLE_CUDA)
