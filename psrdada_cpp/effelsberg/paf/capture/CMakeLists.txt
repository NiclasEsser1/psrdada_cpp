if(ENABLE_CUDA)

set(PSRDADA_CPP_EFFELSBERG_PAF_CAPTURING_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_effelsberg_paf_capture
    ${DEPENDENCY_LIBRARIES}
    -lpsrdada
    -lpsrdada_cpp
    -lboost_system
    -lboost_iostreams
    -lboost_serialization)

set(psrdada_cpp_effelsberg_paf_capture_src
    Types.hpp
    UDPSocket.hpp
    Receive.hpp
    Transmit.hpp
    )

cuda_add_library(${CMAKE_PROJECT_NAME}_effelsberg_paf_capture ${psrdada_cpp_effelsberg_paf_capture_src})

cuda_add_executable(codifrecv src/codifrecv_cli.cpp)
target_link_libraries(codifrecv ${PSRDADA_CPP_EFFELSBERG_PAF_CAPTURING_LIBRARIES})
install(TARGETS codifrecv DESTINATION bin)

cuda_add_executable(codifsend src/codifsend_cli.cpp)
target_link_libraries(codifsend ${PSRDADA_CPP_EFFELSBERG_PAF_CAPTURING_LIBRARIES})
install(TARGETS codifsend DESTINATION bin)

#add_subdirectory(test)
endif(ENABLE_CUDA)
