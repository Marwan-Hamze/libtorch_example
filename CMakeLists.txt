cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch_example)

find_package(Torch REQUIRED)

# set(CMAKE_PREFIX_PATH /home/devel/libtorch)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(libtorch_example cpp_policy.cpp)
target_link_libraries(libtorch_example "${TORCH_LIBRARIES}")
set_property(TARGET libtorch_example PROPERTY CXX_STANDARD 17)

#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)