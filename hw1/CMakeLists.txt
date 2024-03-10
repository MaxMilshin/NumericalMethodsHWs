cmake_minimum_required(VERSION 3.28)
project(hw1)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_executable(hw1 main.cpp
        consecutive.cpp
        parallel.cpp
        preset.h
        parallel.h
        consecutive.h)
