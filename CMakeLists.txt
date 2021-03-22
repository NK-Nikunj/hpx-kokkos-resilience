cmake_minimum_required(VERSION 3.13)

project(Prototype CXX)

# Setting up dependencies
find_package(HPX 1.5.0 REQUIRED)
find_package(Kokkos REQUIRED)
find_package(HPXKokkos REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)


include_directories(src)

enable_testing()
add_subdirectory(tests)
