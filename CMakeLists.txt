cmake_minimum_required(VERSION 3.8.0)
project(FEM_NS)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/home/adityak/PACKAGES/petsc-3.6.4/arch-linux2-cxx-debug/lib/petsc/conf")
include_directories("/home/adityak/PACKAGES/petsc-3.6.4/include" "/home/adityak/PACKAGES/petsc-3.6.4/arch-linux2-cxx-debug/include" "/home/adityak/PACKAGES/petsc-3.6.4/arch-linux2-cxx-debug/lib")
LINK_DIRECTORIES("/home/adityak/PACKAGES/petsc-3.6.4/arch-linux2-cxx-debug/lib")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11") 

find_package(PETSc 3.6 COMPONENTS C REQUIRED)

add_executable(FEM_NS ${SOURCES} include/GaussQuad.h)

target_link_libraries(FEM_NS ${PETSC_LIBRARIES})


