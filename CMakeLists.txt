cmake_minimum_required(VERSION 3.8.0)
project(FEM_NS)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/petsc/conf")
include_directories("$ENV{PETSC_DIR}/include" "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include" "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib")
LINK_DIRECTORIES("$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11") 

find_package(PETSc 3.6 COMPONENTS C REQUIRED)

add_executable(FEM_NS ${SOURCES} include/GaussQuad.h)

target_link_libraries(FEM_NS ${PETSC_LIBRARIES})


