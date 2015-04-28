#The name of the project
PROJECT(gpu-fan)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# default installation directory 
IF(CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
  SET(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}" )
ENDIF(CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")

#Searching CUDA
SET(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
FIND_PACKAGE(CUDA)

INCLUDE_DIRECTORIES(src)

#Adding src directory to the project
ADD_SUBDIRECTORY(src)
