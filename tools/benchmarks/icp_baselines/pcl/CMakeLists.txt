# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.10)
project(pcl_icp)
set(CMAKE_CXX_STANDARD 17)
find_package(PCL 1.12 REQUIRED COMPONENTS common registration)
include_directories(${PCL_INCLUDE_DIRS})
add_executable(pcl_icp pcl_icp.cpp)
target_link_libraries(pcl_icp ${PCL_LIBRARIES})
