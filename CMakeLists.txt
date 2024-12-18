# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(CUDAX_SAMPLES CUDA CXX)

# This example uses the CMake Package Manager (CPM) to simplify fetching CCCL from GitHub
# For more information, see https://github.com/cpm-cmake/CPM.cmake
include(cmake/CPM.cmake)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
  NAME CCCL
  GITHUB_REPOSITORY "nvidia/cccl"
  GIT_TAG "main"
  # The following is required to make the `CCCL::cudax` target available:
  OPTIONS "CCCL_ENABLE_UNSTABLE ON"
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
find_package(NCCL REQUIRED)

# Default to building for the GPU on the current system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# If you're building an executable
add_executable(example1 example1.cu)
target_link_libraries(example1 PUBLIC cuda)

if (CMAKE_CUDA_COMPILER)
    target_compile_options(example1 PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_compile_options(example1 PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_compile_options(example1 PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

target_link_libraries(example1 PRIVATE CCCL::CCCL CCCL::cudax)
target_link_libraries(example1 PRIVATE ${NCCL_LIBRARY})
