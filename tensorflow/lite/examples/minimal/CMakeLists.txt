#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Builds the minimal Tensorflow Lite example.

cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../"
    ABSOLUTE
  )
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

include(ExternalProject)
externalproject_add(benchmark
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/benchmark
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -DGOOGLETEST_PATH=${PROJECT_SOURCE_DIR}/googletest -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR} -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=29 ${PROJECT_SOURCE_DIR}/benchmark
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --target install
    INSTALL_COMMAND "")

add_executable(minimal
  minimal.cc
  ${PROJECT_SOURCE_DIR}/../../profiling/atrace_profiler.cc
)

add_dependencies(minimal benchmark)

add_library(benchmark_lib STATIC IMPORTED)
set_property(TARGET benchmark_lib PROPERTY IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/lib/libbenchmark.a)

target_include_directories(minimal PUBLIC ${PROJECT_BINARY_DIR}/include)

target_link_libraries(minimal
  tensorflow-lite
  ${CMAKE_DL_LIBS}
  benchmark_lib
)
