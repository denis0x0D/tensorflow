/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_VULKAN_COMPILER_H
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_VULKAN_COMPILER_H

#include <memory>
#include <string>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
//#include "tensorflow/stream_executor/vulkan/vulkan_platform_id.h"

namespace xla {
namespace gpu {

class VulkanAotCompilationOptions : public AotCompilationOptions {
 public:
  VulkanAotCompilationOptions(string triple, string vulkan_name,
                              string features, string entry_point_name);
  ~VulkanAotCompilationOptions() override;

  se::Platform::Id PlatformId() const override;

  const string& triple() const { return triple_; }
  const string& vulkan_name() const { return vulkan_name_; }
  const string& features() const { return features_; }
  const string& entry_point_name() const { return entry_point_name_; }

 private:
  const string triple_;
  const string vulkan_name_;
  const string features_;
  const string entry_point_name_;
};

class VulkanAotCompilationResult : public AotCompilationResult {
 public:
  VulkanAotCompilationResult(ObjectFileData object_file_data,
                             int64 result_buffer_index);

  ~VulkanAotCompilationResult();

  const ObjectFileData& object_file_data() const { return object_file_data_; }
  int64 result_buffer_index() const { return result_buffer_index_; }

 private:
  // Contains the compiled computation: an object file.
  const ObjectFileData object_file_data_;
  // result of the computation.  This buffer should be passed into the output
  // parameter when calling the compiled computation.
  const int64 result_buffer_index_;
};

class VulkanCompiler : public Compiler {
 public:
  VulkanCompiler();
  ~VulkanCompiler() override {}

  using Compiler::RunBackend;
  using Compiler::RunHloPasses;

  Status RunHloPassesOnModuleGroup(
      HloModuleGroup* module_group,
      absl::Span<se::StreamExecutor* const> executors,
      DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::vector<std::unique_ptr<Executable>>> RunBackendOnModuleGroup(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_exec,
      DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_execs,
      DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options);

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
      DeviceMemoryAllocator* device_allocator) override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      DeviceMemoryAllocator* device_allocator) override;

 private:
  Status RunHloPassesThroughLayoutAssn(HloModule* module, bool is_aot_compile);
  Status RunHloPassesAfterLayoutAssn(HloModule* module, bool is_aot_compile);
  Status RunHloPasses(HloModule* module, bool is_aot_compile);
  se::Platform::Id PlatformId() const override;

  tensorflow::mutex mutex_;
  TF_DISALLOW_COPY_AND_ASSIGN(VulkanCompiler);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_VULAKN_COMPILER_H
