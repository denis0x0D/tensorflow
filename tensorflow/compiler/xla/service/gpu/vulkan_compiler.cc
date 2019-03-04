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

#include "tensorflow/compiler/xla/service/gpu/vulkan_compiler.h"

#include <stddef.h>
#include <string.h>
#include <map>
#include <mutex>  // NOLINT(build/c++11): only using std::call_once, not mutex.
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
//#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
//#include "tensorflow/compiler/xla/service/batch_dot_simplification.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
//#include "tensorflow/compiler/xla/service/cholesky_expander.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/convolution_group_converter.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_element_type_converter.h"
#include "tensorflow/compiler/xla/service/hlo_get_dimension_size_rewriter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
//#include "tensorflow/compiler/xla/service/indexed_array_analysis.h"
//#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/reduce_precision_insertion.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
//#include "tensorflow/compiler/xla/service/scatter_expander.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
//#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
//#include "tensorflow/compiler/xla/service/while_loop_invariant_code_motion.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/dynamic_annotations.h"

namespace xla {
namespace gpu {

VulkanAotCompilationOptions::VulkanAotCompilationOptions(
    string triple, string vulkan_name, string features, string entry_point_name)
    : triple_(std::move(triple)),
      vulkan_name_(std::move(vulkan_name)),
      features_(std::move(features)),
      entry_point_name_(std::move(entry_point_name)) {}

VulkanAotCompilationOptions::~VulkanAotCompilationOptions() = default;

se::Platform::Id VulkanAotCompilationOptions::PlatformId() const {
//  return se::vulkan::kVulkanPlatformId;
return se::host::kHostPlatformId;
}

VulkanAotCompilationResult::VulkanAotCompilationResult(
    ObjectFileData object_file_data, int64 result_buffer_index)
    : object_file_data_(std::move(object_file_data)),
      result_buffer_index_(result_buffer_index) {}

VulkanAotCompilationResult::~VulkanAotCompilationResult() = default;

VulkanCompiler::VulkanCompiler() {}

Status VulkanCompiler::RunHloPassesThroughLayoutAssn(HloModule* module,
                                                     bool is_aot_compile) {
  return Status::OK();
}

Status VulkanCompiler::RunHloPassesAfterLayoutAssn(HloModule* module,
                                                   bool is_aot_compile) {
  return Status::OK();
}

Status VulkanCompiler::RunHloPasses(HloModule* module, bool is_aot_compile) {
  return Status::OK();
}

// Align buffers to 16-byte boundaries.
constexpr int64 kMemoryAlignment = 16;
auto memory_alignment = [](LogicalBuffer::Color) { return kMemoryAlignment; };

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
VulkanCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                   const AotCompilationOptions& aot_options) {
  TF_RET_CHECK(!module_group->empty());
  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();

  // TODO: Check for Vulkan
  if (aot_options.PlatformId() != se::host::kHostPlatformId) {
    return InvalidArgument("Incompatible AOT compilation platform");
  }

  const VulkanAotCompilationOptions& options =
      static_cast<const VulkanAotCompilationOptions&>(aot_options);
  std::vector<std::unique_ptr<AotCompilationResult>> results;

  for (size_t i = 0; i < modules.size(); ++i) {
    HloModule* module = modules[i].get();
    VLOG(2) << "Compiling ahead-of-time: " << module->name();

    VLOG(2) << "Before optimization:";
    XLA_VLOG_LINES(2, module->ToString());

    TF_RETURN_IF_ERROR(RunHloPasses(module, /*is_aot_compile=*/true));

    VLOG(2) << "After optimization:";
    XLA_VLOG_LINES(2, module->ToString());

    TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                        ScheduleModule(module, BufferSizeBytesFunction()));

        // Run buffer analysis on the HLO graph. This analysis figures out which
    // temporary buffers are required to run the computation.
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<BufferAssignment> assignment,
        BufferAssigner::Run(module,
                            absl::make_unique<SequentialHloOrdering>(schedule),
                            BufferSizeBytesFunction(),
                            // TODO: Figure out memory alignment for Vulkan
                            memory_alignment,
                            /*allow_input_output_aliasing=*/false,
                            /*allocate_buffers_for_constants=*/true));
    // BufferAssignment::ToString() includes a header, so no need for us to
    // print one ourselves.
    XLA_VLOG_LINES(2, assignment->ToString());

    // const string xla_dump_optimized_hlo_proto_to =
    //    module->config().debug_options().xla_dump_optimized_hlo_proto_to();
    // if (!xla_dump_optimized_hlo_proto_to.empty()) {
    // HloProto proto = MakeHloProto(*module, *assignment);
    // TF_RETURN_IF_ERROR(protobuf_util::DumpProtoToDirectory(
    //   proto, xla_dump_optimized_hlo_proto_to, module->name()));
    // }
    // TODO: implement Vulkan Emitter
    // IrEmitter ir_emitter(*module, *assignment, &llvm_module,
    //                     std::move(instruction_to_profile_idx),
    //                    std::move(computation_to_profile_idx),
    //                   &target_machine_features,
    //                  /*emit_code_for_msan=*/false);
    //    TF_RETURN_IF_ERROR(ir_emitter.EmitConstantGlobals());
    // HloComputation* computation = module->entry_computation();
    // for (auto embedded_computation :
    //         computation->MakeEmbeddedComputationsList()) {
    //    if (embedded_computation->IsFusionComputation()) {
    //       continue;
    //     }
    //     TF_RETURN_IF_ERROR(
    //        ir_emitter
    //           .EmitComputation(
    //              embedded_computation, embedded_computation->name(),
    //             /*is_top_level_computation=*/false,
    //            schedule.sequence(embedded_computation).instructions())
    //       .status());
    //    }
    //   const string& entry_point_name = options.entry_point_name();
    //  ObjectFileData object_file_data(object_file->getBufferStart(),
    //                                 object_file->getBufferEnd());

    // TODO: How to create the buffers for Vulkan
    // CreateBufferInfosFromBufferAssignment(*assignment);

    //    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
    //                       assignment->GetUniqueTopLevelOutputSlice());

    // results.emplace_back(absl::make_unique<VulkanAotCompilationResult>(
    //   std::move(object_file_data), std::move(buffer_infos),
    //  result_slice.index(), std::move(hlo_profile_printer_data)));
  }

  return std::move(results);
}

se::Platform::Id VulkanCompiler::PlatformId() const {
  //  return se::vulkan::kVulkanPlatformId;
  return se::host::kHostPlatformId;
}

Status VulkanCompiler::RunHloPassesOnModuleGroup(
    HloModuleGroup* module_group,
    absl::Span<se::StreamExecutor* const> executors,
    DeviceMemoryAllocator* device_allocator) {
  return Unimplemented(
      "RunHloPassesOnModuleGroup is not supported for Vulkan backend");
}

StatusOr<std::vector<std::unique_ptr<Executable>>>
VulkanCompiler::RunBackendOnModuleGroup(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_exec,
    DeviceMemoryAllocator* device_allocator) {
  return Unimplemented(
      "RunBackendOnModuleGroup is not supported for Vulkan backend");
}

StatusOr<std::vector<std::unique_ptr<Executable>>> VulkanCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_execs,
    DeviceMemoryAllocator* device_allocator) {
  return Unimplemented("Compiler is not supported for Vulkan backend");
}

}  // namespace gpu
}  // namespace xla
