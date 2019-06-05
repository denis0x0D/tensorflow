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
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batch_dot_simplification.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/cholesky_expander.h"
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
#include "tensorflow/compiler/xla/service/indexed_array_analysis.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/reduce_precision_insertion.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/scatter_expander.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_invariant_code_motion.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/compiler/xla/service/gpu/spirv_ir_emitter.h"

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
  HloPassPipeline pipeline("HLO passes through layout assignment");
  pipeline.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                            /*allow_mixed_precision=*/false);
  pipeline.AddPass<DynamicIndexSplitter>();
  // TODO: VulkanHloSupportChecker
  //  pipeline.AddPass<CpuHloSupportChecker>();

  ReducePrecisionInsertion::AddPasses(
      &pipeline, module->config().debug_options(),
      ReducePrecisionInsertion::PassTiming::BEFORE_OPTIMIZATION);

  pipeline.AddPass<MapInliner>();

  pipeline.AddPass<CholeskyExpander>();
  pipeline.AddPass<TriangularSolveExpander>();

  // TODO(b/65775800): Fix wrong output bug in Call and remove the CallInliner
  // pass.
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<BatchDotSimplification>();
  pipeline.AddPass<DotDecomposer>(/*decompose_batch_dot=*/false);
  auto cost_model = [](HloInstruction* conv) {
    // We need a cost model for CPUs. Currently, do nothing.
    return false;
  };
  pipeline.AddPass<ConvolutionGroupConverter>(
      cost_model,
      /*convert_batch_groups_only=*/true);
  pipeline.AddPass<ConvolutionGroupConverter>(
      cost_model,
      /*convert_batch_groups_only=*/false);
  // TODO: Figure out ConvCanonicalization for Vulkan
  //  pipeline.AddPass<ConvCanonicalization>(target_machine_features);
  {
    auto& pass =
        pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification");
    pass.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                          /*allow_mixed_precision=*/false);

    pass.AddPass<BatchNormExpander>(
        /*rewrite_training_op=*/true,
        /*rewrite_inference_op=*/true,
        /*rewrite_grad_op=*/true);
    pipeline.AddPass<HloGetDimensionSizeRewriter>();
    AlgebraicSimplifierOptions options;
    options.set_enable_dot_strength_reduction(false);
    pass.AddPass<AlgebraicSimplifier>(options);
    pass.AddPass<SortSimplifier>();
    pass.AddPass<HloDCE>();

    // BatchNormExpander can create zero-sized ops, so zero-sized HLO
    // elimination has to come after that pass.
    pass.AddPass<ZeroSizedHloElimination>();

    pass.AddPass<WhileLoopInvariantCodeMotion>();
    pass.AddPass<TupleSimplifier>();
    pass.AddPass<WhileLoopConstantSinking>();
    pass.AddPass<WhileLoopSimplifier>();
    pass.AddPass<HloDCE>();
    pass.AddPass<ReshapeMover>();
    pass.AddPass<HloConstantFolding>();
    pass.AddPass<ConditionalSimplifier>();
  }
  pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();

  //TODO: Vulkan Transpose folding
  //  pipeline.AddPass<TransposeFolding>(
  //     [&](const HloInstruction& dot,
  //        const TransposeFolding::OperandIndices& candidate_operands) {
  //     return DotImplementationCanHandleTranspose(dot,
  //                                               *target_machine_features)
  //              ? candidate_operands
  //             : TransposeFolding::OperandIndices{};
  // },
  // TransposeFolding::NeverFoldTranspose);
  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
  // TODO: Vulkan Insturcion Fusion
  //  pipeline.AddPass<CpuInstructionFusion>();

  pipeline.AddPass<ScatterExpander>();

  ReducePrecisionInsertion::AddPasses(
      &pipeline, module->config().debug_options(),
      ReducePrecisionInsertion::PassTiming::AFTER_FUSION);
  // TODO: Implement Vulkan layout assignment
  //  pipeline.AddPass<VulkanLayoutAssignment>(
  //     module->mutable_entry_computation_layout(),
  //    LayoutAssignment::InstructionCanChangeLayout, target_machine_features);
  return pipeline.Run(module).status();
}

Status VulkanCompiler::RunHloPassesAfterLayoutAssn(HloModule* module,
                                                   bool is_aot_compile) {
  HloPassPipeline pipeline("HLO passes after layout assignment");
  // After layout assignment, use a layout-sensitive verifier.
  auto& after_layout_assn =
      pipeline.AddPass<HloPassPipeline>("after layout assignment");
  after_layout_assn.AddInvariantChecker<HloVerifier>(
      /*layout_sensitive=*/true,
      /*allow_mixed_precision=*/false);

  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  {
    auto& pass = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
        "simplification after layout assignement");
    pass.AddInvariantChecker<HloVerifier>(
        /*layout_sensitive=*/true,
        /*allow_mixed_precision=*/false);
    AlgebraicSimplifierOptions options;
    options.set_is_layout_sensitive(true);
    options.set_enable_dot_strength_reduction(false);
    pass.AddPass<HloPassFix<AlgebraicSimplifier>>(options);
    pass.AddPass<HloDCE>();
    pass.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
  }

  pipeline.AddPass<HloElementTypeConverter>(BF16, F32);

  // Outline ops in the entry computation into calls to subcomputations.
  // Copy insertion should be performed immediately before IR emission to
  // avoid inserting unnecessary copies (later pass adds an instruction which
  // materializes the value) or missing a necessary copy (later pass removes
  // an instruction which materializes a value). DCE must be run immediately
  // before (and sometime after) copy insertion, to avoid dead code from
  // interfering with the rewrites.
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<FlattenCallGraph>();
  // TODO: VulkanCopyInsertion
//  pipeline.AddPass<CpuCopyInsertion>();
  pipeline.AddPass<HloDCE>();
  return pipeline.Run(module).status();
}

Status VulkanCompiler::RunHloPasses(HloModule* module, bool is_aot_compile) {
  TF_RETURN_IF_ERROR(RunHloPassesThroughLayoutAssn(module, is_aot_compile));
  return RunHloPassesAfterLayoutAssn(module, is_aot_compile);
}

// Align buffers to 16-byte boundaries.
constexpr int64 kMemoryAlignment = 16;
auto memory_alignment = [](LogicalBuffer::Color) { return kMemoryAlignment; };

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
VulkanCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                   const AotCompilationOptions& aot_options) {
  LOG(INFO) << "Compile Ahead Of Time";
  TF_RET_CHECK(!module_group->empty());
  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();

  // TODO: Check for Vulkan
  LOG(INFO) << "Check platfrom id ";
  if (aot_options.PlatformId() != se::host::kHostPlatformId) {
    return InvalidArgument("Incompatible AOT compilation platform");
  }

  LOG(INFO) << "Create Vulkan Compilation Options";
  const VulkanAotCompilationOptions& options =
      static_cast<const VulkanAotCompilationOptions&>(aot_options);
  std::vector<std::unique_ptr<AotCompilationResult>> results;

  for (size_t i = 0; i < modules.size(); ++i) {
    HloModule* module = modules[i].get();
    TF_RETURN_IF_ERROR(RunHloPasses(module, /*is_aot_compile=*/true));
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

    LOG(INFO) << "BufferAssignment : ";
    LOG(INFO) << assignment->ToString();

    const string xla_dump_optimized_hlo_proto_to =
        module->config().debug_options().xla_dump_optimized_hlo_proto_to();
     if (!xla_dump_optimized_hlo_proto_to.empty()) {
       HloProto proto = MakeHloProto(*module, *assignment);
       TF_RETURN_IF_ERROR(protobuf_util::DumpProtoToDirectory(
           proto, xla_dump_optimized_hlo_proto_to, module->name()));
     }

     spirv::Module vulkan_module("compute_kernel");
     SPIRVIrEmitter ir_emitter(*module, *assignment, &vulkan_module);
     // At first we have to emit global allocations
     TF_RETURN_IF_ERROR(ir_emitter.EmitGlobalAllocations());
     TF_RETURN_IF_ERROR(ir_emitter.EmitConstantGlobals());
     HloComputation* computation = module->entry_computation();
     LOG(INFO) << computation->ToString();
     TF_RETURN_IF_ERROR(computation->Accept(&ir_emitter));

     // const string& entry_point_name = options.entry_point_name();
     // ObjectFileData object_file_data(object_file->getBufferStart(),
     //                                object_file->getBufferEnd());

     // TODO: How to create the buffers for Vulkan
     // CreateBufferInfosFromBufferAssignment(*assignment);

     //    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
     //                       assignment->GetUniqueTopLevelOutputSlice());

     // results.emplace_back(absl::make_unique<VulkanAotCompilationResult>(
     //   std::move(object_file_data), std::move(buffer_infos),
     //  result_slice.index(), std::move(hlo_profile_printer_data)));
  }
  LOG(INFO) << "End of CompileAheadOfTime";
  return std::move(results);
}

se::Platform::Id VulkanCompiler::PlatformId() const {
  //  return se::vulkan::kVulkanPlatformId;
  return se::host::kHostPlatformId;
}

//FIXME: Move this function to Vulkan Executable when it is done.
static int64 ShapeSizeBytes(const Shape& shape) {
  // On the cpu, opaques are pointers.
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

HloCostAnalysis::ShapeSizeFunction VulkanCompiler::ShapeSizeBytesFunction()
    const {
  return ShapeSizeBytes;
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
  return Unimplemented("Compile is not supported for Vulkan backend");
}

StatusOr<std::unique_ptr<Executable>> VulkanCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
    DeviceMemoryAllocator* device_allocator) {
  return Unimplemented("RunBackend is not supported for Vulkan backend");
}

StatusOr<std::unique_ptr<HloModule>> VulkanCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    DeviceMemoryAllocator* device_allocator) {
  return Unimplemented("RunHloPasses is not supported for Vulkan backend");
}

}  // namespace gpu
}  // namespace xla

static bool InitModule() {
  // Register Vulkan compiler
  xla::Compiler::RegisterCompilerFactory(
      stream_executor::host::kHostPlatformId,
      []() { return absl::make_unique<xla::gpu::VulkanCompiler>(); });
  return true;
}
static bool module_initialized = InitModule();
