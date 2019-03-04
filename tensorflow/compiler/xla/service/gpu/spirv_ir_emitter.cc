/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/spirv_ir_emitter.h"

#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_options.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/ir_function.h"
#include "tensorflow/compiler/xla/service/cpu/parallel_loop_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/shape_partition.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {
// using spirv::AsStringRef;
// using spirv::IrName;
// using spirv::SetToFirstInsertPoint;
}  // namespace

namespace gpu {

SPIRVIrEmitter::SPIRVIrEmitter(
    const HloModule& hlo_module, const BufferAssignment& assignment,
    spirv::Module* spirv_module,
    std::unordered_map<const HloInstruction*, int64> instruction_to_profile_idx,
    std::unordered_map<const HloComputation*, int64> computation_to_profile_idx,
    const TargetMachineFeatures* target_machine_features) {}
}  // namespace gpu

StatusOr<spirv::Function*> SPIRVIrEmitter::EmitComputation(
    HloComputation* computation, const string& function_name_prefix,
    bool is_top_level_computation,
    absl::Span<HloInstruction* const> instruction_order) {}

void SPIRVIrEmitter::InitializeIrFunction(const string& function_name) {}

SPIRVIrEmitter::~SPIRVIrEmitter() {}

Status SPIRVIrEmitter::HandleBitcast(HloInstruction* bitcast) {}

spirv::Constant* SPIRVIrEmitter::EmitGlobalForLiteral(const Literal& literal) {}

Status SPIRVIrEmitter::EmitConstantGlobals() {}

Status SPIRVIrEmitter::HandleConstant(HloInstruction* constant) {}

Status SPIRVIrEmitter::HandleCopy(HloInstruction* copy) {}

// Calculate the alignment of a buffer allocated for a given primitive type.
int SPIRVIrEmitter::MinimumAlignmentForPrimitiveType(PrimitiveType primitive_type) {}
int64 SPIRVIrEmitter::ByteSizeOf(const Shape& shape) const {}
// Calculate the alignment of a buffer allocated for a given shape.
int SPIRVIrEmitter::MinimumAlignmentForShape(const Shape& shape) {}
void SPIRVIrEmitter::AttachAlignmentMetadataForLoad(spirv::LoadInst* load,
                                               const Shape& shape) {}

void SPIRVIrEmitter::AttachAlignmentMetadataForLoad(spirv::LoadInst* load,
                                               int64 buffer_size) {}

void SPIRVIrEmitter::AttachDereferenceableMetadataForLoad(spirv::LoadInst* load,
                                                     const Shape& shape) {}

void SPIRVIrEmitter::AttachDereferenceableMetadataForLoad(spirv::LoadInst* load,
                                                     int64 buffer_size) {}

Status SPIRVIrEmitter::HandleGetTupleElement(HloInstruction* get_tuple_element) {}

Status SPIRVIrEmitter::HandleSelect(HloInstruction* select) {}

Status SPIRVIrEmitter::HandleTupleSelect(HloInstruction* tuple_select) {}

Status SPIRVIrEmitter::HandleInfeed(HloInstruction* instruction) {}

Status SPIRVIrEmitter::EmitXfeedTransfer(XfeedKind kind, const Shape& shape,
                                    spirv::Value* program_buffer_address) {}

Status SPIRVIrEmitter::HandleOutfeed(HloInstruction* outfeed) {}

Status SPIRVIrEmitter::HandleSort(HloInstruction* hlo) {}

Status SPIRVIrEmitter::HandleTuple(HloInstruction* tuple) {}

spirv::Value* SPIRVIrEmitter::EmitElementalMap(
    const HloMapInstruction& map_instr,
    absl::Span<spirv::Value* const> elemental_operands,
    absl::string_view name) {}

StatusOr<spirv::Value*> SPIRVIrEmitter::EmitElementalReduceWindow(
    const HloReduceWindowInstruction* reduce_window,
    const spirv::ElementGenerator& input_generator,
    const spirv::IrArray::Index& index) {}

Status SPIRVIrEmitter::HandleReduceWindow(HloInstruction* reduce_window) {}

Status SPIRVIrEmitter::HandleSelectAndScatter(HloInstruction* select_and_scatter) {}

Status SPIRVIrEmitter::HandleDot(HloInstruction* dot) {}

StatusOr<spirv::Value*> SPIRVIrEmitter::EmitElementalConvolution(
    const HloConvolutionInstruction* convolution,
    const spirv::ElementGenerator& input_generator,
    const spirv::ElementGenerator& kernel_generator,
    const spirv::IrArray::Index& index) {}

Status SPIRVIrEmitter::HandleFft(HloInstruction* fft) {}

Status SPIRVIrEmitter::HandleAllReduce(HloInstruction* crs) {}

Status SPIRVIrEmitter::HandleParameter(HloInstruction* parameter) {}

// Returns true if the relative order of the unreduced dimensions stays the same
// through the reduce operation.
static bool ReductionPreservesLayout(const HloInstruction& reduce) {}

SPIRVIrEmitter::ReductionGenerator SPIRVIrEmitter::MatchReductionGenerator(
    HloComputation* function, string* failure_reason) const {}

SPIRVIrEmitter::ShardedVectorType SPIRVIrEmitter::CreateShardedVectorType(
    PrimitiveType element_type, unsigned element_count) {}

StatusOr<SPIRVIrEmitter::ShardedVector>
SPIRVIrEmitter::EmitInnerLoopForVectorizedReduction(
    const ReductionGenerator& reduction_generator,
    const spirv::IrArray::Index& output_index,
    const ShardedVectorType& accumulator_type, HloInstruction* init_value,
    HloInstruction* arg, absl::Span<const int64> dimensions,
    unsigned element_alignment) {}

void SPIRVIrEmitter::EmitShardedVectorStore(
    spirv::Value* store_address,
    const std::vector<spirv::Value*>& value_to_store, const int alignment,
    const spirv::IrArray& containing_array) {}

StatusOr<bool> SPIRVIrEmitter::EmitVectorizedReduce(
    HloInstruction* reduce, HloInstruction* arg, HloInstruction* init_value,
    absl::Span<const int64> dimensions, HloComputation* function,
    string* failure_reason) {}

StatusOr<spirv::Value*> SPIRVIrEmitter::EmitElementalReduce(
    const HloReduceInstruction* reduce,
    const spirv::ElementGenerator& input_generator,
    const spirv::ElementGenerator& initial_value_generator,
    const spirv::IrArray::Index& index) {}

Status SPIRVIrEmitter::HandleAllToAll(HloInstruction*) {
  return Unimplemented("AllToAll is not implemented for Vulkan.");
}

Status SPIRVIrEmitter::HandleSend(HloInstruction* send) {
  return Unimplemented("Send is not implemented for Vulkan.");
}

Status SPIRVIrEmitter::HandleSendDone(HloInstruction* send_done) {
  return Unimplemented("Send-done is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleScatter(HloInstruction*) {}
Status SPIRVIrEmitter::HandleSlice(HloInstruction* slice) {}
Status SPIRVIrEmitter::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {}
Status SPIRVIrEmitter::HandleRecv(HloInstruction* recv) {}
Status SPIRVIrEmitter::HandleRecvDone(HloInstruction* recv_done) {}
Status SPIRVIrEmitter::HandlePad(HloInstruction* pad) {}
Status SPIRVIrEmitter::HandleFusion(HloInstruction* fusion) {}
Status SPIRVIrEmitter::HandleCall(HloInstruction* call) {}
Status SPIRVIrEmitter::HandleCustomCall(HloInstruction* custom_call) {}
Status SPIRVIrEmitter::HandleWhile(HloInstruction* xla_while) {}
void SPIRVIrEmitter::EmitTransferElements(spirv::Value* target, spirv::Value* source,
                                     int64 element_count,
                                     PrimitiveType primitive_type,
                                     const spirv::IrArray& target_array,
                                     const spirv::IrArray& source_array) {}
Status SPIRVIrEmitter::HandleAfterAll(HloInstruction* after_all) {}
Status SPIRVIrEmitter::HandleAddDependency(HloInstruction* add_dependency) {}
Status SPIRVIrEmitter::HandleRng(HloInstruction* rng) {}
Status SPIRVIrEmitter::FinishVisit(HloInstruction* root) {}
Status SPIRVIrEmitter::Preprocess(HloInstruction* hlo) {}
Status SPIRVIrEmitter::Postprocess(HloInstruction* hlo) {}
spirv::IrArray SPIRVIrEmitter::GetIrArrayFor(const HloInstruction* hlo) {}
std::vector<spirv::IrArray> SPIRVIrEmitter::GetIrArraysForOperandsOf(
    const HloInstruction* hlo) {}
spirv::Value* SPIRVIrEmitter::GetEmittedValueFor(const HloInstruction* hlo) {}
spirv::Type* SPIRVIrEmitter::IrShapeType(const Shape& shape) {}
spirv::Value* SPIRVIrEmitter::GetProfileCountersArgument() {}
spirv::Value* SPIRVIrEmitter::GetBufferTableArgument() {}
spirv::Value* SPIRVIrEmitter::GetExecutableRunOptionsArgument() {}
spirv::Value* SPIRVIrEmitter::EmitThreadLocalBufferPointer(
    const BufferAllocation::Slice& slice, const Shape& target_shape) {}
spirv::Value* SPIRVIrEmitter::EmitGlobalBufferPointer(
    const BufferAllocation::Slice& slice, const Shape& target_shape) {}
spirv::Value* SPIRVIrEmitter::EmitBufferPointer(const BufferAllocation::Slice& slice,
                                           const Shape& target_shape) {}
Status SPIRVIrEmitter::EmitTargetAddressForOp(const HloInstruction* op) {}
Status SPIRVIrEmitter::EmitTargetElementLoop(
    HloInstruction* target_op,
    const spirv::ElementGenerator& element_generator) {}
Status SPIRVIrEmitter::EmitTargetElementLoop(
    HloInstruction* target_op, absl::string_view desc,
    const spirv::ElementGenerator& element_generator) {}
Status SPIRVIrEmitter::EmitMemcpy(const HloInstruction& source,
                             const HloInstruction& destination) {}
Status SPIRVIrEmitter::ElementTypesSameAndSupported(
    const HloInstruction& instruction,
    absl::Span<const HloInstruction* const> operands,
    absl::Span<const PrimitiveType> supported_types) {}
Status SPIRVIrEmitter::DefaultAction(HloInstruction* hlo) {}
spirv::Value* SPIRVIrEmitter::EmitThreadLocalCall(
    const HloComputation& callee, absl::Span<spirv::Value* const> parameters,
    absl::string_view name) {}

void SPIRVIrEmitter::EmitGlobalCall(const HloComputation& callee,
                               absl::string_view name) {}
spirv::Value* SPIRVIrEmitter::GetBufferForGlobalCallReturnValue(
    const HloComputation& callee) {}

}  // namespace xla
}  // namespace xla
