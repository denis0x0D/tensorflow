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
namespace gpu {

SPIRVIrEmitter::SPIRVIrEmitter(const HloModule& hlo_module,
                               const BufferAssignment& assignment,
                               spirv::Module* spirv_module)
    : hlo_module_config_(hlo_module.config()),
      assignment_(assignment),
      module_(spirv_module) {
  spirv::BasicBlock* entry = new spirv::BasicBlock("entry");
  b_ = new spirv::IRBuilder(entry, SPIRVModule());
  //LOG(INFO) << "Create SPIRV IR Emitter";
  spirv_module->InitHeader();
}

SPIRVIrEmitter::~SPIRVIrEmitter() {}

// TODO: Implement constants generations into spir-v.
// From high-level view it should be a OpConstantComposite.
Status SPIRVIrEmitter::EmitConstantGlobals() {
  //LOG(INFO) << "SPIRVIrEmitter::EmitGlobals\n";
  for (const BufferAllocation& allocation : assignment_.Allocations()) {
    //LOG(INFO) << allocation.ToString();
    if (!allocation.is_constant()) {
      continue;
    }

    //  const Literal& literal =
    //  llvm_ir::LiteralForConstantAllocation(allocation);
    // llvm::Constant* global_for_const;
    // auto it = emitted_literals_.find(&literal);
    // if (it != emitted_literals_.end()) {
    //  global_for_const = it->second;
    // } else {
    //  global_for_const = EmitGlobalForLiteral(literal);
    //   InsertOrDie(&emitted_literals_, &literal, global_for_const);
    // }
  }
  return Status::OK();
}

Status SPIRVIrEmitter::InitGlobalInvocationId() {
  spv::Id v3_int32_t = SPIRVModule()->GetOrCreateCustomType(
      spv::Op::OpTypeVector, SPIRVModule()->GetOrCreateInt32TypeId(), {"3"},
      "v3_int");
  spv::Id ptr_input_v3int = SPIRVModule()->GetOrCreatePointerTypeId(
      v3_int32_t, "Input", "ptr_input_v3int");
  spv::Id global_invocation_id = SPIRVModule()->GetOrCreateGlobalVariable(
      ptr_input_v3int, false, {"Input"}, "global_invoc_id");
  SPIRVModule()->CreateEntryPoint(SPIRVFunction(), global_invocation_id);
  SPIRVModule()->CreateExecutionMode(SPIRVFunction(),
                                     {"LocalSize", "1", "1", "1"});
  return Status::OK();
}

Status SPIRVIrEmitter::EmitComputation(
    const HloComputation* computation, const string& function_name,
    bool is_top_level_computation,
    absl::Span<HloInstruction* const> instruction_order) {
  spv::Id void_t = SPIRVModule()->GetOrCreateVoidTypeId();
  TF_RETURN_IF_ERROR(InitFunction(void_t, function_name));
  TF_RETURN_IF_ERROR(InitGlobalInvocationId());

  // 3 dim vector.
  spirv::BasicBlock* entry = new spirv::BasicBlock("entry");
  spirv::BasicBlock* ret = new spirv::BasicBlock("ret");
  SPIRVFunction()->AddEntryBlock(entry);
  SPIRVFunction()->AddRetBlock(ret);
  SPIRVBuilder()->SetInsertPoint(entry);

  computation->AcceptOrdered(this, instruction_order);
  return Status::OK();
}

Status SPIRVIrEmitter::EmitGlobalAllocations() {
  //LOG(INFO) << "EmitGlobalAllocations";

  for (const auto& allocation : assignment_.Allocations()) {
    if (allocation.is_constant()) {
      continue;
    }
    TF_RETURN_IF_ERROR(EmitGlobalAllocation(allocation));
  }
  return Status::OK();
}

Status SPIRVIrEmitter::EmitGlobalAllocation(
    const BufferAllocation& allocation) {
  // Create global array using ir builder and add it to hash map.
  //LOG(INFO) << "Create allocaiton for " << allocation.ToString();
  spv::Id int_64_t = SPIRVModule()->GetOrCreateInt32TypeId();
  // FIXME: Find out how to get actual buffer type, it should depend on buffer
  // allocation.
  spv::Id float_32_t = SPIRVModule()->GetOrCreateFloat32TypeId();
  std::string allocation_size_str = std::to_string(allocation.size());
  std::string allocation_size_prefix = "allocation_size";
  spv::Id allocation_size = SPIRVModule()->GetOrCreateGlobalVariable(
      int_64_t, true, {allocation_size_str},
      allocation_size_prefix + allocation_size_str);

  std::string array_type_prefix = "array_type";
  spv::Id array_type = SPIRVModule()->GetOrCreateArrayTypeId(
      float_32_t, allocation_size, array_type_prefix + allocation_size_str);
  
  std::string struct_type_prefix = "struct_type";
  spv::Id struct_type = SPIRVModule()->GetOrCreateStructTypeId(
      array_type, struct_type_prefix + array_type_prefix + allocation_size_str);
  std::string ptr_struct_type_prefix = "ptr_struct";
  spv::Id ptr_struct_type = SPIRVModule()->GetOrCreatePointerTypeId(
      struct_type, "Uniform",
      ptr_struct_type_prefix + array_type_prefix + allocation_size_str);
  std::string array_prefix = "array_prefix";

  spv::Id array_id = SPIRVModule()->GetOrCreateGlobalVariable(
      ptr_struct_type, false, {"Uniform"},
      array_prefix + std::to_string(allocation.index()));

  allocation_map_.insert({allocation.index(), array_id});

  // Can't decorate same type more than one time.
  if (!array_type_set_.count(array_type)) {
    array_type_set_.insert(array_type);
    SPIRVModule()->Decorate(array_type, {"ArrayStride", "4"});
    SPIRVModule()->MemberDecorate(struct_type, {"0", "Offset", "0"});
    SPIRVModule()->Decorate(struct_type, {"BufferBlock"});
  }

  SPIRVModule()->Decorate(array_id, {"DescriptorSet", "0"});
  SPIRVModule()->Decorate(array_id,
                          {"Binding", std::to_string(binding_counter_++)});

  return Status::OK();
}

Status SPIRVIrEmitter::HandleBitcast(HloInstruction* bitcast) {
  return Unimplemented("Bitcast Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleConstant(HloInstruction* constant) {
  return Unimplemented("Constant Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleCopy(HloInstruction* copy) {
  return Unimplemented("Copy Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  return Unimplemented("GetTupleElement Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleSelect(HloInstruction* select) {
  return Unimplemented("Select Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleTupleSelect(HloInstruction* tuple_select) {
  return Unimplemented("TupleSelect Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleInfeed(HloInstruction* instruction) {
  return Unimplemented("Infeed Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleOutfeed(HloInstruction* outfeed) {
  return Unimplemented("OutFeed Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleSort(HloInstruction* hlo){
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleTuple(HloInstruction* tuple) {
  //LOG(INFO) << tuple->ToString();
  return Status::OK();
}
Status SPIRVIrEmitter::HandleReduceWindow(HloInstruction* reduce_window) {
  return Unimplemented("ReduceWindow Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleSelectAndScatter(
    HloInstruction* select_and_scatter) {
  return Unimplemented("SelectAndScatter Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleDot(HloInstruction* dot) {
  return Unimplemented("Dot Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleFft(HloInstruction* fft) {
  return Unimplemented("Fft Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleAllReduce(HloInstruction* crs) {
  return Unimplemented("AllReduce Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleParameter(HloInstruction* parameter) {
  return Status::OK();
}
Status SPIRVIrEmitter::HandleAllToAll(HloInstruction*) {
  return Unimplemented("AllToAll is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleSend(HloInstruction* send) {
  return Unimplemented("Send is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleSendDone(HloInstruction* send_done) {
  return Unimplemented("Send-done is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleScatter(HloInstruction*) {
  return Unimplemented("Scatter Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleSlice(HloInstruction* slice) {
  return Unimplemented("Slice Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  return Unimplemented("DynamicUpdateSlice Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleRecv(HloInstruction* recv) {
  return Unimplemented("Recv Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleRecvDone(HloInstruction* recv_done) {
  return Unimplemented("RecvDone Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandlePad(HloInstruction* pad) {
  return Unimplemented("Pad Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleFusion(HloInstruction* fusion) {
  return Unimplemented("Fusion Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleCall(HloInstruction* call) {
  return Unimplemented("Call Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleCustomCall(HloInstruction* custom_call) {
  return Unimplemented("CustomCall Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleWhile(HloInstruction* xla_while) {
  return Unimplemented("While Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleAfterAll(HloInstruction* after_all) {
  return Unimplemented("AfterAll Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleAddDependency(HloInstruction* add_dependency) {
  return Unimplemented("AddDependency Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleRng(HloInstruction* rng) {
  return Unimplemented("Rng Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleConcatenate(HloInstruction* concatenate) {
  return Unimplemented("Concatenate Op is not implemented");
}
Status SPIRVIrEmitter::HandleConvolution(HloInstruction* convolution) {
  return Unimplemented("Convolution Op is not implemented");
}
Status SPIRVIrEmitter::HandleReduce(HloInstruction* reduce) {
  return Unimplemented("Reduce Op is not implemented");
}
Status SPIRVIrEmitter::HandleDynamicSlice(HloInstruction* dynamic_slice) {
  return Unimplemented("Dynamic Slice Op is not implemented");
}
Status SPIRVIrEmitter::HandleConditional(HloInstruction* conditional) {
  return Unimplemented("Conditional Op is not implemented");
}
Status SPIRVIrEmitter::FinishVisit(HloInstruction* root) {
  return Status::OK();
}

Status SPIRVIrEmitter::Preprocess(HloInstruction* hlo) { return Status::OK(); }

Status SPIRVIrEmitter::Postprocess(HloInstruction* hlo) { return Status::OK(); }

Status SPIRVIrEmitter::DefaultAction(HloInstruction* hlo) {
  // TODO: Make this handle more generic
  // For this moment just handle the elementwise operations.
  const HloInstruction* lhs = hlo->operand(0);
  const HloInstruction* rhs = hlo->operand(1);

  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice1,
                      assignment_.GetUniqueTopLevelSlice(lhs));
  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice2,
                      assignment_.GetUniqueTopLevelSlice(rhs));

  const Shape& target_shape = hlo->shape();
  int64 first_dim = target_shape.dimensions(0);
  spirv::BasicBlock* entry_block = SPIRVBuilder()->GetCurrentInsertPoint();
  // The lower bound for the tensor.
  spv::Id lower_bound = SPIRVModule()->GetOrCreateGlobalVariable(
      SPIRVModule()->GetOrCreateInt32TypeId(), true, {"0"}, "const_int64_0");

  std::string first_dim_str = std::to_string(first_dim);
  // The upper bound for the tensor.
  spv::Id upper_bound = SPIRVModule()->GetOrCreateGlobalVariable(
      SPIRVModule()->GetOrCreateInt32TypeId(), true, {first_dim_str},
      "const_int64_" + first_dim_str);
  // FIXME: The step should depends on GPU global and local blocks count.
  std::string step_str = "1";
  spv::Id step = SPIRVModule()->GetOrCreateGlobalVariable(
      SPIRVModule()->GetOrCreateInt32TypeId(), true, {step_str},
      "const_int64_" + step_str);
  // Create new basic block. 
  spirv::BasicBlock* current_block = new spirv::BasicBlock("current");
  spirv::BasicBlock* body_block = new spirv::BasicBlock("body_block");
  spirv::BasicBlock* tail_block = new spirv::BasicBlock("tail_block");

  SPIRVFunction()->AddBasicBlock(current_block);
  SPIRVFunction()->AddBasicBlock(body_block);
  SPIRVFunction()->AddBasicBlock(tail_block);

  SPIRVBuilder()->CreateBr(current_block);
  SPIRVBuilder()->SetInsertPoint(current_block);
  spv::Id phi_index =
      SPIRVBuilder()->CreatePhi(SPIRVModule()->GetOrCreateInt32TypeId());
  SPIRVBuilder()->AddIncoming(current_block, phi_index, lower_bound,
                              entry_block);

  spv::Id cmp = SPIRVBuilder()->CreateBinOp(
      spv::Op::OpSLessThan, SPIRVModule()->GetOrCreateBoolTypeId(), phi_index,
      upper_bound);

  spirv::BasicBlock* ret = SPIRVFunction()->GetRetBlock();
  SPIRVBuilder()->CreateLoopMerge(ret, tail_block, {"None"});
  SPIRVBuilder()->CreateCondBr(cmp, body_block, ret);
  SPIRVBuilder()->SetInsertPoint(body_block);

  spv::Id lhs_array = allocation_map_[slice1.index()];
  spv::Id rhs_array = allocation_map_[slice2.index()];
  spv::Id target_array = allocation_map_[slice2.index()];

  spv::Id ptr_type = SPIRVModule()->GetOrCreatePointerTypeId(
      SPIRVModule()->GetOrCreateFloat32TypeId(), "Uniform", "float_32_ptr");

  spv::Id lhs_ptr = SPIRVBuilder()->CreateAccessChain(
      spv::Op::OpInBoundsAccessChain, ptr_type, lhs_array,
      {lower_bound, phi_index});
  spv::Id rhs_ptr = SPIRVBuilder()->CreateAccessChain(
      spv::Op::OpInBoundsAccessChain, ptr_type, rhs_array,
      {lower_bound, phi_index});

  spv::Id lhs_value = SPIRVBuilder()->CreateLoad(
      SPIRVModule()->GetOrCreateFloat32TypeId(), lhs_ptr, {"None"});
  spv::Id rhs_value = SPIRVBuilder()->CreateLoad(
      SPIRVModule()->GetOrCreateFloat32TypeId(), rhs_ptr, {"None"});

  // Add all elementwise operations.
  spv::Id target_value = SPIRVBuilder()->CreateBinOp(
      spv::Op::OpFAdd, SPIRVModule()->GetOrCreateFloat32TypeId(), lhs_value,
      rhs_value);
  spv::Id target_ptr = SPIRVBuilder()->CreateAccessChain(
      spv::Op::OpInBoundsAccessChain, ptr_type, target_array,
      {lower_bound, phi_index});

  SPIRVBuilder()->CreateStore(target_ptr, target_value, {"None"});
  SPIRVBuilder()->CreateBr(tail_block);
  SPIRVBuilder()->SetInsertPoint(tail_block);

  spv::Id index = SPIRVBuilder()->CreateBinOp(
      spv::Op::OpIAdd, SPIRVModule()->GetOrCreateInt32TypeId(), phi_index,
      step);
  SPIRVBuilder()->CreateBr(current_block);
  SPIRVBuilder()->AddIncoming(current_block, phi_index, index, tail_block);

  return Status::OK();
}

int64 SPIRVIrEmitter::ByteSizeOf(const Shape& shape) const {
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

Status SPIRVIrEmitter::InitFunction(spv::Id ret_type,
                                    std::string function_name) {
  spv::Id function_type = SPIRVModule()->GetOrCreateFunctionTypeId(
      ret_type, function_name + "func_type");

  function_ = SPIRVModule()->GetOrCreateFunction(function_name, ret_type,
                                                 function_type, "None");
  return Status::OK();
}
}  // namespace gpu
}  // namespace xla
