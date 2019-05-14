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
    : hlo_module_config_(hlo_module.config()), assignment_(assignment) {
  LOG(INFO) << "Create SPIRV IR Emitter";
}

SPIRVIrEmitter::~SPIRVIrEmitter() {}

Status SPIRVIrEmitter::EmitConstantGlobals() {
  LOG(INFO) << "SPIRVIrEmitter::EmitGlobals";
  for (const auto& allocation : assignment_.Allocations()) {
    LOG(INFO) << allocation.ToString();
    if (!allocation.is_constant()) {
      continue;
    }
  }
  return Status::OK();
}

Status SPIRVIrEmitter::HandleBitcast(HloInstruction* bitcast) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleConstant(HloInstruction* constant) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleCopy(HloInstruction* copy) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleSelect(HloInstruction* select) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleTupleSelect(HloInstruction* tuple_select) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleInfeed(HloInstruction* instruction) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleOutfeed(HloInstruction* outfeed) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleSort(HloInstruction* hlo){
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleTuple(HloInstruction* tuple) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleReduceWindow(HloInstruction* reduce_window) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleSelectAndScatter(
    HloInstruction* select_and_scatter) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleDot(HloInstruction* dot) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleFft(HloInstruction* fft) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleAllReduce(HloInstruction* crs) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleParameter(HloInstruction* parameter) {
  return Unimplemented("Op is not implemented for Vulkan.");
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
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleSlice(HloInstruction* slice) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleRecv(HloInstruction* recv) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleRecvDone(HloInstruction* recv_done) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandlePad(HloInstruction* pad) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleFusion(HloInstruction* fusion) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleCall(HloInstruction* call) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleCustomCall(HloInstruction* custom_call) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleWhile(HloInstruction* xla_while) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleAfterAll(HloInstruction* after_all) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleAddDependency(HloInstruction* add_dependency) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleRng(HloInstruction* rng) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::HandleConcatenate(HloInstruction* concatenate) {
  return Unimplemented("Op is not implemented");
}
Status SPIRVIrEmitter::HandleConvolution(HloInstruction* convolution) {
  return Unimplemented("Op is not implemented");
}
Status SPIRVIrEmitter::HandleReduce(HloInstruction* reduce) {
  return Unimplemented("Op is not implemented");
}
Status SPIRVIrEmitter::HandleDynamicSlice(HloInstruction* dynamic_slice) {
  return Unimplemented("Op is not implemented");
}
Status SPIRVIrEmitter::HandleConditional(HloInstruction* conditional) {
  return Unimplemented("Op is not implemented");
}
Status SPIRVIrEmitter::FinishVisit(HloInstruction* root) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::Preprocess(HloInstruction* hlo) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::Postprocess(HloInstruction* hlo) {
  return Unimplemented("Op is not implemented for Vulkan.");
}
Status SPIRVIrEmitter::DefaultAction(HloInstruction* hlo) {
  return Unimplemented("Op is not implemented for Vulkan.");
}

int64 SPIRVIrEmitter::ByteSizeOf(const Shape& shape) const { return 0; }
}  // namespace gpu
}  // namespace xla
