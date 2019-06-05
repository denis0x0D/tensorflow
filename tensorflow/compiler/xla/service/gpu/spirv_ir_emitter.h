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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPIRV_IR_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPIRV_IR_EMITTER_H_

#include <stddef.h>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/spirv.hpp>

#include "tensorflow/compiler/xla/service/spirv_ir/ir_builder.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {
// This class is the top-level API for the XLA HLO --> SPIRV IR compiler.  It
// implements the DfsHloVisitor interface and emits HLO computations as SPIRV IR
// functions.
class SPIRVIrEmitter : public DfsHloVisitorWithDefault {
 public:
  SPIRVIrEmitter(const HloModule& hlo_module,
                 const BufferAssignment& assignment,
                 spirv::Module* spirv_module);
  ~SPIRVIrEmitter() override;
  Status EmitConstantGlobals();
  Status EmitGlobalAllocations();
  Status EmitGlobalAllocation(const BufferAllocation& allocation);
  StatusOr<spirv::Function*> EmitComputation(
      HloComputation* computation, const string& function_name_prefix,
      absl::Span<HloInstruction* const> instruction_order);
  spirv::IRBuilder* b() { return b_; }
  spirv::IRBuilder* builder() { return b_; }
 protected:
  Status DefaultAction(HloInstruction* hlo) override;
  Status HandleAllToAll(HloInstruction* instruction) override;
  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleConstant(HloInstruction* constant) override;
  Status HandleCopy(HloInstruction* copy) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
  Status HandleSelect(HloInstruction* select) override;
  Status HandleTupleSelect(HloInstruction* tuple_select) override;
  Status HandleDot(HloInstruction* dot) override;
  Status HandleConvolution(HloInstruction* convolution) override;
  Status HandleFft(HloInstruction* fft) override;
  Status HandleAllReduce(HloInstruction* crs) override;
  Status HandleInfeed(HloInstruction* infeed) override;
  Status HandleOutfeed(HloInstruction* outfeed) override;
  Status HandleSort(HloInstruction* sort) override;
  Status HandleParameter(HloInstruction* parameter) override;
  Status HandleReduce(HloInstruction* reduce) override;
  Status HandleReduceWindow(HloInstruction* reduce_window) override;
  Status HandleSelectAndScatter(HloInstruction* select_and_scatter) override;
  Status HandleSend(HloInstruction* send) override;
  Status HandleSendDone(HloInstruction* send_done) override;
  Status HandleSlice(HloInstruction* slice) override;
  Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
  Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  Status HandleRecv(HloInstruction* recv) override;
  Status HandleRecvDone(HloInstruction* recv_done) override;
  Status HandlePad(HloInstruction* pad) override;
  Status HandleTuple(HloInstruction* tuple) override;
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleCall(HloInstruction* call) override;
  Status HandleCustomCall(HloInstruction* custom_call) override;
  Status HandleWhile(HloInstruction* xla_while) override;
  Status HandleConcatenate(HloInstruction* concatenate) override;
  Status HandleConditional(HloInstruction* conditional) override;
  Status HandleScatter(HloInstruction* scatter) override;
  Status HandleAfterAll(HloInstruction* after_all) override;
  Status HandleAddDependency(HloInstruction* add_dependency) override;
  Status HandleRng(HloInstruction* rng) override;
  Status FinishVisit(HloInstruction* root) override;
  Status Preprocess(HloInstruction* hlo) override;
  Status Postprocess(HloInstruction* hlo) override;
 private:
  const HloModuleConfig& hlo_module_config_;
  // Assignment of the buffers needed by the computation and their shape
  // information.
  const BufferAssignment& assignment_;
  // The SPIRV module into which IR will be emitted.
  spirv::Module* module_;
  // Used to produce unique names for generated functions.
  NameUniquer name_uniquer_;
  // Map containing all previously emitted computations.
  std::map<const HloComputation*, spirv::Function*> emitted_functions_;
  spirv::IRBuilder* b_;
  // Maps the buffer allocation slices for the parameters to the computation
  // being compiled to their parameter numbers.  Only relevant for thread local
  // computations.
  absl::flat_hash_map<BufferAllocation::Index, int64>
      computation_parameter_allocations_;
  // Maps HLO instructions to their index into the profile counter array.
  absl::flat_hash_map<const HloInstruction*, spv::Id> emitted_value_;
  // Returns the number of bytes within the shape.
  int64 ByteSizeOf(const Shape& shape) const;
    struct LiteralPtrHashFunctor {
    size_t operator()(const Literal* literal) const { return literal->Hash(); }
  };
  struct LiteralPtrEqualityFunctor {
    bool operator()(const Literal* lhs, const Literal* rhs) const {
      return *lhs == *rhs;
    }
  };
  absl::flat_hash_map<const Literal*, spv::Id, LiteralPtrHashFunctor,
                      LiteralPtrEqualityFunctor>
      emitted_literals_;
  absl::flat_hash_map<BufferAllocation::Index, spv::Id>
      constant_buffer_to_global_;
  std::vector<const HloComputation*> global_computations_;
  TF_DISALLOW_COPY_AND_ASSIGN(SPIRVIrEmitter);
};
}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPIRV_IR_EMITTER_H_
