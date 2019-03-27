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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/ir_function.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
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
class SPIRVIrEmitter : public DfsHloVisitorWithDefault,
                       public SPIRVBuilderMixin<SPIRVIrEmitter> {
 public:
  // TODO(denis0x0D): Add supoort for Vulkan
  using GeneratorForOperandIrArrays =
      std::function<std::vector<spirv::IrArray>()>;

  // hlo_module: the HLO module we are emitting IR for.
  // assignment: a BufferAssignment from which we know which buffers are used by
  //             the HLO nodes.
  // vulkan_module: the vulkan module to emit IR into.
  // instruction_to_profile_idx: the mapping from HLO instructions to their
  //              index in the profiling array.
  // computation_to_profile_idx: the mapping from HLO computations to their
  //              index in the profiling array.
  // emit_code_for_msan: whether emitted code should be compatible with msan.
  SPIRVIrEmitter(const HloModule& hlo_module, const BufferAssignment& assignment,
            spirv::Module* vulkan_module,
            std::unordered_map<const HloInstruction*, int64>
                instruction_to_profile_idx,
            std::unordered_map<const HloComputation*, int64>
                computation_to_profile_idx,
            const TargetMachineFeatures* target_machine);
  ~SPIRVIrEmitter() override;

  // Emit and return the given HLO computation as an SPIRV IR
  // function.
  //
  // function_name_prefix is the desired name of the function. If the name is
  // not unique among already emitted functions then a suffix is appended to
  // make the name unique.
  //
  // 'is_top_level_computation' has the following meanings for each CPU backend:
  // *) sequential: indicates that this is the entry computation of the HLO
  //    module.
  // *) parallel: indices that this is the callee of a kCall HLO in the entry
  //    computation of the HLO module.
  //
  // If 'instruction_order' is not NULL, then the HLO instructions are emitted
  // in the given order.  In this case, 'instruction_order' must be a
  // topological sort of the set of nodes accessible from the root of the
  // computation.
  StatusOr<spirv::Function*> EmitComputation(
      HloComputation* computation, const string& function_name_prefix,
      bool is_top_level_computation,
      absl::Span<HloInstruction* const> instruction_order);

  spirv::IRBuilder<>* b() { return &b_; }

  // builder() is for IrBuilderMixin.
  spirv::IRBuilder<>* builder() { return &b_; }

  // Emit an SPIRV global variable for every constant buffer allocation.
  Status EmitConstantGlobals();

  // Emit code to map one element according to `map_instr`.
  //  TODO: Support those operations.
  //  spirv::Value* EmitElementalMap(
  //      const HloMapInstruction& map_instr,
  //      absl::Span<spirv::Value* const> elemental_operands,
  //      absl::string_view name);
  //  // Emit code to emit the element at `index` for a reduce window
  //  instruction. StatusOr<spirv::Value*> EmitElementalReduceWindow(
  //      const HloReduceWindowInstruction* reduce_window,
  //      const spirv::ElementGenerator& input_generator,
  //      const spirv::IrArray::Index& index);
  //  // Emit code to emit the element at `index` for a convolution instruction.
  //  StatusOr<spirv::Value*> EmitElementalConvolution(
  //      const HloConvolutionInstruction* convolution,
  //      const llvm_ir::ElementGenerator& input_generator,
  //      const llvm_ir::ElementGenerator& kernel_generator,
  //      const llvm_ir::IrArray::Index& index);
  //  // Emit code to emit the element at `index` for a reduce instruction.
  //  StatusOr<llvm::Value*> EmitElementalReduce(
  //      const HloReduceInstruction* reduce,
  //      const llvm_ir::ElementGenerator& input_generator,
  //      const llvm_ir::ElementGenerator& initial_value_generator,
  //      const llvm_ir::IrArray::Index& index);

 protected:
  //
  // The following methods implement the DfsHloVisitor interface.
  //
  // Default action which emits code for most operations. Operations which are
  // special in some way are handled explicitly in HandleFoo methods.
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

  // A convenient helper for calling BufferAssignment::GetUniqueSlice.
  BufferAllocation::Slice GetAllocationSlice(
      const HloInstruction& hlo, const ShapeIndex& index = {}) const {
    return assignment_.GetUniqueSlice(&hlo, index).ConsumeValueOrDie();
  }

 private:
  // Private helper to initialize an IR function for the computation.
  void InitializeIrFunction(const string& function_name);
  //  template <typename T>
  //  llvm::Value* GetProfileCounterCommon(
  //      const T& hlo,
  //      const std::unordered_map<const T*, int64>& profile_index_map);
  //
  //  // Convenience functions to generate a GEP into the profile counter
  //  parameter
  //  // which would correspond to the index for a given HLO instruction or
  //  // computation.
  //  llvm::Value* GetProfileCounterFor(const HloInstruction& instruction) {
  //    return GetProfileCounterCommon<HloInstruction>(instruction,
  //                                                   instruction_to_profile_idx_);
  //  }
  //
  //  llvm::Value* GetProfileCounterFor(const HloComputation& computation) {
  //    return GetProfileCounterCommon<HloComputation>(computation,
  //                                                   computation_to_profile_idx_);
  //  }
  
  // Gets the IR Value emitted previously for the given hlo.
  //
  // Prefer calling GetIrArrayFor if the value you're reading is a buffer,
  // because GetIrArrayFor annotates buffer's loads/stores with noalias
  // metadata.
 
  // Make sure to call this only when you're certain a value *was* emitted - if
  // not found, this will log a fatal error.
  spirv::Value* GetEmittedValueFor(const HloInstruction* hlo);

  // Gets an IrArray representing the given hlo.
  spirv::IrArray GetIrArrayFor(const HloInstruction* hlo);

  // Gets a list of IrArrays, one for each of hlo's operands.
  std::vector<spirv::IrArray> GetIrArraysForOperandsOf(
      const HloInstruction* hlo);

  GeneratorForOperandIrArrays GetGeneratorForOperandIrArrays(
      HloInstruction* unnested_hlo) {
    return [=]() { return GetIrArraysForOperandsOf(unnested_hlo); };
  }

  // Augments IrArray with aliasing information.
  void AddAliasingInformationToIrArray(const HloInstruction& hlo,
                                       spirv::IrArray* array) {
    alias_analysis_.AddAliasingInformationToIrArray(hlo, array);
  }

  // Convenience function to get the IR type matching the given shape.
  spirv::Type* IrShapeType(const Shape& shape);

  // Get the spirv::Value* that represents the "prof_counters" argument of the
  // computation function being emitted by this emitter.
  spirv::Value* GetProfileCountersArgument();

  // Get the xla::ExecutableRunOptions that represents the "run_options"
  // argument of the computation function being emitted by this emitter.
  spirv::Value* GetExecutableRunOptionsArgument();

  // Get the spirv::Value* that represents the "buffer_table" argument of the
  // computation function being emitted by this emitter.
  spirv::Value* GetBufferTableArgument();

  // Helper for EmitBufferPointer.
  spirv::Value* EmitGlobalBufferPointer(const BufferAllocation::Slice& slice,
                                       const Shape& target_shape);

  // Helper for EmitBufferPointer.
  spirv::Value* EmitThreadLocalBufferPointer(
      const BufferAllocation::Slice& slice, const Shape& target_shape);

  // Emits code that computes the address of the given buffer allocation slice.
  spirv::Value* EmitBufferPointer(const BufferAllocation::Slice& slice,
                                  const Shape& target_shape);
  // Emits a call to a "global" function (e.g. to the computation nested within
  // a kWhile or a kCall).  Buffer assignment unabiguously assignes buffers to
  // the parameters and return values for these computations so there is no need
  // to explicitly pass parameters or return results.
  void EmitGlobalCall(const HloComputation& callee, absl::string_view name);

  // Returns the buffer to which a global call to `callee` would have written
  // its result.
  spirv::Value* GetBufferForGlobalCallReturnValue(const HloComputation& callee);

  // Verifies that the element types of all of the given operand instructions
  // match and are of one of the given supported types.
  Status ElementTypesSameAndSupported(
      const HloInstruction& instruction,
      absl::Span<const HloInstruction* const> operands,
      absl::Span<const PrimitiveType> supported_types);

  // Emit IR to perform a computation for every element in the given target op.
  // This produces a series of nested loops (one for each dimension of the op's
  // shape). The body of the inner-most loop is provided by the body_emitter
  // function.
  //
  // desc is an optional human-readable string that's added to the loop name in
  // IR.  Regardless of whether desc is provided, target_op->name() is included
  // in the loop name.
  //
  // TODO(jingyue): target_op should be a `const HloInstruction*`.
  Status EmitTargetElementLoop(
      HloInstruction* target_op,
      const spirv::ElementGenerator& element_generator);
  Status EmitTargetElementLoop(
      HloInstruction* target_op, absl::string_view desc,
      const spirv::ElementGenerator& element_generator);

  // Emits a memcpy from the source instruction's result value to the
  // destination's.  Both source and destination must have an entry in the
  // emitted_value_ table.
  Status EmitMemcpy(const HloInstruction& source,
                    const HloInstruction& destination);

  // Emits IR to compute the target address of the buffer for the given op.
  // After calling this function, you can get a pointer to this buffer by
  // calling GetIrArrayForOp or GetEmittedValueFor.
  Status EmitTargetAddressForOp(const HloInstruction* op);

  // Structurizes "array_elements" into an MD array that represents "shape".
  // This is a recursive function, and "dimension_index" indicates the index of
  // the current dimension that the function is considering (0 means the
  // most-minor dimension).
  spirv::Constant* CreateInitializerForConstantArray(
      const std::vector<spirv::Constant*>& array_elements, const Shape& shape,
      int64 dimension_index);

  // Tries to codegen a reduction operation using vectorized instructions.
  // Returns true if successful, and false on failure.  On failure, sets
  // "failure_reason" to a string describing why it could not vectorize the
  // reduction.
  //
  // TODO(sanjoy): Some of the things we do here can be abstracted out into
  // concepts that generalize over other vectorizable operations.  We should
  // consider pulling out these abstractions into a VectorizingIrEmitter or
  // something similar.
  StatusOr<bool> EmitVectorizedReduce(HloInstruction* reduce,
                                      HloInstruction* arg,
                                      HloInstruction* init_value,
                                      absl::Span<const int64> dimensions,
                                      HloComputation* function,
                                      string* failure_reason);

  // We'd like to keep one or two one cache-line's worth of data in registers
  // without generating IR with illegal (e.g. excessively large or
  // non-power-of-two) vector types.  We do this by introducing a layer of
  // abstraction: we introduce a high level vector-like concept called a
  // "sharded vector" that models data paralleism, and is mapped to a sequence
  // scalar and vector spirv::Value s.
  //
  // For example, we can represent 29 f32 elements by a sharded vector mapped to
  // a sequence of SPIRV values of types [<16 x f32>, <8 x f32>, <4 x f32>, f32].
  // Note that the last element is scalar.
  //
  // There is no requirement on the ordering or the uniqueness of the elements
  // mapped to sharded vectors -- we allow repeated elements, and we allow
  // elements to appear in any order.
  using ShardedVector = std::vector<spirv::Value*>;

  // A sharded vector type is the element-wise spirv::Type's of some
  // ShardedVector.
  using ShardedVectorType = std::vector<spirv::Type*>;

  // Create a sharded vector type corresponding to a "element_count" long
  // sequence of "element_type" values.
  ShardedVectorType CreateShardedVectorType(PrimitiveType element_type,
                                            unsigned element_count);

  // Emit SPIRV IR to store the sharded vector "value_to_store" to
  // "store_address".
  void EmitShardedVectorStore(spirv::Value* store_address,
                              const ShardedVector& value_to_store,
                              const int alignment,
                              const spirv::IrArray& containing_array);

  using ReductionGenerator = std ::function<spirv::Value*(
      spirv::IRBuilder<>*, spirv::Value*, spirv::Value*)>;

  // Tries to match the reduction function "function" to a known reduction
  // pattern.  Returns a non-null ReductionGenerator on a successful match,
  // which can be used to generate the SPIRV IR corresponding to said reduction.
  // On failure, this stores a reason string into "failure_reason".
  ReductionGenerator MatchReductionGenerator(HloComputation* function,
                                             string* failure_reason) const;

  // Emits the inner loop nest that runs the reduction.  Helper function for
  // EmitVectorizedReduce.
  StatusOr<ShardedVector> EmitInnerLoopForVectorizedReduction(
      const ReductionGenerator& reduction_generator,
      const spirv::IrArray::Index& output_index,
      const ShardedVectorType& accumulator_type, HloInstruction* init_value,
      HloInstruction* arg, absl::Span<const int64> dimensions,
      unsigned element_alignment);

  // Tries to emit a fast concatenate operation using memcpy.  Returns true if
  // successful, and false on failure.  On failure, sets "failure_reason" to a
  // string describing why it could not emit a fast concatenate.
  StatusOr<bool> EmitFastConcatenate(HloInstruction* concatenate,
                                     absl::Span<HloInstruction* const> operands,
                                     string* failure_reason);

  // Emits SPIRV IR to transfer "element_count" elements of type "primitive_type"
  // from the address "source" to the address "target".
  void EmitTransferElements(spirv::Value* target, spirv::Value* source,
                            int64 element_count, PrimitiveType primitive_type,
                            const spirv::IrArray& target_array,
                            const spirv::IrArray& source_array);

  // Assignment of the buffers needed by the computation and their shape
  // information.
  const BufferAssignment& assignment_;

  // The SPIRV module into which IR will be emitted.
  spirv::Module* module_;

  // The target architecture.

  // Used to produce unique names for generated functions.
  NameUniquer name_uniquer_;

  // Map containing all previously emitted computations.
  std::map<const HloComputation*, spirv::Function*> emitted_functions_;

  // The following fields track the IR emission state. According to SPIRV memory
  // management rules, their memory is owned by the module (Note that IrFunction
  // creates the encapsulated spirv::Function s.t. it is added to the spirv
  // module's function list).
  std::unique_ptr<IrFunction> compute_function_;
  spirv::IRBuilder<> b_;

  // The buffer allocation slice for the root of the computation being compiled.
  // Only relevant for thread local computations.
  BufferAllocation::Slice computation_root_allocation_;

  // Maps the buffer allocation slices for the parameters to the computation
  // being compiled to their parameter numbers.  Only relevant for thread local
  // computations.
  absl::flat_hash_map<BufferAllocation::Index, int64>
      computation_parameter_allocations_;

  // Maps HLO instructions to their index into the profile counter array.
  const std::unordered_map<const HloInstruction*, int64>
      instruction_to_profile_idx_;

  // Maps HLO computations to their index into the profile counter array.
  const std::unordered_map<const HloComputation*, int64>
      computation_to_profile_idx_;

  // Maps HLOs to Values emitted for them.
  absl::flat_hash_map<const HloInstruction*, spirv::Value*> emitted_value_;

  // The number of root instruction outer dimensions used in parallel loop
  // emission (ParallelLoopEmitter).
  int64 num_dynamic_loop_bounds_ = 0;

  // Returns whether the given instruction should be emitted as a parallel loop.
  bool ShouldEmitParallelLoopFor(const HloInstruction& op) const {
    // Emit parallel loop for root instruction if dynamic outer-dimension loop
    // bounds were specified.
    return num_dynamic_loop_bounds_ > 0 &&
           op.parent()->root_instruction() == &op;
  }

  // Given a load instruction and a shape or buffer size, annotate the load's
  // result with the alignment required by the shape or size.
  void AttachAlignmentMetadataForLoad(spirv::LoadInst* load, const Shape& shape);
  void AttachAlignmentMetadataForLoad(spirv::LoadInst* load, int64 buffer_size);

  // Given a load instruction and a shape or buffer size, annotate the load's
  // result with the dereferenceable bytes required by the shape / buffer size.
  void AttachDereferenceableMetadataForLoad(spirv::LoadInst* load,
                                            const Shape& shape);
  void AttachDereferenceableMetadataForLoad(spirv::LoadInst* load,
                                            int64 buffer_size);

  // Calculate the alignment of a buffer allocated for a given shape.
  int MinimumAlignmentForShape(const Shape& shape);

  // Calculate the alignment of a buffer allocated for a given primitive type.
  int MinimumAlignmentForPrimitiveType(PrimitiveType primitive_type);

  // Returns the number of bytes within the shape.
  int64 ByteSizeOf(const Shape& shape) const;

  enum class XfeedKind {
    kInfeed,
    kOutfeed,
  };

  // Emit IR to transfer between a {infeed,outfeed} buffer and an in-program
  // address.
  Status EmitXfeedTransfer(XfeedKind kind, const Shape& shape,
                           spirv::Value* program_buffer_address);

  // Returns a ConstExpr bitcast.
  spirv::Constant* EmitGlobalForLiteral(const Literal& literal);

  const HloModuleConfig& hlo_module_config_;

  bool is_top_level_computation_;

  const TargetMachineFeatures& target_machine_features_;

  struct LiteralPtrHashFunctor {
    size_t operator()(const Literal* literal) const { return literal->Hash(); }
  };

  struct LiteralPtrEqualityFunctor {
    bool operator()(const Literal* lhs, const Literal* rhs) const {
      return *lhs == *rhs;
    }
  };

  absl::flat_hash_map<const Literal*, spirv::Constant*, LiteralPtrHashFunctor,
                      LiteralPtrEqualityFunctor>
      emitted_literals_;

  absl::flat_hash_map<BufferAllocation::Index, spirv::Constant*>
      constant_buffer_to_global_;

  std::vector<const HloComputation*> thread_local_computations_;
  std::vector<const HloComputation*> global_computations_;

  bool emit_code_for_msan_;

  TF_DISALLOW_COPY_AND_ASSIGN(SPIRVIrEmitter);
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPIRV_IR_EMITTER_H_
