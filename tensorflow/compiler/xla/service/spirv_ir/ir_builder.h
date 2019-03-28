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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPIRV_IR_IR_BUILDER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPIRV_IR_IR_BUILDER_H_

// TODO: Figure out how to integrate it into bazel build system.
#include <vulkan/spirv.hpp>

#include <memory>
#include <vector>
// TODO: Fix this, when integrates with XLA.
#include <cstdint>
using uint32 = uint32_t;

namespace xla {
namespace spirv {

// Forward declaration.
class BasicBlock;
class Function;
class Module;

enum class LiteralType { kLiteralNumber, kLiteralString };

// TODO: Needs a OpDecorate.


// TODO: Think about how to represent literal.
class Literal {
 public:
  Literal() {}
  ~Literal() {}
  // TODO move, copy constr.

 private:
  std::vector<uint32> value;
  LiteralType kind;
};

class Operand {
 public:
  Operand(spv::Id id) : id_(id) {}
  Operand(Literal *literal) : literal_(literal), is_id_(true) {}
  // TODO move, copy constr.

  bool IsId() { return is_id_; }

 private:
  bool is_id_{true};
  Literal *literal_{nullptr};
  spv::Id id_{0};
};

// Represents an spir-v instruction. Based on section 2.2.1 from spir-v
// specification.
class Instruction {
 public:
  Instruction(spv::Id result_id, spv::Id type_id, spv::Op op_code,
              std::vector<Operand> operands)
      : result_id_(result_id),
        type_id_(type_id),
        op_code_(op_code),
        operands_(std::move(operands)) {}

  ~Instruction() {}

  void SetBasicBlock(BasicBlock *parent_bb) {}

  void AddOperand(spv::Id operand_id) { operands_.push_back(operand_id); }

  std::vector<uint32> ToBinary() {
    // TODO: Implement a binary format converter.
  }

  size_t WordCount() { return word_count_; }

 private:
  size_t word_count_{0};
  spv::Op op_code_;
  spv::Id result_id_{0};
  spv::Id type_id_{0};
  std::vector<Operand> operands_;
  BasicBlock *parent_bb_{nullptr};
};

// Represents list of instructions.
class BasicBlock {
  BasicBlock(spv::Id label_id, Function *func) {}

  void AddInsruction(Instruction *instruction) {
    instructions_.push_back(instruction);
  }

  void AddPredecessor(BasicBlock *predecessor) {
    predecessors_.push_back(predecessor);
  }

  void AddSucessors(BasicBlock *sucessor) { sucessors_.push_back(sucessor); }

  std::vector<uint32> ToBianry() {
    // TODO: Implement binary converter.
  }

  ~BasicBlock() {}

 private:
  std::vector<Instruction *> instructions_;
  std::vector<BasicBlock *> predecessors_;
  std::vector<BasicBlock *> sucessors_;
};

// Represents a spir-v function, consists of basic blocks.
class Function {
 public:
  void AddBasicBlock(BasicBlock *bb) { cfg_.push_back(bb); }

 private:
  std::vector<BasicBlock *> cfg_;
};

// Module contains functions, each function contains entry point.
class Module {
  private:
   std::vector<Function *> functions_;
};

// Class which builds the IR.
class IRBuilder {
 public:
  IRBuilder() {}
  ~IRBuilder() {}

  // Create Load operations.
  spv::Id CreateLoad(spv::Op type, spv::Id ptr) {}

  // Create store operations.
  spv::Id CreateStore(spv::Op ptr, spv::Id value) {}

  // Create an unconditional branch.
  void CreateBr(BasicBlock *dest) {}

  // Create conditional branch.
  void CreateCondBr(spv::Id cond, BasicBlock *true_block,
                    BasicBlock *false_block) {}

  void CreateLoopMerge(BasicBlock *merge_block, BasicBlock *continue_block) {}

  spv::Id CreateAccesssChain(spv::Id ptr_type, spv::Id ptr,
                             const std::vector<spv::Id> &offsets) {}

  spv::Id CreateInBoundsAccessChain(spv::Id ptr_type, spv::Id ptr,
                                    const std::vector<spv::Id> &offsets) {}

  // Create binary op.
  spv::Id CreateBinOp(spv::Op op_code, spv::Id lhs, spv::Id rhs) {}
  // Phi function.
  spv::Id CreatePhi(
      spv::Id type,
      const std::vector<std::pair<spv::Id, BasicBlock *>> &phi_values) {}

  spv::Id CreateType(spv::Op kind) {}
  spv::Id CreatePointerType(spv::Id type) {}

  spv::Id CreateVariable(spv::Op type, spv::StorageClass storage_class) {}
  spv::Id CreateConstant(spv::Op type, int value) {}
  spv::Id CreateFunction() {}
};
}  // namespace spriv
}  // namespace xla
#endif
