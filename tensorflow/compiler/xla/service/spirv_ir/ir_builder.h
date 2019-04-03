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
#include <iostream>
#include <cstdint>
using uint32 = uint32_t;

namespace xla {
namespace spirv {

// Forward declaration.
class Instruction;
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

// The global context.
class SPIRVContext {
 public:
  SPIRVContext() {}
  spv::Id GetUniqueId() { return unique_id++; }

 private:
  spv::Id unique_id{0};
};

SPIRVContext *GetSPIRVContext() {
  static SPIRVContext *context = new SPIRVContext ();
  return context;
}

// The operand could be an id or literal, in case of literal it's an id
// for lookup table.
class Operand {
 public:
  Operand() : id_(0), is_id_(true) {}
  Operand(spv::Id id, bool is_id = true) : id_(id), is_id_(is_id) {}
  bool IsId() { return is_id_; }
 private:
  spv::Id id_;
  bool is_id_;
};

class IRVisitor {
 public:
  IRVisitor() {}
  virtual ~IRVisitor() = 0;
  virtual void Visit(Instruction *instruction) = 0;
};

// Represents an spir-v instruction. Based on section 2.2.1 from spir-v
// specification.
class Instruction {
 public:
  Instruction(spv::Op op_code, spv::Id result_id, spv::Id type_id,
              std::vector<Operand> operands)
      : op_code_(op_code),
        result_id_(result_id),
        type_id_(type_id),
        operands_(std::move(operands)) {}

  Instruction(spv::Op op_code, spv::Id result_id, spv::Id type_id,
              spv::StorageClass storage_class)
      : op_code_(op_code),
        result_id_(result_id),
        type_id_(type_id),
        storage_class_(storage_class) {}

  Instruction(const Instruction &other) = delete;
  Instruction(Instruction &&other) = delete;
  Instruction &operator=(const Instruction &other) = delete;
  Instruction &operator=(Instruction &&other) = delete;
  ~Instruction() {}

  void SetBasicBlock(BasicBlock *parent_bb) {}
  void AddOperand(spv::Id operand_id, bool is_id = true) {
    operands_.push_back(Operand(operand_id, is_id));
  }

  spv::Op GetOpCode() { return op_code_; }
  void Accept(IRVisitor *visitor) { visitor->Visit(this); }
  size_t WordCount() { return word_count_; }

 private:
  spv::Op op_code_;
  spv::Id result_id_{0};
  spv::Id type_id_{0};
  std::vector<Operand> operands_;
  size_t word_count_{0};
  BasicBlock *parent_bb_{nullptr};
  spv::StorageClass storage_class_{spv::StorageClass::StorageClassUniform};
};

class IRPrinter : public IRVisitor {
  IRPrinter() {}
  ~IRPrinter () {}
  void Visit(Instruction *instruction) {
    switch (instruction->GetOpCode()) {
      case spv::Op::OpVariable:
        std::cout << "OpVariable " << std::endl;
        break;
      case spv::Op::OpLoad:
        std::cout << "OpLoad " << std::endl;
        break;
      case spv::Op::OpStore:
        std::cout << "OpStore " << std::endl;
        break;
      case spv::Op::OpBranch:
        std::cout << "OpBranch " << std::endl;
        break;
      case spv::Op::OpLabel:
        std::cout << "OpLabel " << std::endl;
        break;
      case spv::Op::OpBranchConditional:
        std::cout << "OpBranchConditional  " << std::endl;
        break;
      case spv::Op::OpAccessChain:
        std::cout << "OpAccessChain " << std::endl;
        break;
      case spv::Op::OpInBoundsAccessChain:
        std::cout << "OpInBoundsAccessChain  " << std::endl;
        break;
      case spv::Op::OpPhi:
        std::cout << "OpPhi " << std::endl;
        break;
      default:
        std::cout << "not implemented " << std::endl;
        break;
    }
  }
};

class BasicBlock {
 public:
  BasicBlock(std::string name, Function *function, SPIRVContext *context)
      : name_(std::move(name)), function_(function), ctx_(context) {
    // Each basic blocks starts from unique label.
    spv::Id id = ctx_->GetUniqueId();
    Instruction *instruction = new Instruction(spv::Op::OpLabel, 0, 0, {id});
    instructions_.push_back(instruction);
    label_id_ = id;
  }

  void AddInstruction(Instruction *instruction) {
    instructions_.push_back(instruction);
  }

  void AddPredeccessor(BasicBlock *predeccessor) {
    predeccessors_.push_back(predeccessor);
  }

  void AddSuccessor(BasicBlock *successor) { successors_.push_back(successor); }

  spv::Id Label() { return label_id_; }

  ~BasicBlock() {}

 private:
  std::vector<Instruction *> instructions_;
  std::vector<BasicBlock *> predeccessors_;
  std::vector<BasicBlock *> successors_;
  std::string name_;
  Function *function_;
  SPIRVContext *ctx_;
  spv::Id label_id_{0};
};

// Represents a spir-v function, consists of basic blocks.
class Function {
 public:
  Function(std::string function_name)
      : function_name_(std::move(function_name)) {}
  void AddBasicBlock(BasicBlock *bb) { cfg_.push_back(bb); }
 private:
  std::vector<BasicBlock *> cfg_;
  std::string function_name_;
  BasicBlock *entry_point;
};

// Module contains functions, each function contains entry point.
class Module {
  Module(std::string module_name) : module_name_(std::move(module_name)) {}
  Function *CreateFunction(std::string name) { return nullptr; }
 private:
  std::vector<Function *> functions_;
  std::string module_name_;
};

// Class which builds the IR.
class IRBuilder {
 public:
  IRBuilder(BasicBlock *insert_point, SPIRVContext *context)
      : insert_point_(insert_point), ctx_(context) {}
  IRBuilder(const IRBuilder &other) = delete;
  IRBuilder(IRBuilder &&other) = delete;
  IRBuilder &operator=(const IRBuilder &other) = delete;
  IRBuilder &operator=(IRBuilder &&other) = delete;
  // Clean all.
  ~IRBuilder() {}

  // Sets the insert point
  void SetInsertPoint(BasicBlock *insert_point) {
    insert_point_ = insert_point;
  }

  // %id = OpLoad %type %ptr MemAccesstype
  spv::Id CreateLoad(spv::Op type, spv::Id ptr) {
    spv::Id id = ctx_->GetUniqueId();
    Instruction *instruction =
        new Instruction(spv::Op::OpLoad, id, type, {ptr});
    insert_point_->AddInstruction(instruction);
    return id;
  }

  // OpStore %ptr %value MemAccessType
  void CreateStore(spv::Op ptr, spv::Id value) {
    Instruction *instruction =
        new Instruction(spv::Op::OpStore, 0, 0, {ptr, value});
    insert_point_->AddInstruction(instruction);
  }

  // OpBranch %label_id
  void CreateBr(BasicBlock *dest) {
    dest->AddPredeccessor(insert_point_);
    insert_point_->AddSuccessor(dest);
    Instruction *instruction =
        new Instruction(spv::Op::OpBranch, 0, 0, {dest->Label()});
    insert_point_->AddInstruction(instruction);
  }

  // OpBranchConditional %condition %true_id %false_id
  void CreateCondBr(spv::Id condition, BasicBlock *true_block,
                    BasicBlock *false_block) {
    insert_point_->AddSuccessor(true_block);
    insert_point_->AddSuccessor(false_block);
    true_block->AddPredeccessor(insert_point_);
    false_block->AddPredeccessor(insert_point_);

    Instruction *instruction =
        new Instruction(spv::Op::OpBranchConditional, 0, 0,
                        {condition, true_block->Label(), false_block->Label()});
    insert_point_->AddInstruction(instruction);
  }

  // OpLoopMerge %merge_id %continue_id
  void CreateLoopMerge(BasicBlock *merge_block, BasicBlock *continue_block) {
    Instruction *instruction =
        new Instruction(spv::Op::OpLoopMerge, 0, 0,
                        {merge_block->Label(), continue_block->Label()});
    insert_point_->AddInstruction(instruction);
  }

  // %id = OpAccessChain %ptr_type %ptr {%offsets}
  spv::Id CreateAccesssChain(spv::Op op_code, spv::Id ptr_type, spv::Id ptr,
                             const std::vector<spv::Id> &offsets) {
    spv::Id id = ctx_->GetUniqueId();
    std::vector<Operand> operands{ptr_type, ptr};
    for (size_t i = 0; i < offsets.size(); ++i) {
      operands.push_back(offsets[i]);
    }
    Instruction *instruction = new Instruction(op_code, id, 0, operands);
    insert_point_->AddInstruction(instruction);
    return id;
  }

  // %id = OpBinOp %type %lhs %rhs
  spv::Id CreateBinOp(spv::Op op_code, spv::Id type, spv::Id lhs, spv::Id rhs) {
    spv::Id id = ctx_->GetUniqueId();
    Instruction *instruction = new Instruction(op_code, id, type, {lhs, rhs});
    insert_point_->AddInstruction(instruction);
    return id;
  }

  // %id = OpPhi %type {%value %label}
  spv::Id CreatePhi(
      spv::Id type,
      const std::vector<std::pair<spv::Id, BasicBlock *>> &phi_values) {
    spv::Id id = ctx_->GetUniqueId();
    std::vector<Operand> operands;

    for (size_t i = 0; i < phi_values.size(); ++i) {
      // Value
      operands.push_back(phi_values[i].first);
      // Label
      operands.push_back(phi_values[i].second->Label());
    }

    Instruction *instruction =
        new Instruction(spv::Op::OpPhi, id, type, operands);
    insert_point_->AddInstruction(instruction);
    return id;
  }

  // Could takes a literals.
  // Figure out how to implement it.
  spv::Id CreateType(spv::Op kind) { return 0; }
  spv::Id CreatePointerType(spv::Id type) { return 0; }

  // %id = OpVariable %ptr StorageClass
  spv::Id CreateVariable(spv::Id type, spv::StorageClass storage_class) {
    spv::Id id = ctx_->GetUniqueId();
    Instruction *instruction =
        new Instruction(spv::Op::OpVariable, id, type, storage_class);
    insert_point_->AddInstruction(instruction);
    return id;
  }

  spv::Id CreateConstant(spv::Op type, int value) { return 0; }
  spv::Id CreateFunction() { return 0; }

 private:
  BasicBlock *insert_point_;
  SPIRVContext *ctx_;
};
}  // namespace spriv
}  // namespace xla
#endif
