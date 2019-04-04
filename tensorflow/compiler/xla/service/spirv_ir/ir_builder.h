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
#include <iostream>
#include <sstream>
#include <unordered_map>

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

// Simple singleton.
SPIRVContext *GetSPIRVContext() {
  static SPIRVContext *context = new SPIRVContext();
  return context;
}

// The operand could be an id or literal, in case of literal it's an id
// for lookup table.
class Operand {
 public:
  Operand() : id_(0), is_id_(true) {}
  Operand(spv::Id id, bool is_id = true) : id_(id), is_id_(is_id) {}
  bool IsId() { return is_id_; }
  spv::Id GetId() { return id_; }

 private:
  spv::Id id_;
  bool is_id_;
};

class IRVisitor {
 public:
  IRVisitor() {}
  virtual ~IRVisitor() {}
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

  Instruction(const Instruction &other) = delete;
  Instruction(Instruction &&other) = delete;
  Instruction &operator=(const Instruction &other) = delete;
  Instruction &operator=(Instruction &&other) = delete;
  ~Instruction() {}

  void AddOperand(spv::Id operand_id, bool is_id = true) {
    operands_.push_back(Operand(operand_id, is_id));
  }

  spv::Op GetOpCode() { return op_code_; }
  spv::Id GetResultId() { return result_id_; }
  spv::Id GetTypeId() { return type_id_; }
  std::vector<Operand> GetOperands() { return operands_; }
  void Accept(IRVisitor *visitor) { visitor->Visit(this); }
  size_t WordCount() { return word_count_; }

 private:
  spv::Op op_code_;
  spv::Id result_id_{0};
  spv::Id type_id_{0};
  std::vector<Operand> operands_;
  size_t word_count_{0};
};

class IRPrinter : public IRVisitor {
 public:
  IRPrinter() {}
  ~IRPrinter() {}

  void Visit(Instruction *instruction) {
    switch (instruction->GetOpCode()) {
      case spv::Op::OpVariable:
        ProcessInstruction(instruction, "OpVariable");
        break;
      case spv::Op::OpLoad:
        ProcessInstruction(instruction, "OpLoad");
        break;
      case spv::Op::OpStore:
        ProcessInstruction(instruction, "OpStore");
        break;
      case spv::Op::OpBranch:
        ProcessInstruction(instruction, "OpBranch");
        break;
      case spv::Op::OpLabel:
        ProcessInstruction(instruction, "OpLabel");
        break;
      case spv::Op::OpBranchConditional:
        ProcessInstruction(instruction, "OpBranchConditional");
        break;
      case spv::Op::OpAccessChain:
        ProcessInstruction(instruction, "OpAccessChain");
        break;
      case spv::Op::OpInBoundsAccessChain:
        ProcessInstruction(instruction, "OpInBoundsAccessChain");
        break;
      case spv::Op::OpPhi:
        ProcessInstruction(instruction, "OpPhi");
        break;
      case spv::Op::OpConstant:
        ProcessInstruction(instruction, "OpConstant");
        break;
      default:
        break;
    }
  }

  void ProcessInstruction(Instruction *instruction, std::string op_code) {
    if (instruction->GetResultId()) {
      stream_ << ident_ << instruction->GetResultId() << white_space_ << assign_
              << white_space_;
    }
    stream_ << op_code << white_space_;
    if (instruction->GetTypeId()) {
      stream_ << ident_ << instruction->GetTypeId() << white_space_;
    }
    std::vector<Operand> operands = instruction->GetOperands();
    for (size_t i = 0; i < operands.size(); ++i) {
      if (operands[i].IsId()) {
        stream_ << ident_ << operands[i].GetId() << white_space_;
      } else {
        // TODO: Lookup for literal
      }
    }

    stream_ << new_line_;
  }

 private:
  std::stringstream stream_;
  char ident_ = '%';
  char white_space_ = ' ';
  char assign_ = '=';
  char new_line_ = '\n';
};

class BasicBlock {
 public:
  BasicBlock(std::string name)
      : name_(std::move(name)) {
    // Each basic blocks starts from unique label.
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    Instruction *instruction = new Instruction(spv::Op::OpLabel, 0, 0, {id});
    instructions_.push_back(instruction);
    label_id_ = id;
  }
  ~BasicBlock() {}

  void AddInstruction(Instruction *instruction) {
    instructions_.push_back(instruction);
  }

  void AddPredeccessor(BasicBlock *predeccessor) {
    predeccessors_.push_back(predeccessor);
  }

  void AddSuccessor(BasicBlock *successor) { successors_.push_back(successor); }

  spv::Id Label() { return label_id_; }

  std::vector<Instruction *> &GetInstructions() { return instructions_; }

 private:
  std::vector<Instruction *> instructions_;
  std::vector<BasicBlock *> predeccessors_;
  std::vector<BasicBlock *> successors_;
  std::string name_;
  spv::Id label_id_{0};
};

// Represents a spir-v function, consists of basic blocks.
class Function {
 public:
  Function(std::string function_name)
      : function_name_(std::move(function_name)) {}
  void AddBasicBlock(BasicBlock *bb) { basic_blocks_.push_back(bb); }

  void AddEntryBlock(BasicBlock *bb) {
    entry_block_ = bb;
    basic_blocks_.push_back(bb);
  }

  void AddRetBlock(BasicBlock *bb) {
    ret_block_ = bb;
    basic_blocks_.push_back(bb);
  }

  std::vector<BasicBlock *> &GetBasicBlocks() { return basic_blocks_; }

 private:
  std::vector<BasicBlock *> basic_blocks_;
  std::string function_name_;
  Instruction *entry_point_;
  BasicBlock *entry_block_;
  BasicBlock *ret_block_;
};

// Module contains functions, each function contains entry point.
class Module {
 public:
  Module(std::string module_name) : module_name_(std::move(module_name)) {}

  Module (const Module &other) = delete;
  Module &operator=(const Module &other) = delete;
  Module(Module &&other) = delete;
  Module &operator=(Module &&other) = delete;

  ~Module() {
    // clear all
  }

  void Accept(IRVisitor *visitor) {
    // TODO:
    // Visit Decorator table at first.
    // Visit Custom types table at second.

    for (auto &table_instance : functions_) {
      // Think about reordering of the basic blocks.
      for (BasicBlock *bb : table_instance.second->GetBasicBlocks()) {
        for (Instruction *instruction : bb->GetInstructions()) {
          visitor->Visit(instruction);
        }
      }
    }
  }

  void InitBasicTypesAndConstants() {}
  // TODO: Init all basic types by default.

  // TODO: Implement it.
  spv::Id GetOrCreateCustomType(std::string type_name) { return 0; }
  spv::Id GetOrCreateConstant(std::string value) { return 0; }
  spv::Id GetOrCreateVariable(std::string var_name) { return 0; }
  Function *GetOrCreateFunction(std::string name,
                                std::string function_control = "None") {
    /*
      spv::Id id = ctx_->GetUniqueId();
      spv::Id literal_id = ctx_->GetUniqueId();
      literal_pool_id.insert({literal_id, function_control});
      Instruction *instruction = new Instruction(spv::Op::OpFunction, id, type,
                                                 {literal_id, function_type});
      instert_point_->
      */
    return nullptr;
  }

 private:
  std::unordered_map<std::string, Function *> functions_;
  std::string module_name_;
  std::unordered_map<std::string, Instruction *> user_types_table_;
  std::unordered_map<std::string, Instruction *> user_variales_table_;
};

// Class which builds the IR.
class IRBuilder {
 public:
  IRBuilder(BasicBlock *insert_point, Module *module)
      : insert_point_(insert_point), module_(module) {}
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
    spv::Id id = GetSPIRVContext()->GetUniqueId();
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
    spv::Id id = GetSPIRVContext()->GetUniqueId();
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
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    Instruction *instruction = new Instruction(op_code, id, type, {lhs, rhs});
    insert_point_->AddInstruction(instruction);
    return id;
  }

  // %id = OpPhi %type {%value %label}
  spv::Id CreatePhi(
      spv::Id type,
      const std::vector<std::pair<spv::Id, BasicBlock *>> &phi_values) {
    spv::Id id = GetSPIRVContext()->GetUniqueId();
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

  // %id = OpVariable %ptr StorageClass
  // The storage class defines, the "scope" of the variable.
  spv::Id CreateVariable(spv::Id type, std::string storage_class) {
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    spv::Id literal_id = GetSPIRVContext()->GetUniqueId();
    literal_pool_.insert({literal_id, std::move(storage_class)});
    Instruction *instruction =
        new Instruction(spv::Op::OpVariable, id, type, {literal_id});
    insert_point_->AddInstruction(instruction);
    return id;
  }

  // %id = OpConstant %type %literal_value
  spv::Id CreateConstant(spv::Id type, std::string value) {
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    spv::Id literal_id = GetSPIRVContext()->GetUniqueId();
    literal_pool_.insert({literal_id, std::move(value)});
    Instruction *instruction =
        new Instruction(spv::Op::OpConstant, id, type, {literal_id});
    insert_point_->AddInstruction(instruction);
    return id;
  }

 private:
  BasicBlock *insert_point_;
  Module *module_;
  std::unordered_map<spv::Id, std::string> literal_pool_;
};
}  // namespace spriv
}  // namespace xla
#endif
