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
#include <cassert>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>

using uint32 = uint32_t;

namespace xla {
namespace spirv {

// Forward declaration.
class Operand;
class Instruction;
class BasicBlock;
class Function;
class Module;

// The operand could be an id or literal, in case of literal it's an id
// for lookup table.
class Operand {
 public:
  Operand();
  Operand(spv::Id id, bool is_id = true);
  bool IsId() const ;
  spv::Id GetId() const;

 private:
  spv::Id id_;
  bool is_id_;
};

// The global context.
class SPIRVContext {
 public:
  SPIRVContext() {}
  spv::Id GetUniqueId();
  std::vector<Operand> CreateOperandsFromLiterals(
      std::vector<std::string> literals);
  std::string LookUpForLiteral(spv::Id literal_id);

 private:
  spv::Id unique_id{0};
  std::unordered_map<spv::Id, std::string> literal_pool_;
  std::mutex mutex_;
};

// The class which provides the forward declaration for the Visit method.
class IRVisitorBase {
  public:
   IRVisitorBase() {}
   virtual ~IRVisitorBase() {}
   virtual void Visit(Instruction *instruction) = 0;
};

// Represents an spir-v instruction. Based on section 2.2.1 from spir-v
// specification.
class Instruction {
 public:
  // In case global variable or other type of instruction which does not
  // require to be inserted into basic block - parent basic block is nullptr.
  Instruction(spv::Op op_code, spv::Id result_id, spv::Id type_id,
              std::vector<Operand> operands, BasicBlock *parent = nullptr);
  Instruction(spv::Op op_code, spv::Id result_id, spv::Id type_id,
              BasicBlock *parent = nullptr);
  Instruction(const Instruction &other) = delete;
  Instruction(Instruction &&other) = delete;
  Instruction &operator=(const Instruction &other) = delete;
  Instruction &operator=(Instruction &&other) = delete;
  ~Instruction();

  void AddOperand(spv::Id operand_id, bool is_id = true);
  spv::Op GetOpCode() const;
  spv::Id GetResultId() const;
  spv::Id GetTypeId() const;
  std::vector<Operand> &GetOperands();
  void Accept(IRVisitorBase *visitor);
  std::string GetStringOpCode();

 private:
  spv::Op op_code_;
  spv::Id result_id_{0};
  spv::Id type_id_{0};
  std::vector<Operand> operands_;
  BasicBlock *parent_{nullptr};
};

class IRVisitor : public IRVisitorBase {
 public:
  IRVisitor() {}
  virtual ~IRVisitor() override {}
  virtual void Visit(Instruction *instrcution);
  virtual void HandleHeaderOp(Instruction *instruction) = 0;
  virtual void HandleVariableOp(Instruction *instruction) = 0;
  virtual void HandleTypeOp(Instruction *instruction) = 0;
  virtual void HandlePointerTypeOp(Instruction *instruction) = 0;
  virtual void HandleFunctionOp(Instruction *instruction) = 0;
  virtual void HandleMemAccessOp(Instruction *instruction) = 0;
  virtual void HandleControlFlowOp(Instruction *instruction) = 0;
  virtual void HandleAccessChainOp(Instruction *instruction) = 0;
  virtual void HandleBinOp(Instruction *instruction) = 0;
  virtual void HandleDecorate(Instruction *instruction) = 0;
};

class IRPrinter : public IRVisitor {
 public:
  IRPrinter();
  ~IRPrinter() override;

  void Visit(Instruction *instruction) override;
  void HandleHeaderOp(Instruction *instruction) override;
  void HandleVariableOp(Instruction *instruction) override;
  void HandleTypeOp(Instruction *instruction) override;
  void HandlePointerTypeOp(Instruction *instruction) override;
  void HandleFunctionOp(Instruction *instruction) override;
  void HandleMemAccessOp(Instruction *instruction) override;
  void HandleControlFlowOp(Instruction *instruction) override;
  void HandleAccessChainOp(Instruction *instruction) override;
  void HandleBinOp(Instruction *instruction) override;
  void HandleDecorate(Instruction *instruction) override;
  void HandleInstruction(Instruction *instruction);
  void HandleOperands(const std::vector<Operand> &operands);
  void Dump();
  void AddMetaInfo();

 private:
  std::stringstream stream_;
  char ident_ = '%';
  // Should be platfomr specific.
  char white_space_ = ' ';
  char assign_ = '=';
  char new_line_ = '\n';
  char quote_ = '"';
  char tab_ = '\t';
};

class BasicBlock {
 public:
  BasicBlock(std::string name);
  ~BasicBlock();
  Instruction *GetInstructionById(spv::Id id);
  void AddInstruction(Instruction *instruction);
  void AddPredeccessor(BasicBlock *predeccessor);
  void AddSuccessor(BasicBlock *successor);
  spv::Id Label();
  std::vector<Instruction *> &GetInstructions();
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
  Function(std::string function_name);
  void AddBasicBlock(BasicBlock *bb);
  void AddEntryBlock(BasicBlock *bb);
  BasicBlock *GetEntryBlock();
  BasicBlock *GetRetBlock();
  void AddRetBlock(BasicBlock *bb);
  void SetEntryPoint(Instruction *entry_point);
  Instruction *GetEntryPoint();
  std::vector<BasicBlock *> &GetBasicBlocks();
  std::string GetFunctionName();

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
  Module(std::string module_name);
  Module(const Module &other) = delete;
  Module &operator=(const Module &other) = delete;
  Module(Module &&other) = delete;
  Module &operator=(Module &&other) = delete;
  ~Module();

  void Accept(IRVisitor *visitor);
  void InitHeader();
  void CreateEntryPoint(Function *func, spv::Id work_group_id);
  void CreateExecutionMode(Function *func,
                           std::vector<std::string> local_sizes);
  void Decorate(spv::Id target_id, std::vector<std::string> literals);
  void MemberDecorate(spv::Id struct_type, std::vector<std::string> literals);
  spv::Id GetOrCreateCustomType(spv::Op type_code, spv::Id type_id,
                                std::string type_name);
  spv::Id GetOrCreateCustomType(spv::Op type_code, spv::Id type_id,
                                std::vector<std::string> literals,
                                std::string type_name);
  spv::Id GetOrCreateCustomTypeLen(spv::Op type_code, spv::Id type_id,
                                   std::vector<Operand> member_types,
                                   std::string type_name);
  spv::Id GetOrCreateGlobalVariable(spv::Id type_id, bool is_constant,
                                    std::vector<std::string> literals,
                                    std::string global_var_name,
                                    spv::Id initializer = 0);
  spv::Id GetOrCreateConstantComposite(spv::Id type_id,
                                       std::vector<Operand> consituents,
                                       std::string constant_name);
  Function *GetOrCreateFunction(std::string name, spv::Id ret_type,
                                spv::Id func_type,
                                std::string function_control);

  spv::Id GetOrCreateInt32TypeId();
  spv::Id GetOrCreateUInt32TypeId();
  spv::Id GetOrCreateFloat32TypeId();
  spv::Id GetOrCreateFloat64TypeId();
  spv::Id GetOrCreateInt64TypeId();
  spv::Id GetOrCreateUInt64TypeId();
  spv::Id GetOrCreateVoidTypeId();
  spv::Id GetOrCreateBoolTypeId();

 private:
  std::unordered_map<std::string, Function *> functions_;
  std::string module_name_;
  std::vector<Instruction *> header_;
  std::vector<Instruction *> decoration_table_;
  std::unordered_map<std::string, Instruction *> user_vars_types_table_;
};

// Class which builds the IR.
class IRBuilder {
 public:
  IRBuilder(BasicBlock *insert_point, Module *module);
  IRBuilder(const IRBuilder &other) = delete;
  IRBuilder(IRBuilder &&other) = delete;
  IRBuilder &operator=(const IRBuilder &other) = delete;
  IRBuilder &operator=(IRBuilder &&other) = delete;
  ~IRBuilder();

  // Sets the insert point
  void SetInsertPoint(BasicBlock *insert_point);
  // %id = OpLoad %type %ptr MemAccesstype
  spv::Id CreateLoad(spv::Id type, spv::Id ptr,
                     std::vector<std::string> literals);
  // OpStore %ptr %value MemAccessType
  void CreateStore(spv::Id ptr, spv::Id value,
                   std::vector<std::string> literals);
  // OpBranch %label_id
  void CreateBr(BasicBlock *dest);
  // OpBranchConditional %condition %true_id %false_id
  void CreateCondBr(spv::Id condition, BasicBlock *true_block,
                    BasicBlock *false_block);
  // OpLoopMerge %merge_id %continue_id
  void CreateLoopMerge(BasicBlock *merge_block, BasicBlock *continue_block,
                       std::vector<std::string> literals);
  // %id = OpAccessChain %ptr_type %ptr {%offsets}
  spv::Id CreateAccessChain(spv::Op op_code, spv::Id ptr_type, spv::Id ptr,
                            const std::vector<spv::Id> &offsets);
  // %id = OpBinOp %type %lhs %rhs
  spv::Id CreateBinOp(spv::Op op_code, spv::Id type, spv::Id lhs, spv::Id rhs);
  // %id = OpPhi %type {%value %label}
  spv::Id CreatePhi(spv::Id type);
  void AddIncoming(BasicBlock *phi_block, spv::Id phi_id, spv::Id id,
                   BasicBlock *bb);
  // %id = OpVariable %type {literals}
  // %id = OpConstant %type {literals}
  spv::Id CreateLocalVariable(spv::Id type_id, bool is_constant,
                              std::vector<std::string> literals);

 private:
  BasicBlock *insert_point_;
  Module *module_;
};
}  // namespace spirv
}  // namespace xla
#endif
