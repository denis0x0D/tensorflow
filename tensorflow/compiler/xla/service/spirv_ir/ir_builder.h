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
  Operand() : id_(0), is_id_(true) {}
  Operand(spv::Id id, bool is_id = true) : id_(id), is_id_(is_id) {}
  bool IsId() const { return is_id_; }
  spv::Id GetId() const { return id_; }

 private:
  spv::Id id_;
  bool is_id_;
};

// The global context.
class SPIRVContext {
 public:
  SPIRVContext() {}
  spv::Id GetUniqueId() { return ++unique_id; }

  std::vector<Operand> CreateOperandsFromLiterals(
      std::vector<std::string> literals) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::vector<Operand> operands;
    for (size_t i = 0; i < literals.size(); ++i) {
      spv::Id id = GetUniqueId();
      literal_pool_.insert({id, literals[i]});
      operands.push_back(Operand(id, false));
    }
    return operands;
  }

  std::string LookUpForLiteral(spv::Id literal_id) {
    std::unique_lock<std::mutex> lock (mutex_);
    // TODO: insert assert
    if (literal_pool_.count(literal_id)) {
      return literal_pool_[literal_id];
    }
    return "0";
  }

 private:
  spv::Id unique_id{0};
  std::unordered_map<spv::Id, std::string> literal_pool_;
  std::mutex mutex_;
};

// Simple singleton.
SPIRVContext *GetSPIRVContext() {
  static SPIRVContext *context = new SPIRVContext();
  return context;
}

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
  // In case global variable or other type of instruction which does not
  // require to be inserted into basic block - parent basic block is nullptr.
  Instruction(spv::Op op_code, spv::Id result_id, spv::Id type_id,
              std::vector<Operand> operands, BasicBlock *parent = nullptr)
      : op_code_(op_code),
        result_id_(result_id),
        type_id_(type_id),
        operands_(std::move(operands)),
        parent_(parent) {}

  Instruction(spv::Op op_code, spv::Id result_id, spv::Id type_id,
              BasicBlock *parent = nullptr)
      : op_code_(op_code),
        result_id_(result_id),
        type_id_(type_id),
        parent_(parent) {}

  Instruction(const Instruction &other) = delete;
  Instruction(Instruction &&other) = delete;
  Instruction &operator=(const Instruction &other) = delete;
  Instruction &operator=(Instruction &&other) = delete;
  ~Instruction() {}

  void AddOperand(spv::Id operand_id, bool is_id = true) {
    operands_.push_back(Operand(operand_id, is_id));
  }

  spv::Op GetOpCode() const { return op_code_; }
  spv::Id GetResultId() const { return result_id_; }
  spv::Id GetTypeId() const { return type_id_; }
  std::vector<Operand> &GetOperands() { return operands_; }
  void Accept(IRVisitor *visitor) { visitor->Visit(this); }

  std::string GetStringOpCode() {
    switch (op_code_) {
      case spv::Op::OpVariable:
        return "OpVariable";
      case spv::Op::OpConstant:
        return "OpConstant";
      case spv::Op::OpTypeVoid:
        return "OpTypeVoid";
      case spv::Op::OpTypeInt:
        return "OpTypeInt";
      case spv::Op::OpTypeFloat:
        return "OpTypeFloat";
      case spv::Op::OpTypeBool:
        return "OpTypeBool";
      case spv::Op::OpTypeFunction:
        return "OpTypeFunction";
      case spv::Op::OpTypePointer:
        return "OpTypePointer";
      case spv::Op::OpTypeRuntimeArray:
        return "OpTypeRuntimeArray";
      case spv::Op::OpTypeStruct:
        return "OpTypeStruct";
      case spv::Op::OpTypeVector:
        return "OpTypeVector";
      case spv::Op::OpLoad:
        return "OpLoad";
      case spv::Op::OpStore:
        return "OpStore";
      case spv::Op::OpBranch:
        return "OpBranch";
      case spv::Op::OpBranchConditional:
        return "OpBranchConditional";
      case spv::Op::OpLoopMerge:
        return "OpLoopMerge";
      case spv::Op::OpLabel:
        return "OpLabel";
      case spv::Op::OpPhi:
        return "OpPhi";
      case spv::Op::OpAccessChain:
        return "OpAccessChain";
      case spv::Op::OpInBoundsAccessChain:
        return "OpInBoundsAccessChain";
      case spv::Op::OpDecorate:
        return "OpDecorate";
      case spv::Op::OpMemberDecorate:
        return "OpMemberDecorate";
      case spv::Op::OpFunction:
        return "OpFunction";
      case spv::Op::OpIMul:
        return "OpIMul";
      case spv::Op::OpIAdd:
        return "OpIAdd";
      case spv::Op::OpISub:
        return "OpISub";
      case spv::Op::OpUDiv:
        return "OpUdiv";
      case spv::Op::OpSDiv:
        return "OpSDiv";
      case spv::Op::OpUMod:
        return "OpUMod";
      case spv::Op::OpSRem:
        return "OpSRem";
      case spv::Op::OpSMod:
        return "OpSMod";
      case spv::Op::OpShiftRightLogical:
        return "OpShiftRightLogical";
      case spv::Op::OpShiftLeftLogical:
        return "OpShitfLeftLogical";
      case spv::Op::OpBitwiseOr:
        return "OpBitwiseOr";
      case spv::Op::OpBitwiseXor:
        return "OpBitwiseXor";
      case spv::Op::OpBitwiseAnd:
        return "OpBitwiseAnd";
      case spv::Op::OpLogicalOr:
        return "OpLogicalOr";
      case spv::Op::OpLogicalNot:
        return "OpLogicalNot";
      case spv::Op::OpLogicalEqual:
        return "OpLogicalEqual";
      case spv::Op::OpLogicalNotEqual:
        return "OpLogicalNotEqual";
      case spv::Op::OpIEqual:
        return "OpIEqual";
      case spv::Op::OpINotEqual:
        return "OpINotEqual";
      case spv::Op::OpULessThan:
        return "OpULessThan";
      case spv::Op::OpSLessThan:
        return "OpSLessThan";
      case spv::Op::OpUGreaterThan:
        return "OpUGreaterThan";
      case spv::Op::OpSGreaterThan:
        return "OpSGreaterThan";
      case spv::Op::OpULessThanEqual:
        return "OpULessThanEqual";
      case spv::Op::OpSLessThanEqual:
        return "OpSLessThanEqual";
      case spv::Op::OpUGreaterThanEqual:
        return "OpUGreaterThanEqual";
      case spv::Op::OpSGreaterThanEqual:
        return "OpSGreaterThanEqual";
      case spv::Op::OpReturn:
        return "OpReturn";
      case spv::Op::OpFunctionEnd:
        return "OpFunctionEnd";
      case spv::Op::OpCapability:
        return "OpCapability";
      case spv::Op::OpMemoryModel:
        return "OpMemoryModel";
      case spv::Op::OpExtInstImport:
        return "OpExtInstImport";
      case spv::Op::OpEntryPoint:
        return "OpEntryPoint";
      case spv::Op::OpExecutionMode:
        return "OpExecutionMode";
      default:
        return "Unknown";
    }
  }

 private:
  spv::Op op_code_;
  spv::Id result_id_{0};
  spv::Id type_id_{0};
  std::vector<Operand> operands_;
  BasicBlock *parent_{nullptr};
};

class IRPrinter : public IRVisitor {
 public:
  IRPrinter() {}
  ~IRPrinter() {}
  // Instruction processing is split based on instruction semantics.
  // It could make sence, because some instructions have different text layout
  // than binary form.
  void Visit(Instruction *instruction) {
    switch (instruction->GetOpCode()) {
      case spv::Op::OpVariable:
      case spv::Op::OpConstant:
        ProcessVariableOp(instruction);
        break;
      case spv::Op::OpTypeVoid:
      case spv::Op::OpTypeInt:
      case spv::Op::OpTypeFloat:
      case spv::Op::OpTypeBool:
      case spv::Op::OpTypeVector:
      case spv::Op::OpTypeRuntimeArray:
      case spv::Op::OpTypeStruct:
      case spv::Op::OpTypeFunction:
        ProcessTypeOp(instruction);
        break;
      case spv::Op::OpTypePointer:
        ProcessPointerTypeOp(instruction);
        break;
      case spv::Op::OpFunction:
        ProcessFunctionOp(instruction);
        break;
      case spv::Op::OpLoad:
      case spv::Op::OpStore:
        ProcessMemAccessOp(instruction);
        break;
      case spv::Op::OpBranch:
      case spv::Op::OpBranchConditional:
      case spv::Op::OpLabel:
      case spv::Op::OpPhi:
      case spv::Op::OpLoopMerge:
      case spv::Op::OpReturn:
      case spv::Op::OpFunctionEnd:
        ProcessControlFlowOp(instruction);
        break;
      case spv::Op::OpAccessChain:
      case spv::Op::OpInBoundsAccessChain:
        ProcessAccessChainOp(instruction);
        break;
      case spv::Op::OpDecorate:
      case spv::Op::OpMemberDecorate:
        ProcessDecorate(instruction);
        break;
      case spv::Op::OpIMul:
      case spv::Op::OpIAdd:
      case spv::Op::OpSLessThan:
      case spv::Op::OpUDiv:
      case spv::Op::OpSDiv:
      case spv::Op::OpUMod:
      case spv::Op::OpSRem:
      case spv::Op::OpSMod:
      case spv::Op::OpShiftRightLogical:
      case spv::Op::OpShiftLeftLogical:
      case spv::Op::OpBitwiseOr:
      case spv::Op::OpBitwiseXor:
      case spv::Op::OpBitwiseAnd:
      case spv::Op::OpLogicalAnd:
      case spv::Op::OpLogicalOr:
      case spv::Op::OpLogicalNot:
      case spv::Op::OpLogicalEqual:
      case spv::Op::OpLogicalNotEqual:
      case spv::Op::OpIEqual:
      case spv::Op::OpINotEqual:
      case spv::Op::OpULessThan:
      case spv::Op::OpUGreaterThan:
      case spv::Op::OpSGreaterThan:
      case spv::Op::OpULessThanEqual:
      case spv::Op::OpSLessThanEqual:
      case spv::Op::OpUGreaterThanEqual:
      case spv::Op::OpSGreaterThanEqual:
        ProcessBinOp(instruction);
        break;
      case spv::Op::OpCapability:
      case spv::Op::OpExtInstImport:
      case spv::Op::OpMemoryModel:
      case spv::Op::OpEntryPoint:
      case spv::Op::OpExecutionMode:
        ProcessHeaderOp(instruction);
        break;
      default:
        break;
    }
  }

  void ProcessHeaderOp(Instruction *instruction) {
    assert(instruction && "Instruction is nullptr");
    if (instruction->GetOpCode() == spv::Op::OpEntryPoint) {
      unsigned index = 0;
      auto operands = instruction->GetOperands();
      assert(operands.size() == 3 &&
             "Operand size for OpEntryPoint should be equal to 3");
      stream_ << tab_ << instruction->GetStringOpCode() << white_space_;
            stream_ << GetSPIRVContext()->LookUpForLiteral(operands[index].GetId())
              << white_space_;
      ++index;
      stream_ << ident_ << instruction->GetResultId() << white_space_;
      stream_ << quote_
              << GetSPIRVContext()->LookUpForLiteral(operands[index].GetId())
              << quote_ << white_space_;
      ++index;
      stream_ << ident_ << operands[index].GetId() << new_line_;
    } else if (instruction->GetOpCode() == spv::Op::OpExecutionMode) {
      stream_ << tab_ << instruction->GetStringOpCode() << white_space_
              << ident_ << instruction->GetResultId() << white_space_;
      ProcessOperands(instruction->GetOperands());
    } else if (instruction->GetOpCode() == spv::Op::OpExtInstImport) {
      ProcessInstruction(instruction);
      assert(instruction->GetOperands().size() == 1 &&
             "Operand size for OpExtInstImport should be equal to 1 ");
      stream_ << quote_
              << GetSPIRVContext()->LookUpForLiteral(
                     instruction->GetOperands()[0].GetId())
              << quote_ << new_line_;
    } else {
      ProcessInstruction(instruction);
      ProcessOperands(instruction->GetOperands());
    }
  }

  void ProcessVariableOp(Instruction *instruction) {
    ProcessInstruction(instruction);
    ProcessOperands(instruction->GetOperands());
  }

  void ProcessTypeOp(Instruction *instruction) {
    ProcessInstruction(instruction);
    ProcessOperands(instruction->GetOperands());
  }

  // The layout is different agaist usual type op.
  // %id = OpTypePointer literal %type_id
  void ProcessPointerTypeOp(Instruction *instruction) {
    stream_ << ident_ << instruction->GetResultId() << white_space_ << assign_
            << white_space_ << instruction->GetStringOpCode() << white_space_;
    assert(instruction->GetOperands().size() == 1 &&
           "Operand size for OpTypePointer should be equal to 1");
    Operand op = instruction->GetOperands()[0];
    stream_ << GetSPIRVContext()->LookUpForLiteral(op.GetId()) << white_space_;
    stream_ << ident_ << instruction->GetTypeId() << white_space_;
    stream_ << new_line_;
  }

  void ProcessFunctionOp(Instruction *instruction) {
    stream_ << ident_ << instruction->GetResultId() << white_space_ << assign_
            << white_space_ << instruction->GetStringOpCode() << white_space_
            << ident_ << instruction->GetTypeId() << white_space_;
    ProcessOperands(instruction->GetOperands());
  }

  void ProcessMemAccessOp(Instruction *instruction) {
    ProcessInstruction(instruction);
    ProcessOperands(instruction->GetOperands());
  }

  void ProcessControlFlowOp(Instruction *instruction) {
    ProcessInstruction(instruction);
    ProcessOperands(instruction->GetOperands());
  }

  void ProcessAccessChainOp(Instruction *instruction) {
    ProcessInstruction(instruction);
    ProcessOperands(instruction->GetOperands());
  }

  void ProcessBinOp(Instruction *instruction) {
    ProcessInstruction(instruction);
    ProcessOperands(instruction->GetOperands());
  }

  void ProcessDecorate(Instruction *instruction) {
    stream_ << tab_ << instruction->GetStringOpCode() << white_space_ << ident_
            << instruction->GetResultId() << white_space_;
    ProcessOperands(instruction->GetOperands());
  }

  void ProcessInstruction(Instruction *instruction) {
    if (instruction->GetResultId()) {
      stream_ << ident_ << instruction->GetResultId() << white_space_ << assign_
              << white_space_;
    } else {
      stream_ << tab_;
    }
    stream_ << instruction->GetStringOpCode() << white_space_;
    if (instruction->GetTypeId()) {
      stream_ << ident_ << instruction->GetTypeId() << white_space_;
    }
  }

  void ProcessOperands(const std::vector<Operand> &operands) {
    for (auto &op : operands) {
      if (op.IsId()) {
        stream_ << ident_ << op.GetId();
      } else {
        stream_ << GetSPIRVContext()->LookUpForLiteral(op.GetId());
      }
      stream_ << white_space_;
    }
    stream_ << new_line_;
  }

  void Dump() { std::cout << stream_.str(); }

  void AddMetaInfo() {
    // TODO: I did not find any official specification of SPIR-V
    // text format. This meta is the same as SPIR-V tools generate.
    stream_ << "; SPIR-V" << new_line_;
    stream_ << "; Version: 1.3" << new_line_;
    stream_ << "; Generator: Khronos; 0" << new_line_;
    stream_ << "; Bound: 0" << new_line_;
    stream_ << "; Schema: 0" << new_line_;
  }

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
  BasicBlock(std::string name)
      : name_(std::move(name)) {
    // Each basic blocks starts from unique label.
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    Instruction *instruction = new Instruction(spv::Op::OpLabel, id, 0);
    AddInstruction(instruction);
    label_id_ = id;
  }

  ~BasicBlock() {
    for (auto *it : instructions_) {
      delete it;
    }
  }

  // Liner search.
  Instruction *GetInstructionById(spv::Id id) {
    for (auto *it: instructions_) {
      if (it->GetResultId() == id) {
        return it;
      }
    }
    return nullptr;
  }

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

  BasicBlock *GetEntryBlock() { return entry_block_; }
  BasicBlock *GetRetBlock() { return ret_block_; }

  void AddRetBlock(BasicBlock *bb) {
    Instruction *ret_instruction = new Instruction(spv::Op::OpReturn, 0, 0);
    Instruction *func_end_instruction =
        new Instruction(spv::Op::OpFunctionEnd, 0, 0);
    bb->AddInstruction(ret_instruction);
    bb->AddInstruction(func_end_instruction);
    ret_block_ = bb;
    basic_blocks_.push_back(bb);
  }

  void SetEntryPoint(Instruction *entry_point) { entry_point_ = entry_point; }
  Instruction *GetEntryPoint() { return entry_point_; }
  std::vector<BasicBlock *> &GetBasicBlocks() { return basic_blocks_; }
  std::string GetFunctionName() { return function_name_; }

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
    // Frees header at first.
    for (auto *header_inst : header_) {
      delete header_inst;
    }
    for (auto *dec : decoration_table_) {
      delete dec;
    }
    for (auto *type : user_types_table_) {
      delete type;
    }
    for (auto *var : user_var_table_) {
      delete var;
    }

    // So, at this moment just process the array and free the
    // basic block in order it was added.
    for (auto &table_instance : functions_) {
      for (auto *bb : table_instance.second->GetBasicBlocks()) {
        delete bb;
      }
      delete table_instance.second->GetEntryPoint();
      // Delete function.
      delete table_instance.second;
    }
  }

  void Accept(IRVisitor *visitor) {
    // Visit header.
    for (auto *header_inst : header_) {
      visitor->Visit(header_inst);
    }
    // Visit decoration table.
    for (auto *dec : decoration_table_) {
      visitor->Visit(dec);
    }
    // Visit custom types.
    for (auto *type : user_types_table_) {
      visitor->Visit(type);
    }
    // Visit variables.
    for (auto *var : user_var_table_) {
      visitor->Visit(var);
    }

    for (auto &table_instance : functions_) {
      // Entry point at first.
      auto *entry = table_instance.second->GetEntryPoint();
      visitor->Visit(entry);

      for (BasicBlock *bb : table_instance.second->GetBasicBlocks()) {
        for (auto *it : bb->GetInstructions()) {
          visitor->Visit(it);
        }
      }
    }
  }

  void InitHeader() {
    auto cap_operands =
        GetSPIRVContext()->CreateOperandsFromLiterals({"Shader"});
    Instruction *cap_instruction =
        new Instruction(spv::Op::OpCapability, 0, 0, std::move(cap_operands));
    header_.push_back(cap_instruction);
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    auto ext_operands =
        GetSPIRVContext()->CreateOperandsFromLiterals({"GLSL.std.450"});
    Instruction *ext_import = new Instruction(spv::Op::OpExtInstImport, id, 0,
                                              std::move(ext_operands));
    header_.push_back(ext_import);
    auto mem_operands =
        GetSPIRVContext()->CreateOperandsFromLiterals({"Logical", "GLSL450"});
    Instruction *mem_model =
        new Instruction(spv::Op::OpMemoryModel, 0, 0, std::move(mem_operands));
    header_.push_back(mem_model);
  }

  void CreateEntryPoint(Function *func, spv::Id work_group_id) {
    assert(func && "Function for entry point could not be null");
    std::vector<Operand> operands =
        GetSPIRVContext()->CreateOperandsFromLiterals(
            {"GLCompute", func->GetFunctionName()});
    operands.push_back(Operand(work_group_id));
    Instruction *global_entry =
        new Instruction(spv::Op::OpEntryPoint,
                        func->GetEntryPoint()->GetResultId(), 0, operands);
    header_.push_back(global_entry);
  }

  void CreateExecutionMode(Function *func,
                           std::vector<std::string> local_sizes) {
    auto operands =
        GetSPIRVContext()->CreateOperandsFromLiterals(std::move(local_sizes));
    Instruction *exec_mode = new Instruction(
        spv::Op::OpExecutionMode, func->GetEntryPoint()->GetResultId(), 0,
        std::move(operands));
    header_.push_back(exec_mode);
  }

  void Decorate(spv::Id target_id, std::vector<std::string> literals) {
    std::vector<Operand> operands =
        GetSPIRVContext()->CreateOperandsFromLiterals(std::move(literals));
    Instruction *instruction =
        new Instruction(spv::Op::OpDecorate, target_id, 0, std::move(operands));
    decoration_table_.push_back(instruction);
  }

  void MemberDecorate(spv::Id struct_type, std::vector<std::string> literals) {
    std::vector<Operand> operands =
        GetSPIRVContext()->CreateOperandsFromLiterals(std::move(literals));
    Instruction *instruction = new Instruction(
        spv::Op::OpMemberDecorate, struct_type, 0, std::move(operands));
    decoration_table_.push_back(instruction);
  }

  spv::Id CreateCustomType(spv::Op type_code, spv::Id type_id) {
    return CreateCustomType(type_code, type_id, {});
  }

  spv::Id CreateCustomType(spv::Op type_code, spv::Id type_id,
                           std::vector<std::string> literals) {
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    std::vector<Operand> operands =
        GetSPIRVContext()->CreateOperandsFromLiterals(std::move(literals));
    Instruction *instruction =
        new Instruction(type_code, id, type_id, std::move(operands));
    user_types_table_.push_back(instruction);
    return id;
  }

  spv::Id CreateGlobalVariable(spv::Id type_id, bool is_constant,
                               std::vector<std::string> literals) {
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    std::vector<Operand> operands =
        GetSPIRVContext()->CreateOperandsFromLiterals(std::move(literals));
    Instruction *instruction = new Instruction(
        !is_constant ? spv::Op::OpVariable : spv::Op::OpConstant, id, type_id,
        std::move(operands));
    user_var_table_.push_back(instruction);
    return id;
  }

  Function *GetOrCreateFunction(std::string name, spv::Id ret_type,
                                spv::Id func_type,
                                std::string function_control) {
    // Check the table at first.
    if (functions_.count(name)) {
      return functions_[name];
    }
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    std::vector<Operand> operands =
        GetSPIRVContext()->CreateOperandsFromLiterals(
            {std::move(function_control)});
    operands.push_back(func_type);
    Function *function = new Function(std::move(name));
    Instruction *entry_point =
        new Instruction(spv::Op::OpFunction, id, ret_type, operands);
    function->SetEntryPoint(entry_point);
    functions_.insert({name, function});
    return function;
  }

 private:
  std::unordered_map<std::string, Function *> functions_;
  std::string module_name_;
  std::vector<Instruction *> header_;
  std::vector<Instruction *> decoration_table_;
  std::vector<Instruction *> user_types_table_;
  std::vector<Instruction *> user_var_table_;
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
  spv::Id CreateLoad(spv::Id type, spv::Id ptr,
                     std::vector<std::string> literals) {
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    std::vector<Operand> operands =
        GetSPIRVContext()->CreateOperandsFromLiterals(std::move(literals));
    operands.insert(operands.begin(), Operand(ptr));
    Instruction *instruction = new Instruction(
        spv::Op::OpLoad, id, type, std::move(operands), insert_point_);
    insert_point_->AddInstruction(instruction);
    return id;
  }

  // OpStore %ptr %value MemAccessType
  void CreateStore(spv::Id ptr, spv::Id value,
                   std::vector<std::string> literals) {
    std::vector<Operand> operands{ptr, value};
    std::vector<Operand> other_operands =
        GetSPIRVContext()->CreateOperandsFromLiterals(std::move(literals));
    if (other_operands.size()) {
      operands.push_back(other_operands.front());
    }
    Instruction *instruction = new Instruction(
        spv::Op::OpStore, 0, 0, std::move(operands), insert_point_);
    insert_point_->AddInstruction(instruction);
  }

  // OpBranch %label_id
  void CreateBr(BasicBlock *dest) {
    dest->AddPredeccessor(insert_point_);
    insert_point_->AddSuccessor(dest);
    Instruction *instruction = new Instruction(spv::Op::OpBranch, 0, 0,
                                               {dest->Label()}, insert_point_);
    insert_point_->AddInstruction(instruction);
  }

  // OpBranchConditional %condition %true_id %false_id
  void CreateCondBr(spv::Id condition, BasicBlock *true_block,
                    BasicBlock *false_block) {
    insert_point_->AddSuccessor(true_block);
    insert_point_->AddSuccessor(false_block);
    true_block->AddPredeccessor(insert_point_);
    false_block->AddPredeccessor(insert_point_);

    Instruction *instruction = new Instruction(
        spv::Op::OpBranchConditional, 0, 0,
        {condition, true_block->Label(), false_block->Label()}, insert_point_);
    insert_point_->AddInstruction(instruction);
  }

  // OpLoopMerge %merge_id %continue_id
  void CreateLoopMerge(BasicBlock *merge_block, BasicBlock *continue_block,
                       std::vector<std::string> literals) {
    std::vector<Operand> operands{Operand(merge_block->Label()),
                                  Operand(continue_block->Label())};
    std::vector<Operand> other_operands =
        GetSPIRVContext()->CreateOperandsFromLiterals(std::move(literals));
    assert(other_operands.size());
    operands.push_back(other_operands[0]);
    Instruction *instruction = new Instruction(
        spv::Op::OpLoopMerge, 0, 0, std::move(operands), insert_point_);
    insert_point_->AddInstruction(instruction);
  }

  // %id = OpAccessChain %ptr_type %ptr {%offsets}
  spv::Id CreateAccessChain(spv::Op op_code, spv::Id ptr_type, spv::Id ptr,
                             const std::vector<spv::Id> &offsets) {
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    std::vector<Operand> operands{ptr_type, ptr};
    for (size_t i = 0; i < offsets.size(); ++i) {
      operands.push_back(offsets[i]);
    }
    Instruction *instruction =
        new Instruction(op_code, id, 0, operands, insert_point_);
    insert_point_->AddInstruction(instruction);
    return id;
    }

  // %id = OpBinOp %type %lhs %rhs
  spv::Id CreateBinOp(spv::Op op_code, spv::Id type, spv::Id lhs, spv::Id rhs) {
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    Instruction *instruction =
        new Instruction(op_code, id, type, {lhs, rhs}, insert_point_);
    insert_point_->AddInstruction(instruction);
    return id;
  }

  // %id = OpPhi %type {%value %label}
  spv::Id CreatePhi(spv::Id type) {
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    Instruction *instruction =
        new Instruction(spv::Op::OpPhi, id, type, {}, insert_point_);
    insert_point_->AddInstruction(instruction);
    return id;
  }

  void AddIncoming(BasicBlock *phi_block, spv::Id phi_id, spv::Id id,
                   BasicBlock *bb) {
    if (!phi_block) return;
    Instruction *phi = phi_block->GetInstructionById(phi_id);
    if (!phi) return;
    phi->AddOperand(id);
    phi->AddOperand(bb->Label());
  }

  // %id = OpVariable %type {literals}
  // %id = OpConstant %type {literals}
  spv::Id CreateLocalVariable(spv::Id type_id, bool is_constant,
                              std::vector<std::string> literals) {
    spv::Id id = GetSPIRVContext()->GetUniqueId();
    std::vector<Operand> operands =
        GetSPIRVContext()->CreateOperandsFromLiterals(std::move(literals));
    Instruction *instruction =
        new Instruction(is_constant ? spv::Op::OpConstant : spv::Op::OpVariable,
                        id, type_id, std::move(operands), insert_point_);
    insert_point_->AddInstruction(instruction);
    return id;
  }

 private:
  BasicBlock *insert_point_;
  Module *module_;
};
}  // namespace spriv
}  // namespace xla
#endif
