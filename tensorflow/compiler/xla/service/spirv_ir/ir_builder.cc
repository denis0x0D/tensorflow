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
#include "ir_builder.h"

namespace xla {
namespace spirv {
// Simple singleton.
SPIRVContext *GetSPIRVContext() {
  static SPIRVContext *context = new SPIRVContext();
  return context;
}

Operand::Operand() : id_(0), is_id_(true) {}

Operand::Operand(spv::Id id, bool is_id) : id_(id), is_id_(is_id) {}

bool Operand::IsId() const { return is_id_; }

spv::Id Operand::GetId() const { return id_; }

spv::Id SPIRVContext::GetUniqueId() { return ++unique_id; }

std::vector<Operand> SPIRVContext::CreateOperandsFromLiterals(
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

std::string SPIRVContext::LookUpForLiteral(spv::Id literal_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  // TODO: insert assert
  if (literal_pool_.count(literal_id)) {
    return literal_pool_[literal_id];
  }
  return "0";
}

Instruction::Instruction(spv::Op op_code, spv::Id result_id, spv::Id type_id,
                         std::vector<Operand> operands, BasicBlock *parent)
    : op_code_(op_code),
      result_id_(result_id),
      type_id_(type_id),
      operands_(std::move(operands)),
      parent_(parent) {}

Instruction::Instruction(spv::Op op_code, spv::Id result_id, spv::Id type_id,
                         BasicBlock *parent)
    : op_code_(op_code),
      result_id_(result_id),
      type_id_(type_id),
      parent_(parent) {}

Instruction::~Instruction() {}

void Instruction::AddOperand(spv::Id operand_id, bool is_id) {
  operands_.push_back(Operand(operand_id, is_id));
}

spv::Op Instruction::GetOpCode() const { return op_code_; }

spv::Id Instruction::GetResultId() const { return result_id_; }

spv::Id Instruction::GetTypeId() const { return type_id_; }

std::vector<Operand> &Instruction::GetOperands() { return operands_; }

void Instruction::Accept(IRVisitorBase *visitor) { visitor->Visit(this); }

std::string Instruction::GetStringOpCode() {
  switch (op_code_) {
    case spv::Op::OpVariable:
      return "OpVariable";
    case spv::Op::OpConstant:
      return "OpConstant";
    case spv::Op::OpConstantComposite:
      return "OpConstantComposite";
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
    case spv::Op::OpTypeArray:
      return "OpTypeArray";
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

void IRVisitor::Visit(Instruction *instruction) {
  switch (instruction->GetOpCode()) {
    case spv::Op::OpVariable:
    case spv::Op::OpConstant:
    case spv::Op::OpConstantComposite:
      HandleVariableOp(instruction);
      break;
    case spv::Op::OpTypeVoid:
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeBool:
    case spv::Op::OpTypeVector:
    case spv::Op::OpTypeRuntimeArray:
    case spv::Op::OpTypeStruct:
    case spv::Op::OpTypeFunction:
    case spv::Op::OpTypeArray:
      HandleTypeOp(instruction);
      break;
    case spv::Op::OpTypePointer:
      HandlePointerTypeOp(instruction);
      break;
    case spv::Op::OpFunction:
      HandleFunctionOp(instruction);
      break;
    case spv::Op::OpLoad:
    case spv::Op::OpStore:
      HandleMemAccessOp(instruction);
      break;
    case spv::Op::OpBranch:
    case spv::Op::OpBranchConditional:
    case spv::Op::OpLabel:
    case spv::Op::OpPhi:
    case spv::Op::OpLoopMerge:
    case spv::Op::OpReturn:
    case spv::Op::OpFunctionEnd:
      HandleControlFlowOp(instruction);
      break;
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
      HandleAccessChainOp(instruction);
      break;
    case spv::Op::OpDecorate:
    case spv::Op::OpMemberDecorate:
      HandleDecorate(instruction);
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
      HandleBinOp(instruction);
      break;
    case spv::Op::OpCapability:
    case spv::Op::OpExtInstImport:
    case spv::Op::OpMemoryModel:
    case spv::Op::OpEntryPoint:
    case spv::Op::OpExecutionMode:
      HandleHeaderOp(instruction);
      break;
    default:
      break;
  }
}

IRPrinter::IRPrinter() {}

IRPrinter::~IRPrinter() {}

void IRPrinter::Visit(Instruction *instruction) {
  IRVisitor::Visit(instruction);
}
// Instruction processing is split based on instruction semantics.
// It could make sence, because some instructions have different text layout
// than binary form.
void IRPrinter::HandleHeaderOp(Instruction *instruction) {
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
    stream_ << tab_ << instruction->GetStringOpCode() << white_space_ << ident_
            << instruction->GetResultId() << white_space_;
    HandleOperands(instruction->GetOperands());
  } else if (instruction->GetOpCode() == spv::Op::OpExtInstImport) {
    HandleInstruction(instruction);
    assert(instruction->GetOperands().size() == 1 &&
           "Operand size for OpExtInstImport should be equal to 1 ");
    stream_ << quote_
            << GetSPIRVContext()->LookUpForLiteral(
                   instruction->GetOperands()[0].GetId())
            << quote_ << new_line_;
  } else {
    HandleInstruction(instruction);
    HandleOperands(instruction->GetOperands());
  }
}

void IRPrinter::HandleVariableOp(Instruction *instruction) {
  HandleInstruction(instruction);
  HandleOperands(instruction->GetOperands());
}

void IRPrinter::HandleTypeOp(Instruction *instruction) {
  HandleInstruction(instruction);
  HandleOperands(instruction->GetOperands());
}

// The layout is different agaist usual type op.
// %id = OpTypePointer literal %type_id
void IRPrinter::HandlePointerTypeOp(Instruction *instruction) {
  stream_ << ident_ << instruction->GetResultId() << white_space_ << assign_
          << white_space_ << instruction->GetStringOpCode() << white_space_;
  assert(instruction->GetOperands().size() == 1 &&
         "Operand size for OpTypePointer should be equal to 1");
  Operand op = instruction->GetOperands()[0];
  stream_ << GetSPIRVContext()->LookUpForLiteral(op.GetId()) << white_space_;
  stream_ << ident_ << instruction->GetTypeId() << white_space_;
  stream_ << new_line_;
}

void IRPrinter::HandleFunctionOp(Instruction *instruction) {
  stream_ << ident_ << instruction->GetResultId() << white_space_ << assign_
          << white_space_ << instruction->GetStringOpCode() << white_space_
          << ident_ << instruction->GetTypeId() << white_space_;
  HandleOperands(instruction->GetOperands());
}

void IRPrinter::HandleMemAccessOp(Instruction *instruction) {
  HandleInstruction(instruction);
  HandleOperands(instruction->GetOperands());
}

void IRPrinter::HandleControlFlowOp(Instruction *instruction) {
  HandleInstruction(instruction);
  HandleOperands(instruction->GetOperands());
}

void IRPrinter::HandleAccessChainOp(Instruction *instruction) {
  HandleInstruction(instruction);
  HandleOperands(instruction->GetOperands());
}

void IRPrinter::HandleBinOp(Instruction *instruction) {
  HandleInstruction(instruction);
  HandleOperands(instruction->GetOperands());
}

void IRPrinter::HandleDecorate(Instruction *instruction) {
  stream_ << tab_ << instruction->GetStringOpCode() << white_space_ << ident_
          << instruction->GetResultId() << white_space_;
  HandleOperands(instruction->GetOperands());
}

void IRPrinter::HandleInstruction(Instruction *instruction) {
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

void IRPrinter::HandleOperands(const std::vector<Operand> &operands) {
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

void IRPrinter::Dump() { std::cout << stream_.str(); }

void IRPrinter::AddMetaInfo() {
  // TODO: I did not find any official specification of SPIR-V
  // text format. This meta is the same as SPIR-V tools generate.
  stream_ << "; SPIR-V" << new_line_;
  stream_ << "; Version: 1.3" << new_line_;
  stream_ << "; Generator: Khronos; 0" << new_line_;
  stream_ << "; Bound: 0" << new_line_;
  stream_ << "; Schema: 0" << new_line_;
}

Module::Module(std::string module_name)
    : module_name_(std::move(module_name)) {}

Module::~Module() {
  // Frees header at first.
  for (auto *header_inst : header_) {
    delete header_inst;
  }
  for (auto *dec : decoration_table_) {
    delete dec;
  }
  for (auto *var : user_vars_types_table_) {
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

void Module::Accept(IRVisitor *visitor) {
  // Visit header.
  for (auto *header_inst : header_) {
    visitor->Visit(header_inst);
  }
  // Visit decoration table.
  for (auto *dec : decoration_table_) {
    visitor->Visit(dec);
  }
  // Visit variables and types.
  for (auto *var : user_vars_types_table_) {
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

void Module::InitHeader() {
  auto cap_operands = GetSPIRVContext()->CreateOperandsFromLiterals({"Shader"});
  Instruction *cap_instruction =
      new Instruction(spv::Op::OpCapability, 0, 0, std::move(cap_operands));
  header_.push_back(cap_instruction);
  spv::Id id = GetSPIRVContext()->GetUniqueId();
  auto ext_operands =
      GetSPIRVContext()->CreateOperandsFromLiterals({"GLSL.std.450"});
  Instruction *ext_import =
      new Instruction(spv::Op::OpExtInstImport, id, 0, std::move(ext_operands));
  header_.push_back(ext_import);
  auto mem_operands =
      GetSPIRVContext()->CreateOperandsFromLiterals({"Logical", "GLSL450"});
  Instruction *mem_model =
      new Instruction(spv::Op::OpMemoryModel, 0, 0, std::move(mem_operands));
  header_.push_back(mem_model);
}

void Module::CreateEntryPoint(Function *func, spv::Id work_group_id) {
  assert(func && "Function for entry point could not be null");
  std::vector<Operand> operands = GetSPIRVContext()->CreateOperandsFromLiterals(
      {"GLCompute", func->GetFunctionName()});
  operands.push_back(Operand(work_group_id));
  Instruction *global_entry = new Instruction(
      spv::Op::OpEntryPoint, func->GetEntryPoint()->GetResultId(), 0, operands);
  header_.push_back(global_entry);
}

void Module::CreateExecutionMode(Function *func,
                                 std::vector<std::string> local_sizes) {
  auto operands =
      GetSPIRVContext()->CreateOperandsFromLiterals(std::move(local_sizes));
  Instruction *exec_mode = new Instruction(spv::Op::OpExecutionMode,
                                           func->GetEntryPoint()->GetResultId(),
                                           0, std::move(operands));
  header_.push_back(exec_mode);
}

void Module::Decorate(spv::Id target_id, std::vector<std::string> literals) {
  std::vector<Operand> operands =
      GetSPIRVContext()->CreateOperandsFromLiterals(std::move(literals));
  Instruction *instruction =
      new Instruction(spv::Op::OpDecorate, target_id, 0, std::move(operands));
  decoration_table_.push_back(instruction);
}

void Module::MemberDecorate(spv::Id struct_type,
                            std::vector<std::string> literals) {
  std::vector<Operand> operands =
      GetSPIRVContext()->CreateOperandsFromLiterals(std::move(literals));
  Instruction *instruction = new Instruction(
      spv::Op::OpMemberDecorate, struct_type, 0, std::move(operands));
  decoration_table_.push_back(instruction);
}

spv::Id Module::CreateCustomType(spv::Op type_code, spv::Id type_id) {
  return CreateCustomType(type_code, type_id, {});
}

spv::Id Module::CreateCustomType(spv::Op type_code, spv::Id type_id,
                                 std::vector<std::string> literals) {
  spv::Id id = GetSPIRVContext()->GetUniqueId();
  std::vector<Operand> operands =
      GetSPIRVContext()->CreateOperandsFromLiterals(std::move(literals));
  Instruction *instruction =
      new Instruction(type_code, id, type_id, std::move(operands));
  user_vars_types_table_.push_back(instruction);
  return id;
}

spv::Id Module::CreateCustomTypeLen(spv::Op type_code, spv::Id type_id,
                                    std::vector<Operand> member_types) {
  spv::Id id = GetSPIRVContext()->GetUniqueId();
  Instruction *instruction =
      new Instruction(type_code, id, type_id, std::move(member_types));
  user_vars_types_table_.push_back(instruction);
  return id;
}

spv::Id Module::CreateGlobalVariable(spv::Id type_id, bool is_constant,
                                     std::vector<std::string> literals,
                                     spv::Id initializer) {
  spv::Id id = GetSPIRVContext()->GetUniqueId();
  std::vector<Operand> operands =
      GetSPIRVContext()->CreateOperandsFromLiterals(std::move(literals));
  if (initializer) operands.push_back(Operand(initializer));
  Instruction *instruction =
      new Instruction(!is_constant ? spv::Op::OpVariable : spv::Op::OpConstant,
                      id, type_id, std::move(operands));
  user_vars_types_table_.push_back(instruction);
  return id;
}

spv::Id Module::CreateConstantComposite(spv::Id type_id, std::vector<Operand> operands) {
  spv::Id id = GetSPIRVContext()->GetUniqueId();
  Instruction *instruction = new Instruction(spv::Op::OpConstantComposite, id,
                                             type_id, std::move(operands));
  user_vars_types_table_.push_back(instruction);
  return id;
}

Function *Module::GetOrCreateFunction(std::string name, spv::Id ret_type,
                                      spv::Id func_type,
                                      std::string function_control) {
  // Check the table at first.
  if (functions_.count(name)) {
    return functions_[name];
  }
  spv::Id id = GetSPIRVContext()->GetUniqueId();
  std::vector<Operand> operands = GetSPIRVContext()->CreateOperandsFromLiterals(
      {std::move(function_control)});
  operands.push_back(func_type);
  Function *function = new Function(std::move(name));
  Instruction *entry_point =
      new Instruction(spv::Op::OpFunction, id, ret_type, operands);
  function->SetEntryPoint(entry_point);
  functions_.insert({name, function});
  return function;
}

BasicBlock::BasicBlock(std::string name) : name_(std::move(name)) {
  // Each basic blocks starts from unique label.
  spv::Id id = GetSPIRVContext()->GetUniqueId();
  Instruction *instruction = new Instruction(spv::Op::OpLabel, id, 0);
  AddInstruction(instruction);
  label_id_ = id;
}

BasicBlock::~BasicBlock() {
  for (auto *it : instructions_) {
    delete it;
  }
}

Instruction *BasicBlock::GetInstructionById(spv::Id id) {
  for (auto *it : instructions_) {
    if (it->GetResultId() == id) {
      return it;
    }
  }
  return nullptr;
}

void BasicBlock::AddInstruction(Instruction *instruction) {
  instructions_.push_back(instruction);
}

void BasicBlock::AddPredeccessor(BasicBlock *predeccessor) {
  predeccessors_.push_back(predeccessor);
}

void BasicBlock::AddSuccessor(BasicBlock *successor) {
  successors_.push_back(successor);
}
spv::Id BasicBlock::Label() { return label_id_; }

std::vector<Instruction *> &BasicBlock::GetInstructions() {
  return instructions_;
}

Function::Function(std::string function_name)
    : function_name_(std::move(function_name)) {}
void Function::AddBasicBlock(BasicBlock *bb) { basic_blocks_.push_back(bb); }
void Function::AddEntryBlock(BasicBlock *bb) {
  entry_block_ = bb;
  basic_blocks_.push_back(bb);
}

BasicBlock *Function::GetEntryBlock() { return entry_block_; }

BasicBlock *Function::GetRetBlock() { return ret_block_; }

void Function::AddRetBlock(BasicBlock *bb) {
  Instruction *ret_instruction = new Instruction(spv::Op::OpReturn, 0, 0);
  Instruction *func_end_instruction =
      new Instruction(spv::Op::OpFunctionEnd, 0, 0);
  bb->AddInstruction(ret_instruction);
  bb->AddInstruction(func_end_instruction);
  ret_block_ = bb;
  basic_blocks_.push_back(bb);
}

void Function::SetEntryPoint(Instruction *entry_point) {
  entry_point_ = entry_point;
}

Instruction *Function::GetEntryPoint() { return entry_point_; }

std::vector<BasicBlock *> &Function::GetBasicBlocks() { return basic_blocks_; }

std::string Function::GetFunctionName() { return function_name_; }

IRBuilder::IRBuilder(BasicBlock *insert_point, Module *module)
    : insert_point_(insert_point), module_(module) {}

IRBuilder::~IRBuilder() {}

void IRBuilder::SetInsertPoint(BasicBlock *insert_point) {
  insert_point_ = insert_point;
}

// %id = OpLoad %type %ptr MemAccesstype
spv::Id IRBuilder::CreateLoad(spv::Id type, spv::Id ptr,
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
void IRBuilder::CreateStore(spv::Id ptr, spv::Id value,
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
void IRBuilder::CreateBr(BasicBlock *dest) {
  dest->AddPredeccessor(insert_point_);
  insert_point_->AddSuccessor(dest);
  Instruction *instruction =
      new Instruction(spv::Op::OpBranch, 0, 0, {dest->Label()}, insert_point_);
  insert_point_->AddInstruction(instruction);
}

// OpBranchConditional %condition %true_id %false_id
void IRBuilder::CreateCondBr(spv::Id condition, BasicBlock *true_block,
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
void IRBuilder::CreateLoopMerge(BasicBlock *merge_block,
                                BasicBlock *continue_block,
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
spv::Id IRBuilder::CreateAccessChain(spv::Op op_code, spv::Id ptr_type,
                                     spv::Id ptr,
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
spv::Id IRBuilder::CreateBinOp(spv::Op op_code, spv::Id type, spv::Id lhs,
                               spv::Id rhs) {
  spv::Id id = GetSPIRVContext()->GetUniqueId();
  Instruction *instruction =
      new Instruction(op_code, id, type, {lhs, rhs}, insert_point_);
  insert_point_->AddInstruction(instruction);
  return id;
}

// %id = OpPhi %type {%value %label}
spv::Id IRBuilder::CreatePhi(spv::Id type) {
  spv::Id id = GetSPIRVContext()->GetUniqueId();
  Instruction *instruction =
      new Instruction(spv::Op::OpPhi, id, type, {}, insert_point_);
  insert_point_->AddInstruction(instruction);
  return id;
}

void IRBuilder::AddIncoming(BasicBlock *phi_block, spv::Id phi_id, spv::Id id,
                            BasicBlock *bb) {
  if (!phi_block) return;
  Instruction *phi = phi_block->GetInstructionById(phi_id);
  if (!phi) return;
  phi->AddOperand(id);
  phi->AddOperand(bb->Label());
}

// %id = OpVariable %type {literals}
// %id = OpConstant %type {literals}
spv::Id IRBuilder::CreateLocalVariable(spv::Id type_id, bool is_constant,
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
}  // namespace spirv
}  // namespace xla
 
