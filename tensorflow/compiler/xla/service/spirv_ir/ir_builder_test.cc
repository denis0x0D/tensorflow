#include "ir_builder.h"

int main() {
  using namespace xla;
  using namespace spirv;
  IRPrinter *printer = new IRPrinter();
  Module *module = new Module("test");

  // Types.
  spv::Id int_32_t =
      module->CreateCustomType(spv::Op::OpTypeInt, 0, {"32", "1"});
  spv::Id uint_32_t =
      module->CreateCustomType(spv::Op::OpTypeInt, 0, {"32", "0"});
  spv::Id void_t = module->CreateCustomType(spv::Op::OpTypeVoid, 0);
  spv::Id bool_t = module->CreateCustomType(spv::Op::OpTypeBool, 0);
  spv::Id func_type = module->CreateCustomType(spv::Op::OpTypeFunction, void_t);

  // Variables and constants.
  spv::Id var1 = module->CreateGlobalVariable(int_32_t, true, {"40"});

  Function *function =
      module->GetOrCreateFunction("test_function", void_t, func_type, "None");

  BasicBlock *entry = new BasicBlock("entry");
  BasicBlock *next_block = new BasicBlock("next");
  BasicBlock *ret = new BasicBlock("ret");

  function->AddEntryBlock(entry);
  function->AddRetBlock(ret);

  IRBuilder *builder = new IRBuilder(entry, module);
  builder->SetInsertPoint(entry);
  builder->CreateBr(next_block);
  builder->SetInsertPoint(next_block);
  builder->CreateBr(ret);
  // entry -> next -> ret
  //
  module->Accept(printer);
  return 0;
}
