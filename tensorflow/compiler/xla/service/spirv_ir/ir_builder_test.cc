#include "ir_builder.h"

int main() {
  using namespace xla;
  using namespace spirv;
  IRPrinter *printer = new IRPrinter();
  Module *module = new Module("test");

  spv::Id void_id = module->GetOrCreateCustomType("void");
  spv::Id int_id = module->GetOrCreateCustomType("int");

  Function *function = module->GetOrCreateFunction("test_function", "None");

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
