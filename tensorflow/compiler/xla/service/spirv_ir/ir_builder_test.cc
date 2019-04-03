#include "ir_builder.h"

int main() {
  using namespace xla;
  using namespace spirv;
  IRPrinter printer;
  Function *function = new Function("shader");
  BasicBlock *entry = new BasicBlock("entry", function,
                                     GetSPIRVContext());  // new BasicBlock();
  IRBuilder builder(entry, GetSPIRVContext());
  spv::Id some_type = 21;
  spv::Id var =
      builder.CreateVariable(some_type, spv::StorageClass::StorageClassUniform);
  return 0;
}
