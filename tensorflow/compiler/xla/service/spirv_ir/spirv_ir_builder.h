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


namespace xla {
namespace spriv {

class IRBuilder {
  IRBuilder();
  ~IRBuilder();
 protected:
  template <class... Args>
  spirv::Value* CreateAdd(Args&&... args);
  template <class... Args>
  spirv::LoadInst* CreateAlignedLoad(Args&&... args);
  template <class... Args>
  spirv::StoreInst* CreateAlignedStore(Args&&... args);
  template <class... Args>
  spirv::Value* CreateAnd(Args&&... args);
  template <class... Args>
  spirv::Value* CreateAtomicRMW(Args&&... args);
  template <class... Args>
  spirv::Value* CreateBitCast(Args&&... args);
  template <class... Args>
  spirv::Value* CreateBr(Args&&... args);
  spirv::CallInst* CreateCall(spirv::Value* callee,
                              spirv::ArrayRef<spirv::Value*> args = spirv::None,
                              const spirv::Twine& name = "",
                              spirv::MDNode* fp_math_tag = nullptr);
  template <class... Args>
  spirv::BranchInst* CreateCondBr(Args&&... args);
  template <class... Args>
  spirv::Value* CreateConstInBoundsGEP1_32(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFAdd(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFMul(Args&&... args);
  spirv::Value* CreateGEP(spirv::Value* ptr, spirv::ArrayRef<spirv::Value*> idx_list,
                    const spirv::Twine& name = "");
  template <class... Args>
  spirv::Value* CreateICmpEQ(Args&&... args);
  template <class... Args>
  spirv::Value* CreateICmpNE(Args&&... args);
  template <class... Args>
  spirv::Value* CreateICmpULE(Args&&... args);
  template <class... Args>
  spirv::Value* CreateICmpULT(Args&&... args);
  spirv::Value* CreateInBoundsGEP(spirv::Value* ptr,
                            spirv::ArrayRef<spirv::Value*> idx_list,
                            const spirv::Twine& name = "");
  spirv::Value* CreateExtractValue(spirv::Value* agg, spirv::ArrayRef<unsigned> idxs,
                             const spirv::Twine& name = "");
  spirv::Value* CreateInsertValue(spirv::Value* agg, spirv::Value* val,
                                  spirv::ArrayRef<unsigned> idxs,
                                  const spirv::Twine& name = "");
  template <class... Args>
  spirv::Value* CreateIntToPtr(Args&&... args);
  template <class... Args>
  spirv::LoadInst* CreateLoad(Args&&... args);
  template <class... Args>
  spirv::CallInst* CreateMemCpy(Args&&... args);
  template <class... Args>
  spirv::Value* CreateMul(Args&&... args);
  template <class... Args>
  spirv::Value* CreateNSWAdd(Args&&... args);
  template <class... Args>
  spirv::Value* CreateNSWMul(Args&&... args);
  template <class... Args>
  spirv::Value* CreateNSWSub(Args&&... args);
  template <class... Args>
  spirv::Value* CreateOr(Args&&... args);
  template <class... Args>
  spirv::Value* CreatePointerCast(Args&&... args);
  template <class... Args>
  spirv::Value* CreatePtrToInt(Args&&... args);
  template <class... Args>
  spirv::Value* SDiv(Args&&... args);
  template <class... Args>
  spirv::Value* CreateSelect(Args&&... args);
  template <class... Args>
  spirv::Value* CreateSRem(Args&&... args);
  template <class... Args>
  spirv::StoreInst* CreateStore(Args&&... args);
  template <class... Args>
  spirv::Value* CreateUDiv(Args&&... args);
  template <class... Args>
  spirv::Value* CreateURem(Args&&... args);
  template <class... Args>
  spirv::Value* CreateVectorSplat(Args&&... args);
  template <class... Args>
  spirv::Value* CreateZExtOrTrunc(Args&&... args);
  template <class... Args>
  spirv::Value* CreateaAShr(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFCmpOEQ(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFCmpOLT(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFCmpOLE(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFCmpONE(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFCmpUNE(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFCmpUNO(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFDiv(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFNeg(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFPCast(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFPToSI(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFPToUI(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFPTrunc(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFRem(Args&&... args);
  template <class... Args>
  spirv::Value* CreateFSub(Args&&... args);
  template <class... Args>
  spirv::Value* CreateICmpSGE(Args&&... args);
  template <class... Args>
  spirv::Value* CreateICmpSLT(Args&&... args);
  template <class... Args>
  spirv::Value* CreateIntCast(Args&&... args);
  template <class... Args>
  spirv::Value* CreateLShr(Args&&... args);
  template <class... Args>
  spirv::Value* CreateMemSet(Args&&... args);
  template <class... Args>
  spirv::Value* CreateNeg(Args&&... args);
  template <class... Args>
  spirv::Value* CreateNot(Args&&... args);
  template <class... Args>
  spirv::PHINode* CreatePHI(Args&&... args);
  template <class... Args>
  spirv::Value* RetVoid(Args&&... args);
  template <class... Args>
  spirv::Value* CreateSExtOrTrunc(Args&&... args);
  template <class... Args>
  spirv::Value* CreateShl(Args&&... args);
  template <class... Args>
  spirv::Value* CreateSIToFP(Args&&... args);
  template <class... Args>
  spirv::Value* CreateSub(Args&&... args);
  template <class... Args>
  spirv::Value* CreateTrunc(Args&&... args);
  template <class... Args>
  spirv::Value* CreateUIToFP(Args&&... args);
  template <class... Args>
  spirv::Value* CreateUnreachable(Args&&... args);
  template <class... Args>
  spirv::Value* CreateXor(Args&&... args);
};
}  // namespace spriv
}  // namespace xla
