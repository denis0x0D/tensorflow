/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPIRV_IR_IR_BUILDER_MIXIN_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPIRV_IR_IR_BUILDER_MIXIN_H_

#include "spirv_ir_builder.h"
namespace xla {

// Mixin class that injects more ergonomic versions of spirv::IRBuilder methods
// into a class.  Intended to be used as a CRTP base class, like:
//
//  class MyIrEmitter : public IrBuilderMixin<MyIrEmitter> {
//    spirv::IRBuilder<>* builder() { return builder_; }
//
//    void EmitFoo(HloInstruction* foo) {
//      Add(Mul(...), FPToUI(...));
//    }
//  };

template <typename Derived>
class SPIRVIrBuilderMixin {
 protected:
  template <class... Args>
  spirv::Value* Add(Args&&... args) {
    return mixin_builder()->CreateAdd(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::LoadInst* AlignedLoad(Args&&... args) {
    return mixin_builder()->CreateAlignedLoad(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::StoreInst* AlignedStore(Args&&... args) {
    return mixin_builder()->CreateAlignedStore(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::AllocaInst* Alloca(Args&&... args) {
    return mixin_builder()->CreateAlloca(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* And(Args&&... args) {
    return mixin_builder()->CreateAnd(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* AtomicCmpXchg(Args&&... args) {
    return mixin_builder()->CreateAtomicCmpXchg(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* AtomicRMW(Args&&... args) {
    return mixin_builder()->CreateAtomicRMW(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* BitCast(Args&&... args) {
    return mixin_builder()->CreateBitCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* Br(Args&&... args) {
    return mixin_builder()->CreateBr(std::forward<Args>(args)...);
  }

  spirv::CallInst* Call(spirv::Value* callee,
                       spirv::ArrayRef<spirv::Value*> args = spirv::None,
                       const spirv::Twine& name = "",
                       spirv::MDNode* fp_math_tag = nullptr) {
    return mixin_builder()->CreateCall(callee, args, name, fp_math_tag);
  }

  template <class... Args>
  spirv::BranchInst* CondBr(Args&&... args) {
    return mixin_builder()->CreateCondBr(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* ConstInBoundsGEP1_32(Args&&... args) {
    return mixin_builder()->CreateConstInBoundsGEP1_32(
        std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FAdd(Args&&... args) {
    return mixin_builder()->CreateFAdd(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FMul(Args&&... args) {
    return mixin_builder()->CreateFMul(std::forward<Args>(args)...);
  }

  spirv::Value* GEP(spirv::Value* ptr, spirv::ArrayRef<spirv::Value*> idx_list,
                   const spirv::Twine& name = "") {
    return mixin_builder()->CreateGEP(ptr, idx_list, name);
  }

  template <class... Args>
  spirv::Value* ICmpEQ(Args&&... args) {
    return mixin_builder()->CreateICmpEQ(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* ICmpNE(Args&&... args) {
    return mixin_builder()->CreateICmpNE(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* ICmpULE(Args&&... args) {
    return mixin_builder()->CreateICmpULE(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* ICmpULT(Args&&... args) {
    return mixin_builder()->CreateICmpULT(std::forward<Args>(args)...);
  }

  spirv::Value* InBoundsGEP(spirv::Value* ptr,
                           spirv::ArrayRef<spirv::Value*> idx_list,
                           const spirv::Twine& name = "") {
    return mixin_builder()->CreateInBoundsGEP(ptr, idx_list, name);
  }

  spirv::Value* ExtractValue(spirv::Value* agg, spirv::ArrayRef<unsigned> idxs,
                            const spirv::Twine& name = "") {
    return mixin_builder()->CreateExtractValue(agg, idxs, name);
  }

  spirv::Value* InsertValue(spirv::Value* agg, spirv::Value* val,
                           spirv::ArrayRef<unsigned> idxs,
                           const spirv::Twine& name = "") {
    return mixin_builder()->CreateInsertValue(agg, val, idxs, name);
  }

  template <class... Args>
  spirv::Value* IntToPtr(Args&&... args) {
    return mixin_builder()->CreateIntToPtr(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::LoadInst* Load(Args&&... args) {
    return mixin_builder()->CreateLoad(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::CallInst* MemCpy(Args&&... args) {
    return mixin_builder()->CreateMemCpy(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* Mul(Args&&... args) {
    return mixin_builder()->CreateMul(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* NSWAdd(Args&&... args) {
    return mixin_builder()->CreateNSWAdd(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* NSWMul(Args&&... args) {
    return mixin_builder()->CreateNSWMul(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* NSWSub(Args&&... args) {
    return mixin_builder()->CreateNSWSub(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* Or(Args&&... args) {
    return mixin_builder()->CreateOr(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* PointerCast(Args&&... args) {
    return mixin_builder()->CreatePointerCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* PtrToInt(Args&&... args) {
    return mixin_builder()->CreatePtrToInt(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* SDiv(Args&&... args) {
    return mixin_builder()->CreateSDiv(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* Select(Args&&... args) {
    return mixin_builder()->CreateSelect(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* SRem(Args&&... args) {
    return mixin_builder()->CreateSRem(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::StoreInst* Store(Args&&... args) {
    return mixin_builder()->CreateStore(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* UDiv(Args&&... args) {
    return mixin_builder()->CreateUDiv(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* URem(Args&&... args) {
    return mixin_builder()->CreateURem(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* VectorSplat(Args&&... args) {
    return mixin_builder()->CreateVectorSplat(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* ZExtOrTrunc(Args&&... args) {
    return mixin_builder()->CreateZExtOrTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* AShr(Args&&... args) {
    return mixin_builder()->CreateAShr(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FCmpOEQ(Args&&... args) {
    return mixin_builder()->CreateFCmpOEQ(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FCmpOLT(Args&&... args) {
    return mixin_builder()->CreateFCmpOLT(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FCmpOLE(Args&&... args) {
    return mixin_builder()->CreateFCmpOLE(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FCmpONE(Args&&... args) {
    return mixin_builder()->CreateFCmpONE(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FCmpUNE(Args&&... args) {
    return mixin_builder()->CreateFCmpUNE(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FCmpUNO(Args&&... args) {
    return mixin_builder()->CreateFCmpUNO(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FDiv(Args&&... args) {
    return mixin_builder()->CreateFDiv(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FNeg(Args&&... args) {
    return mixin_builder()->CreateFNeg(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FPCast(Args&&... args) {
    return mixin_builder()->CreateFPCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FPToSI(Args&&... args) {
    return mixin_builder()->CreateFPToSI(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FPToUI(Args&&... args) {
    return mixin_builder()->CreateFPToUI(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FPTrunc(Args&&... args) {
    return mixin_builder()->CreateFPTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FRem(Args&&... args) {
    return mixin_builder()->CreateFRem(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* FSub(Args&&... args) {
    return mixin_builder()->CreateFSub(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* ICmpSGE(Args&&... args) {
    return mixin_builder()->CreateICmpSGE(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* ICmpSLT(Args&&... args) {
    return mixin_builder()->CreateICmpSLT(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* IntCast(Args&&... args) {
    return mixin_builder()->CreateIntCast(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* LShr(Args&&... args) {
    return mixin_builder()->CreateLShr(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* MemSet(Args&&... args) {
    return mixin_builder()->CreateMemSet(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* Neg(Args&&... args) {
    return mixin_builder()->CreateNeg(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* Not(Args&&... args) {
    return mixin_builder()->CreateNot(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::PHINode* PHI(Args&&... args) {
    return mixin_builder()->CreatePHI(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* RetVoid(Args&&... args) {
    return mixin_builder()->CreateRetVoid(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* SExtOrTrunc(Args&&... args) {
    return mixin_builder()->CreateSExtOrTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* Shl(Args&&... args) {
    return mixin_builder()->CreateShl(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* SIToFP(Args&&... args) {
    return mixin_builder()->CreateSIToFP(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* Sub(Args&&... args) {
    return mixin_builder()->CreateSub(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* Trunc(Args&&... args) {
    return mixin_builder()->CreateTrunc(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* UIToFP(Args&&... args) {
    return mixin_builder()->CreateUIToFP(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* Unreachable(Args&&... args) {
    return mixin_builder()->CreateUnreachable(std::forward<Args>(args)...);
  }

  template <class... Args>
  spirv::Value* Xor(Args&&... args) {
    return mixin_builder()->CreateXor(std::forward<Args>(args)...);
  }

 private:
  spirv::IRBuilder<>* mixin_builder() {
    return static_cast<Derived*>(this)->builder();
  }
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPIRV_IR_IR_BUILDER_MIXIN_H_
