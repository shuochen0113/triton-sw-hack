//===----------------------------------------------------------------------===//
// MemoryOpToLLVM.cpp (TritonNVIDIAGPUToLLVM)
//
// [Context] Shuochen’s hack for sw_kernel v1 — 2025/09/10
//   No longer need custom hacking here
//   Since we have defined our custom Op `ttg.local_load_slice` / `ttg.local_store_slice`
//   They will be lowered to LLVM IR directly, carrying MemDesc + OffsetTensor
//   without any high-level lowering optimization for NVIDIA backend.
//===----------------------------------------------------------------------===//

#include "Dialect/NVGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
// #include "triton/Dialect/Triton/IR/Dialect.h" // for recognize tt.view
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"   // for recognize LLVM::ExtractValueOp

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::NVIDIA;
using namespace mlir::LLVM::NVIDIA;

// =================================================================
// Helper
// =================================================================
// static mlir::triton::gpu::LocalAllocOp findAllocFromMemDesc(mlir::Value memDesc) {
//     // Use a set to prevent infinite loops in case of cycles in the IR
//     llvm::SmallPtrSet<mlir::Value, 8> visited;

//     while (memDesc) {
//         if (!visited.insert(memDesc).second) {
//             return nullptr;
//         }

//         mlir::Operation* definingOp = memDesc.getDefiningOp();
//         if (!definingOp) {
//             return nullptr;
//         }

//         // case1: Success. We found the root: LocalAllocOp.
//         if (auto allocOp = llvm::dyn_cast<mlir::triton::gpu::LocalAllocOp>(definingOp)) {
//             return allocOp;
//         }
        
//         // case2: Trace back through MemDescIndexOp. the correct API is getSrc().
//         if (auto indexOp = llvm::dyn_cast<mlir::triton::gpu::MemDescIndexOp>(definingOp)) {
//             memDesc = indexOp.getSrc();
//             continue;
//         }
        
//         // case3: Trace back through LLVM struct passing of memdesc.
//         // In LLVM dialect, the correct Op is ExtractValueOp, source is getContainer().
//         if (auto extractOp = llvm::dyn_cast<mlir::LLVM::ExtractValueOp>(definingOp)) {
//             memDesc = extractOp.getContainer();
//             continue;
//         }

//         return nullptr;
//     }
//     return nullptr;
// }

LogicalResult lowerLdStMatrix(
    Location loc, const LinearLayout &regLayout, MemDescType memDescType,
    SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
    SharedMemoryObject smemObj, ConversionPatternRewriter &rewriter,
    const NVIDIA::TargetInfo &targetInfo,
    const LLVMTypeConverter *typeConverter) {
  bool isStore = !vals.empty();

  // Remove broadcasting from regLayout
  auto removeBroadcast = actionRemoveBroadcastedRegs(regLayout);
  if (!removeBroadcast.isIdentity()) {
    if (isStore) {
      auto newRegLayout = removeBroadcast.apply(regLayout);
      vals = removeBroadcast.apply(vals);
      return lowerLdStMatrix(loc, newRegLayout, memDescType, vals, smemObj,
                             rewriter, targetInfo, typeConverter);
    } else {
      auto newRegLayout = removeBroadcast.apply(regLayout);
      auto result =
          lowerLdStMatrix(loc, newRegLayout, memDescType, vals, smemObj,
                          rewriter, targetInfo, typeConverter);
      if (succeeded(result)) {
        vals = broadcastAs(vals, regLayout);
      }
      return result;
    }
  }
  if (isa<PaddedSharedEncodingAttr>(memDescType.getEncoding())) {
    return failure();
  }
  auto memLayout = toLinearLayout(memDescType);
  auto cvt = regLayout.invertAndCompose(memLayout);
  auto kBlock = StringAttr::get(loc.getContext(), "block");
  auto maybeSublayout = cvt.quotient({kBlock});
  if (!maybeSublayout) {
    return failure();
  }
  cvt = maybeSublayout.value();
  auto smemBase = smemObj.getBase();
  auto affineOffset = smemObj.getShmemOffset(loc, rewriter, memDescType);
  auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(memDescType);
  auto llvmElemTy = typeConverter->convertType(memDescType.getElementType());
  for (bool transpose : {false, true}) {
    auto result = LLVM::NVIDIA::lowerLdStMatrix(
        loc, cvt, transpose, vals, smemBase, affineOffset, maskSpanAffineOffset,
        llvmElemTy, rewriter, targetInfo);
    if (succeeded(result)) {
      return result;
    }
  }
  return failure();
}

struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  LocalLoadOpConversion(const LLVMTypeConverter &converter,
                        const NVIDIA::TargetInfo &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // =================================================================
    // FIX
    // =================================================================
    // chech if the source of this Load is from our specially marked smem alloc.
    // if so, this pattern actively declines to match (return failure),
    // forcing MLIR to fall back to using a lower-priority, generic, linear Load Pattern.
    // if (auto allocOp = findAllocFromMemDesc(op.getSrc())) {
    //     if (allocOp->hasAttr("sw.smem.linear_layout")) {
    //         return failure();
    //     }
    // }
    // =================================================================
    // FIX: END
    // =================================================================


    if (!op.getSrc())
      return failure();
    MemDescType memDescType = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Type llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getSrc(), llvmElemTy, rewriter);

    auto *typeConverter = getTypeConverter();
    llvm::SmallVector<Value> values;
    auto regLayout = toLinearLayout(dstTy);
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter()); // lower to 
    if (failed(result)) {
      return failure();
    }
    auto structTy = LLVM::LLVMStructType::getLiteral(
        op.getLoc().getContext(), SmallVector<Type>(values.size(), llvmElemTy));
    auto value =
        packLLElements(op.getLoc(), typeConverter, values, rewriter, structTy);
    rewriter.replaceOp(op, value);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const NVIDIA::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    MemDescType memDescType = op.getType();
    RankedTensorType regTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(regTy.getElementType());
    Value smemBase =
        LLVM::getSharedMemoryBase(op.getLoc(), rewriter, targetInfo, op);
    auto smemObj = SharedMemoryObject(
        smemBase, llvmElemTy, memDescType.getRank(), op.getLoc(), rewriter);

    auto regLayout = toLinearLayout(regTy);
    auto values = unpackLLElements(op.getLoc(), adaptor.getSrc(), rewriter);
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    if (failed(result)) {
      return failure();
    }

    auto retVal =
        getStructFromSharedMemoryObject(op.getLoc(), smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const NVIDIA::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // =================================================================
    // FIX (Symmetric for Store)
    // =================================================================
    // if (auto allocOp = findAllocFromMemDesc(op.getDst())) {
    //     if (allocOp->hasAttr("sw.smem.linear_layout")) {
    //         return failure();
    //     }
    // }
    // =================================================================
    // FIX (Symmetric for Store): END
    // =================================================================
    MemDescType memDescType = op.getDst().getType();
    RankedTensorType srcTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    SharedMemoryObject smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);

    auto regLayout = toLinearLayout(srcTy);
    auto values = unpackLLElements(op.getLoc(), adaptor.getSrc(), rewriter);
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    if (failed(result)) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};
} // namespace

void mlir::triton::NVIDIA::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Backend optimized memory ops get higher benefit
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo,
                                      benefit.getBenefit() + 1);
  mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, benefit);
}
