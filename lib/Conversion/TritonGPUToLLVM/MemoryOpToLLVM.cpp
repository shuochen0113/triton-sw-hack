#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

LogicalResult lowerLocalStore(Location loc, MLIRContext *ctx, Value regVal,
                              MemDescType memDescTy, SharedMemoryObject smemObj,
                              ArrayRef<Value> inVals,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              const TargetInfoBase &targetInfo) {
  auto regTy = cast<RankedTensorType>(regVal.getType());
  auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());

  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kOffset = str_attr("offset");
  auto regLayout = toLinearLayout(regTy);
  auto paddedLayout =
      dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(memDescTy.getEncoding());
  LinearLayout cvt = LinearLayout::empty();
  if (paddedLayout) {
    cvt = regLayout.reshapeOuts({{kOffset, regLayout.getTotalOutDimSize()}});
  } else {
    auto sharedLayout = toLinearLayout(memDescTy);
    cvt = regLayout.invertAndCompose(sharedLayout);
    auto kBlock = str_attr("block");
    // NYI. We would need to emit a map.shared::cluster instruction.
    if (!cvt.isTrivialOver({kBlock})) {
      return failure();
    }
  }
  cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
  lowerLocalLdSt(loc, ctx, cvt, inVals, llvmElemTy, memDescTy, smemObj,
                 rewriter, targetInfo);

  return success();
}

struct GlobalScratchAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::GlobalScratchAllocOp> {
  const TargetInfoBase *targetInfo;

  GlobalScratchAllocOpConversion(LLVMTypeConverter &converter,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::GlobalScratchAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto opOffsetAttr = op->getAttrOfType<mlir::IntegerAttr>(
        "ttg.global_scratch_memory_offset");
    assert(opOffsetAttr);
    auto opOffset = opOffsetAttr.getValue().getZExtValue();

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!funcOp) {
      return failure();
    }
    Value ptr = LLVM::getGlobalScratchPtr(loc, rewriter, *targetInfo, funcOp,
                                          b.i32_val(opOffset));

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc())
      return failure();
    Location loc = op->getLoc();
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto memDescTy = cast<MemDescType>(op.getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, memDescTy.getRank(),
                                      loc, rewriter);
    // If there is an initial tensor, store it into the shared memory.
    if (op.getSrc()) {
      auto *ctx = op.getContext();
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      if (failed(lowerLocalStore(loc, ctx, op.getSrc(), memDescTy, smemObj,
                                 inVals, typeConverter, rewriter,
                                 targetInfo))) {
        return failure();
      }
    }
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalDeallocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalDeallocOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalDeallocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalDeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct LocalLoadOpConversion : public ConvertOpToLLVMPattern<LocalLoadOp> {
public:
  LocalLoadOpConversion(LLVMTypeConverter &typeConverter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescVal = op.getSrc();
    auto regVal = op.getResult();
    auto memDescTy = cast<MemDescType>(memDescVal.getType());
    auto regTy = cast<RankedTensorType>(regVal.getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);

    auto sharedEnc =
        cast<triton::gpu::SharedEncodingTrait>(memDescTy.getEncoding());
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kOffset = str_attr("offset");
    auto regLayout = toLinearLayout(regTy);
    auto paddedLayout =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(sharedEnc);
    LinearLayout cvt = LinearLayout::empty();
    if (paddedLayout) {
      cvt = regLayout.reshapeOuts({{kOffset, regLayout.getTotalOutDimSize()}});
    } else {
      auto sharedLayout = toLinearLayout(memDescTy);
      cvt = regLayout.invertAndCompose(sharedLayout);
      auto kBlock = str_attr("block");
      // NYI. We would need to emit a map.shared::cluster instruction.
      if (!cvt.isTrivialOver({kBlock})) {
        return failure();
      }
    }
    cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});

    auto outVals = lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, memDescTy,
                                  smemObj, rewriter, targetInfo, op);

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, regTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalStoreOp>::ConvertOpToLLVMPattern;

  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    Value regVal = op.getSrc();
    Value memDescVal = op.getDst();
    auto typeConverter = getTypeConverter();
    auto memDescTy = cast<MemDescType>(memDescVal.getType());
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDst(),
                                                         llvmElemTy, rewriter);
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (failed(lowerLocalStore(loc, ctx, regVal, memDescTy, smemObj, inVals,
                               typeConverter, rewriter, targetInfo))) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

// ============================================================================
// Lowering Pattern for ttg.local_load_slice
// [Context] Shuochen’s hack for sw_kernel v1 — 2025/09/10
// [MODIFIED: 2025/09/10 - Correctness Fix]
//
// [Purpose]
//   Deterministic SMEM load lowering. This pattern consumes the preserved
//   (MemDesc, OffsetTensor) from the middle-end.
//
// [Contract]
//   - Matches only #ttg.linear_shared MemDesc in #triton_gpu.shared.
//
// [Invariants]
//   - To prevent illegal memory access, this pattern implements "address clamping".
//     An unconditional load is performed, but the address is selected: if the
//     logical offset is in-bounds, use it; otherwise, load from a safe
//     default address (e.g., offset 0). The correctness is guaranteed by a
//     higher-level `select` operation that discards this garbage data when the
//     original mask was false. This avoids crashes from out-of-bounds GEPs.
//
// [Lowering Steps]
//   1) Guard on #ttg.linear_shared encoding.
//   2) Extract SMEM base address.
//   3) Unpack the offset tensor into per-lane LLVM scalar values.
//   4) For each lane, check if `0 <= logical_offset < bound`.
//   5) Select `safe_offset = in_bounds ? logical_offset : 0`.
//   6) Emit GEP(base, safe_offset) + Load. This is always a safe access.
//   7) Repack loaded scalars.
// ============================================================================

struct LocalLoadSliceOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadSliceOp> {
public:
  LocalLoadSliceOpConversion(LLVMTypeConverter &typeConverter,
                             const TargetInfoBase &targetInfo,
                             PatternBenefit benefit = 10)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();

    // --- Get types from new Op signature ---
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    auto memDescTy = cast<triton::gpu::MemDescType>(op.getSrc().getType());
    auto offsetTy = cast<RankedTensorType>(op.getOffset().getType());

    // 1. Guard: This pattern only matches our custom linear encoding.
    auto linearSharedEnc = dyn_cast<triton::gpu::LinearSharedEncodingAttr>(memDescTy.getEncoding());
    if (!linearSharedEnc)
      return failure();
      
    // --- Get shared memory info ---
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), llvmElemTy, rewriter);
    Value smemBase = smemObj.getBase();
    int64_t wrapBound = memDescTy.getShape()[0];

    TritonLLVMOpBuilder b(loc, rewriter);
    SmallVector<Value> loadedVals;

    // --- Unpack the offset tensor to get per-thread logical offsets ---
    auto logicalOffsets = unpackLLElements(loc, adaptor.getOffset(), rewriter);
    
    // --- Create constants for boundary checks ---
    Value cZero = b.i32_val(0);
    Value cWrapBound = b.i32_val(wrapBound);

    // For each element this thread handles...
    for (unsigned i = 0; i < logicalOffsets.size(); ++i) {
      Value logicalOffset = logicalOffsets[i];
      
      // 2. [FIX] Implement address clamping to prevent illegal memory access.
      //    Create a predicate to check if the offset is within bounds [0, wrapBound).
      Value predGTEZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, logicalOffset, cZero);
      Value predLTBound = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, logicalOffset, cWrapBound);
      Value predInBounds = rewriter.create<arith::AndIOp>(loc, predGTEZero, predLTBound);

      //    If the offset is out of bounds, use a safe offset (0) for the load.
      //    The loaded value will be garbage, but it will be correctly discarded
      //    by the higher-level `select` op generated in the MaterializeSWSmem pass.
      Value safePhysicalOffset = rewriter.create<arith::SelectOp>(loc, predInBounds, logicalOffset, cZero);
      
      // 3. Generate GEP + Load using the GUARANTEED-TO-BE-SAFE physical offset.
      Value ptr = rewriter.create<LLVM::GEPOp>(
          loc, smemBase.getType(), llvmElemTy, smemBase, ValueRange{safePhysicalOffset});
      Value val = rewriter.create<LLVM::LoadOp>(loc, llvmElemTy, ptr);
      loadedVals.push_back(val);
    }

    Value result = packLLElements(loc, typeConverter, loadedVals, rewriter, resultTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

// ============================================================================
// Lowering Pattern for ttg.local_store_slice
// [Context] Shuochen’s hack for sw_kernel v1 — 2025/09/10
// [MODIFIED: 2025/09/10 - Correctness Fix]
//
// [Purpose]
//   Deterministic SMEM store lowering symmetrical to local_load_slice.
//
// [Invariants]
//   - Implements "address clamping" for the same safety reasons as the load
//     pattern. If an offset is out of bounds, the store is redirected to a safe
//     address (offset 0). Since masked stores are implemented as RMW,
//     this garbage store is harmless: the original value at offset 0 is read,
//     and then immediately written back, resulting in no net change. This
//     prevents crashes while preserving correctness.
//
// [Lowering Steps]
//   1) Guard on #ttg.linear_shared encoding.
//   2) Extract SMEM base address.
//   3) Unpack source values and offsets into per-lane scalars.
//   4) For each lane, check if `0 <= logical_offset < bound`.
//   5) Select `safe_offset = in_bounds ? logical_offset : 0`.
//   6) Emit GEP(base, safe_offset) + Store.
//   7) Erase the op.
// ============================================================================

struct LocalStoreSliceOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreSliceOp> {
public:
  LocalStoreSliceOpConversion(LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit = 10)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();

    // --- Get types from new Op signature ---
    auto srcDataTy = cast<RankedTensorType>(op.getSrc().getType());
    auto memDescTy = cast<triton::gpu::MemDescType>(op.getDst().getType());
    auto offsetTy = cast<RankedTensorType>(op.getOffset().getType());
    
    // 1. Guard: This pattern only matches our custom linear encoding.
    auto linearSharedEnc = dyn_cast<triton::gpu::LinearSharedEncodingAttr>(memDescTy.getEncoding());
    if (!linearSharedEnc)
      return failure();
      
    // --- Get shared memory info ---
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDst(), llvmElemTy, rewriter);
    Value smemBase = smemObj.getBase();
    int64_t wrapBound = memDescTy.getShape()[0];
    
    TritonLLVMOpBuilder b(loc, rewriter);
    
    // --- Unpack both the data tensor and the offset tensor ---
    auto valsToStore = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto logicalOffsets = unpackLLElements(loc, adaptor.getOffset(), rewriter);
    
    // --- Create constants for boundary checks ---
    Value cZero = b.i32_val(0);
    Value cWrapBound = b.i32_val(wrapBound);

    for (unsigned i = 0; i < valsToStore.size(); ++i) {
      Value logicalOffset = logicalOffsets[i];

      // 2. [FIX] Implement address clamping to prevent illegal memory access.
      Value predGTEZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, logicalOffset, cZero);
      Value predLTBound = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, logicalOffset, cWrapBound);
      Value predInBounds = rewriter.create<arith::AndIOp>(loc, predGTEZero, predLTBound);
      
      //    If the offset is out of bounds, use a safe offset (0).
      Value safePhysicalOffset = rewriter.create<arith::SelectOp>(loc, predInBounds, logicalOffset, cZero);

      // 3. Generate GEP + Store using the GUARANTEED-TO-BE-SAFE physical offset.
      Value ptr = rewriter.create<LLVM::GEPOp>(
          loc, smemBase.getType(), llvmElemTy, smemBase, ValueRange{safePhysicalOffset});
      rewriter.create<LLVM::StoreOp>(loc, valsToStore[i], ptr);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, targetInfo,
                                               benefit);
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalDeallocOpConversion>(typeConverter, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo, benefit);

  // [MODIFIED] Register our new patterns
  patterns.add<LocalLoadSliceOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalStoreSliceOpConversion>(typeConverter, targetInfo, benefit);
}
