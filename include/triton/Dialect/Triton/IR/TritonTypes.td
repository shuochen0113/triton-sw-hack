#ifndef TRITON_TYPES
#define TRITON_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"

//
// Types
//
class TritonTypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<Triton_Dialect, name, traits> {
    // Used by printer/parser
    let mnemonic = _mnemonic;
}

// Floating-point Type
def TT_Float : AnyTypeOf<[F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64], "floating-point">;
def TT_FloatTensor : RankedTensorOf<[TT_Float]>;
def TT_FloatLike : AnyTypeOf<[TT_Float, TT_FloatTensor]>;

// Boolean Type
// TT_Bool -> I1
def TT_BoolTensor : RankedTensorOf<[I1]>;
def TT_BoolLike : AnyTypeOf<[I1, TT_BoolTensor]>;

// Integer Type
def I4 : I<4>;
def TT_Int : AnyTypeOf<[I1, I4, I8, I16, I32, I64], "integer">;
def TT_IntTensor : RankedTensorOf<[TT_Int]>;
def TT_IntLike : AnyTypeOf<[TT_Int, TT_IntTensor]>;

// I32 Type
// TT_I32 -> I32
// TT_I32Tensor -> I32Tensor
def TT_I32Like : AnyTypeOf<[I32, I32Tensor]>;

// I64 Type
// TT_I64 -> I64
// TT_I64Tensor -> I64Tensor
def TT_I64Like : AnyTypeOf<[I64, I64Tensor]>;

// Pointer Type in TableGen
class TT_PtrOf<list<Type> pointeeTypes> :
    DialectType<Triton_Dialect,
                And<[CPred<"::mlir::isa<::mlir::triton::PointerType>($_self)">,
                     Concat<"[](::mlir::Type pointeeType) { return ",
                            SubstLeaves<"$_self", "pointeeType", AnyTypeOf<pointeeTypes>.predicate>,
                                        "; }(::mlir::cast<::mlir::triton::PointerType>($_self).getPointeeType())">]>,
                "ptr", "::mlir::triton::PointerType">;

// Pointer Type in C++ (corresponding to `TT_PtrOf`)
def TT_PtrType : TritonTypeDef<"Pointer", "ptr"> {
    let summary = "Pointer type (`::mlir::triton::PointerType`) in Triton IR type system";

    let description = [{
        Pointer type in Triton IR type system, which could be pointing to scalars or tensors.
    }];

    let parameters = (ins "Type":$pointeeType, "int":$addressSpace);

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "Type":$pointeeType,
            "int":$addressSpace
        ), [{
            return $_get(pointeeType.getContext(), pointeeType, addressSpace);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;

    let skipDefaultBuilders = 1;
}

// Scalar Pointer Type: `ptr<>`
def TT_Ptr : TT_PtrOf<[AnyType]>;

// Tensor of Pointer Type: `tensor<ptr<>>`
def TT_PtrTensor : RankedTensorOf<[TT_Ptr]>;

// Tensor of Pointer Type or Pointer type: `tensor<ptr<>>` or `ptr<>`
def TT_PtrLike : AnyTypeOf<[TT_Ptr, TT_PtrTensor]>;

// Tensor Type
def TT_FpIntTensor : RankedTensorOf<[TT_Float, TT_Int]>;
def TT_Tensor : RankedTensorOf<[TT_Float, TT_Int, TT_Ptr]>;

// Pointer Type to Tensor Type: `ptr<tensor<>>`
def TT_TensorPtr : TT_PtrOf<[TT_Tensor]>;

// Any Type in Triton IR
def TT_Type : AnyTypeOf<[TT_FloatLike, TT_IntLike, TT_PtrLike, TT_TensorPtr]>;

// ── Shared scratch-buffer handle (generalizable SMEM frontend API) ──────────
//
// [Context] shuochen 2025-09 generalizable follow-up to sw_kernel hack.
//
// An opaque handle representing a fixed-size scratch buffer in shared memory,
// allocated at JIT compile-time. Unlike TT_PtrType (which uses raw address
// arithmetic) this type is recognized by dedicated load/store ops and is
// lowered cleanly to ttg.local_alloc + ttg.local_{load,store}_slice without
// any pointer-backtracking heuristics.
//
// [Contract]
//   - `size` is the number of elements, known at compile time (constexpr).
//   - `elemType` is the element type (e.g. i32, f32).
//   - The handle is not a pointer and must not be used with tt.load/tt.store.
//   - Lifetime: function scope; each block gets its own private copy.
def TT_SharedBufType : TritonTypeDef<"SharedBuf", "shared_buf", []> {
  let summary = "Opaque handle to a per-block shared-memory scratch buffer";

  let description = [{
    Represents a fixed-size scratch buffer allocated in CUDA shared memory
    at JIT compile time. Used by tt.alloc_shared / tt.load_shared /
    tt.store_shared for algorithm-agnostic SMEM promotion.

    Unlike a raw pointer, this type is opaque to alias analysis, GVN, and
    CSE, preventing unintended transformations on SMEM accesses.
  }];

  let parameters = (ins "int64_t":$size, "Type":$elemType);

  let assemblyFormat = "`<` $size `,` $elemType `>`";
}

// Convenience constraint for TableGen op arguments.
def TT_SharedBuf : Type<CPred<"::mlir::isa<::mlir::triton::SharedBufType>($_self)">,
                         "shared_buf", "::mlir::triton::SharedBufType">;

// Result type of MakeTensorDescriptor
def TT_TensorDescType : TritonTypeDef<"TensorDesc", "tensordesc", []> {
  let summary = "Tensor descriptor type (`::mlir::triton::TensorDescType`) in Triton IR type system";

  let description = [{
      A portable abstraction for nvidia-TMA descriptors.
  }];

  let parameters = (ins "RankedTensorType":$blockType);
  let assemblyFormat = "`<` $blockType `>`";

  let builders = [
    TypeBuilder<(ins "RankedTensorType":$blockType, "bool":$isSigned), [{
      if (auto intTy = llvm::dyn_cast<IntegerType>(blockType.getElementType())) {
        auto sem = isSigned ? IntegerType::Signed : IntegerType::Unsigned;
        auto elemTy = IntegerType::get($_ctxt, intTy.getWidth(), sem);
        blockType = blockType.clone(elemTy);
      }
      return Base::get($_ctxt, blockType);
    }]>,
  ];
  let extraClassDeclaration = [{
    RankedTensorType getSignlessBlockType() const {
      auto resTy = getBlockType();
      if (auto intTy = llvm::dyn_cast<IntegerType>(resTy.getElementType())) {
        auto width = resTy.getElementTypeBitWidth();
        auto signlessTy = IntegerType::get(getContext(), width);
        resTy = resTy.clone(signlessTy);
      }
      return resTy;
    }
  }];
}

#endif
