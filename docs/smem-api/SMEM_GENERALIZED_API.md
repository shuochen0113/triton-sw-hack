# Generalized Shared-Memory API for Triton
## `tl.allocate_shared` / `tl.load_shared` / `tl.store_shared`

**Author:** Shuochen Chen
**Date:** 2025-09
**Branch:** `hack/smem-api-v2`
**Status:** ✅ Full-stack working — correctness verified on H100 (16,384 pairs)

---

## Overview

This document describes the generalized per-block shared-memory (SMEM) API added to the
`triton-sw-hack` compiler fork.  The API lets a Triton JIT kernel explicitly allocate,
read, and write per-block SMEM scratch buffers using three new language primitives:

```python
buf  = tl.allocate_shared(size: constexpr, dtype: constexpr) -> shared_buf
data = tl.load_shared(buf, offsets, mask=None, other=None)   -> tensor
       tl.store_shared(buf, offsets, value, mask=None)
```

This was built to support the Smith-Waterman / KSW2-ESTZ anti-diagonal wavefront DP kernel
(`sw_kernel_smem`, OPv9) where each GPU thread block needs its own H/E/F ring buffers in
shared memory, replacing the V1 global-memory ring-buffer arguments.

---

## Motivation

### V1 ("hack") Limitations

The original `hack/sw_kernel-v1` approach used three compiler passes
(`SeqAlignDetect` → `PromoteSeqAlignToShared` → `MaterializeSWSmem`) that **automatically**
promoted global-memory H/E/F ring-buffer accesses in `sw_kernel` to shared memory. This
worked but had fundamental limitations:

| Limitation | Detail |
|---|---|
| Hardcoded params | `STRIDE=768`, `BLOCK=256` burnt into C++ pass; wrong results for other band widths |
| Fragile pattern matching | Must match exact `addptr(splat(base), offset_tensor)` IR shape |
| Implicit API | No user-visible contract; compiler "magically" rewrites global accesses |
| No generality | Only works for one specific kernel function (`@sw_kernel`) |

### V2 Goals

- **Explicit API** that the user calls directly in Python/Triton code
- **Arbitrary params** (BAND, STRIDE, BLOCK) as long as STRIDE ≥ BLOCK
- **Full-stack correctness** from Python frontend → PTX `ld.shared`/`st.shared`
- **No pre-allocation** of global ring-buffer tensors at the host side

---

## Full-Stack Implementation

### Layer 1: Python Frontend (`python/triton/language/`)

**`core.py`** — Three new `@builtin` functions + two new types:

```python
@dataclass(frozen=True)
class shared_buf_type(base_type):
    size: int          # number of elements (compile-time)
    elem_type: dtype   # e.g. tl.int32

    def to_ir(self, builder):
        return builder.create_shared_buf_type(self.size, self.elem_type.to_ir(builder))

    def _unflatten_ir(self, handles, cursor):
        return shared_buf(handles[cursor], self), cursor + 1  # must return shared_buf, not raw handle

class shared_buf(base_value):
    def __init__(self, handle, buf_type: shared_buf_type): ...

@builtin
def allocate_shared(size: constexpr, dtype: constexpr, _semantic=None) -> shared_buf: ...

@builtin
def load_shared(buf, offsets, mask=None, other=None, _semantic=None) -> tensor: ...

@builtin
def store_shared(buf, offsets, value, mask=None, _semantic=None): ...
```

**`semantic.py`** — Delegates to builder C++ methods, returns `tl.shared_buf` (not `tl.tensor`):

```python
def alloc_shared(self, size, dtype):
    handle = self.builder.create_alloc_shared(size, elem_ir)
    return tl.shared_buf(handle, tl.shared_buf_type(size, dtype))   # NOT tl.tensor()

def load_shared(self, buf, offsets, mask, other):
    handle = self.builder.create_load_shared(buf.handle, offsets.handle, mask_h, other_h)
    result_ty = tl.block_type(buf.type.elem_type, offsets.type.get_block_shapes())
    return tl.tensor(handle, result_ty)
```

**Key bug fixed:** `_unflatten_ir` in `shared_buf_type` must return a `shared_buf` object —
not a raw `ir.value` — otherwise scope clones (for-loop bodies) degrade the handle.

---

### Layer 2: TTIR — Custom Dialect Ops (`Dialect/Triton/IR/`)

Three new ops in `TritonOps.td`:

```mlir
%buf  = tt.alloc_shared : !tt.shared_buf<2304, i32>
%data = tt.load_shared %buf[%offsets],
          mask = %mask : tensor<256xi1>,
          other = %other : i32
        : !tt.shared_buf<2304, i32>, tensor<256xi32> -> tensor<256xi32>
        tt.store_shared %buf[%offsets], %value
        : !tt.shared_buf<2304, i32>, tensor<256xi32>, tensor<256xi32>
```

**`TT_AllocSharedOp`**: Must NOT be `[Pure]` — marking it Pure allows CSE to merge two
same-size allocs (e.g. `Esmem` and `Fsmem`), causing aliasing bugs.  Uses
`DeclareOpInterfaceMethods<MemoryEffectsOpInterface>` with `Allocate + Write` effects.

**`TT_LoadSharedOp`**: Needs `AttrSizedOperandSegments` trait for optional `mask`/`other`
operands.  Assembly format requires explicit `type($mask)` and `type($other)` in optional
groups (otherwise MLIR can't infer types for non-buildable optional operands).

**`TT_SharedBufType`**: Custom opaque type `!tt.shared_buf<size, elemType>`.  Registered in
`TritonTypes.td`.

**`SharedMemory` resource**: Defined in `mlir::triton` namespace in `Dialect.h` (not
`mlir::triton::gpu`) to avoid layering violation when compiling `Triton/IR/Ops.cpp`.

---

### Layer 3: TTIR → TTGIR Conversion (`Conversion/TritonToTritonGPU/`)

Three new `OpConversionPattern`s in `TritonToTritonGPUPass.cpp`:

| Pattern | Input | Output |
|---|---|---|
| `AllocSharedPattern` | `tt.alloc_shared : !tt.shared_buf<N, T>` | `ttg.local_alloc : MemDescType<[N], T, LinearSharedEncoding>` |
| `LoadSharedPattern` | `tt.load_shared %buf[%off], mask=..., other=...` | `ttg.local_load_slice %memdesc, %off` + `arith.select` |
| `StoreSharedPattern` | `tt.store_shared %buf[%off], %val, mask=...` | `ttg.local_store_slice %val, %memdesc, %off` (+ RMW if masked) |

**Key details:**
- `AllocSharedPattern`: creates `MemDescType` with `LinearSharedEncodingAttr` (not swizzled).
  Shape is `[N]` (1D flat) — the `MemDescType::verify` only checks `drop_front(1)` is
  power-of-2; for 1D this is vacuously true, avoiding the `STRIDE=768` non-power-of-2 issue.
- `LoadSharedPattern`: uses `getTypeConverter()->convertType(origResultTy)` to produce a
  TTGIR tensor with `BlockedEncodingAttr`. Without this, tensors lack encoding and later
  passes fail verification.  Splats scalar `other` to tensor type before `arith.select`.

---

### Layer 4: TTGIR Ops (`Dialect/TritonGPU/IR/`)

Two new ops in `TritonGPUOps.td`:

```mlir
%result = ttg.local_load_slice %src, %offset
          : !tt.memdesc<[2304], i32, #ttg.linear_shared, #smem, mutable>,
            tensor<256xi32, #blocked> -> tensor<256xi32, #blocked>

ttg.local_store_slice %src, %dst, %offset
          : tensor<256xi32, #blocked>,
            !tt.memdesc<[2304], i32, #ttg.linear_shared, #smem, mutable>,
            tensor<256xi32, #blocked>
```

Verifier enforces:
- MemDesc must be in shared memory space, rank-1, mutable (for stores)
- `offset` and `result`/`src` shapes must match
- `MemDesc.shape[0] >= result.shape[0]` (STRIDE ≥ BLOCK)
- Element types must match between MemDesc and tensor

---

### Layer 5: LLVM Lowering (`Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp`)

`LocalLoadSliceOpConversion` and `LocalStoreSliceOpConversion` lower to PTX:

```
Per thread t:
  flat_addr = smem_base + offset[t]  (no wrap needed for flat addressing)
  ld.shared.b32 %reg, [flat_addr]    // for load
  st.shared.b32 [flat_addr], %reg    // for store
```

The `wrapBound = memDescTy.getShape()[0]` provides the modular bound for safe addressing.

---

## Usage (OPv9 Kernel)

```python
@triton.jit
def sw_kernel_smem(q_ptrs, r_ptrs, m_arr, n_arr, outs,
                   match_score, mismatch_score, gap_open_penalty, gap_extend_penalty,
                   drop_threshold,
                   SCORING_MODEL: tl.constexpr,
                   PRUNING_BAND: tl.constexpr, PRUNING_DROP: tl.constexpr,
                   IS_EXTENSION: tl.constexpr,
                   STRIDE: tl.constexpr, BAND: tl.constexpr, BLOCK: tl.constexpr):

    # Allocate per-block SMEM ring buffers (no global buffer args needed)
    Hsmem = tl.allocate_shared(3 * STRIDE, tl.int32)   # 3-slot H ring
    Esmem = tl.allocate_shared(2 * STRIDE, tl.int32)   # 2-slot E ring
    Fsmem = tl.allocate_shared(2 * STRIDE, tl.int32)   # 2-slot F ring

    # ... DP loop ...
    # Load: slot base + per-lane index
    slot_base = prev_slot_h   # = ((d-1) % 3) * STRIDE
    Hleft = tl.load_shared(Hsmem, slot_base + idx_prev,
                            mask=lane_mask & valid_prev, other=MINF)
    # Store:
    tl.store_shared(Hsmem, curr_slot_h + lane_off, H_new, mask=lane_mask)
```

**Constraint:** `STRIDE >= BLOCK` (enforced by `LocalLoadSliceOp` verifier).

---

## Verification Results

Tested on H100 80GB, CUDA 13.1, PyTorch 2.6.0+cu124.

### Correctness (16,384 pairs, standard dataset)
All `(score, best_i, best_j)` tuples match OPv6 (global memory baseline) exactly.

### Arbitrary BAND/STRIDE Robustness
| Config | OPv9 (SMEM) | OPv6 (global) | Match |
|--------|-------------|----------------|-------|
| BAND=127, STRIDE=128 | ✓ | ✓ | ✓ |
| BAND=251, STRIDE=256 | ✓ | ✓ | ✓ |
| BAND=501, STRIDE=512 | ✓ | ✓ | ✓ |
| BAND=751, STRIDE=768 | ✓ | ✓ | ✓ |
| BAND=1023, STRIDE=1024 | ✓ | ✓ | ✓ |
| BAND=51, STRIDE=64 | Rejected (STRIDE<BLOCK) | — | Expected |

### Performance (122.6 GCells, BAND=751)
| Kernel | Time | Throughput |
|--------|------|------------|
| OPv6 (global mem, L2-cached) | 97.4 ms | 1260 GCUPS |
| OPv9 (SMEM, tl.allocate_shared) | 99.6 ms | 1232 GCUPS |

Performance parity (~0.98×) is expected: at this sequence length and batch size, OPv6's
ring buffers (21 KB/block) stay resident in H100 L2 cache, so L2 and SMEM provide similar
effective bandwidth. OPv9 eliminates ~344 MB of pre-allocated global ring-buffer memory.

### PTX Verification
```
// OPv9 PTX (sw_kernel_smem.ptx)
.extern .shared .align 16 .b8 global_smem[];
st.shared.b32   [%r4], -10000000;      // SMEM init (H ring)
ld.shared.b32   %r133, [%r132];        // H ring read (main loop)
st.shared.b32   [%r266], %r272;        // H ring write (main loop)
ld.shared.b32   %r214, [%r213+9216];   // E ring read (offset = 2304*4 = Esmem base)
st.shared.b32   [%r251+15360], %r261;  // F ring write (offset = 3840*4 = Fsmem base)
```
Real `ld.shared`/`st.shared` confirmed: 20 loads, 52 stores (42 init + 10 main loop).

---

## Files Changed

### New Files
- `include/triton/Dialect/Triton/IR/TritonTypes.td` — `TT_SharedBufType`
- `docs/smem-api/SMEM_GENERALIZED_API.md` — this document
- `python/triton/language/core.py` — `shared_buf_type`, `shared_buf`, `allocate_shared`, `load_shared`, `store_shared`
- `python/triton/language/semantic.py` — `alloc_shared`, `load_shared`, `store_shared`

### Modified Files
| File | Change |
|------|--------|
| `include/triton/Dialect/Triton/IR/TritonOps.td` | `TT_AllocSharedOp`, `TT_LoadSharedOp`, `TT_StoreSharedOp`; `SharedMemory` resource |
| `include/triton/Dialect/Triton/IR/Dialect.h` | `struct SharedMemory` in `mlir::triton` namespace |
| `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` | `TTG_LocalLoadSliceOp`, `TTG_LocalStoreSliceOp` |
| `lib/Dialect/Triton/IR/Ops.cpp` | `AllocSharedOp::getEffects`, `LoadSharedOp::verify`, `StoreSharedOp::verify` |
| `lib/Dialect/TritonGPU/IR/Ops.cpp` | `LocalLoadSliceOp::getEffects/verify`, `LocalStoreSliceOp::getEffects/verify` |
| `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` | `AllocSharedPattern`, `LoadSharedPattern`, `StoreSharedPattern` |
| `lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp` | `LocalLoadSliceOpConversion`, `LocalStoreSliceOpConversion` |
| `python/src/ir.cc` | `create_alloc_shared`, `create_load_shared`, `create_store_shared` pybind bindings |
| `python/triton/language/__init__.py` | Export `shared_buf`, `shared_buf_type`, `allocate_shared`, `load_shared`, `store_shared` |

### V1 Hack (kept but secondary)
- `lib/Dialect/TritonGPU/Transforms/SeqAlignDetect.cpp` — unchanged
- `lib/Dialect/TritonGPU/Transforms/PromoteSeqAlignToShared.cpp` — unchanged
- `lib/Dialect/TritonGPU/Transforms/MaterializeSWSmem.cpp` — updated to 1D flat MemDesc
  (fixes `MemDescType::verify` power-of-2 constraint), but still hardcoded to STRIDE=768

---

## Known Limitations / Future Work

1. **V1 MaterializeSWSmem** remains hardcoded to `STRIDE=768, BLOCK=256`. It should either
   be removed or made parameter-aware by reading the kernel's constexpr attributes.

2. **`LocalStoreSliceOp` in MaterializeSWSmem** may not be firing correctly for OPv6
   (further debugging needed — global mem accesses still appear in OPv6 PTX).

3. **Performance** could be improved with vectorized `ld.shared.v4.b32` when offsets are
   128-bit aligned — currently generates scalar `ld.shared.b32` per thread.

4. **STRIDE < BLOCK** configurations are correctly rejected but could be supported by
   splitting the offset tensor into multiple iterations inside the lowering.
