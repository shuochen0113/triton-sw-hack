> **This is `triton-sw-hack` — a custom Triton compiler fork for GPU-accelerated Smith-Waterman alignment.**
> It adds a generalized shared-memory API (`tl.allocate_shared` / `tl.load_shared` / `tl.store_shared`) and a Smith-Waterman SMEM kernel to the upstream Triton compiler. See [Triton-Seq](https://github.com/shuochen0113/Triton-Seq) for the full application stack.

---

# triton-sw-hack — Custom Shared-Memory API for Smith-Waterman

**Branch:** `hack/smem-api-v2` (active; latest A6000 investigation: March 16, 2026)
**Author:** Shuochen
**Status:** Full-stack working. Shared-memory allocation/load/store is correct and
benchmarked; the remaining open issue is compacting initialization so the final
PTX store count gets closer to the manual hacked PTX.

## Design Goal

Standard Triton kernels can use `tl.load` / `tl.store` for global memory, and `ttg.local_alloc` for shared memory — but only through implicitly managed shared memory (e.g., via `triton.language.tensor` with swizzled encodings). There is no first-class Python-level API to:

- Explicitly allocate a per-block shared-memory scratch buffer of arbitrary size
- Load and store individual elements (or small vectors) from it using integer offsets
- Use non-power-of-2 strides (e.g., STRIDE=768 for a band-width-751 alignment kernel)

This fork adds exactly that: a **generalized SMEM allocation API** exposed through three new Triton builtins, lowered all the way to real `ld.shared` / `st.shared` PTX instructions.

The primary use-case is the Smith-Waterman / KSW2-ESTZ anti-diagonal wavefront DP kernel (OPv9), where each GPU thread block maintains H/E/F ring buffers in shared memory instead of pre-allocated global memory.

---

## Latest Status (A6000, March 16, 2026)

The newest compiler comparison runs in Triton-Seq show:

| Run | Time | Throughput | Static PTX profile |
|-----|------|------------|--------------------|
| Upstream OPv6 | `303.13 ms` | `404.80 GCUPS` | `ld_shared=4`, `st_shared=4`, `ld_global=18`, `st_global=9`, `bar_sync=6` |
| Hack-v2 OPv9 | `147.49 ms` | `832.74 GCUPS` | `ld_shared=14`, `st_shared=52`, `ld_global=8`, `st_global=3`, `bar_sync=7` |

This means:

- the V2 SMEM path is clearly working and materially faster on A6000
- the March 16 compiler fixes for **masked stores** and **membar insertion**
  already worked, because `ld_shared` and `bar_sync` now match the manual PTX target
- the remaining mismatch is `st_shared=52`, which analysis traced mostly to
  **initialization code shape**

The manual PTX target in `experiments/ptx_modification/ptx/hacked_HEF.ptx`
still has the best reference profile:

- `ld_shared=14`
- `st_shared=14`
- `ld_global=8`
- `st_global=3`
- `bar_sync=7`

Important: the newest init-compaction patches were added **after** the A6000
measurement above. Those patches still need a rebuild and rerun:

- Triton-Seq OPv9 source now uses strip-mined runtime fill loops for H/E/F init
- `MaterializeSWSmem.cpp` now uses the same runtime-fill shape for the V1 path

---

## What Was Added

### New Language Primitives (`tl.*`)

```python
buf  = tl.allocate_shared(size: constexpr, dtype: constexpr) -> tl.shared_buf
data = tl.load_shared(buf, offsets, mask=None, other=None)   -> tensor
       tl.store_shared(buf, offsets, value, mask=None)
```

**Usage in the OPv9 kernel:**

```python
@triton.jit
def sw_kernel_smem(..., STRIDE: tl.constexpr, BAND: tl.constexpr, BLOCK: tl.constexpr):
    # Per-block SMEM ring buffers — no global buffer args needed
    Hsmem = tl.allocate_shared(3 * STRIDE, tl.int32)   # 3-slot H ring
    Esmem = tl.allocate_shared(2 * STRIDE, tl.int32)   # 2-slot E ring
    Fsmem = tl.allocate_shared(2 * STRIDE, tl.int32)   # 2-slot F ring

    # Load from ring buffer:
    Hleft = tl.load_shared(Hsmem, slot_base + lane_off, mask=lane_mask, other=MINF)

    # Store to ring buffer:
    tl.store_shared(Hsmem, curr_slot + lane_off, H_new, mask=lane_mask)
```

### Full-Stack Implementation

| Layer | What was added |
|-------|---------------|
| **Python frontend** (`core.py`, `semantic.py`, `__init__.py`) | `shared_buf_type`, `shared_buf`, `@builtin allocate_shared/load_shared/store_shared` |
| **TTIR dialect** (`TritonOps.td`, `TritonTypes.td`, `Dialect.h`, `Ops.cpp`) | `!tt.shared_buf<N, T>` type; `tt.alloc_shared`, `tt.load_shared`, `tt.store_shared` ops with correct memory effects (non-Pure to prevent CSE aliasing) |
| **TTIR→TTGIR conversion** (`TritonToTritonGPUPass.cpp`, `TritonGPUConversion.cpp`) | `AllocSharedPattern` → `ttg.local_alloc` with 1D flat `LinearSharedEncodingAttr`; `LoadSharedPattern` → `ttg.local_load_slice` + `arith.select`; `StoreSharedPattern` → `ttg.local_store_slice` with optional mask |
| **TTGIR ops** (`TritonGPUOps.td`, `lib/Dialect/TritonGPU/IR/Ops.cpp`) | `ttg.local_load_slice`, `ttg.local_store_slice` — offset-indexed slice access into a flat 1D MemDesc |
| **LLVM lowering** (`MemoryOpToLLVM.cpp`) | `LocalLoadSliceOpConversion` / `LocalStoreSliceOpConversion` → `ld.shared.b32` / predicated `st.shared.b32` PTX |
| **C++ pybind** (`python/src/ir.cc`) | `create_alloc_shared`, `create_load_shared`, `create_store_shared` |

### Key Design Decisions

**1. Non-power-of-2 STRIDE support**
`MemDescType::verify` checks that all dims except the first are powers of 2. By using a 1D flat `[N]` shape for the MemDesc (rather than a 2D `[slots, stride]`), the `drop_front(1)` check is vacuously empty — so STRIDE=768, 192, etc. all work.

**2. Non-Pure `TT_AllocSharedOp`**
If `TT_AllocSharedOp` were marked `[Pure]`, MLIR’s CSE pass would merge two same-size allocations (e.g. `Esmem` and `Fsmem` both `!tt.shared_buf<1536, i32>`) into one, causing aliasing and incorrect results. The op is instead declared with `Allocate + Write` memory effects so CSE treats each call as distinct.

**3. `shared_buf_type._unflatten_ir` must return `shared_buf`**
Triton’s JIT for-loop compiler clones loop-body scopes by calling `_unflatten_ir` on every live value’s type. If `_unflatten_ir` returns a raw `ir.value` instead of a `shared_buf` wrapper, the cloned handle loses its `.handle` attribute and causes a `’ir.value’ has no attribute ‘handle’` error on the first loop iteration.

**4. BlockedEncodingAttr on load result**
`LoadSharedPattern` must call `getTypeConverter()->convertType(origResultTy)` to obtain the TTGIR tensor type with `BlockedEncodingAttr`. Using the raw TTIR type (no encoding) causes downstream pass verification failures.

**5. Masked shared stores must remain stores**
`StoreSharedPattern` and `MaterializeSWSmem` originally lowered masked shared
stores through a read-modify-write sequence. `ttg.local_store_slice` now carries
an optional mask and lowers directly to predicated `st.shared`, removing the old
`ld.shared + selp + st.shared` shape.

**6. Explicit shared slice ops must bypass generic membar**
`ttg.local_load_slice` / `ttg.local_store_slice` represent explicit shared-memory
 scheduling. Generic whole-buffer membar insertion was adding many unnecessary
 `bar.sync` instructions; `lib/Analysis/Membar.cpp` now skips that automatic path
 for these ops.

### V1 Hack (Legacy Path)

An earlier, V1 approach uses three automatic MLIR passes (`SeqAlignDetect` → `PromoteSeqAlignToShared` → `MaterializeSWSmem`) to auto-promote global H/E/F ring-buffer accesses in the original `sw_kernel` to shared memory. `MaterializeSWSmem.cpp` is kept in this fork for reference. **It is hardcoded to STRIDE=768, BLOCK=256** and only handles `@sw_kernel` specifically — the V2 generalized API (above) supersedes it. The original V1 design doc is preserved below.

---

## Verification Results (H100 80GB, CUDA 13.1)

### Correctness
All `(score, best_i, best_j)` tuples for 16,384 real sequence pairs match the global-memory OPv6 baseline exactly.

### Arbitrary BAND/STRIDE Robustness
| Config | OPv9 SMEM | OPv6 Global | Match |
|--------|-----------|-------------|-------|
| BAND=127, STRIDE=128 | ✓ | ✓ | ✓ |
| BAND=251, STRIDE=256 | ✓ | ✓ | ✓ |
| BAND=501, STRIDE=512 | ✓ | ✓ | ✓ |
| BAND=751, STRIDE=768 | ✓ | ✓ | ✓ |
| BAND=1023, STRIDE=1024 | ✓ | ✓ | ✓ |
| BAND=51, STRIDE=64 | Rejected (STRIDE < BLOCK=256) | — | Expected |

**Constraint:** `STRIDE ≥ BLOCK` (enforced by `LocalLoadSliceOp` verifier).

### Performance (122.6 GCells, BAND=751)
| Kernel | Time | Throughput |
|--------|------|------------|
| OPv6 (global mem, L2-cached) | 97.4 ms | 1260 GCUPS |
| OPv9 (SMEM, tl.allocate_shared) | 99.6 ms | 1232 GCUPS |

Performance parity (~0.98×) is expected: at this sequence length and batch size, OPv6’s ring buffers (21 KB/block) stay resident in H100 L2 cache, providing similar effective bandwidth to SMEM. **OPv9 eliminates ~344 MB of pre-allocated global ring-buffer memory.**

### PTX Evidence
```
.extern .shared .align 16 .b8 global_smem[];
ld.shared.b32   %r133, [%r132];          // H ring read (main loop)
st.shared.b32   [%r266], %r272;          // H ring write (main loop)
ld.shared.b32   %r214, [%r213+9216];     // E ring read (base = 2304*4 bytes)
st.shared.b32   [%r251+15360], %r261;    // F ring write (base = 3840*4 bytes)
```
Real `ld.shared`/`st.shared` are confirmed. In the latest A6000 benchmark the
static profile is `14` shared loads and `52` shared stores. The remaining store
excess is mostly initialization, not the core DP recurrence.

---

## Modified Files (relative to upstream Triton)

| File | Change |
|------|--------|
| `include/triton/Dialect/Triton/IR/TritonTypes.td` | `TT_SharedBufType` |
| `include/triton/Dialect/Triton/IR/TritonOps.td` | `TT_AllocSharedOp`, `TT_LoadSharedOp`, `TT_StoreSharedOp`; `SharedMemory` resource |
| `include/triton/Dialect/Triton/IR/Dialect.h` | `struct SharedMemory` in `mlir::triton` namespace |
| `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` | `TTG_LocalLoadSliceOp`, `TTG_LocalStoreSliceOp` |
| `lib/Dialect/Triton/IR/Ops.cpp` | `AllocSharedOp::getEffects`, verifiers |
| `lib/Dialect/TritonGPU/IR/Ops.cpp` | Slice op verifiers and effects |
| `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` | `AllocSharedPattern`, `LoadSharedPattern`, `StoreSharedPattern` |
| `lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp` | `SharedBufType → MemDescType` type conversion |
| `lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp` | `LocalLoadSliceOpConversion`, `LocalStoreSliceOpConversion` |
| `lib/Analysis/Membar.cpp` | skip generic membar insertion for explicit local slice ops |
| `lib/Dialect/TritonGPU/Transforms/MaterializeSWSmem.cpp` | V1 hack (legacy, kept for reference) |
| `python/src/ir.cc` | `create_alloc_shared`, `create_load_shared`, `create_store_shared` pybind |
| `python/triton/language/core.py` | `shared_buf_type`, `shared_buf`, builtins |
| `python/triton/language/semantic.py` | `alloc_shared`, `load_shared`, `store_shared` |
| `python/triton/language/__init__.py` | Exports |

Full implementation details: [`docs/smem-api/SMEM_GENERALIZED_API.md`](docs/smem-api/SMEM_GENERALIZED_API.md)

---

# V1 Design Doc (Legacy Reference)

The section below is preserved as historical context from the original V1 work.
Some details there are now outdated for the current branch, notably:

- masked shared stores no longer use IR-level read-modify-write
- explicit local slice ops now bypass generic membar insertion
- current initialization work is moving toward compact runtime fill loops

# SW Kernel - Compiler Extensions for Triton (Shared Memory Materialization)
**Shuochen’s hack for `sw_kernel` v1 - 2025/09/10**

---

## Why a Bypass Is Necessary

- Triton’s **swizzled shared encodings** introduce XOR/non-linear address transforms -> **breaks SW’s linear pointer arithmetic** and produced numerical errors in early attempts.
- **Blocked encodings** require tiling divisibility that our ring buffer shapes (e.g., `[3, 768]`) do not satisfy.
- Conclusion: a **fully controlled, linear SMEM path** is required.

---

## System Overview

> **Pipeline:** `SeqAlignDetect` -> `PromoteSeqAlignToShared` -> `MaterializeSWSmem` -> *(standard codegen)*  
> **Core contract:** The middle end must carry **both**:  
> (1) a **MemDesc** in shared memory with **linear** encoding, and  
> (2) the **per-thread offset tensor** (unchanged)  
> to the final lowering site.

### 1) `ttg.LinearSharedEncodingAttr` (Attribute)
- **[Purpose]** Passive, *inert* identity tag for MemDesc in `#triton_gpu.shared` to **isolate** it from generic swizzling/padding.
- **[Design]** Holds only `order` (fastest->slowest) and `CTALayout`. **No** swizzle/XOR/padding semantics.
- **[Contract]** Applied to MemDesc in shared space; lowering must match this encoding explicitly.
- **[Diagnostics]** Printed form:  
  `#triton_gpu.linear_shared<{order=[...], CTALayout=...}>`.

### 2) `SeqAlignDetect` & `PromoteSeqAlignToShared` (Pre-materialization passes)
- **[Purpose]** Detect tt.load/store traffic to H/E/F in `@sw_kernel` and insert **anchors**.
- **[Design]**
  - `SeqAlignDetect`: targeted backtracking from `tt.load/store` pointers to function args; tags ops with `seq.buffer = "H"|"E"|"F"`.
  - `PromoteSeqAlignToShared`: wraps each site with identity `ttg.convert_layout` carrying `smem.anchor = "H"|"E"|"F"`.
- **[Invariants]** No layout change yet; anchors serve as control points.
- **[Diagnostics]** Uniform stderr prefixes:  
  `[SeqAlignDetect][…]`, `[PromoteSeqAlignToShared][…]`.

### 3) `MaterializeSWSmem` (Traffic materialization; **critical pass**)
- **[Purpose]** Allocate H/E/F in shared memory with **linear** encoding and **rewrite** anchored loads/stores into `ttg.local_*_slice`.
- **[Design]**
  - Allocates **2D MemDesc** per element type: `[SLOTS, STRIDE]` with `#ttg.linear_shared`, then takes 1D slot views via `ttg.memdesc.index`.
  - **Initialization (i32 path):** vectorized fill with `-INF` via `LocalStoreSliceOp`, then `gpu.barrier`.
  - **Pattern match:** expects `tt.addptr(tt.splat(base_scalar_ptr), offset_tensor)`.
  - **Slot math:** derive `slotIdx = (slotOffset / STRIDE) % SLOTS` from the **base pointer’s addptr offset** (no guessing).
  - **Information forwarding:** pass the **MemDesc** and the **unaltered `offset_tensor`** to the op.
  - **Masked store:** RMW at IR level (`local_load_slice` -> `arith.select` -> `local_store_slice`) for correctness.
- **[Contract]** Fails fast if the pointer pattern is missing; emits diagnostics.
- **[Diagnostics]** Uniform stderr prefix `[MaterializeSWSmem][INFO/WARN/OK/FAIL/DEBUG]`.

### 4) `ttg.local_load_slice` / `ttg.local_store_slice` (Ops)
- **[Purpose]** The **information safes** that carry *(MemDesc, offset_tensor)* across the middle end **without mutation**.
- **[Signature]**
  - `local_load_slice(%smem: MemDesc, %offset: tensor) -> tensor`
  - `local_store_slice(%src: tensor, %smem: MemDesc, %offset: tensor)`
- **[Contract]**
  - MemDesc: 1D or slot-indexed view in **shared** space with **linear_shared** encoding.
  - `src/result` and `offset` are **rank-1** tensors with **identical shapes**.
  - Element types between MemDesc and data tensor **match**.
- **[Verify]** Enforces space/rank/shape/etype consistency; prevents silent drift.

### 5) Lowering: `Local*SliceOpConversion`
- **[Purpose]** The **only** lowering path for the above ops when MemDesc is `#ttg.linear_shared`.
- **[Design]**
  - Unpacks `offset_tensor` into per-lane scalars.
  - Computes `phys = offset & (WRAP - 1)` (assuming POT stride); `gep + ld/st` from SMEM base.
  - Re-packs results for loads.
- **[Invariants]** No linear-tid reconstruction. No hidden swizzling.  
  *(Future)* Provide a `urem` fallback if stride is not power-of-two.

---

## Key Contracts & Invariants

- **MemDesc isolation.** Any MemDesc with `#ttg.linear_shared` must only be handled by our path.
- **Lossless information flow.** The middle end must not alter `offset_tensor`.
- **Ring buffer slot rule.** `slotIdx = (slotOffset / STRIDE) % SLOTS`, with
  - `H : SLOTS=3`, `E : 2`, `F : 2`
  - `STRIDE = 768`, `BLOCK = 256`, `SEGS = STRIDE/BLOCK`.
- **Lowering wrap.** `phys = offset & (BLOCK-1)` (POT assumption); otherwise consider `urem`.
- **Masked store correctness.** RMW sequence at IR is the source of truth.

---

## Minimal IR Sketch (Before/After)

**Before (generic):**
```mlir
%p    = tt.splat %P : (ptr) -> tensor<… x ptr>
%ptr  = tt.addptr %p, %O
%val  = tt.load %ptr : tensor<… x etype>
tt.store %X, %ptr : tensor<… x etype>
```

**After (materialized to SMEM):**
```mlir
%sm   = ttg.local_alloc : memdesc<[SLOTS, STRIDE], etype, #ttg.linear_shared, #triton_gpu.shared>
%slot = ttg.memdesc.index %sm[%slotIdx] : memdesc<[STRIDE], etype, #ttg.linear_shared, #triton_gpu.shared>

%r    = ttg.local_load_slice %slot, %O : (memdesc<[STRIDE], etype>, tensor<… x i32>) -> tensor<… x etype>
ttg.local_store_slice %X, %slot, %O
```

---

## Diagnostics & Logging

- **Stderr prefixes (uniform):**
  - `[SeqAlignDetect][INFO/WARN/OK/FAIL] …`
  - `[PromoteSeqAlignToShared][INFO/WARN/OK/FAIL] …`
  - `[MaterializeSWSmem][INFO/WARN/OK/FAIL/DEBUG] …`
- **Typical failure:**  
  `MaterializeSWSmem: requires addptr(splat(base), offset_tensor) pattern`  
  -> indicates non-matching pointer shape; inspect producer chain.

---

## Repro (Pipeline & IR Checks)

1. **Run passes in order:**  
   `SeqAlignDetect` -> `PromoteSeqAlignToShared` -> `MaterializeSWSmem`.
2. **Inspect IR after each phase:**
   - After **Detect**: loads/stores tagged with `seq.buffer = "H"|"E"|"F"`.
   - After **Promote**: `ttg.convert_layout` with `smem.anchor = "H"|"E"|"F"`.
   - After **Materialize**:  
     - 2D shared allocs with `#ttg.linear_shared`.  
     - Slot-indexed 1D MemDesc via `ttg.memdesc.index`.  
     - `ttg.local_load_slice/store_slice` carrying *(MemDesc, offset_tensor)*.
3. **Module attributes used in init path (optional):**  
   `ttg.num-warps`, `ttg.threads-per-warp` (for vectorized fill).

> *Note:* The pass names/creators are available as `createSeqAlignDetectPass`, `createPromoteSeqAlignToShared`, `createMaterializeSWSmem`.

---

## Results

| Path                                 | Runtime (ms) |
|--------------------------------------|--------------|
| Triton baseline (global memory)      | ~90          |
| **This work (linear SMEM, compiler)**| **~75**      |
| Hand-tuned PTX (“gold” ceiling)      | ~55          |

**Gain vs. baseline:** ~17% so far; headroom remains to reach the gold ceiling.

---

## Appendix

- **Constants & Shapes:**  
  `STRIDE=768`, `BLOCK=256`, `SEGS=STRIDE/BLOCK`, `H:3 slots`, `E:2`, `F:2`.
- **Attributes & Tags:**  
  `seq.buffer`, `smem.anchor`, `#ttg.linear_shared`.
- **Invariants Recap:**  
  - No transformation may alter `offset_tensor`.  
  - MemDesc with `#ttg.linear_shared` must not be swizzled/padded.  
  - Slot math derives strictly from base pointer offset.  
  - Lowering performs `gep + ld/st` only.
