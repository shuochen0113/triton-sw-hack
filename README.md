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
