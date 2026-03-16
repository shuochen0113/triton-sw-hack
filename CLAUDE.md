# CLAUDE.md — triton-sw-hack (Custom Triton Compiler Fork)

Guidance for Claude Code working in this compiler repository.

## What This Is

A fork of [triton-lang/triton](https://github.com/triton-lang/triton) that adds a
generalized shared-memory (SMEM) allocation API for use in the Triton-Seq
Smith-Waterman kernel project.

**Active branch:** `hack/smem-api-v2`
**Application repo:** [Triton-Seq](https://github.com/shuochen0113/Triton-Seq)

## Critical Rules

- **NEVER run `pip install -e .`** unless the user explicitly asks.
  Compilation takes 10-20 minutes.  The user always builds manually.
- **Use `-DTRITON_BUILD_TESTING=OFF`** if running cmake to avoid network fetch failures
  for googletest.
- This repo is used as a **git submodule** of Triton-Seq.  When pushing, push to
  `triton-sw-hack` first, then update the submodule pointer in `Triton-Seq`.

## What Was Added (V2 Generalized SMEM API)

Three new `tl.*` Python builtins + full compiler pipeline support:

```python
buf  = tl.allocate_shared(size: constexpr, dtype: constexpr)
data = tl.load_shared(buf, offsets, mask=None, other=None)
       tl.store_shared(buf, offsets, value, mask=None)
```

### Files Modified (relative to upstream Triton)

| File | What changed |
|------|-------------|
| `include/triton/Dialect/Triton/IR/TritonTypes.td` | `TT_SharedBufType` custom type |
| `include/triton/Dialect/Triton/IR/TritonOps.td` | `TT_AllocSharedOp`, `TT_LoadSharedOp`, `TT_StoreSharedOp`; `SharedMemory` resource |
| `include/triton/Dialect/Triton/IR/Dialect.h` | `struct SharedMemory` in `mlir::triton` namespace |
| `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` | `TTG_LocalLoadSliceOp`, `TTG_LocalStoreSliceOp` |
| `lib/Dialect/Triton/IR/Ops.cpp` | `AllocSharedOp::getEffects` (Allocate+Write) |
| `lib/Dialect/TritonGPU/IR/Ops.cpp` | Slice op effects + verifiers |
| `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` | `AllocSharedPattern`, `LoadSharedPattern`, `StoreSharedPattern` |
| `lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp` | `SharedBufType → MemDescType` type conversion |
| `lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp` | `LocalLoadSliceOpConversion`, `LocalStoreSliceOpConversion` |
| `lib/Analysis/Membar.cpp` | bypass generic membar insertion for explicit local slice ops |
| `lib/Dialect/TritonGPU/Transforms/MaterializeSWSmem.cpp` | V1 legacy auto-promotion pass (kept for reference) |
| `python/src/ir.cc` | `create_alloc_shared/load_shared/store_shared` pybind bindings |
| `python/triton/language/core.py` | `shared_buf_type`, `shared_buf`, `@builtin` functions |
| `python/triton/language/semantic.py` | `alloc_shared`, `load_shared`, `store_shared` |
| `python/triton/language/__init__.py` | Exports for all new symbols |

Full implementation details: `docs/smem-api/SMEM_GENERALIZED_API.md`

## Latest Known State (March 16, 2026)

### Measured on A6000

- **Upstream OPv6:** `303.13 ms`, `404.80 GCUPS`
- **Hack-v2 OPv9:** `147.49 ms`, `832.74 GCUPS`
- **Hack-v2 OPv9 PTX stats:** `ld_shared=14`, `st_shared=52`, `ld_global=8`, `st_global=3`, `bar_sync=7`

### Manual PTX target

From `Triton-Seq/experiments/ptx_modification/ptx/hacked_HEF.ptx`:

- `ld_shared=14`
- `st_shared=14`
- `ld_global=8`
- `st_global=3`
- `bar_sync=7`

### Interpretation

- The generalized SMEM API is working correctly end-to-end.
- The masked-store and membar fixes already worked:
  - `ld_shared` now matches the manual PTX target
  - `bar_sync` now matches the manual PTX target
- The remaining issue is **too many static `st.shared` sites**, mostly from
  shared-buffer initialization shape rather than the DP recurrence itself.

### Important: measured vs implemented

The latest A6000 benchmark artifacts were collected **before** the newest
init-compaction patches were rebuilt. These patches are now in tree but still
need a rebuild/rerun to verify:

- `lib/Dialect/TritonGPU/Transforms/MaterializeSWSmem.cpp`
- `Triton-Seq/src/kernel/experimental/local_dp_kernel_OPv9_smem.py`

## Known Gotchas

1. **`TT_AllocSharedOp` must NOT be `[Pure]`**
   If it is, MLIR CSE merges two same-size allocs (e.g. Esmem and Fsmem both
   `!tt.shared_buf<1536, i32>`) into one, causing aliasing and wrong kernel output.

2. **`shared_buf_type._unflatten_ir` must return `shared_buf`, not raw `ir.value`**
   Triton's for-loop JIT calls `_unflatten_ir` when cloning scope.  Returning a raw
   handle causes `'ir.value' has no attribute 'handle'` on first loop iteration.

3. **1D flat MemDesc to bypass power-of-2 constraint**
   `MemDescType::verify` checks all dims except the first are power-of-2.  Using shape
   `[N]` (1D) makes `drop_front(1)` vacuously empty — so STRIDE=768, 192, etc. work.

4. **`LoadSharedPattern` must use type converter for result type**
   Use `getTypeConverter()->convertType(origResultTy)` to get `BlockedEncodingAttr`
   on the TTGIR tensor.  Using the raw TTIR type causes downstream pass failures.

5. **`STRIDE ≥ BLOCK` constraint**
   Enforced by `LocalLoadSliceOp::verify`.  Configurations with STRIDE < BLOCK are
   correctly rejected.

6. **`ttg.local_store_slice` now carries an optional mask**
   Masked stores must lower to predicated `st.shared`, not to a read-modify-write
   sequence. Relevant files:
   - `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`
   - `lib/Dialect/TritonGPU/IR/Ops.cpp`
   - `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp`
   - `lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp`

7. **Explicit local slice ops bypass generic membar insertion**
   `lib/Analysis/Membar.cpp` now treats `ttg.local_load_slice` /
   `ttg.local_store_slice` as explicit shared-memory primitives. If barriers
   suddenly come back, inspect this file first.

8. **The remaining `st.shared` inflation is mostly init shape**
   If the next rerun still shows too many static stores, inspect whether the new
   strip-mined runtime fill loops were preserved or unrolled away. The next likely
   step would be a dedicated `tl.fill_shared` / TTIR op if source-level strip-mining
   is still not compact enough.

## Branch History

| Branch | Description |
|--------|-------------|
| `main` | Mirror of upstream Triton main |
| `hack/sw_kernel-v1` | Original V1 automatic MLIR passes (SeqAlignDetect + MaterializeSWSmem) |
| `hack/smem-api-v2` | V2 generalized SMEM API (based on `hack/sw_kernel-v1`) |
| `hack/smem-api-v2` | **Current** — V2 generalized SMEM API, verified on H100 |

## Build & Verify

```bash
# Build (user runs this manually)
pip install -r python/requirements.txt
TRITON_BUILD_TESTING=OFF pip install -e .

# Verify API available
python -c "import triton.language as tl; print(tl.allocate_shared)"

# Run OPv9 correctness test (from Triton-Seq root)
python benchmarks/scripts/test_opv9_correctness.py

# Run latest compiler experiment (from Triton-Seq root)
python benchmarks/scripts/experiment_triton_compiler.py \
  --compiler-label hack-v2 \
  --kernel opv9
```

## Git Workflow for This Submodule

```bash
# All git commands from workspace root:
GIT_DIR=/workspace/Triton-Seq/.git/modules/compiler/triton \
GIT_WORK_TREE=/workspace/triton-custom \
git <command>

# Push compiler changes first:
#   git push origin hack/smem-api-v2-rebased
# Then update submodule pointer in Triton-Seq:
#   cd /workspace/Triton-Seq && git add compiler/triton && git commit && git push
```
