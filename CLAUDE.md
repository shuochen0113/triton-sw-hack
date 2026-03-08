# CLAUDE.md — triton-sw-hack (Custom Triton Compiler Fork)

Guidance for Claude Code working in this compiler repository.

## What This Is

A fork of [triton-lang/triton](https://github.com/triton-lang/triton) that adds a
generalized shared-memory (SMEM) allocation API for use in the Triton-Seq
Smith-Waterman kernel project.

**Active branch:** `hack/smem-api-v2` (verified working on H100, 2026-03)
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
| `lib/Dialect/TritonGPU/Transforms/MaterializeSWSmem.cpp` | V1 legacy auto-promotion pass (kept for reference) |
| `python/src/ir.cc` | `create_alloc_shared/load_shared/store_shared` pybind bindings |
| `python/triton/language/core.py` | `shared_buf_type`, `shared_buf`, `@builtin` functions |
| `python/triton/language/semantic.py` | `alloc_shared`, `load_shared`, `store_shared` |
| `python/triton/language/__init__.py` | Exports for all new symbols |

Full implementation details: `docs/smem-api/SMEM_GENERALIZED_API.md`

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
