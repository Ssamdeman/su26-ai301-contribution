# Contribution #2: Feature Request: Implement missing ops from backends

**Contribution Number:** 2
**Student:** Samuel Damon
**Issue:** [ggml-org/llama.cpp#14909](https://github.com/ggml-org/llama.cpp/issues/14909)
**Pull Request:** [ggml-org/llama.cpp#26642](https://github.com/ggml-org/llama.cpp/pull/26642)
**Status:** Phase IV In Progress — PR submitted and open; awaiting review (first maintainer comment received, response in progress)

> **Note on the issue number:** #14909 is an *umbrella* issue ("Implement missing ops from backends") — each contributor picks one op per PR. This is the same issue as my Contribution #1; what differs is the **op** (Contribution #1 was `COL2IM_1D`, this is `REPEAT`), not the issue.

## Why I Chose This Issue

I stayed in the same umbrella issue (#14909) for my second contribution, which CodePath recommends — the workflow is proven, my build environment is healthy, and staying in one project deepens my understanding of it rather than restarting cold in a new codebase.

For the op itself I deliberately picked a **bounded, data-movement-only** gap this time: extending CUDA `GGML_OP_REPEAT` to cover the integer/bfloat16 types it currently doesn't support. My first contribution (`COL2IM_1D`) was a full new kernel; this one is a type-extension of existing machinery — a cleaner second win that still touches real CUDA internals without the risk of a large, easily-superseded surface.

Carrying forward the hardest lesson from Contribution #1 (my approved PR was superseded by a parallel implementation that merged first), I searched open and merged PRs for this exact op **before** committing to it, and confirmed no one is working the dtype gap.

## Understanding the Issue

**Target op:** `GGML_OP_REPEAT` (broadcast/tile a tensor along one or more dimensions)
**Backend:** CUDA

### Problem Description
CUDA's `REPEAT` path only implemented the `F32` and `F16` element types. When a graph asked it to repeat an `i32`, `i16`, or `bf16` tensor, the backend reported the op as unsupported (and would assert at runtime if forced past the gate). REPEAT is pure data movement — it copies/tiles bytes and does no arithmetic on the values — so the missing types are a natural, low-risk extension rather than new math.

### Expected Behavior
CUDA should execute `REPEAT` for `i32`, `i16`, and `bf16` and produce output matching the CPU reference. In the test harness, the previously-skipped `i32` / `i16` / `bf16` cases should all report `OK`.

### Current Behavior (before fix)
On `master`, `REPEAT` passed all `f32` cases but reported `not supported [CUDA0]` for every `i32`, `i16`, and `bf16` case (6 skipped cases in the `-o REPEAT` run).

### Affected Components
- `ggml/src/ggml-cuda/binbcast.cu` — `ggml_cuda_op_repeat` routes through the shared `bin_bcast` copy machinery, whose type dispatch was instantiated only for `f32` / `f16`. This is where the new element-sizes are handled.
- `ggml/src/ggml-cuda/ggml-cuda.cu` — the `supports_op` gate for `GGML_OP_REPEAT` (around line 4930) explicitly returned true only for `F32` / `F16`; it was widened to match the extended kernel.
- Reference only (no change needed): `ggml/src/ggml-cpu/ops.cpp` (CPU `REPEAT` implementation — the numerical ground truth the test harness compares against).
- **Excluded from the PR:** `docs/ops.md` / `docs/ops/CUDA.csv`. In Contribution #1 the maintainer (@am17an) explicitly asked to drop docs changes since they are regenerated separately. This PR touches only CUDA source.

## Reproduction Process

### Environment Setup

| Component | Version / Notes |
|-----------|-----------------|
| OS | Windows 11 |
| Compiler | MSVC (VS 2022 BuildTools, 14.44) |
| Build system | CMake + Ninja |
| CUDA Toolkit | 12.8 (required for Blackwell `sm_120a` / RTX 5060 Laptop GPU, compute capability 12.0) |
| Test binary | `build\bin\test-backend-ops.exe` |
| Build config | `-DGGML_CUDA=ON` |

**Challenges and fixes (carried from Contribution #1):**
- **`cl.exe not found` from PowerShell.** The CUDA build only succeeds from the **x64 Native Tools Command Prompt for VS 2022**, which runs `vcvars64.bat` to put MSVC on the path. Fix: always build/test from that prompt. (Git and file-inspection commands run fine in PowerShell.)
- **`docs/ops.md` cannot be trusted for op selection.** The support table lags reality. Ground truth is the test harness. The target was chosen by dumping live CUDA support with `test-backend-ops support --output csv`, diffing to find ops with missing non-view variants, then re-verifying against open/merged GitHub PRs.

### Steps to Reproduce
1. Open the **x64 Native Tools Command Prompt for VS 2022** (not PowerShell) for build/test; use PowerShell for the git sync.
2. Sync `master` with upstream so the result reflects current reality:

   ```
   git checkout master
   git fetch upstream
   git reset --hard upstream/master
   ```

3. Confirm the whole project builds clean against current upstream:

   ```
   cmake --build build --config Release
   ```

4. Run the op-filtered test:

   ```
   build\bin\test-backend-ops.exe -o REPEAT
   ```

5. **Observed (on master):** all `f32` cases print `OK`; the `i32`, `i16`, and `bf16` cases print `not supported [CUDA0]`.

> **Reading the result correctly:** the harness prints `Backend CUDA0: OK` even though three types are missing, because unsupported cases are *skipped*, not *failed*. The real signal is the per-line `not supported [CUDA0]` text on the `i32` / `i16` / `bf16` rows — **not** the final `OK`. Those skipped rows are the work.

## Solution Approach

### Analysis
This is a missing-feature issue, not a bug. `REPEAT` on CUDA is implemented via the shared `bin_bcast` (broadcast-copy) machinery: `ggml_cuda_op_repeat` (in `binbcast.cu`) calls `ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_repeat, 0>>`. That dispatcher's type ladder was instantiated only for `f32` and `f16`, so any other element type fell through to an unsupported/assert path. The `supports_op` gate in `ggml-cuda.cu` mirrored this limit exactly, and its inline comment confirmed it: *"the CUDA REPEAT path only implements F32/F16; other types assert at runtime."*

Because REPEAT only moves bytes, the type matters solely for its **element size** (`i32` = 4 bytes; `i16` and `bf16` = 2 bytes). No value arithmetic is involved. The risk in a naive fix is pushing integer/byte data through the existing *float* arithmetic stack, which would trigger implicit conversion and FTZ corruption. The correct approach is to treat the data as opaque bytes.

### Implemented Solution
The fix matched the Phase II plan: a **parallel, bitwise-only broadcast path** (`bin_bcast_cuda_bitwise`) that never routes non-float data through the float arithmetic stack, plus a widened `supports_op` gate.

The bitwise kernels and launchers are templated on the **payload size**, not the semantic type — `uint32_t` for 4-byte payloads (`i32`, and `f32`) and `uint16_t` for 2-byte payloads (`i16`, `bf16`, `f16`). This guarantees the GPU treats the data as opaque bytes and copies it verbatim, which is exactly what REPEAT should do.

**Files modified — exactly two, no new files** (to contain the blast radius):
- **`ggml/src/ggml-cuda/binbcast.cu`** — added the strictly-typed bitwise operators (`op_repeat_bitwise_u32`, `op_repeat_bitwise_u16`), the bitwise kernels (`k_bin_bcast_unravel_bitwise`, `k_bin_bcast_bitwise`), the bitwise launcher (`launch_bin_bcast_pack_bitwise`), and the wrapper. Updated `ggml_cuda_op_repeat` to route 4-byte and 2-byte tensors down this new pipeline.
- **`ggml/src/ggml-cuda/ggml-cuda.cu`** — widened the `supports_op` gate for `GGML_OP_REPEAT` to authorize `GGML_TYPE_I32`, `GGML_TYPE_I16`, and `GGML_TYPE_BF16`.

## Testing Strategy

The op is verified through ggml's built-in backend test harness, which auto-compares CUDA output against the CPU reference. The `i32` / `i16` / `bf16` cases already ship with the op, so no new test cases were needed — the goal was to make the previously-skipped cases pass.

### Results
- **Primary:** `test-backend-ops.exe -o REPEAT` → all `i32`, `i16`, and `bf16` cases now route through the new bitwise pipeline and pass (18/18, up from 10/10), with **no runtime assertions and no data corruption**. Passing = byte-exact parity with the CPU reference across all three types.
- **Regression:** full `test-backend-ops.exe` (no `-o`) → **13,000 / 13,000 tests passed**, clean, no regressions in the ops that share the `bin_bcast` machinery (`ADD`, `SUB`, `MUL`, `DIV`).

## Implementation Notes

### Blockers encountered and resolved (compilation)

1. **NVCC template-argument misalignment.** A `<unknown-type>` deduction failure on `ggml_cuda_kernel_launch`: the kernel definitions expected `typename op_t` as the first template argument, but the launch call omitted it, so NVCC pushed the function pointer (`bin_op`) into the `typename` slot. **Fix:** explicitly added `op_t` as the first argument inside the `< >` of the launch calls.
2. **NVCC C++17 `auto` bug / signature mismatch.** To bypass a known NVCC template-instantiation bug, the strict function-pointer type `op_t (*bin_op)(...)` was replaced with `auto bin_op` in the new bitwise kernels. `k_bin_bcast_unravel_bitwise` was initially missed, causing a signature mismatch with the launcher. **Fix:** standardized `auto bin_op` across all bitwise kernels and wrappers.
3. **Enum identifier typo.** A syntax error in `ggml-cuda.cu` — `GGML_TYPE_I26` written instead of `GGML_TYPE_I16` in the `supports_op` gate — halted the compiler on an undefined identifier. **Fix:** corrected to `GGML_TYPE_I16`.

### Guardrails carried from Contribution #1
- **One symbol, one translation unit.** Duplicate definitions cause `LNK2005` and a silently stale binary. All changes stayed in their proper files.
- **Full-project relink, not single-target.** Rebuild the whole project after any wiring change, and confirm a known-good op (`ADD`) runs before trusting a REPEAT result.
- **LF line endings** on any edited file, to match Linux CI.

## Pull Request

**PR Link:** [ggml-org/llama.cpp#26642](https://github.com/ggml-org/llama.cpp/pull/26642)
**Title:** Support i32, i16, and bf16 for GGML_OP_REPEAT on CUDA
**Branch:** `Ssamdeman:cuda-repeat-types` → `ggml-org:master`
**Diff:** 2 files changed (+386 / −3), single commit, DCO signed-off.

**Summary:** Adds CUDA support for `i32`, `i16`, and `bf16` in `GGML_OP_REPEAT` by introducing a separate bitwise broadcast pipeline that moves raw bytes instead of routing non-float data through the float arithmetic stack. Passes all `REPEAT` cases (18/18) and the full regression suite (13,000/13,000).

**AI-usage disclosure (as submitted):** Declared **YES** — AI was used assistively to read and understand the existing code, and to help troubleshoot specific bugs during compiling and testing. I authored the implementation myself.

**Automated checks (at submission):**
- Pull Request Labeler → passed; `ggml` and `CUDA` labels applied automatically.
- CI workflows → awaiting maintainer approval (standard for a non-write-access contributor; workflows do not auto-run on first-time external PRs).
- No AI-disclosure bot objection raised.

**Maintainer Feedback (Round 1 — @am17an, contributor):**
- @am17an noted that extending the repeat kernel to more types doesn't fit cleanly into the existing `f32 -> (f32, f32)` regime, and suggested creating a separate path that operates on bytes.
- This is directionally aligned with the byte-based approach already implemented (the `uint32_t` / `uint16_t` bitwise pipeline). **Response in progress** — I'll reply per-thread confirming the byte-operating design and clarifying how the current diff already separates the bitwise path from the float arithmetic path, adjusting the structure if the reviewer wants a cleaner separation.

**Status:** Open, under review — awaiting the 2 required approving reviews. First maintainer comment received; response and any requested changes to follow within 24h.

## Learnings & Reflections

**Technical skills gained.** This contribution taught me how a single dispatcher (`bin_bcast`) can back multiple ops, and why type support there is about *element size* rather than semantics for a pure data-movement op. Building a separate bitwise path — instead of forcing bytes through the existing float arithmetic — was the key insight: it treats the payload as opaque `uint32_t` / `uint16_t` and sidesteps implicit conversion and FTZ corruption entirely. I also got real practice diagnosing NVCC-specific template failures, which behave differently from host-side C++ template errors.

**Hardest part.** Not the design — which matched the plan — but the NVCC template quirks during compilation: the argument-slot misalignment and the C++17 `auto` workaround, both of which produce confusing error messages that point away from the real cause.

**Carry-forward from Contribution #1.** My first PR earned two approvals but was superseded by a parallel implementation that merged first. The process changes I applied here: (1) searched open **and** merged PRs for the exact op before starting, (2) chose a bounded, lower-collision op over a large new kernel, and (3) rebased on upstream immediately before submitting. This contribution is the test of whether those changes produce a cleaner outcome — the review cycle is now underway.

### Resources Used
- llama.cpp `test-backend-ops` harness and its `support --output csv` mode (ground-truth op support)
- `CONTRIBUTING.md` and `AGENTS.md` (AI policy, commit conventions, CPU-first rule)
- The existing `binbcast.cu` `bin_bcast` machinery and the CPU `REPEAT` reference in `ggml-cpu/ops.cpp`
- Contribution #1's own review history (docs-exclusion, AI-disclosure, and stale-build lessons)
