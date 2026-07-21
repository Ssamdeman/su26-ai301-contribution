# Contribution #2: Feature Request: Implement missing ops from backends

**Contribution Number:** 2
**Student:** Samuel Damon
**Issue:** [ggml-org/llama.cpp#14909](https://github.com/ggml-org/llama.cpp/issues/14909)
**Status:** Phase II Complete — Reproduce & Plan done; Phase III (Build) starting this week

## Why I Chose This Issue

I stayed in the same umbrella issue (#14909) for my second contribution, which CodePath recommends — the workflow is proven, my build environment is healthy, and staying in one project deepens my understanding of it rather than restarting cold in a new codebase.

For the op itself I deliberately picked a **bounded, data-movement-only** gap this time: extending CUDA `GGML_OP_REPEAT` to cover the integer/bfloat16 types it currently doesn't support. My first contribution (`COL2IM_1D`) was a full new kernel; this one is a type-extension of existing machinery — a cleaner second win that still touches real CUDA internals without the risk of a large, easily-superseded surface.

Carrying forward the hardest lesson from Contribution #1 (my approved PR was superseded by a parallel implementation that merged first), I searched open and merged PRs for this exact op **before** committing to it, and confirmed no one is working the dtype gap.

## Understanding the Issue

**Target op:** `GGML_OP_REPEAT` (broadcast/tile a tensor along one or more dimensions)
**Backend:** CUDA

### Problem Description
CUDA's `REPEAT` path only implements the `F32` and `F16` element types. When a graph asks it to repeat an `i32`, `i16`, or `bf16` tensor, the backend reports the op as unsupported (and would assert at runtime if forced past the gate). REPEAT is pure data movement — it copies/tiles bytes and does no arithmetic on the values — so the missing types are a natural, low-risk extension rather than new math.

### Expected Behavior
CUDA should execute `REPEAT` for `i32`, `i16`, and `bf16` and produce output matching the CPU reference. In the test harness, the currently-skipped `i32` / `i16` / `bf16` cases should all report `OK`.

### Current Behavior (before fix)
On current `master`, `REPEAT` passes all `f32` cases but reports `not supported [CUDA0]` for every `i32`, `i16`, and `bf16` case (6 skipped cases in the `-o REPEAT` run).

### Affected Components
- `ggml/src/ggml-cuda/binbcast.cu` — `ggml_cuda_op_repeat` routes through the shared `bin_bcast` copy machinery, whose type dispatch is instantiated only for `f32` / `f16`. This is where the new element-sizes must be handled.
- `ggml/src/ggml-cuda/ggml-cuda.cu` — the `supports_op` gate for `GGML_OP_REPEAT` (around line 4930) explicitly returns true only for `F32` / `F16`; it must be widened to match the extended kernel.
- Reference only (no change needed): `ggml/src/ggml-cpu/ops.cpp` (CPU `REPEAT` implementation — the numerical ground truth the test harness compares against).
- **Excluded from the PR:** `docs/ops.md` / `docs/ops/CUDA.csv`. In Contribution #1 the maintainer (@am17an) explicitly asked to drop docs changes since they are regenerated separately. This PR will touch only CUDA source.

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

> **Reading the result correctly:** the harness prints `10/10 tests passed` and `Backend CUDA0: OK` even though three types are missing, because unsupported cases are *skipped*, not *failed*. The real signal is the per-line `not supported [CUDA0]` text on the `i32` / `i16` / `bf16` rows — **not** the final `OK`. Those skipped rows are the work.

## Solution Approach

### Analysis (Phase II)
This is a missing-feature issue, not a bug. `REPEAT` on CUDA is implemented via the shared `bin_bcast` (broadcast-copy) machinery: `ggml_cuda_op_repeat` (in `binbcast.cu`) simply calls `ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_repeat, 0>>`. That dispatcher's type ladder is instantiated only for `f32` and `f16`, so any other element type falls through to an unsupported/assert path. The `supports_op` gate in `ggml-cuda.cu` mirrors this limit exactly, and its inline comment confirms it: *"the CUDA REPEAT path only implements F32/F16; other types assert at runtime."*

Because REPEAT only moves bytes, the type matters solely for its **element size** (`i32` = 4 bytes; `i16` and `bf16` = 2 bytes). No value arithmetic is involved, which is what makes this a safe, bounded extension.

### Planned Fix (Phase III)
Two coordinated edits, data-movement only:
1. **Extend the `bin_bcast` type dispatch** so the copy path handles the new element-sizes (`i32` / `i16` / `bf16`) in addition to `f32` / `f16`.
2. **Widen the `supports_op` gate** for `GGML_OP_REPEAT` to allow the newly-supported types, matching the kernel's real capability (surgical edit — no fall-through into adjacent cases).

Then rebuild the whole project (not a single target — a stale relink masqueraded as a logic bug in Contribution #1) and iterate until the previously-skipped cases pass.

### Concurrency Check (lesson carried from Contribution #1)
Before committing to REPEAT I searched the repository's pull requests for `ggml_repeat` and `repeat cuda` (without an `is:open` filter, to also catch anything merged recently). Every hit was a model/attention/conv PR that merely mentions "repeat" or "cuda" in passing — **no open or recently-merged PR targets the REPEAT dtype gap.** The op is clear to claim. I will keep my branch rebased on upstream throughout Phase III to surface any collision early.

## Testing Strategy

*(Planned — no results yet. This section will be filled in during Phase III.)*

The op will be verified through ggml's built-in backend test harness, which auto-compares CUDA output against the CPU reference. The `i32` / `i16` / `bf16` cases already ship with the op, so no new test cases are needed — the goal is to make the existing skipped cases pass.

- **Primary (planned):** `test-backend-ops.exe -o REPEAT` → the `i32` / `i16` / `bf16` cases should move from `not supported [CUDA0]` to `OK`, confirming numerical parity with the CPU reference.
- **Regression (planned):** full `test-backend-ops.exe` (no `-o`) → confirm no regressions in other ops that share the `bin_bcast` machinery (`ADD`, `SUB`, `MUL`, `DIV`).

## Implementation Notes

*(Pending — Phase III not yet started.)*

Guardrails carried from Contribution #1:
- **One symbol, one translation unit.** Duplicate definitions cause `LNK2005` and a silently stale binary. Keep any change in its proper home.
- **Full-project relink, not single-target.** `cmake --build` on one target skipped the relink last time and ran tests against a stale DLL. Rebuild the whole project after any wiring change, and confirm a known-good op (`ADD`) runs before trusting a REPEAT result.
- **LF line endings** on any edited file, to match Linux CI.

## Pull Request

*(Pending — Phase IV not yet reached.)*

- PR title convention: `CUDA: add i32/i16/bf16 support to REPEAT` (or lowercase `cuda : ...`).
- Docs (`docs/ops.md`, `docs/ops/CUDA.csv`) will **not** be included, per @am17an's guidance in Contribution #1.
- AI-usage disclosure will be written first-person and honest: I author the code; AI assists with architecture explanation, layout verification, and debugging my own code.
- DCO sign-off (`-s`) will be added if the DCO check goes red.

## Learnings & Reflections

*(In progress — will be completed after Phase IV.)*

**Carry-forward from Contribution #1.** My first PR earned two approvals but was superseded by a parallel implementation that merged first. The concrete process changes I applied to this contribution: (1) search open **and** merged PRs for the exact op before starting, (2) prefer a bounded, lower-collision op over a large new kernel, and (3) rebase on upstream early and often throughout the build phase. This contribution is the test of whether those changes produce a cleaner outcome.

### Resources Used
- llama.cpp `test-backend-ops` harness and its `support --output csv` mode (ground-truth op support)
- `CONTRIBUTING.md` and `AGENTS.md` (AI policy, commit conventions, CPU-first rule)
- The existing `binbcast.cu` `bin_bcast` machinery and the CPU `REPEAT` reference in `ggml-cpu/ops.cpp`
- Contribution #1's own review history (docs-exclusion, AI-disclosure, and stale-build lessons)
