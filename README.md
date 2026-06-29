# Contribution #14909: Feature Request: Implement missing ops from backends

**Contribution Number:** 1
**Student:** Samuel Damon
**Issue:** [ggml-org/llama.cpp#14909](https://github.com/ggml-org/llama.cpp/issues/14909)
**Status:** Phase IV Complete — PR submitted, awaiting review

## Why I Chose This Issue

I chose this issue because llama.cpp is one of the most impactful open source projects in the AI space — it enables local LLM inference for millions of users. Contributing to something with real-world value matters to me, and this project is exactly that. I want to expand my knowledge from high-level AI into lower-level systems programming. Implementing a missing backend op means working in C/C++, understanding how inference engines execute operations, and writing code that directly affects performance. This is the kind of challenge I'm looking for — practical, technical, and meaningful.

## Understanding the Issue

**Target op:** `COL2IM_1D` (1D column-to-image — the inverse of `IM2COL`)
**Backend:** CUDA

### Problem Description
The CUDA backend has no implementation for the `GGML_OP_COL2IM_1D` operation. The op exists in core ggml (CPU reference merged in #24206), but when a graph containing it runs on a CUDA device, the backend reports the op as unsupported rather than executing it. To run on CUDA it needs a kernel plus an entry in the backend's `supports_op` so it gets dispatched.

### Expected Behavior
CUDA should execute `COL2IM_1D` and produce output matching the CPU reference. In the test harness, all 33 `COL2IM_1D` cases should report `OK`, ending with `33/33 tests passed`.

### Current Behavior (before fix)
On current `master`, every `COL2IM_1D` case is skipped on CUDA. The harness prints `not supported [CUDA0]` for all 33 cases and ends with `0/0 tests passed`.

### Affected Components
- `ggml/src/ggml-cuda/` — needs a new `col2im_1d.cu` / `.cuh` kernel and dispatch wiring.
- `ggml/src/ggml-cuda/ggml-cuda.cu` — op dispatch switch and `supports_op` switch.
- `docs/ops.md` — regenerated support matrix.
- Reference only (no change needed): `ggml/src/ggml-cpu/ops.cpp` (CPU implementation).

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

**Challenges and fixes:**
- **`cl.exe not found` / missing `stdbool.h` from PowerShell.** The CUDA build only succeeds from the **x64 Native Tools Command Prompt for VS 2022**, which runs `vcvars64.bat` to put MSVC on the path. A normal shell fails. Fix: always build/test from that prompt.
- **`docs/ops.md` cannot be trusted for op selection.** The support table lags reality. Ground truth is the test harness. The target was chosen by running `test-backend-ops` and dumping live support, then re-verified against open/merged GitHub PRs.

### Steps to Reproduce
1. Open the **x64 Native Tools Command Prompt for VS 2022** (not PowerShell).
2. From the repo root, sync `master` with upstream so the result reflects current reality:

git checkout master
git fetch upstream
git reset --hard upstream/master

(CPU `COL2IM_1D` is present as of PR #24206 — the reference implementation
3. Build the test target:
cmake --build build --target test-backend-ops --config Release
4. Run the op-filtered support test:
build\bin\test-backend-ops.exe -o COL2IM_1D
5. **Observed (on master):** all 33 cases print `not supported [CUDA0]`, ending `0/0 tests passed`.

> **Reading the result correctly:** the harness prints `Backend CUDA0: OK` even when the op is missing, because unsupported cases are *skipped*, not *failed*. The real signal is the per-line `not supported [CUDA0]` text and the `0/0 tests passed` count — **not** the final `OK`.

## Solution Approach

### Analysis
This is a missing-feature issue, not a bug. `COL2IM_1D` was added to core ggml and the CPU backend but never to CUDA, so the CUDA backend's `supports_op` returns false and the op is skipped. The fix is additive: implement the kernel and register support. `COL2IM_1D` is the inverse of `IM2COL` — where im2col **gathers** input patches into columns, col2im **reassembles** an output signal from those columns, summing the contributions of overlapping kernel windows.

### Implemented Solution
A CUDA kernel for `COL2IM_1D` modeled on the existing `im2col.cu`, using an **output-centric (gather)** strategy so no atomics are required and all three dtypes share one code path. Wired into the op dispatch with a `supports_op` case.

**Why gather, not scatter:** the naive inverse scatters (each column writes multiple overlapping outputs), needing `atomicAdd`; half/bf16 atomics are unsupported or slow on many archs, and the tests include `f16`/`bf16`. Gather needs no atomics and unifies all dtypes. Accumulation is done in fp32 for precision on half types.

### Implementation
- **Files added:** `ggml/src/ggml-cuda/col2im_1d.cu`, `ggml/src/ggml-cuda/col2im_1d.cuh` — a self-contained gather kernel templated over `float` / `half` / `nv_bfloat16`, plus the host wrapper that reads `s0` / `OC` / `p0` from `op_params` and derives `K = src->ne[0] / OC`.
- **File modified:** `ggml/src/ggml-cuda/ggml-cuda.cu` — `#include "ggml-cuda/col2im_1d.cuh"`, a `case GGML_OP_COL2IM_1D:` in the op dispatch, and a standalone `case GGML_OP_COL2IM_1D: return true;` in `supports_op` (surgical edit, no fall-through into adjacent cases).
- **Docs:** regenerated `docs/ops.md` via `scripts/create_ops_docs.py` after appending the CUDA support rows to `docs/ops/CUDA.csv` (per the project's `update-ops-docs` workflow — never hand-edited).

## Testing Strategy

The op is verified entirely through ggml's built-in backend test harness, which auto-compares CUDA output against the CPU reference. The 33 `COL2IM_1D` cases ship with the op (added by the CPU PR #24206), so no new test cases were needed — the goal was to make the existing ones pass.

### Results
- **Primary:** `test-backend-ops.exe -o COL2IM_1D` → **33/33 tests passed** (`f32` / `f16` / `bf16` × kernel/stride/padding combos, including edge cases `K=1`, `T_in=1`, and `p0=5` with small input). Passing = numerical parity with the CPU reference across all dtypes and params.
- **Regression:** full `test-backend-ops.exe` (no `-o`) → **12868/12868 tests passed**, no regressions in other ops. Confirmed the `im2col` sibling files were reverted to a pristine state (zero diff).

## Implementation Notes

### Resolved Blocker — duplicate symbol at link time

Early Phase III runs showed `0/0 not supported [CUDA0]` even after the wiring was in place. Diagnosis (by elimination):

1. **Ruled out backend-load failure.** A startup line `load_backend: failed to find ggml_backend_init in ...ggml-cuda.dll` looked like the cause, but running `-o ADD` returned `99/99 OK` — the CUDA backend *was* loading and executing. That line is cosmetic for this statically-linked build.
2. **Ruled out missing wiring.** The real, CMake-compiled `ggml-cuda.cu` already had the include, dispatch case, and `supports_op` case correctly in place.
3. **Found the real cause.** The op implementation had been authored into the wrong file (`im2col.cu`) *and* into the dedicated `col2im_1d.cu`, so `ggml_cuda_op_col2im_1d` was defined twice. The link failed with `LNK2005: ... already defined`, no DLL was produced, and the test kept running against a stale binary.
4. **Fix.** Kept the implementation in its proper home (`col2im_1d.cu`), reverted `im2col.cu` / `.cuh` to pristine upstream, and did a full project rebuild (not a single target — `cmake --build` on one target had been skipping the relink). After that the op executed and passed 33/33.

**Lesson:** a stale build masquerades as a logic bug. Confirm a known-good op (`ADD`) loads and the whole project relinks cleanly before trusting any test result — and keep one symbol in exactly one translation unit.

### Approach decisions
Output-centric gather over scatter+atomics (clean f16/bf16 path); single `<typename T>` template matching the CPU dispatch; fp32 accumulation for precision on half types; sibling `im2col` left untouched for a minimal, reviewable diff.

## Pull Request

**PR Link:** [ggml-org/llama.cpp#25151](https://github.com/ggml-org/llama.cpp/pull/25151)
**Summary:** Implements `GGML_OP_COL2IM_1D` on the CUDA backend — the inverse of IM2COL and the missing counterpart to the merged CPU reference (#24206). Output-centric gather kernel templated over f32/f16/bf16 with fp32 accumulation. Passes 33/33 `test-backend-ops` cases; full regression suite clean (12868/12868). Diff is 5 files, +126/−1.
**Maintainer Feedback:** None yet — awaiting initial review.
**Status:** Awaiting review.

## Learnings & Reflections

**Technical skills gained.** I went from high-level AI work into the systems layer of an inference engine: writing a CUDA kernel, understanding how ggml dispatches ops per-backend and gates them through `supports_op`, and reasoning about why a gather formulation avoids atomics that would break on half/bf16. I also learned the project's real contribution mechanics — the CSV-driven `docs/ops.md` regeneration, the surgical-diff expectation, and the CPU-first / CUDA-follow-up convention.

**Hardest part.** Not the kernel math (it verified against the CPU reference on the first pass) but everything *around* it: Windows build-environment friction (wrong shell, stale Ninja objects), and the duplicate-symbol link failure that disguised itself as `0/0 not supported`. Almost all of my debugging time went into build-system and tooling problems, not algorithm logic.

**What I'd do differently.** Author the implementation in its dedicated file from the start (one symbol, one translation unit), and after every wiring change do a full clean relink and confirm a known-good op runs *before* concluding anything about my own code. Trusting a test result without first trusting the build cost me the most time.

### Resources Used
- llama.cpp `docs/ops.md` and the `test-backend-ops` harness (ground-truth op support)
- `CONTRIBUTING.md` and `AGENTS.md` (AI policy, commit conventions, CPU-first rule)
- CPU reference PR [#24206](https://github.com/ggml-org/llama.cpp/pull/24206) and the existing CUDA `im2col.cu`
- The `update-ops-docs` CI workflow and `scripts/create_ops_docs.py` (docs regeneration)
- Prior CUDA op PRs [#22297](https://github.com/ggml-org/llama.cpp/pull/22297), [#21361](https://github.com/ggml-org/llama.cpp/pull/21361) (review-convention and AI-disclosure lessons)

