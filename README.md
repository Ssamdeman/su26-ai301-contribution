# Contribution #14909: Feature Request: Implement missing ops from backends

**Contribution Number:** 1
**Student:** Samuel Damon
**Issue:** [ggml-org/llama.cpp#14909](https://github.com/ggml-org/llama.cpp/issues/14909)
**Status:** Phase II Completed

## Why I Chose This Issue

I chose this issue because llama.cpp is one of the most impactful open source projects in the AI space — it enables local LLM inference for millions of users. Contributing to something with real-world value matters to me, and this project is exactly that. I want to expand my knowledge from high-level AI into lower-level systems programming. Implementing a missing backend op means working in C/C++, understanding how inference engines execute operations, and writing code that directly affects performance. This is the kind of challenge I'm looking for — practical, technical, and meaningful.

## Understanding the Issue

**Target op:** `COL2IM_1D` (1D column-to-image — the inverse of `IM2COL`)
**Backend:** CUDA

### Problem Description
The CUDA backend has no implementation for the `GGML_OP_COL2IM_1D` operation. The op exists in core ggml (CPU reference merged in #24206), but when a graph containing it runs on a CUDA device, the backend reports the op as unsupported rather than executing it. To run on CUDA it needs a kernel plus an entry in the backend's `supports_op` so it gets dispatched.

### Expected Behavior
CUDA should execute `COL2IM_1D` and produce output matching the CPU reference. In the test harness, all 33 `COL2IM_1D` cases should report `OK`, ending with `33/33 tests passed`.

### Current Behavior
On current `master`, every `COL2IM_1D` case is skipped on CUDA. The harness prints `not supported [CUDA0]` for all 33 cases and ends with `0/0 tests passed`.

### Affected Components
- `ggml/src/ggml-cuda/` — needs a new `col2im_1d.cu` / `.cuh` kernel and dispatch wiring.
- `ggml/src/ggml-cuda/ggml-cuda.cu` — op dispatch switch and `supports_op` switch.
- `docs/ops.md` — regenerated support matrix (Phase III).
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
- **`docs/ops.md` cannot be trusted for op selection.** The support table lags reality. Ground truth is the test harness. The target was chosen by running `test-backend-ops` and dumping live support to `cuda_support.csv`, then re-verified against open/merged GitHub PRs.

### Steps to Reproduce
1. Open the **x64 Native Tools Command Prompt for VS 2022** (not PowerShell).
2. From the repo root, sync `master` with upstream so the result reflects current reality:
   ```
   git checkout master
   git fetch upstream
   git reset --hard upstream/master
   ```
   (CPU `COL2IM_1D` is present as of commit `26021699b`, tag `b9575`, PR #24206 — the reference implementation.)
3. Build the test target:
   ```
   cmake --build build --target test-backend-ops --config Release
   ```
4. Run the op-filtered support test:
   ```
   build\bin\test-backend-ops.exe -o COL2IM_1D
   ```
5. **Expected (if implemented):** every case prints `OK`, ending `33/33 tests passed`.
6. **Observed:** all 33 cases print `not supported [CUDA0]`, ending `0/0 tests passed`.

> **Reading the result correctly:** the harness prints `Backend CUDA0: OK` even when the op is missing, because unsupported cases are *skipped*, not *failed*. The real signal is the per-line `not supported [CUDA0]` text and the `0/0 tests passed` count — **not** the final `OK`.

### Reproduction Evidence
- **Branch:** https://github.com/Ssamdeman/llama.cpp/tree/cuda-col2im1d-op
- **Logs:** `col2im1d_repro.txt` — full `0/0 not supported [CUDA0]` dump.
- **My findings:** All 33 cases span the three dtypes the CPU reference supports (`f32`, `f16`, `bf16`) and exercise the full parameter surface — `K` (kernel size), `OC` (output channels), `T_in` (input length), `s0` (stride), `p0` (padding). The op is genuinely missing on CUDA, not partially implemented. Prior-art check: CPU (#24206) and Vulkan (#24425) have merged; the only open CUDA attempt (#24417) is early / has requested changes, so the op is still contestable.

## Solution Approach

### Analysis
This is a missing-feature issue, not a bug. `COL2IM_1D` was added to core ggml and the CPU backend but never to CUDA, so the CUDA backend's `supports_op` returns false and the op falls back / is skipped. The fix is additive: implement the kernel and register support. `COL2IM_1D` is the inverse of `IM2COL` — where im2col **gathers** input patches into columns, col2im **reassembles** an output signal from those columns, summing the contributions of overlapping kernel windows.

### Proposed Solution
Add a CUDA kernel for `COL2IM_1D` modeled on the existing `im2col.cu`, run in reverse, using an **output-centric (gather)** strategy so no atomics are required and all three dtypes share one code path. Wire it into the op dispatch and add a `supports_op` case.

### Implementation Plan

Using the UMPIRE framework (adapted):

**Understand:** CUDA cannot execute `GGML_OP_COL2IM_1D`. It needs (a) a CUDA kernel computing 1D col2im, and (b) a `supports_op` entry so the op is dispatched instead of skipped. Inverse of im2col: reassemble an output signal from overlapping column windows.

**Match:** Existing code that anchors the work —
- *CPU reference (math to port):* `ggml/src/ggml-cpu/ops.cpp` — `ggml_compute_forward_col2im_1d_impl` (templated, ~L6741); dtype dispatch `ggml_compute_forward_col2im_1d` (~L6794) for `F32`/`F16`/`BF16`.
- *Central op registration (already merged by #24206 — reused, not re-added):* enum `ggml/include/ggml.h:538`; name string `ggml/src/ggml.c:1034`; constructor `ggml/src/ggml.c:4575`.
- *Closest CUDA analogue (structural template):* `ggml/src/ggml-cuda/im2col.cu` / `im2col.cuh` — `im2col_kernel`, host wrapper `ggml_cuda_op_im2col`, dispatched from `ggml-cuda.cu:3076`. col2im mirrors this layout in reverse.

**Plan:**
1. Add `ggml/src/ggml-cuda/col2im_1d.cu` and `col2im_1d.cuh`, structured after `im2col.cu`.
2. Implement an **output-centric (gather) kernel** — one thread per output element, looping over the kernel windows that map onto that position and summing their column contributions. *Why gather, not scatter:* the naive inverse scatters (each column writes multiple overlapping outputs), needing `atomicAdd`; half/bf16 atomics are unsupported or slow on many archs, and the tests include `f16`/`bf16`. Gather needs no atomics and unifies all dtypes.
3. Read `K`, `OC`, `T_in`, `s0`, `p0` from `op_params`; compute output length from stride/padding exactly as the CPU reference does.
4. Template the kernel over `float` / `half` / `nv_bfloat16` so all three dtypes build from one impl.
5. Add `#include "col2im_1d.cuh"` and a `case GGML_OP_COL2IM_1D:` to the op dispatch in `ggml-cuda.cu`.
6. Add `case GGML_OP_COL2IM_1D:` to the CUDA `supports_op` switch. **Edit only this case** — keep the diff surgical (the POOL_1D PR #22297 was flagged for an accidental `case` fall-through that changed unrelated ops).

**Implement:** Phase III. Code will land on https://github.com/Ssamdeman/llama.cpp/tree/cuda-col2im1d-op *(placeholder — no kernel written yet).*

**Review:**
- Self-review against `CONTRIBUTING.md` before opening the PR.
- Commit / PR title convention: lowercase, colon-scoped — e.g. `cuda : add col2im_1d op` (matches merged CPU PR `ggml : add GGML_OP_COL2IM_1D`).
- **AI-usage disclosure is mandatory** (both prior CUDA op PRs #22297/#21361 included one; the PR template requires it).
- Regenerate `docs/ops.md` **with the project's script**, not by hand (a reviewer rejected a manual docs edit on #22297). Locate the script via `CONTRIBUTING.md` / the `update-ops-docs` CI job.

**Evaluate:**
- *Primary:* `test-backend-ops.exe -o COL2IM_1D` goes `0/0` (all `not supported`) → **`33/33 tests passed`**. The harness checks CUDA output against the CPU reference, so passing = numerical parity across all dtypes and params.
- *Regression:* run the full `test-backend-ops.exe` (no `-o`) to confirm nothing else broke.

## Testing Strategy

The op is verified entirely through ggml's built-in backend test harness, which auto-compares CUDA output against the CPU reference. (Detailed results filled in Phase III.)

### Unit Tests
- *Test case 1:* `test-backend-ops -o COL2IM_1D` — all 33 cases (`f32`/`f16`/`bf16` × kernel/stride/padding combos) must pass.
- *Test case 2:* edge params present in the set — `K=1` (degenerate kernel), `T_in=1` (minimal input), `p0=5` with small input (padding ≥ input).
- *Test case 3:* [Phase III]

### Integration Tests
- Full `test-backend-ops` run — no regressions in other ops.
- [Phase III]

### Manual Testing
[Phase III — to be filled during implementation.]

## Implementation Notes

### Week [X] Progress
[Phase III — to be filled during implementation.]

### Code Changes
- Files modified: [Phase III]
- Key commits: [Phase III]
- Approach decisions: [Phase III]

## Pull Request

**PR Link:** [Phase III]
**PR Description:** [Phase III — much of the Solution Approach above will be adapted]
**Maintainer Feedback:** [Phase III/IV]
**Status:** Not yet submitted (Phase II complete; implementation begins in Phase III)

## Learnings & Reflections

[Phase IV — to be filled at the end.]

### Resources Used
- llama.cpp `docs/ops.md` and the `test-backend-ops` harness (ground-truth op support)
- CPU reference PR [#24206](https://github.com/ggml-org/llama.cpp/pull/24206) and the existing CUDA `im2col.cu`
- Prior CUDA op PRs [#22297](https://github.com/ggml-org/llama.cpp/pull/22297), [#21361](https://github.com/ggml-org/llama.cpp/pull/21361) (review-convention and AI-disclosure lessons)
