# sd-scripts fork Anima BF16 fix and local installation

## Authorized scope

- This follows the user's approved implementation plan, superseding the earlier
  instruction to revert temporary sd-scripts changes. The target is now
  `/home/munsy0227/sd-scripts`; `/home/munsy0227/tempsd/sd-scripts` stays unchanged.
- Back up the old fork, synchronize main with upstream, add the minimal BF16 fix
  on fork main, delete the old work branch, and publish both repositories.
  No new PR or merge is requested.
- The complete old history and five refs were verified in
  `.internal/backups/sd-scripts-pre-reset-20261008.bundle` outside sd-scripts.
  SHA-256: `a8c8a9c0b39f05866603406e2be7bf8d3c9af09b1118c79719035327c6b639e1`.
- Local sd-scripts main was reset from `c9224ba8dfd924002eed9d7022fe18fbfeb6d012`
  to latest upstream `690ea7f96c23182352ec63def76d431c6120bd2f`; neither old
  `aa763e0` nor `c9224ba` remains in its main ancestry. Existing `.venv` and
  ignored/user files were preserved. The first remote force-with-lease attempt
  was rejected by main's branch protection; no unleased overwrite was attempted.
- GitHub's branch page confirmed deletion of `agent/bucket-no-upscale-hybrid`,
  with its Restore action and merged PR #1 retained. `git ls-remote` also
  confirmed the branch ref is absent.

## Source changes

- sd-scripts: both Block and FinalLayer AdaLN contexts now use `nullcontext()`
  when `use_fp32=False`, preserving the caller's BF16 autocast. The original
  FP32 autocast stays enabled when `use_fp32=True`. RMSNorm and timestep
  precision are unchanged, as are parameter names, shapes, dtypes and API.
- LyCORIS: removed the sd-scripts VCS installation and setup.py comment from
  `requirements-kohya.txt`. The pin previously supplied a reproducible upstream
  package revision, but would install a separate original package when using a
  locally patched checkout. `setup.py` also does not install training requirements.
- README now specifies: install the chosen sd-scripts checkout's requirements,
  then install local LyCORIS in the same environment. All upstream dependency
  specifications still match; the requirements header identifies their revision.
- Added an opt-in real Anima integration at `test/test_anima_sd_scripts.py`,
  enabled by `LYCORIS_SD_SCRIPTS_PATH`. It checks the actual imported source path
  and restores the global preset configuration after execution.

## Environment and validation

- Linux, RTX 4070, Python 3.12.15, PyTorch 2.13.0+cu130, Transformers 5.17.0,
  Accelerate 1.15.0. CPU thread counts were 1; LyCORIS used the torch backend.
- The existing Python/CUDA installation was read only. Updated dependencies and
  local LyCORIS were installed into a disposable overlay. Import from `/tmp`
  confirmed the installed LyCORIS package and actual fork source, and an
  installed-package LoKR DoRA forward/backward passed. No original sd-scripts
  VCS package was installed.
- sd-scripts precision suite: before the source fix, 18 failed and 12 passed.
  This includes direct AdaLN autocast-state failures and the real model's
  CPU/CUDA BF16 dtype collision with AdaLN-LoRA, before LyCORIS adapters exist.
  After the fix, **all 30 passed**. Both AdaLN structures, CPU FP32/BF16 and CUDA
  FP32/FP16/BF16, finite forward/backward, caller state, parameter identity/schema,
  FP32 timestep features, and exact checkpointing/non-checkpointing output and
  gradient equality are covered.
- Independent original-versus-patched model comparison: CPU FP32 and CUDA
  FP32/FP16, both AdaLN structures, all six cases had bitwise-identical outputs
  and gradients. This specifically verifies the retained FP16 path.
- Real Anima: width 64, 4 heads, 28 blocks, diffusion-only full-matrix LoKR
  rank 100000, factor 4, DoRA, scalar, 448 adapters. Twelve cases cover CPU/CUDA
  FP32 and BF16 (FP32 and BF16 adapters), checkpointing off/on. Every case had
  exact initial no-op, all 448 scalar gradients finite/nonzero, finite parameter
  gradients, output change after SGD, unchanged base weights, and bitwise-exact
  native save/reconstruction. Text encoder and full LLM adapter are not loaded.
- Existing LyCORIS selection `test.lokr test.test_kohya_optimizer
  test.torch_compile`: 152 tests, 150 passed, 2 optional skips.
- Existing CUDA `test.test_lokr_cuda`: all 3 passed, including independent DoRA
  math across 24 combinations and the mock 448-adapter BF16 optimizer case.
- sd-scripts offloading/sampling/warning/cv2 selection: 78 passed, 6 hardware skips.
- `anima_train_network.py --help` and `train_network.py --help` passed offline.
- Metadata traversal checked 86 dependency sets including training and installed
  LyCORIS roots: no version/missing conflicts. All 22 direct requirement
  specifications match upstream (21 active on this platform).
- New tests passed Ruff; Black formatting and diff whitespace checks passed.
  Black's multiprocess CLI stalled in the sandbox and was interrupted; the
  completed formatting was retained and verified without changing model code.

## Portable precision measurements and limits

Portable export is an inference representation, folding the scalar into factors
and rounding them to the requested dtype. Native resume retains raw factors,
scalar and FP32 DoRA magnitude. The native assertion stays exact. The existing
portable comparison references remain FP32 `atol=rtol=2e-5` and BF16
`atol=rtol=0.03`; BF16 is a diagnostic, not an assertion of numerical accuracy.

Checkpointing off/on produced the same measurements for each row:

| Device | Base / adapters | Max absolute error | Mean absolute error | Within reference |
| --- | --- | ---: | ---: | --- |
| CPU | FP32 / FP32 | 0.00000119209 | 0.00000022872 | yes |
| CUDA | FP32 / FP32 | 0.00000095367 | 0.00000017766 | yes |
| CPU | BF16 / FP32 | 0.037109375 | 0.0087100267 | no |
| CPU | BF16 / BF16 | 0.0399169922 | 0.0103203654 | no |
| CUDA | BF16 / FP32 | 0.0390625 | 0.0059773326 | yes |
| CUDA | BF16 / BF16 | 0.03515625 | 0.0086379647 | yes |

Four of eight BF16 portable cases exceed the combined absolute/relative reference.
The tolerance was not increased, and portable BF16 accuracy is not claimed fixed.

An extra, broader FP16 adapter/export experiment found that rank 100000 becomes
an infinite alpha buffer when cast/exported to FP16, so checkpoint reconstruction
rejects it (`alpha must be finite, got inf`). This is distinct from the Anima BF16
autocast failure and is not changed here. FP16 base-model forward/backward remains
covered, but high-rank FP16 checkpoint compatibility is not claimed.

This validation uses small randomly initialized models with nonzero gate fixtures.
It does not establish full pretrained checkpoint training, VAE/dataset/trainer-loop
success, convergence, long-duration behavior, Windows or MPS correctness.
Small logs are retained under `.internal/backups/anima-bf16-validation-20261008/`;
the disposable overlay, model files and caches are removed after validation.
