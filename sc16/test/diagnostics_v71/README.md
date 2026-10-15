# v71 diagnostics (2026-10-12)

Probe scripts used for the v71 review in `doc/claude_v71_review.md`.
Run from the `sc15` directory with `KMP_DUPLICATE_LIB_OK=TRUE`.

- `probe_common.py` loads a checkpoint plus scalers, builds Cantera trajectories with the
  runner's exact QSSA inversion, and exposes `step_model` for oracle and pure-AR tests.
- `grad_probe_state_loss.py` measures rate-loss versus state-loss gradient norms and runs
  short controlled trainings. Set `W` to a small database built with the real pipeline.
- `grad_probe_fixes.py` compares the state loss on the raw integrator output with the
  projected state. Set `GP` to the path of `grad_probe_state_loss.py`.
- `build_mini_dataset.py` builds a small dataset with the current sampler and the exact inversion.
