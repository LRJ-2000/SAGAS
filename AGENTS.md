# Running SAGAS with an Agent

Use this runbook when asked to evaluate a task, reproduce a benchmark,
prepare assets, or diagnose a run. [README.md](README.md) is the human
quick start; [ASSETS.md](ASSETS.md) describes the available bundles.

## Run Workflow

1. **Select the task.** Read the matching row in `configs/navigation.json`,
   `configs/overlap.json`, or `configs/scene.json`. For a single task,
   select one preset: use `--env` and `--difficulty` for Navigation,
   `--env` and `--shard` for Overlap, or the default preset for Scene.
   Then pass `--task-id`, which is local to the selected task file. For an
   unspecified demo, choose AntMaze Medium-Navigate Easy task 0 from the
   README. Run a full fixed task file or use `--all` only when the requested
   scope calls for it.
2. **Check the runtime and command.** Use the Linux x86-64/Python 3.9
   environment and pinned dependencies in the README. Confirm that
   `ltl2ba` is executable. Pass `--ltl2ba` when it is not on PATH.
   Run the intended `evaluate` command with `--dry-run`; check the preset,
   task hash, asset IDs, and effective flags. This neither loads assets
   nor executes a rollout.
3. **Verify assets.** Run `python sagas.py assets SUITE` with the same
   preset selectors and asset/data roots as the evaluation. If files
   are missing, look up their IDs, sizes, and sources in
   `assets/manifest.json`. Use `--fetch` for the requested bundle only;
   estimate the missing download size before a large transfer. Require
   a passing asset check before proceeding.
4. **Execute.** Run the checked command without `--dry-run`, with a fresh
   `--output-dir` under `runs/`. Set `WANDB_MODE=disabled` for benchmark
   runs. Keep the saved task, per-task seed, checkpoint pairing, and
   evaluation flags for a reproduction. Use `JAX_PLATFORM_NAME=cpu`
   for a CPU smoke run and report the backend used.
5. **Verify and report.** Check that the preset's `run.json` records
   `"status": "complete"` and inspect the task outcome in the result file
   below. Report the executed selection, backend, output path, outcome,
   and any failure reason. A completed command or found plan is not
   evidence that the task succeeded.

## Results

Result paths are relative to the preset directory containing `run.json`
inside the chosen output root.

| Suite | Result to inspect | Success criterion |
| --- | --- | --- |
| Navigation | `results/**/overall_ltl_results.csv` | Read `success_rate` for the selected task; the benchmark uses the two-suffix outcome. |
| Overlap | `results/batch_summary.json` | Use the independent `oracle_success` in the selected row of `file_aggregates[].results`. Use planner and runtime outcomes for diagnosis. |
| Scene | `results/stress_suite_summary.json` | Inspect `tasks[]`. For the paper's finite-lasso SR, count a task only when `status` is `ok` and both `runtime_success` and `oracle_success` are true. Divide by the full selected task count, including failures. Report runner errors and the two success fields separately when diagnosing failures. |

Use `python sagas.py summarize` with the same suite and selectors, the original
evaluation root as `--results-dir`, and a new directory under `reports/`.
It requires a complete run on the fixed file. Keep all task outcomes,
including failures; report a subset as a smoke test, not benchmark SR or NCC.
Keep the original `run.json` and summarize with the matching release; the
summary checks its asset IDs, dataset IDs, and asset manifest hash.
Check `summary_protocol.json` before reporting the metrics. A
`modified_preset` summary is a custom evaluation, not a fixed-preset result.

## When Blocked

- **Missing bundle:** Match the exact manifest IDs; Giant-Stitch overlap
  cannot use the navigation Giant-Stitch bundle. Report missing or
  mismatched files instead of swapping checkpoints, graphs, or datasets.
- **Private or unreachable Hub:** Ask the user to authenticate locally
  with an account that has access, or provide the exact local files. Keep
  credentials out of commands, logs, and repository files. Only load
  trusted pickle assets.
- **Installation or execution error:** Read the actual error and the
  effective command from `run.json` or `--dry-run`. Check Python, the
  bundled OGBenchTL import, MuJoCo/EGL, and the LTL2BA executable before
  changing dependencies. Keep the pinned experiment environment intact.
- **Unsuccessful rollout:** Preserve the recorded outcome. Separate
  planner infeasibility, tracking failure, and oracle rejection using the
  available result fields. Report any diagnostic rerun separately.

## Other Workflows

- **Render a rollout:** Use `--render` with a single `--task-id` for Overlap
  or Scene. For Navigation, use the rendering options in
  `implementations/navigation/test_ltl.py`.
- **Edit or diagnose code:** Start at `sagas.py` and use `--dry-run` to
  locate the original entry point under `implementations/`. Run
  `python -B -m unittest discover -s tests` at the repository root for the
  launcher, and from the affected implementation directory for its tests.
- **Use custom inputs:** Select one preset and use `--task-file` for a
  separate task file or `--config` (`{"flags": {...}}`) for allowed preset
  overrides. Use a fresh output directory and report deviations from the
  fixed protocol. These options still verify the manifest assets. For your
  own checkpoints or graphs, follow [ASSETS.md](ASSETS.md#fetch-or-verify)
  and pass explicit asset paths to the original evaluator.
- **Train, build, or generate:** Use the corresponding `sagas.py`
  subcommand when requested. Inspect its `--help` and preview with
  `--dry-run`. Keep newly generated tasks, graphs, and checkpoints separate
  from fixed benchmark inputs and the released asset files.
