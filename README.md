# SAGAS: Semantic-Aware Graph-Assisted Stitching for Offline Temporal Logic Planning

**Ruijia Liu, Ancheng Hou, and Xiang Yin** | [NeurIPS 2026](https://neurips.cc/virtual/2026/poster/152343)

[Paper (arXiv)](https://arxiv.org/abs/2512.00775) |
[Project page](https://cps-sjtu.github.io/SAGAS/) |
[Pretrained assets](https://huggingface.co/lrj-2000/SAGAS)

SAGAS uses fixed, task-agnostic offline trajectories to plan and execute
new robot tasks specified in linear temporal logic (LTL). It reuses the
same reachability graph and policy across tasks without retraining.

Start with [one AntMaze task](#run-one-task). To run it through a coding agent,
point the agent to [AGENTS.md](AGENTS.md) and give it a task:

> Read AGENTS.md and run AntMaze Medium-Navigate Easy task 0. Use the saved
> preset and report the result file and rollout outcome.

## Install

Use **Linux x86-64 with Python 3.9**, EGL/OpenGL libraries for MuJoCo, and
[LTL2BA 1.3](https://lsv.ens-paris-saclay.fr/~gastin/ltl2ba/). Put the
`ltl2ba` executable on your PATH or pass its path with `--ltl2ba`.

From the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --no-deps -r requirements.txt
python -m pip install --no-build-isolation --no-deps ./third_party/ogbenchTL
```

These commands install the bundled OGBenchTL environment and the versions
used in the experiments. For GPU runs on a machine with a compatible NVIDIA
driver, add `python -m pip install --no-deps -r requirements-cuda12.txt`.

## Run One Task

Run **AntMaze Medium-Navigate, Easy, task 0** on CPU. The checkpoints, graph,
and datasets for this example total about 414 MB.

**Get the files.** The asset command fetches missing files and checks their
hashes. If the pinned Hub revision is restricted, you need an account with
access. Install the client below, then run `hf auth login` before fetching.

```bash
python -m pip install -c requirements.txt 'huggingface_hub==0.36.0'
python sagas.py assets navigation --env antmaze-medium-navigate-v0 \
  --difficulty easy --fetch
```

Already have the files? Skip the Hub client installation and omit `--fetch`
to verify them locally. The defaults are `assets/` for checkpoints and graphs
and `~/.ogbench/data` for datasets. For custom locations, pass the same
`--assets-dir` and `--data-dir` to `assets` and `evaluate`.
See [ASSETS.md](ASSETS.md) for the available bundles.

**Run the task.** Give each run a new output directory.

```bash
JAX_PLATFORM_NAME=cpu python sagas.py evaluate navigation \
  --env antmaze-medium-navigate-v0 --difficulty easy --task-id 0 \
  --output-dir runs/antmaze-example
```

Under `runs/antmaze-example/`, `run.json` records the task and settings.
Check `overall_ltl_results.csv` for the rollout outcome.

## Try Another Suite

| Suite | Tasks included | How to select a run |
| --- | ---: | --- |
| Navigation | 4,200 AntMaze and HumanoidMaze tasks | `--env` and `--difficulty` |
| Overlap | 320 tasks with overlapping regions | `--env` and `--shard` |
| Scene | 120 manipulation tasks | No additional selector required |

The [asset list](ASSETS.md) shows which settings have pretrained bundles.
Large-Explore overlap shares its navigation bundle. Giant-Stitch overlap
and the remaining navigation settings need matching local assets.

To see a manipulation rollout, run Scene task 0 with video rendering:

```bash
python sagas.py assets scene --fetch
python sagas.py evaluate scene --task-id 0 --render --render-every 10 \
  --output-dir runs/scene-example
```

## Reproduce a Benchmark Setting

To evaluate all 100 saved tasks for AntMaze Medium-Navigate Easy, omit
`--task-id`. Then summarize the run to get the success rate and normalized
capped cost:

```bash
python sagas.py evaluate navigation --env antmaze-medium-navigate-v0 \
  --difficulty easy --output-dir runs/antmaze-easy
python sagas.py summarize navigation --env antmaze-medium-navigate-v0 \
  --difficulty easy --results-dir runs/antmaze-easy \
  --output-dir reports/antmaze-easy
```

Summaries use the complete fixed task file, including failed rollouts.
The asset IDs, dataset IDs, and manifest hash recorded in `run.json` must
match the release used to summarize the run.
Check `summary_protocol.json` for any changes to the preset evaluation
parameters. Scene reports finite-lasso success only when both the runtime
monitor and the independent verifier succeed on the same task.

## Train and Build

This example trains the temporal-distance representation (TD) and low-level
policy and builds a graph for AntMaze Medium-Navigate. Run it from the
repository root after [installation](#install). The
[navigation presets](configs/training.json) cover AntMaze and HumanoidMaze.
Use a fresh output directory for each run.

**1. Download the offline data.** This fetches the training and validation
datasets without downloading pretrained checkpoints. Skip the Python command
if the matching datasets are already in `DATA_DIR`.

```bash
ENV_NAME=antmaze-medium-navigate-v0
DATA_DIR="$HOME/.ogbench/data"
python -c 'import sys; from ogbenchTL import download_datasets; download_datasets([sys.argv[1]], sys.argv[2])' \
  "$ENV_NAME" "$DATA_DIR"
```

**2. Train the TD representation.** The example preset saves a checkpoint at
step 1,000,000. Use the resulting TD checkpoint for both of the next commands.

```bash
python sagas.py train tdr --env "$ENV_NAME" --data-dir "$DATA_DIR" \
  --output-dir runs/antmaze-tdr
TDR_CHECKPOINT="$(find "$PWD/runs/antmaze-tdr/results" -type f -name params_1000000.pkl -print -quit)"
```

**3. Train the policy and build the graph.** These two jobs share the TD
checkpoint; neither requires the other job's output. To skip step 2, set
`TDR_CHECKPOINT` to a compatible existing checkpoint file.

```bash
python sagas.py train policy --env "$ENV_NAME" --data-dir "$DATA_DIR" \
  --tdr-checkpoint "$TDR_CHECKPOINT" --output-dir runs/antmaze-policy
python sagas.py build navigation --env "$ENV_NAME" --data-dir "$DATA_DIR" \
  --tdr-checkpoint "$TDR_CHECKPOINT" --output-dir runs/antmaze-graph
```

Checkpoints (`params_<step>.pkl`) and graphs (`keygraph.pkl`) are saved in
run-specific subdirectories under the corresponding `results/` directory.
Add `--dry-run` to any training or build command to preview its inputs and
flags without starting the job.

**Rebuild a Scene graph.** With the official Scene datasets and the joint
checkpoint from the [Scene bundle](ASSETS.md) available locally, run
`python sagas.py build scene --output-dir runs/scene-graph`. Use
`--tdr-checkpoint` to supply a compatible checkpoint explicitly. The release
does not include a Scene training preset.

**Evaluate new assets.** The unified evaluation command checks the hashes of
the released assets. For a new navigation graph or policy, use the original
[navigation evaluator](implementations/navigation/evaluate_ltl.py) with
`--keygraph_path` and `--policy_path`. Preview the full-file preset first:

```bash
python sagas.py evaluate navigation --env "$ENV_NAME" --difficulty easy \
  --data-dir "$DATA_DIR" --output-dir runs/antmaze-custom-eval --dry-run
```

Replace the asset paths in the printed `command` array and run that command
from its reported `cwd`. Set
`SAGAS_DATA_DIR` and `SAGAS_LTL2BA` to the data directory and translator
path when invoking the evaluator directly. Report these runs separately from
the fixed benchmark.

See `python sagas.py generate --help` to generate new task files.

## Citation

Cite the [arXiv preprint](https://arxiv.org/abs/2512.00775) using the following BibTeX:

```bibtex
@misc{liu2026sagassemanticawaregraphassistedstitching,
  title         = {SAGAS: Semantic-Aware Graph-Assisted Stitching for Offline Temporal Logic Planning},
  author        = {Ruijia Liu and Ancheng Hou and Xiang Yin},
  year          = {2026},
  eprint        = {2512.00775},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2512.00775}
}
```

## License

SAGAS code is released under the [MIT License](LICENSE).
See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream code and resource notices.
