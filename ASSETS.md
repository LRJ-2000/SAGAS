# Assets and Runtime

[The manifest](assets/manifest.json) records each file's asset ID, role,
size, SHA-256 and relative destination. These are the first-release SAGAS
bundles:

| Suite | Environment | Checkpoint/graph files |
| --- | --- | ---: |
| Navigation | `antmaze-medium-navigate-v0` | 3 |
| Navigation | `antmaze-large-explore-v0` | 3 |
| Navigation | `antmaze-giant-stitch-v0` | 3 |
| Navigation | `humanoidmaze-medium-stitch-v0` | 3 |
| Navigation | `humanoidmaze-large-navigate-v0` | 3 |
| Scene | `scene-play-v0` | 2 |

Each navigation bundle has a TD checkpoint, a low-level policy checkpoint
and a SAGAS graph with state support. Scene has one joint checkpoint and one
graph. The 17 files total 933,767,906 bytes, excluding datasets. The
bundles are for 1,500 navigation tasks and 120 Scene tasks. They also
satisfy the Large-Explore overlap preset, which shares that navigation
pairing. Giant-Stitch overlap requires different assets; do not substitute
the navigation bundle.

## Fetch or Verify

```sh
python sagas.py assets navigation --env antmaze-medium-navigate-v0 \
  --difficulty easy --assets-dir "$SAGAS_ASSETS" --data-dir "$OGBENCH_DATA"
```

Add `--fetch` to download missing files when their sources are available.
The datasets come from the official OGBench data server. Pretrained SAGAS
files use the `lrj-2000/SAGAS` model repository on Hugging Face and the
full commit SHA in the manifest's `hub.revision` field. Install
`huggingface_hub` for Hub downloads. For a private revision, use the Hub's
local login; credentials do not belong in the manifest or commands.

Files outside `hub.assets` are not part of the pretrained release. The
fetcher reports an unavailable bundle before downloading part of it and
never overwrites an existing file. The Scene joint checkpoint also has
the exact GAS upstream reference recorded in the manifest. The graphs
retain sampled OGBench states for semantic grounding; the original dataset
files are downloaded separately.

For other settings, use a trusted matching local asset set, or build new
assets using the [Train and Build](README.md#train-and-build) guide.
Use the original evaluator with explicit paths for newly built assets that
do not match the historical manifest. **Only load trusted pickle files.**

## Environment and Translator

Install the frozen OGBenchTL source from
[third_party/ogbenchTL](third_party/ogbenchTL) as shown in the README.
[assets/runtime.json](assets/runtime.json) identifies the captured source,
the packaged documentation changes, and the recorded Linux translator
binary hash. The translator is not bundled: install
[LTL2BA 1.3](https://lsv.ens-paris-saclay.fr/~gastin/ltl2ba/) and supply
`--ltl2ba` (or set `SAGAS_LTL2BA`). The launcher can also find `ltl2ba`
on PATH.
