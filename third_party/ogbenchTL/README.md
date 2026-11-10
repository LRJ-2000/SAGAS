# OGBenchTL

This snapshot is the `ogbenchTL` 0.1.0 environment used by SAGAS. Its
[package metadata](pyproject.toml) records the fork authors and declares MIT.
The simulation source and MuJoCo resources are retained at their original
paths. Install the local distribution from the SAGAS repository root after
installing the pinned dependencies:

```sh
python -m pip install --no-build-isolation --no-deps ./third_party/ogbenchTL
```

Use `ogbenchTL.make_env_and_datasets` for the navigation environments and
their datasets. See [the SAGAS README](../../README.md) for the evaluation
commands and [ASSETS.md](../../ASSETS.md) for the dataset locations.

## Attribution

The [OGBenchTL fork](https://github.com/Pr0ximah/ogbenchTL) builds on
[OGBench](https://github.com/seohongpark/ogbench). The fork authors are
credited in [pyproject.toml](pyproject.toml); [LICENSE](LICENSE) retains the
OGBench Authors' MIT notice. The original
notices for the
[Universal Robots UR5e](ogbenchTL/manipspace/descriptions/universal_robots_ur5e/LICENSE)
and
[Robotiq 2F-85](ogbenchTL/manipspace/descriptions/robotiq_2f85/LICENSE)
descriptions remain next to their resources. The OGBench and OGBenchTL
sources also credit [D4RL](https://github.com/Farama-Foundation/D4RL),
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium),
[dm_control](https://github.com/google-deepmind/dm_control),
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
[Meta-World](https://github.com/Farama-Foundation/Metaworld), and
[Powderworld](https://github.com/kvfrans/powderworld).

Please cite *OGBench: Benchmarking Offline Goal-Conditioned RL* (ICLR 2025)
and acknowledge the OGBenchTL extension when using this environment.
