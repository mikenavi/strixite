# mikenavi Strixite fleet source

This branch is upstream `main` (base recorded in `local/source.lock.json`) with a
merge of each feature branch, then the fleet files (`local/`, this README,
`AGENTS.md`). Each feature lives on its own branch from upstream `main`, ready for
an upstream pull request:

| Branch | Feature | Upstream issue |
|---|---|---|
| `feat/token-probabilities-api` | raw/chat token probabilities and tokenizer endpoints | #34 |
| `fix/qsa-split-deep-batches` | QSA automatic split for deep serving batches | #31 |
| `feat/turn-retention-superseded-branches` | four evictable turn endpoints, superseded branches, CPU tests | #32 |
| `feat/boundary-snapshots` | bounded recent message and protected image predecessor snapshots (on the API branch) | #33 |
| `fix/optional-format-include` | `<format>` only where libstdc++ has it, so the GCC 12 SDK image builds upstream | — |

Update: merge upstream `main` into each feature branch and fix it there, then merge
upstream `main` and the feature branches into this branch. A feature that upstream
accepts drops out of the list. Newer upstream commits are not silently selected.
AGPL-3.0 and upstream notices remain in this repository.

`bash local/build.sh --verify-only` checks source identity. Build with externally
provided pinned STRIX_ROCM_ROOT and STRIX_HIP_LIB and a separate STRIX_BUILD_DIR;
JOBS defaults to1. The existing gfx1151 build uses ROCm10/Python SDK components
and the stddef.h workaround. Other GPUs/toolchains require their own validation.
STRIX_TOKENIZER optionally enables the CPU API assertions with a local tokenizer.

Boundary snapshots are opt-in; the deployment chooses their count, protected
bank and memory cap. Keep explicit Coop available for diagnosis. Regression-test
each feature after an update. Tag a checked
commit and deploy exact image digests. This export preserves code bytes, not a
new GPU,512K or task-quality qualification. No weights or private host state are
included. Runtime/configuration/image distribution belongs to deployment tooling.

Run `bash local/test-cache.sh /path/to/cache-check` for the fleet turn/cache
pressure/superseded/config regressions, using an external C++20 host compiler.
