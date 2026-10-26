# mikenavi Strixite fleet source

The qualified source baseline is upstream0.5.0 atb175d27, with four separate
feature commits: native probability/tokenizer API, QSA automatic split workaround,
four evictable turn endpoints with superseded-branch handling, and bounded recent
message/protected image predecessor snapshots. Newer upstream commits are not
silently selected. AGPL-3.0 and upstream notices remain in this repository.

`bash local/build.sh --verify-only` checks source identity. Build with externally
provided pinned STRIX_ROCM_ROOT and STRIX_HIP_LIB and a separate STRIX_BUILD_DIR;
JOBS defaults to1. The existing gfx1151 build uses ROCm10/Python SDK components
and the stddef.h workaround. Other GPUs/toolchains require their own validation.
STRIX_TOKENIZER optionally enables the CPU API assertions with a local tokenizer.

Boundary snapshots are opt-in; the deployment chooses their count, protected
bank and memory cap. Keep explicit Coop available for diagnosis. Update by
merging/rebasing upstream and regression-testing each feature. Tag a checked
commit and deploy exact image digests. This export preserves code bytes, not a
new GPU,512K or task-quality qualification. No weights or private host state are
included. Runtime/configuration/image distribution belongs to deployment tooling.
