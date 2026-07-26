# New railway scenes acquisition

This directory preserves the supplied acquisition starter byte-for-byte under
`starter/` and records the outer ZIP checksum in `SOURCE_PACKAGE.sha256`.

The starter contributes source metadata and tooling only. It does not contribute
accepted scenes and does not authorize training.

`download_verified_open_sources.sh` is the project-side execution wrapper. It:

- downloads only the open RailGoerl24 annotated RGB archive;
- verifies the current official byte count and records SHA-256;
- clones only the public RailEye3D annotation repository;
- uses DNS-over-HTTPS only when the local resolver cannot resolve the official
  RailGoerl24 host, while retaining normal TLS hostname verification;
- preserves failed RailGoerl24 transfers as `.download` evidence and never
  treats them as archives; the official server does not support HTTP Range, so
  a failed transfer must restart from byte zero;
- never downloads RAIL-BENCH, RailEye3D images, RAWPED, or any benchmark test.

The restricted or terms-gated sources require separate owner action. Their
images must not be committed or added to a public bundle.

The 2026-07-26 local execution downloaded the public RailEye3D annotations but
could not complete RailGoerl24 because the official host reset two
non-resumable transfers. See `ACQUISITION_RUNTIME_STATUS.json`. This is an
external acquisition blocker, not a data-audit pass. Acquisition was
subsequently closed as `CLOSED_DATA_UNAVAILABLE`: accepted scenes remain zero,
automatic download/training are disabled, and railway test remains sealed.
