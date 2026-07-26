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
- never downloads RAIL-BENCH, RailEye3D images, RAWPED, or any benchmark test.

The restricted or terms-gated sources require separate owner action. Their
images must not be committed or added to a public bundle.
