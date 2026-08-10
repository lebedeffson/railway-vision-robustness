# Railway Vision Robustness

Reproducible research on person detection, temporal tracking, false-alarm
trade-offs, and track-fragment association in railway video scenes.

## Research status

```text
Research:                 COMPLETED
Final computational stage: B7-SCF
Scientific gate:          SCIENTIFIC_FAIL
Operational gate:         OPERATIONAL_FAIL
Computation state:        FROZEN_AFTER_B7
B8:                       PROHIBITED BY FROZEN PROTOCOL
Test:                     SEALED
Test access count:        0
```

`SCIENTIFIC_FAIL` means that the prospectively defined hypothesis did not pass
all scientific gates. It does not mean that the software failed or that the
experiment is invalid. `OPERATIONAL_FAIL` is a separate finding: the complete
set of predefined operational requirements was not met. Both are final,
reproducible negative results.

After B7, the computation was frozen. The same evidence base must not be used
for B8, new threshold selection, weight changes, consensus tuning, or opening
the sealed test. A future study would require an independent experimental
basis.

## Overview

The project studies whether missed person detections in independent railway
video scenes can be reduced through temporal processing, tracking, and
track-fragment association without causing an unacceptable increase in false
alarms or cross-person merges.

This is one research project. `TNormFilter` is the historical name of an early
diagnostic code module. `Operator Assistant` is the internal name of a later
human-review experiment. B6 and B7 are sequential track-fragment association
stages; they are not separate products or research projects.

The repository is the public software and reproducibility record for the
associated manuscript. It contains source code, frozen protocol definitions,
aggregated results, threshold analyses, trace audits, and a non-autonomous
review demonstrator. It is not a production or safety-certified system.

## Publication

**ИССЛЕДОВАНИЕ КОМПРОМИССА МЕЖДУ ПРОПУСКАМИ И ЛОЖНЫМИ ТРЕВОГАМИ ПРИ
ОБНАРУЖЕНИИ ЛЮДЕЙ В ЖЕЛЕЗНОДОРОЖНЫХ ВИДЕОСЦЕНАХ**

**A STUDY OF THE TRADE-OFF BETWEEN MISSED DETECTIONS AND FALSE ALARMS IN
RAILWAY PERSON DETECTION**

Yuri V. Trofimov · Alexey N. Averkin · Alexey V. Shevchenko · Tatiana V. Kim ·
Alexander D. Lebedev

- Publication status: manuscript
- DOI: pending
- Public article URL: pending
- Research repository:
  [github.com/lebedeffson/railway-vision-robustness](https://github.com/lebedeffson/railway-vision-robustness)
- Public evidence:
  [`artifacts/operator_assistant_b7_public.zip`](artifacts/operator_assistant_b7_public.zip)

No DOI or publisher URL is assigned in the available final manuscript, and no
placeholder DOI has been created.

## Funding and government assignment

> Исследование выполнено в рамках государственного задания Министерства науки
> и высшего образования Российской Федерации, тема № 124112200072-2.

This research was carried out within the framework of the State Assignment of
the Ministry of Science and Higher Education of the Russian Federation, topic
No. 124112200072-2.

Several authors are affiliated with Dubna State University. This statement does
not claim that the repository is legally owned by the university.

## Research question

Can temporal continuity and fragment association recover people missed by a
frame detector while keeping false alarms and erroneous associations between
different people within prospectively fixed limits?

## Research trajectory

| Stage | Research question | Frozen result |
|---|---|---|
| Detection | Can people be detected reliably across independent railway scenes? | A baseline was obtained; a substantial cross-scene gap remained. |
| Pretraining / robustness | Do CrowdHuman pretraining and robustness methods close that gap? | The tested configurations did not fully close it. |
| T-norm diagnostics | Do T-norm features add stable information beyond detector and representation features? | No additional value was confirmed on the primary endpoint. |
| Threshold analysis | Can misses be recovered by lowering the frame-level operating threshold? | Recall increased together with false alarms. |
| Tracking | Does temporal continuity reduce misses? | FN decreased, but false alarms increased sharply. |
| Verification | Can false tracks be removed while keeping recovered observations? | The trade-off remained. |
| B6-FULL | Can a pairwise classifier associate track fragments reliably? | Many true fragment links were missed. |
| B7-FLOW | Does global path optimization improve recovery? | Recovery improved, but cross-person merges increased. |
| B7-HARD | Do hard constraints control aggressive flow linking? | Aggressiveness decreased, but the scientific gates were not met. |
| B7-CONSENSUS | Does perturbation consensus provide a stable acceptable solution? | Merges decreased, but useful links were nearly suppressed. |
| Final | Were all predefined gates met? | No: `SCIENTIFIC_FAIL`, `OPERATIONAL_FAIL`, `FROZEN_AFTER_B7`. |

Historical protocols and results remain in the repository for provenance. They
are completed stages, not open tasks.

## Experimental setting

The railway experiments use RGB material from
[OSDaR23](https://data.fid-move.de/dataset/osdar23). OSDaR23 is obtained
separately under its own distribution conditions; raw images and annotations
are not distributed in this repository. The project treats a grouped scene,
not an adjacent frame or tile, as the independent unit for splits and scene
statistics.

[CrowdHuman](https://www.crowdhuman.org/) was used only in designated
pretraining experiments under its non-commercial research/education terms.
CrowdHuman images, archives, derived datasets, crops, and checkpoints are not
included in the repository or public evidence.

The final B6/B7 association benchmark has the following frozen scale:

```text
Development scenes:         5
Frames:                   479
Person observations:     4943
TrackFragments:           646
Reference PersonEpisodes:  48 (evaluation only)
```

The railway test contains five held-out grouped scenes. It remained sealed
through the final B7 decision. Test access count: `0`.

## Detection, T-norm, and temporal findings

The V8b primary comparison tested whether Product and Łukasiewicz consistency
features improved prediction of `FN/frame` over detector and standard
representation features:

```text
U2 scene-macro MAE: 1.59549
U3 scene-macro MAE: 1.76798
Relative change:   -10.81%
Scene wins:         5/15
```

The added T-norm features worsened the primary MAE in this setting.

Temporal processing consistently recovered some missed observations. Across
the studied variants, Recall increased by approximately `0.13` and FN/frame
fell by approximately `17–19%`. The same variants increased false alarms by
approximately `276–391%`, so the practical deployment gate failed. These
results document an operating-point trade-off, not a deployable safety system.

## Track-fragment association

The B6 audit restored full link-level observability:

```text
TP links:                  44
FP links:                  16
FN links:                 750
True positive pairs:      794
Predicted positive pairs:  60
Model coefficients:        95
Scaler parameters:         95
B6 pseudo-pair rows:     36864
```

The final comparison uses frozen micro metrics and perturbation stability:

| Method | Micro F1 | Cross-person merge | Split recovery | Median ARI |
|---|---:|---:|---:|---:|
| B6-FULL | 0.10304 | 0.26667 | 0.05542 | 0.53446 |
| B7-FLOW | 0.19600 | 0.40000 | 0.11713 | 0.35724 |
| B7-HARD | 0.10514 | 0.27419 | 0.05668 | 0.61004 |
| B7-CONSENSUS | 0.03206 | 0.23529 | 0.01637 | 0.61780 |

### B7-FLOW

Global flow increased Micro F1 from `0.10304` to `0.19600` and split recovery
from `0.05542` to `0.11713`. This confirms that global path optimization can
recover additional correct links. It was not accepted because cross-person
merge increased from `0.26667` to `0.40000`, while median perturbation ARI fell
from `0.53446` to `0.35724`.

### B7-HARD

Hard constraints reduced the aggressiveness of flow, yielding Micro F1
`0.10514`, cross-person merge `0.27419`, split recovery `0.05668`, and median
ARI `0.61004`. The combined improvement was insufficient for the frozen gate.

### B7-CONSENSUS

The primary consensus variant reduced cross-person merge to `0.23529` and
raised median perturbation ARI to `0.61780`, but its Micro F1 fell to `0.03206`
and split recovery to `0.01637`. It improved `0/5` scenes. Its 10th-percentile
ARI was `0.46926`, coverage was `0.90462`, and the prospectively required
median ARI was at least `0.68446`. Consensus was too conservative and
suppressed most useful links.

## Scientific conclusion

The prospectively registered B7 hypothesis was not confirmed. Global flow
improved recovery but worsened cross-person association. Stability consensus
reduced merges but nearly eliminated correct links. None of the variants
simultaneously improved recovery, controlled cross-person merges, improved
cluster stability, and passed all predefined gates.

```text
Scientific decision: FAIL
Operational decision: FAIL
Compute status:       FROZEN_AFTER_B7
B8 allowed:           false
```

The final scientific result is therefore the documented boundary of this
trade-off, not a claim that track association or railway hazard detection has
been solved.

## Reproducibility

The final B7 provenance commit is
[`f25a751c52104b0353995d2206afbe92a4182ff2`](https://github.com/lebedeffson/railway-vision-robustness/commit/f25a751c52104b0353995d2206afbe92a4182ff2).
The computation lock records its immediate predecessor
`484d5eebf1bdc6a02cfeb7f751e81443d881c2f3`. Both remain reachable from
`main`; their history must not be squashed or rebased.

### Evidence replay (no dataset required)

The tracked public bundle contains the frozen aggregated evidence, protocol
locks, hierarchy, manifest, replay audit, and final decision. Verify it with
Python's standard library:

```bash
git clone https://github.com/lebedeffson/railway-vision-robustness.git
cd railway-vision-robustness
python scripts/operator_assistant_b7/verify_public_evidence.py \
  artifacts/operator_assistant_b7_public.zip --json
```

Expected status:

```text
manifest: PASS
b6_audit: PASS
b7_metrics: PASS
replay: PASS
cross_process_determinism: PASS
scientific_gate: FAIL
operational_gate: FAIL
test_status: SEALED
test_access_count: 0
```

The frozen B7 bundle records the original finalization suite (`481 passed`).
After adding repository-publication checks, the complete source-tree suite at
this publication commit is `483 passed`; the scientific result files were not
changed.

### Full reproduction (permitted inputs required)

Full recomputation requires separately obtained OSDaR23 inputs and the frozen
derived B6 evidence identified by the hashes in
[`B7_PROTOCOL_LOCK.json`](protocol/operator_assistant_b7/B7_PROTOCOL_LOCK.json).
The repository never downloads data or model weights implicitly.

```bash
git clone https://github.com/lebedeffson/railway-vision-robustness.git
cd railway-vision-robustness
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

PYTHONPATH="$PWD:$PWD/scripts" .venv/bin/python \
  scripts/operator_assistant_b7/run_b7.py prepare
PYTHONPATH="$PWD:$PWD/scripts" .venv/bin/python \
  scripts/operator_assistant_b7/run_b7.py run
PYTHONPATH="$PWD:$PWD/scripts" .venv/bin/python \
  scripts/operator_assistant_b7/finalize_b7.py
.venv/bin/python -m pytest -q
```

`prepare` validates and locks the already-frozen inputs. It does not authorize
new model training, threshold tuning, B8, or access to the sealed test.

## Public evidence

- Bundle: [`operator_assistant_b7_public.zip`](artifacts/operator_assistant_b7_public.zip)
- SHA-256: `7f4892b170fbb4da9721077f201dbfeeafa5906bfd16bac8745b31b76c3dc09d`
- Sidecar: [`operator_assistant_b7_public.zip.sha256`](artifacts/operator_assistant_b7_public.zip.sha256)
- Manifest: `MANIFEST.sha256` inside the archive

The bundle includes `B7_HIERARCHY.jsonl`, the file referenced by the
cross-process determinism report. It also includes B6 audit files, B7 candidate
and selected edges, model coefficients, scene normalization, pseudo-pair
indices, per-scene/overall results, protocol locks, final decisions, and
figures.

The bundle excludes raw images, video, restricted annotations, model
checkpoints, crops, feature tensors, the sealed test, local absolute paths,
credentials, and internal project memory.

## Repository layout

```text
configs/       frozen experiment configurations
protocol/      immutable protocol locks and final decisions
scripts/       reproducibility, evaluation, audit, and packaging commands
src/           detector, tracking, verification, and review modules
tests/         regression, integrity, leakage, and release checks
artifacts/     compact public evidence release
reports/       technical reports from completed stages
article/       manuscript-support tooling, not the publication itself
```

`README.md` is the public scientific source of truth. `AGENTS.md` contains
development constraints, while `.codex/` stores internal workflow notes.

## Limitations

- The association evaluation contains five development scenes, 48 reference
  person episodes, and 646 fragments; statistical power and visual diversity
  are limited.
- The results concern the studied railway RGB scenes and do not automatically
  transfer to other cameras, stations, routes, weather, or operating regimes.
- The detector used in parts of the development history was not fully
  detector-OOF for every scene.
- No operator or human-factors study was performed. Fewer cards or detections
  do not establish reduced cognitive workload.
- There is no safety certification, production-readiness claim, autonomous
  alarm output, or safety actuation.
- Temporal modes retain a high false-alarm rate.
- The sealed test was not opened; no test-performance claim is made.
- B8 and further tuning on this evidence base are prohibited by the frozen
  protocol.

## Citation

Until a DOI is assigned, cite the manuscript without invented publication
metadata:

```text
Trofimov, Yuri V.; Averkin, Alexey N.; Shevchenko, Alexey V.;
Kim, Tatiana V.; Lebedev, Alexander D.

“Исследование компромисса между пропусками и ложными тревогами при
обнаружении людей в железнодорожных видеосценах”. Manuscript.

Software and evidence: Railway Vision Robustness,
https://github.com/lebedeffson/railway-vision-robustness
```

Machine-readable software citation metadata is available in
[`CITATION.cff`](CITATION.cff). Add the DOI and publisher URL only after they
exist in an official public record.

## License

**License decision required from repository owner.** No software or data
license has been selected automatically. Third-party datasets and pretrained
weights retain their own terms; repository visibility alone does not grant
permission to reuse them.
