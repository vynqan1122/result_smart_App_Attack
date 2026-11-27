# Six-month research and maintenance roadmap

This is a proposed plan, not a funding commitment or a list of completed features. Priorities may change with contributor feedback, available compute and rights clearance.

| Phase | Proposed deliverables | Completion evidence |
| --- | --- | --- |
| Month 1 — Foundation | Review upstream and local code/data rights; improve English documentation and setup guidance; establish repository checks | Provenance inventory, compatible licensing decision, documented checks |
| Month 2 — Reproducibility | Record dependency versions, dataset splits, model metadata, seeds and hardware; establish a small authorized reference run | Environment manifest and a documented reference run |
| Month 3 — Benchmark coverage | Extend controlled evaluations to additional supported datasets and backbones where artifacts and permissions are available | Versioned results, coverage matrix and run logs |
| Month 4 — Reliability | Refactor result collection and validation; add metric and preprocessing unit tests; preserve historical outputs | Tests, schema documentation and comparison report |
| Month 5 — Defensive evaluation | Evaluate defensive robustness measures under consistent conditions; document trade-offs and limitations | Reproducible evaluation report with clean and robustness metrics |
| Month 6 — Research release | Publish reviewed artifacts and accessible tutorials; improve citation, contribution and release processes | Tagged release if rights clearance allows, release notes and archived results |

## Priorities

- Scientific accuracy over inflated metrics or coverage claims.
- Repeatable, controlled experiments and documented limitations.
- Accessible documentation and contributor onboarding.
- Responsible research, provenance and permission compliance.

## Current baseline

The committed appendix CSV contains 74 rows from GTSRB / MobileNetV2 experiments. Training plots cover additional datasets and architectures. Raw JSON outputs and trained models are not tracked in the current snapshot. This roadmap starts from that evidence rather than treating unrecorded experiments as completed.
