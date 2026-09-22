# Codex Reply Archive — Stage 06 ETTh1 Full Benchmark

- Archived: 2026-09-22 18:50 CST
- Request: Strictly execute `CURRENT_STAGE.md` Stage 06 only; preserve the frozen TSLib branch/SHA and Stage 05 alpha choices; complete both the selected-model Track A and shared-alpha mechanism Track B; write and push all required bridge artifacts; do not execute Stage 07.
- Status: **PASS**

## Final result

- TSLib remained `codex/saema-v1 @ 91735a83b27ce1bf766df76a953cf86ef1e7500d` with no source changes.
- The full matrix is complete: 48/48 cells, comprising 33 new runs, 12 audited Stage 00 baseline reuses, and 3 audited Stage 05 reuses.
- Frozen alphas were not retuned: Fixed EMA `0.02`; Scale-Aware EMA `0.05`; shared-alpha control `0.05`.
- Track A four-horizon mean MSE/MAE:
  - Moving average: `0.450142 / 0.440632`
  - Fixed EMA 0.02: `0.444800 / 0.437647`
  - SAEMA 0.05: `0.451193 / 0.440575`
- Track A MSE wins: Fixed EMA 0.02 vs baseline `8/12`; SAEMA 0.05 vs baseline `6/12`; SAEMA 0.05 vs Fixed EMA 0.02 `6/12`.
- Track B, with alpha fixed at 0.05: SAEMA had lower MSE in `4/12` cells and lower MAE in `4/12`; its mean MSE was higher at all `4/4` horizons. Mechanism direction: **CONSISTENT_NEGATIVE**.
- The primary mechanism evidence does not support proceeding to the multi-dataset pilot. Stage 07 was not started.

## Files added

- [Main Stage 06 report](results/06_etth1_full_benchmark.md)
- [Raw 48-cell table](results/data/06_etth1_full_benchmark.csv)
- [Track A summary](results/data/06_trackA_selected_models_summary.csv)
- [Track B same-alpha summary](results/data/06_trackB_same_alpha_mechanism.csv)
- [Paired deltas](results/data/06_paired_deltas.csv)
- This unique reply archive

## Verification

- 33/33 new commands exited successfully and completed 10 epochs.
- 48/48 checkpoints were present, independently reloadable, and finite.
- All 36 available result-array triplets were present, shape-correct, metric-consistent, and finite; the 12 Stage 00 cells were re-audited from their preserved logs and checkpoints.
- An independent CSV audit confirmed 48 unique config × horizon × seed rows, exactly 12 rows per config, and the required 12 Track B pairs.
- Stage 05 selection artifact SHA-256 remained `8d24db3196126c02146117e9602e31e04682988651c4b593c4086c9c1628473e`.
- Required CSV SHA-256 values are recorded in the main report.
- TSLib final tracked status was clean and `git diff --check` passed.

The bridge artifacts were committed and pushed to the default branch. Execution stopped after Stage 06.
