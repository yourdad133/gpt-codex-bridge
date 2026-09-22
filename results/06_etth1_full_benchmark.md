# Stage 06 — ETTh1 full benchmark: frozen-model generalization and same-alpha mechanism control

Status: **PASS**

- TSLib branch: `codex/saema-v1`
- TSLib SHA: `91735a83b27ce1bf766df76a953cf86ef1e7500d`
- Bridge branch at start: `main`
- Bridge HEAD at report generation: `3316cdf92824652078c2c93f96f082065970bcda`
- Matrix complete: **48/48**
- Newly executed: **33**; reused Stage 00: **12**; reused Stage 05: **3**
- Stage 07: **NOT EXECUTED**

Stage 06 uses two deliberately separate analyses. Track A compares the three frozen, validation-selected final configurations. Track B holds `alpha_base=0.05` fixed and changes only fixed versus scale-aware coefficient conversion; it is the mechanism attribution analysis.

## Environment / Git

- Remote host: `ps`
- Worktree: `/home/ouyangguanghong/projects/Time-Series-Library-saema-v1`
- Python: `/home/ouyangguanghong/miniconda3/envs/tslib/bin/python3.11`
- PyTorch: `2.5.1+cu121`; CUDA available: `True`; visible devices during audit: `2`
- GPUs: `0, NVIDIA GeForce RTX 4090 D, 570.211.01, 24564 MiB ; 1, NVIDIA GeForce RTX 4090 D, 570.211.01, 24564 MiB`
- New-run window: `2026-09-22T18:01:57+08:00` to `2026-09-22T18:46:00+08:00`
- Pre-run and post-run gates: exact branch/SHA, empty tracked status, and `git diff --check` PASS.

Both physical GPUs already hosted an unrelated Python process using about 746 MiB with variable utilization before Stage 06. Track B pairs were launched simultaneously and C/D GPU assignments alternated by cell. Runtime is retained for audit only and is not used to rank methods.

## Frozen hyperparameters from Stage 05

- Fixed EMA selected alpha: **0.02**.
- Scale-Aware EMA selected alpha: **0.05**.
- SAEMA 0.05 derived scale coefficients (k=0..3): **0.050000, 0.097500, 0.18549375, 0.3365795687**; every SAEMA log reported this frozen schedule.
- Same-alpha mechanism control: Fixed EMA **0.05** versus SAEMA **0.05**.
- No alpha was added, removed, reselected, or changed after observing Stage 06 validation/test results.

## Reuse audit

The reuse gate was checked before launching any new run:

| Criterion | Stage 00 / 05 source | Stage 06 frozen contract | Verdict |
|---|---|---|---|
| Dataset / split | same ETTh1 file and loader split | ETTh1, features=M | PASS |
| Lengths | seq=96, label=0; matching pred_len | seq=96, label=0; 96/192/336/720 | PASS |
| Architecture | TimeMixer, e_layers=2, 7→7, d_model=16, d_ff=32 | identical | PASS |
| Optimization | Adam path, MSE, lr=0.01, type1 | identical | PASS |
| Training | 10 epochs, patience=10, batch=128, workers=10 | identical | PASS |
| Multiscale | 3 layers, window=2, avg | identical | PASS |
| Channel / norm | channel_independence=1, use_norm=1 | identical | PASS |
| Seed / loader RNG | frozen deterministic seed and DataLoader logic | same frozen parent logic | PASS |
| Moving average | window=25 | window=25 | PASS |
| Integrated legacy path | Stage 03 bitwise-exact; Stage 05 metric reproduction | unchanged at frozen SHA | PASS |

Accordingly all 12 Stage 00 moving-average cells were reused as one indivisible set. Their historical wall-clock is recorded as NA and is not compared with current EMA runtime. The exact Stage 00 HEAD is recorded in the raw table; its audited dirty runtime state and later clean baseline-freeze provenance remain documented in Stage 00/02.

The three Stage 05 overlap cells also passed exact command/config, log, checkpoint, final metric, best-validation, frozen-SHA, and finite-artifact checks:

| Config | Cell | Source log | MSE / MAE | Best val / epoch | Verdict |
|---|---|---|---|---|---|
| B_fixed_selected_a0p02 | 96 / 2021 | Stage05Alpha_02_ema_a0p02_ETTh1_96_seed2021.log | 0.376850455999374 / 0.397125661373138 | 0.6844822 / 4 | PASS |
| C_saema_selected_a0p05 | 96 / 2021 | Stage05Alpha_05_saema_a0p05_ETTh1_96_seed2021.log | 0.373896032571793 / 0.39574533700943 | 0.6875293 / 4 | PASS |
| D_fixed_control_a0p05 | 96 / 2021 | Stage05Alpha_04_ema_a0p05_ETTh1_96_seed2021.log | 0.377866417169571 / 0.397137433290482 | 0.6907149 / 4 | PASS |

## Experiment matrix

| Config | Method | alpha | Cells | Reused | New | Role |
|---|---|---|---|---|---|---|
| A_moving_avg | moving_avg | — | 12 | 12 | 0 | Track A baseline |
| B_fixed_selected_a0p02 | Fixed EMA | 0.02 | 12 | 1 | 11 | Track A selected |
| C_saema_selected_a0p05 | SAEMA | 0.05 | 12 | 1 | 11 | Track A selected + Track B |
| D_fixed_control_a0p05 | Fixed EMA | 0.05 | 12 | 1 | 11 | Track B control |

All non-method settings stayed fixed: ETTh1, `seq_len=96`, `label_len=0`, four prescribed horizons, seeds 2021/2022/2023, formal 10-epoch training, and the Stage 00 architecture/optimizer/data contract.

## Run completion / finite check

- 33/33 new commands exited with code 0 and completed all 10 epochs.
- 48/48 rows have a saved best checkpoint and a final full-test result after the runner reloaded that checkpoint.
- 48/48 checkpoints were independently loaded on CPU and all state tensors were finite.
- 36/36 target-worktree result triplets (`metrics.npy`, `pred.npy`, `true.npy`) were present, shape-correct, metric-consistent, and finite. Stage 00 arrays were already documented as unavailable; its 12 logs and checkpoints were re-audited.
- No NaN/Inf, shape error, device error, checkpoint failure, or incomplete log was found.

## Raw result integrity

Every seed-level observation is retained in `results/data/06_etth1_full_benchmark.csv`; the table below is a report copy of all 48 cells.

| Config | H | Seed | Best val (ep) | Test MSE | Test MAE | GPU | Source |
|---|---|---|---|---|---|---|---|
| A_moving_avg | 96 | 2021 | 0.6861665 (7) | 0.376217573881149 | 0.399332642555237 | GPU 1 | Stage 00 reused |
| A_moving_avg | 96 | 2022 | 0.6862650 (5) | 0.37336328625679 | 0.393358826637268 | GPU 1 | Stage 00 reused |
| A_moving_avg | 96 | 2023 | 0.6739254 (3) | 0.382440328598022 | 0.39885476231575 | GPU 1 | Stage 00 reused |
| A_moving_avg | 192 | 2021 | 0.9735200 (1) | 0.447378516197205 | 0.432037770748138 | GPU 1 | Stage 00 reused |
| A_moving_avg | 192 | 2022 | 0.9757308 (4) | 0.432128757238388 | 0.431674242019653 | GPU 1 | Stage 00 reused |
| A_moving_avg | 192 | 2023 | 0.9931056 (10) | 0.428995341062546 | 0.432160556316376 | GPU 1 | Stage 00 reused |
| A_moving_avg | 336 | 2021 | 1.3172755 (1) | 0.497218549251556 | 0.461006879806519 | GPU 1 | Stage 00 reused |
| A_moving_avg | 336 | 2022 | 1.3127393 (1) | 0.470660090446472 | 0.44471001625061 | GPU 1 | Stage 00 reused |
| A_moving_avg | 336 | 2023 | 1.3124551 (1) | 0.474299252033234 | 0.449385017156601 | GPU 1 | Stage 00 reused |
| A_moving_avg | 720 | 2021 | 1.5890277 (1) | 0.493089079856873 | 0.475488007068634 | GPU 1 | Stage 00 reused |
| A_moving_avg | 720 | 2022 | 1.5701961 (2) | 0.519978761672974 | 0.489870339632034 | GPU 1 | Stage 00 reused |
| A_moving_avg | 720 | 2023 | 1.5843962 (1) | 0.50593513250351 | 0.479701906442642 | GPU 1 | Stage 00 reused |
| B_fixed_selected_a0p02 | 96 | 2021 | 0.6844822 (4) | 0.376850455999374 | 0.397125661373138 | GPU 0 | Stage 05 reused |
| B_fixed_selected_a0p02 | 96 | 2022 | 0.6913422 (3) | 0.37557178735733 | 0.394205510616302 | GPU 0 | Stage 06 new |
| B_fixed_selected_a0p02 | 96 | 2023 | 0.6864606 (5) | 0.376869440078735 | 0.39528477191925 | GPU 1 | Stage 06 new |
| B_fixed_selected_a0p02 | 192 | 2021 | 0.9682195 (4) | 0.424509704113007 | 0.426906764507294 | GPU 0 | Stage 06 new |
| B_fixed_selected_a0p02 | 192 | 2022 | 0.9712348 (4) | 0.430106908082962 | 0.429674327373505 | GPU 1 | Stage 06 new |
| B_fixed_selected_a0p02 | 192 | 2023 | 0.9848701 (10) | 0.42610689997673 | 0.428973615169525 | GPU 0 | Stage 06 new |
| B_fixed_selected_a0p02 | 336 | 2021 | 1.3133250 (1) | 0.488452762365341 | 0.457583159208298 | GPU 1 | Stage 06 new |
| B_fixed_selected_a0p02 | 336 | 2022 | 1.3051882 (1) | 0.473294675350189 | 0.44321683049202 | GPU 0 | Stage 06 new |
| B_fixed_selected_a0p02 | 336 | 2023 | 1.3010470 (1) | 0.475750476121902 | 0.449468046426773 | GPU 1 | Stage 06 new |
| B_fixed_selected_a0p02 | 720 | 2021 | 1.5847198 (1) | 0.48985156416893 | 0.474126189947128 | GPU 0 | Stage 06 new |
| B_fixed_selected_a0p02 | 720 | 2022 | 1.5864140 (1) | 0.497850179672241 | 0.478335410356522 | GPU 1 | Stage 06 new |
| B_fixed_selected_a0p02 | 720 | 2023 | 1.5816798 (1) | 0.502384185791016 | 0.476865470409393 | GPU 0 | Stage 06 new |
| C_saema_selected_a0p05 | 96 | 2021 | 0.6875293 (4) | 0.373896032571793 | 0.39574533700943 | GPU 0 | Stage 05 reused |
| C_saema_selected_a0p05 | 96 | 2022 | 0.6965093 (2) | 0.396558165550232 | 0.404853612184525 | GPU 0 | Stage 06 new |
| C_saema_selected_a0p05 | 96 | 2023 | 0.6855280 (3) | 0.375583410263062 | 0.395928382873535 | GPU 1 | Stage 06 new |
| C_saema_selected_a0p05 | 192 | 2021 | 0.9750663 (1) | 0.453311562538147 | 0.434914499521255 | GPU 0 | Stage 06 new |
| C_saema_selected_a0p05 | 192 | 2022 | 0.9818384 (1) | 0.454370439052582 | 0.438576906919479 | GPU 1 | Stage 06 new |
| C_saema_selected_a0p05 | 192 | 2023 | 0.9907334 (10) | 0.426603674888611 | 0.430846780538559 | GPU 0 | Stage 06 new |
| C_saema_selected_a0p05 | 336 | 2021 | 1.3274023 (1) | 0.497672915458679 | 0.462675034999847 | GPU 1 | Stage 06 new |
| C_saema_selected_a0p05 | 336 | 2022 | 1.3182979 (1) | 0.470789134502411 | 0.44365006685257 | GPU 0 | Stage 06 new |
| C_saema_selected_a0p05 | 336 | 2023 | 1.3070853 (1) | 0.473091572523117 | 0.447983652353287 | GPU 1 | Stage 06 new |
| C_saema_selected_a0p05 | 720 | 2021 | 1.5977766 (1) | 0.497598528862 | 0.477667510509491 | GPU 0 | Stage 06 new |
| C_saema_selected_a0p05 | 720 | 2022 | 1.5723027 (2) | 0.495793014764786 | 0.479694247245789 | GPU 1 | Stage 06 new |
| C_saema_selected_a0p05 | 720 | 2023 | 1.5832045 (1) | 0.499050498008728 | 0.474363297224045 | GPU 0 | Stage 06 new |
| D_fixed_control_a0p05 | 96 | 2021 | 0.6907149 (4) | 0.377866417169571 | 0.397137433290482 | GPU 0 | Stage 05 reused |
| D_fixed_control_a0p05 | 96 | 2022 | 0.6932668 (3) | 0.377362430095673 | 0.396721929311752 | GPU 1 | Stage 06 new |
| D_fixed_control_a0p05 | 96 | 2023 | 0.6772970 (3) | 0.378625839948654 | 0.398094117641449 | GPU 0 | Stage 06 new |
| D_fixed_control_a0p05 | 192 | 2021 | 0.9758779 (1) | 0.451054006814957 | 0.433939188718796 | GPU 1 | Stage 06 new |
| D_fixed_control_a0p05 | 192 | 2022 | 0.9768373 (4) | 0.430685788393021 | 0.431487023830414 | GPU 0 | Stage 06 new |
| D_fixed_control_a0p05 | 192 | 2023 | 0.9769452 (10) | 0.42453870177269 | 0.4284488260746 | GPU 1 | Stage 06 new |
| D_fixed_control_a0p05 | 336 | 2021 | 1.3123343 (1) | 0.489607930183411 | 0.457280278205872 | GPU 0 | Stage 06 new |
| D_fixed_control_a0p05 | 336 | 2022 | 1.3086876 (1) | 0.470833033323288 | 0.442294389009476 | GPU 1 | Stage 06 new |
| D_fixed_control_a0p05 | 336 | 2023 | 1.3026559 (1) | 0.473850965499878 | 0.44828525185585 | GPU 0 | Stage 06 new |
| D_fixed_control_a0p05 | 720 | 2021 | 1.5897030 (1) | 0.49070817232132 | 0.475344121456146 | GPU 1 | Stage 06 new |
| D_fixed_control_a0p05 | 720 | 2022 | 1.5836200 (2) | 0.489835500717163 | 0.4759541451931 | GPU 0 | Stage 06 new |
| D_fixed_control_a0p05 | 720 | 2023 | 1.5824729 (1) | 0.494191348552704 | 0.474864989519119 | GPU 1 | Stage 06 new |

## Track A — Selected-model performance

Track A compares A moving average, B Fixed EMA 0.02, and C SAEMA 0.05 as frozen final model configurations. Because B and C use different alpha values, C−B is not a pure scale-aware mechanism effect.

## Track A — Per-horizon mean ± std

| H | Config | MSE mean ± sample std | MAE mean ± sample std | ΔMSE vs A | ΔMAE vs A |
|---|---|---|---|---|---|
| 96 | A_moving_avg | 0.377340 ± 0.004642 | 0.397182 ± 0.003320 | +0.000% | +0.000% |
| 96 | B_fixed_selected_a0p02 | 0.376431 ± 0.000744 | 0.395539 ± 0.001477 | -0.241% | -0.414% |
| 96 | C_saema_selected_a0p05 | 0.382013 ± 0.012625 | 0.398842 ± 0.005207 | +1.238% | +0.418% |
| 192 | A_moving_avg | 0.436168 ± 0.009835 | 0.431958 ± 0.000253 | +0.000% | +0.000% |
| 192 | B_fixed_selected_a0p02 | 0.426908 ± 0.002883 | 0.428518 ± 0.001439 | -2.123% | -0.796% |
| 192 | C_saema_selected_a0p05 | 0.444762 ± 0.015734 | 0.434779 ± 0.003867 | +1.970% | +0.653% |
| 336 | A_moving_avg | 0.480726 ± 0.014398 | 0.451701 ± 0.008392 | +0.000% | +0.000% |
| 336 | B_fixed_selected_a0p02 | 0.479166 ± 0.008136 | 0.450089 ± 0.007203 | -0.325% | -0.357% |
| 336 | C_saema_selected_a0p05 | 0.480518 ± 0.014901 | 0.451436 ± 0.009971 | -0.043% | -0.059% |
| 720 | A_moving_avg | 0.506334 ± 0.013449 | 0.481687 ± 0.007394 | +0.000% | +0.000% |
| 720 | B_fixed_selected_a0p02 | 0.496695 ± 0.006346 | 0.476442 ± 0.002136 | -1.904% | -1.089% |
| 720 | C_saema_selected_a0p05 | 0.497481 ± 0.001632 | 0.477242 ± 0.002691 | -1.749% | -0.923% |

Four-horizon mean of the four per-horizon 3-seed means:

| Config | Average MSE | Average MAE |
|---|---|---|
| A_moving_avg | 0.450142 | 0.440632 |
| B_fixed_selected_a0p02 | 0.444800 | 0.437647 |
| C_saema_selected_a0p05 | 0.451193 | 0.440575 |

Because the design is balanced at exactly three seeds per horizon, the 12-cell pooled descriptive mean is numerically equal to this mean-of-horizon-means, but the definitions remain distinct.

## Track A — 12-cell paired analysis

| Comparison (right − left) | Right lower / higher / tie MSE | Mean / median ΔMSE | Mean relative ΔMSE | Right lower / higher / tie MAE | Mean / median ΔMAE | Mean relative ΔMAE |
|---|---|---|---|---|---|---|
| B − A | 8 / 4 / 0 | -0.005342 / -0.003063 | -1.122% | 10 / 2 / 0 | -0.002985 / -0.002522 | -0.659% |
| C − A | 6 / 6 / 0 | +0.001051 / -0.000539 | +0.374% | 7 / 5 / 0 | -0.000057 / -0.001187 | +0.028% |
| C − B | 6 / 6 / 0 | +0.006393 / -0.000395 | +1.526% | 3 / 9 / 0 | +0.002928 / +0.001616 | +0.691% |

## Track B — Same-alpha mechanism control

Track B compares D Fixed EMA 0.05 with C SAEMA 0.05. The paired delta convention is `SAEMA − Fixed`; negative values favor scale-aware conversion.

## Track B — Per-horizon mean ± std

| H | Fixed MSE ± sd | SAEMA MSE ± sd | Mean paired ΔMSE | Rel ΔMSE | Fixed MAE ± sd | SAEMA MAE ± sd | Mean paired ΔMAE | Rel ΔMAE | SAEMA MSE wins |
|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.377952 ± 0.000636 | 0.382013 ± 0.012625 | +0.004061 | +1.074% | 0.397318 ± 0.000704 | 0.398842 ± 0.005207 | +0.001525 | +0.384% | 2/3 |
| 192 | 0.435426 ± 0.013879 | 0.444762 ± 0.015734 | +0.009336 | +2.144% | 0.431292 ± 0.002750 | 0.434779 ± 0.003867 | +0.003488 | +0.809% | 0/3 |
| 336 | 0.478097 ± 0.010082 | 0.480518 ± 0.014901 | +0.002421 | +0.506% | 0.449287 ± 0.007543 | 0.451436 ± 0.009971 | +0.002150 | +0.478% | 2/3 |
| 720 | 0.491578 ± 0.002305 | 0.497481 ± 0.001632 | +0.005902 | +1.201% | 0.475388 ± 0.000546 | 0.477242 ± 0.002691 | +0.001854 | +0.390% | 0/3 |

## Track B — 12-cell paired mechanism analysis

| H | Seed | Fixed MSE | SAEMA MSE | C−D MSE | Fixed MAE | SAEMA MAE | C−D MAE |
|---|---|---|---|---|---|---|---|
| 96 | 2021 | 0.377866417169571 | 0.373896032571793 | -0.003970385 | 0.397137433290482 | 0.39574533700943 | -0.001392096 |
| 96 | 2022 | 0.377362430095673 | 0.396558165550232 | +0.019195735 | 0.396721929311752 | 0.404853612184525 | +0.008131683 |
| 96 | 2023 | 0.378625839948654 | 0.375583410263062 | -0.003042430 | 0.398094117641449 | 0.395928382873535 | -0.002165735 |
| 192 | 2021 | 0.451054006814957 | 0.453311562538147 | +0.002257556 | 0.433939188718796 | 0.434914499521255 | +0.000975311 |
| 192 | 2022 | 0.430685788393021 | 0.454370439052582 | +0.023684651 | 0.431487023830414 | 0.438576906919479 | +0.007089883 |
| 192 | 2023 | 0.42453870177269 | 0.426603674888611 | +0.002064973 | 0.4284488260746 | 0.430846780538559 | +0.002397954 |
| 336 | 2021 | 0.489607930183411 | 0.497672915458679 | +0.008064985 | 0.457280278205872 | 0.462675034999847 | +0.005394757 |
| 336 | 2022 | 0.470833033323288 | 0.470789134502411 | -0.000043899 | 0.442294389009476 | 0.44365006685257 | +0.001355678 |
| 336 | 2023 | 0.473850965499878 | 0.473091572523117 | -0.000759393 | 0.44828525185585 | 0.447983652353287 | -0.000301600 |
| 720 | 2021 | 0.49070817232132 | 0.497598528862 | +0.006890357 | 0.475344121456146 | 0.477667510509491 | +0.002323389 |
| 720 | 2022 | 0.489835500717163 | 0.495793014764786 | +0.005957514 | 0.4759541451931 | 0.479694247245789 | +0.003740102 |
| 720 | 2023 | 0.494191348552704 | 0.499050498008728 | +0.004859149 | 0.474864989519119 | 0.474363297224045 | -0.000501692 |

- SAEMA lower MSE: **4/12**; Fixed lower MSE: **8/12**; ties: **0**.
- SAEMA lower MAE: **4/12**; Fixed lower MAE: **8/12**; ties: **0**.
- Mean / median paired MSE delta: `+0.005429901` / `+0.003558353`; mean relative delta `+1.233%`.
- Mean / median paired MAE delta: `+0.002253970` / `+0.001839533`; mean relative delta `+0.514%`.
- Horizons with lower SAEMA mean MSE: **0/4**; higher: **4/4**.
- Mechanism-level direction: **CONSISTENT_NEGATIVE**.

Under the shared-alpha control, scale-aware conversion shows a consistently negative empirical direction; the fixed-alpha control is lower across all horizon-level mean MSE values and a majority of paired cells.

## Validation stability

| Config | Mean best val | Sample sd | Min | Max | Best-epoch distribution | Finite |
|---|---|---|---|---|---|---|
| A_moving_avg | 1.139567 | 0.354264 | 0.673925 | 1.589028 | e1:6, e2:1, e3:1, e4:1, e5:1, e7:1, e10:1 | 12/12 |
| B_fixed_selected_a0p02 | 1.138249 | 0.353156 | 0.684482 | 1.586414 | e1:6, e3:1, e4:3, e5:1, e10:1 | 12/12 |
| C_saema_selected_a0p05 | 1.143606 | 0.352910 | 0.685528 | 1.597777 | e1:7, e2:2, e3:1, e4:1, e10:1 | 12/12 |
| D_fixed_control_a0p05 | 1.139201 | 0.353564 | 0.677297 | 1.589703 | e1:6, e2:1, e3:2, e4:2, e10:1 | 12/12 |

The Stage 05-selected alphas remained numerically stable for seeds 2022/2023: every run completed, all losses/checkpoints/results were finite, and no alpha was reconsidered even when a seed or horizon was less favorable.

## Runtime caveat

The 33 new commands recorded `5149.05` aggregate command-seconds (mean `156.03`, range `100.60`–`176.15`). Paired jobs overlapped in wall time, and both GPUs had pre-existing variable load. These numbers are operational audit data only, not an efficiency comparison. Stage 00 runtime remains NA in the raw table.

## Source status

- Final TSLib branch/SHA: `codex/saema-v1 @ 91735a83b27ce1bf766df76a953cf86ef1e7500d`.
- Final tracked `git status --short`: empty; `git diff --check`: PASS.
- TSLib source, tests, runner, loader, optimizer, loss, and seed logic changed in Stage 06: **NO**.
- Only ignored runtime artifacts (logs, checkpoints, result arrays, and test images) were created in the TSLib worktree.

## Structured outputs

| Artifact | Rows | SHA-256 |
|---|---|---|
| results/data/06_etth1_full_benchmark.csv | 48 | 5546ff09f95fae34d45d9d2587c3954ba7fa58c7284ab7818a4c7c9cb1631bf9 |
| results/data/06_trackA_selected_models_summary.csv | 12 | 2b29808143aab45300bffe38faefcf0940bccc0b58bfd2adf869f5e17ce85bfa |
| results/data/06_trackB_same_alpha_mechanism.csv | 4 | 01fa40ca39def596a5daa4fb8d73cc1969c40f451c92b8ec0120c224f71d817d |
| results/data/06_paired_deltas.csv | 48 | 30dc92392aeb22946f0b71f9aef48798b8f012dd693c2f3d9493552b4406d5de |

## Interpretation boundaries

- Track A evaluates frozen, validation-selected configurations and therefore includes different alpha values for B and C.
- Track B fixes `alpha_base=0.05` and is the primary evidence for the scale-aware conversion itself.
- The Stage 07 decision rule follows that primary evidence: proceed only for `CONSISTENT_POSITIVE`; `MIXED` or `CONSISTENT_NEGATIVE` does not support the multi-dataset pilot.
- Results are descriptive for ETTh1 with three seeds. No exaggerated significance claim is made, and no test result was used for tuning.
- Runtime is not interpreted because of overlapping runs and pre-existing GPU load.

## Final

- `Status: PASS`
- 48/48 matrix complete: **YES**
- newly executed runs: **33**
- reused Stage 00 baseline cells: **12**
- reused Stage 05 cells: **3**
- source code changed: **NO**

### Track A

- Fixed EMA selected config lower MSE than baseline in: **8/12** cells
- SAEMA selected config lower MSE than baseline in: **6/12** cells
- SAEMA selected config lower MSE than Fixed selected config in: **6/12** cells
- four-horizon mean MSE/MAE — A: **0.450142 / 0.440632**; B: **0.444800 / 0.437647**; C: **0.451193 / 0.440575**

### Track B

- same-alpha SAEMA lower MSE than Fixed EMA in: **4/12** cells
- same-alpha SAEMA lower MAE than Fixed EMA in: **4/12** cells
- horizons with lower mean MSE under SAEMA: **0/4**
- mechanism-level direction: **CONSISTENT_NEGATIVE**

### Decision

- evidence supports proceeding to multi-dataset pilot: **NO**
- alpha retuned in Stage 06: **NO**
- ready for Stage 07: **NO** (Stage 07 was not executed)

> Track A evaluates the frozen, validation-selected model configurations and includes different alpha values.

> Track B holds alpha_base fixed at 0.05 and is the primary Stage 06 evidence for the scale-aware mechanism itself.

Stage 06 is complete. Execution stops here; Stage 07 was not started.
