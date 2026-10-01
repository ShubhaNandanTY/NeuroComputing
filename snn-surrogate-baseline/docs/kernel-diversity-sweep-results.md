# Kernel-diversity sweep results (Phase B3)

Status: single-seed, 1-epoch sweeps completed on `kernel-diversity-pilot`
branch. **No conclusion possible yet** — the apparent improvement from
diversity is at the single-seed noise floor. Multi-seed verification is
the next step before moving to the pilot (Phase B4).

Hardware: RTX 5080, Windows 11, cuDNN benchmark on (not deterministic —
intentional per the README; a known source of same-seed variance).

Reference scripts (all in `snn-surrogate-baseline/scripts/`):
- `lambda_sweep.py` — fine λ grid on mci
- `kernel_sweep.py` — all 5 kernels at fixed λ=1e-3
- `kernel_lambda_sweep.py` — per-kernel λ grid scaled to each kernel's K magnitude

Raw outputs live under `runs/` (git-ignored). Timestamps referenced
below are from the sweep run on 2026-09-30.

---

## TL;DR

- Every kernel has *some* λ at which it nominally beats the baseline by
  **~0.6–0.7 pts val accuracy**.
- Same-seed baseline variance across our sweeps is **~0.6 pts**.
- **The signal and the noise are the same size.** We cannot distinguish them
  without multi-seed replication.
- λ grids reveal clear monotonic accuracy/diversity tradeoffs within each
  kernel, which validates the implementation.

## 1. Fine λ sweep on mci

`runs/lambda-sweep/20260930-201553/` · 8 runs

Baseline (λ=0): val=0.9744, test=0.9792, train=34.8 s, train_energy=2806 J

| λ | val_acc | test_acc | div_term | train_s | train_J |
|---|---|---|---|---|---|
| 1e-5 | **0.9786** | **0.9824** | 72.87 | 35.5 | 3252 |
| 3e-5 | 0.9758 | 0.9823 | 36.58 | 35.2 | 3306 |
| 1e-4 | 0.9738 | 0.9807 | 14.18 | 34.9 | 3336 |
| 3e-4 | 0.9702 | 0.9750 | 6.65 | 35.0 | 3167 |
| 1e-3 | 0.9624 | 0.9642 | 2.32 | 34.9 | 3085 |
| 3e-3 | 0.9590 | 0.9651 | 1.16 | 35.5 | 3094 |
| 1e-2 | 0.9484 | 0.9555 | 0.44 | 35.1 | 3048 |

Script-selected λ₀ (largest λ within 1-pt tolerance): **3e-4**. The actual
best val_acc point was **1e-5** (+0.4 pt over baseline).

- Diversity term shrinks ~160× from λ=1e-5 to λ=1e-2 → the mci kernel
  responds to λ as expected.
- Energy overhead from hook + kernel: ~10–20% roughly flat across λ > 0.

## 2. Kernel sweep at fixed λ=1e-3

`runs/kernel-sweep/20260930-202055/` · 6 runs

Baseline: val=0.9742, test=0.9787, train=34.2 s, train_energy=2730 J

| Kernel | val_acc | test_acc | div_term | train_J |
|---|---|---|---|---|
| none (baseline) | 0.9742 | 0.9787 | 0.000 | 2730 |
| count | **0.4090** | **0.4142** | 5.66 | 2832 |
| binned | 0.9644 | 0.9706 | 2.51 | 2983 |
| mci | 0.9564 | 0.9648 | 2.38 | 3121 |
| vanrossum | 0.9630 | 0.9665 | 2.21 | 3055 |
| schoenberg | 0.9708 | 0.9756 | 0.013 | 3058 |

The count collapse (val 0.41) is a **scale artifact**: count K entries are
~10⁵ at this network size, so λ=1e-3 is effectively ~10⁸× stronger
regularization than mci's. Not an argument against the count kernel —
an argument against sharing λ across kernels with different K magnitudes.

This finding motivated sweep #3.

## 3. Per-kernel λ sweep (apples-to-apples)

`runs/kernel-lambda-sweep/20260930-202843/` · 25 runs

Each kernel got its own λ grid chosen for its natural K magnitude.
`vanrossum` skipped: numerical alias of mci.

Baseline: val=**0.9692**, test=**0.9742**, train=34.3 s, train_energy=2697 J

### count (K ~ 10⁵)

| λ | val_acc | test_acc | div_term |
|---|---|---|---|
| 1e-8 | **0.9760** | **0.9806** | 1.3e5 |
| 1e-7 | 0.9640 | 0.9703 | 2.4e4 |
| 1e-6 | 0.9492 | 0.9584 | 4.0e3 |
| 1e-5 | 0.8104 | 0.8130 | 4.8e2 |
| 1e-4 | 0.6158 | 0.6301 | 80 |

### binned (K ~ 10)

| λ | val_acc | test_acc | div_term |
|---|---|---|---|
| 1e-5 | 0.9754 | **0.9826** | 76 |
| 3e-5 | 0.9724 | 0.9799 | 36 |
| 1e-4 | **0.9762** | 0.9797 | 15 |
| 3e-4 | 0.9714 | 0.9760 | 6.7 |
| 1e-3 | 0.9658 | 0.9717 | 2.7 |
| 3e-3 | 0.9476 | 0.9543 | 1.3 |

### mci (K ~ 1–10)

| λ | val_acc | test_acc | div_term |
|---|---|---|---|
| 1e-5 | **0.9752** | **0.9819** | 70 |
| 3e-5 | 0.9752 | 0.9789 | 34 |
| 1e-4 | 0.9746 | 0.9818 | 14 |
| 3e-4 | 0.9704 | 0.9752 | 6.0 |
| 1e-3 | 0.9592 | 0.9695 | 2.4 |
| 3e-3 | 0.9544 | 0.9604 | 1.2 |
| 1e-2 | 0.9442 | 0.9531 | 0.44 |

### schoenberg (K ∈ [0,1])

| λ | val_acc | test_acc | div_term |
|---|---|---|---|
| 1e-2 | 0.9732 | 0.9760 | 0.014 |
| 3e-2 | **0.9764** | 0.9787 | 0.015 |
| 1e-1 | 0.9678 | **0.9768** | 0.022 |
| 3e-1 | 0.9660 | 0.9756 | 0.079 |
| 1.0 | 0.9570 | 0.9640 | 0.35 |
| 3.0 | 0.9530 | 0.9595 | 0.53 |

### Best per kernel

| Kernel | λ_best | val_best | Δ vs baseline |
|---|---|---|---|
| count | 1e-8 | 0.9760 | +0.68 pt |
| binned | 1e-4 | 0.9762 | +0.70 pt |
| mci | 1e-5 | 0.9752 | +0.60 pt |
| schoenberg | 3e-2 | 0.9764 | +0.72 pt |

Four different kernels, four different λ scales — all improve the
baseline by essentially the same amount. That's either a real effect of
mild diversity regularization or seed noise.

## 4. The noise-floor problem

Same seed (`20260923`), same code, baseline val_acc across our sweeps:

| Run | Baseline val_acc |
|---|---|
| First standalone validation run | 0.9752 |
| Lambda sweep (fine) | 0.9744 |
| Kernel sweep | 0.9742 |
| Per-kernel sweep | **0.9692** |

Same-seed variance: **~0.6 pt**.

Root cause (documented in README §"Reproducibility limits"): cuDNN
benchmark mode is intentionally on for throughput, so kernel
autotuning + reduction ordering vary between runs even with identical
seeds. We accept this at ~0.1-pt scale; here it's ~0.6 pt, likely
because thermal state / concurrent GPU workloads differ run-to-run.

**Consequence:** the apparent +0.6–0.7 pt diversity improvement is at
the floor. No claim can be made on single seed.

## 5. Next steps — multi-seed verification

Minimum viable experiment to answer "does diversity do anything":

- **Conditions:** 5 total
  1. Baseline (λ=0)
  2. count, λ=1e-8
  3. binned, λ=1e-4
  4. mci, λ=1e-5
  5. schoenberg, λ=3e-2
- **Seeds:** 3 per condition, so 15 runs total.
- **Epochs:** 1 (same as the sweeps — cheap confirmation first). If signal
  survives, promote to 5 epochs under Phase B4.
- **Reporting:** mean ± std of val_acc and test_acc per condition.
  Confidence intervals that don't overlap → signal is real.
- **Compute:** ~15 min on this hardware.

Script to add in a follow-up PR: `scripts/multi_seed_verify.py`.

If results are inconclusive at 1 epoch, promote the two most-promising
conditions to Phase B4 proper: 5 epochs × 3 seeds × {baseline, mci@1e-5,
schoenberg@3e-2}. ~30 min.

## 6. Open questions and known gaps

1. **Why does mild diversity (λ at the "no-pressure" end) help across
   kernels, if it does?** Hypothesis: the Leaky₂ layer is somewhat
   under-regularized by default; any small structural prior that
   discourages exact-redundancy pushes the optimizer into a better
   basin. Falsifiable by observing whether multi-seed CIs separate.
2. **Count-kernel results above λ=1e-7 are *always* bad.** This is
   consistent with count's rank-1 math: every entry is `n_i · n_j`, so
   the gradient points at reducing *all* firing to zero, which breaks
   classification. Count is probably not usable as a diversity kernel
   at this network scale.
3. **Schoenberg at the knee (λ=3e-2) has tiny div_term values
   (~0.015).** This is because the kernel's natural output range is
   [0,1] and off-diagonal values are already small for a layer with
   mild sparsity. Harder to tell whether the kernel is "doing work" or
   whether the gradient is just too small to matter.
4. **No sparsity metric logged yet.** The plan mentioned spike_sparsity
   as a reporting field; the current training loop does not capture
   it. Easy to add once we commit to Phase B4.
5. **This is all 1 epoch.** The patterns at 5 epochs may differ — the
   accuracy knee could sharpen, or the mild-regularization effect could
   vanish once the baseline has had time to generalize on its own.
