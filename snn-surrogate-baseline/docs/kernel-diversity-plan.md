# Kernel diversity — related work and implementation plan

Status: draft on branch `kernel-diversity-pilot`. This PR is scoped **purely
to the diversity component** of the proposal. Energy protocol lives in PR #1;
this plan assumes it lands first (or in parallel), because we need its
window-splitting to measure diversity's overhead.

Written in plain English, with the math introduced as needed rather than
front-loaded. Reference paper: Park et al. 2013, "Kernel methods on spike
train space for neuroscience: A tutorial"
([arxiv 1302.5964](https://arxiv.org/abs/1302.5964)).

---

## 1. Why diversity, in one paragraph

Our spiking network on MNIST could, in principle, produce the *same* internal
spike pattern for every image and still get reasonable accuracy by having the
final layer pull those identical patterns apart. That's a bad thing: it means
the intermediate representations aren't doing work — they're overloaded and
brittle. A **diversity term** in the training loss rewards the network for
producing *different* spike patterns for different inputs. If two inputs are
different, their hidden spike trains should look different too. That's the
whole idea. The proposal calls this "Kernel/DPP Diversity" and grounds it in
Park et al. 2013.

## 2. What Park et al. 2013 gives us

The paper is a tutorial on **kernels for spike trains**. A kernel is a way of
measuring similarity between two objects — two vectors, two functions, two
spike trains. What matters for us: Park 2013 defines several mathematically
valid kernels for spike trains (positive semi-definite, differentiable if you
smooth appropriately), so we can talk about "how similar are these two spike
trains" without inventing an ad-hoc metric.

The simplest one — and the one we'll use — is the **memoryless cross-intensity
kernel (mCI)**. In plain English:

1. Take each spike train and smooth it by convolving with a small Gaussian
   (or exponential) window. This turns a sparse set of spike times into a
   smooth function of time.
2. To compare two spike trains, take the inner product of their smoothed
   versions: multiply them pointwise, integrate over time.
3. Two spike trains that spike at similar times → high value. Two that spike
   at different times → low value.

That's it. The formula is:

> κ(s_i, s_j) = ∫ [Σ_a h(t − t_a^i)] · [Σ_b h(t − t_b^j)] dt

where `h(·)` is the smoothing kernel and the sums run over each spike train's
spike times. In discrete steps (T = 50 in our setup) this becomes a simple
dot product between the two smoothed activation traces.

**Why this one:** cheapest (batch dot products), differentiable end-to-end,
directly cited in Park 2013 as the workhorse kernel. Fancier kernels
(nonlinear, memory-full) exist and can come later; the pilot uses mCI.

## 3. Turning a kernel into a diversity loss

Once we have a kernel `κ`, we can score how *redundant* a batch of B spike
trains is:

- Form the B×B **kernel matrix** `K`, where `K[i,j] = κ(s_i, s_j)`.
- Two extreme cases:
  - If every pair has `κ(s_i, s_j) ≈ κ(s_i, s_i)` — every spike train looks
    the same. Redundant.
  - If `κ(s_i, s_j) ≈ 0` for i ≠ j — the spike trains are orthogonal.
    Diverse.

Two candidate loss forms turn this into a scalar we add to the CE loss:

- **Option (a) — pairwise redundancy penalty:**
  `L_div = (1/(B²−B)) · Σ_{i≠j} K(i,j)`
  Meaning: average off-diagonal similarity. Small when spike trains are
  diverse. Cost: O(B²), one dense matrix multiply.

- **Option (b) — DPP log-det:**
  `L_div = −log det(K + εI)`
  This is the **determinantal point process** (DPP) log-likelihood. Higher
  determinant = more diverse (the eigenvalues of K span a bigger volume).
  Matches the proposal's phrasing ("Kernel/DPP Diversity") exactly. Cost:
  O(B³) per batch, and needs stable log-det numerics.

**Default for the pilot: option (a).** It's simpler, faster, and easier to
debug. If it moves the needle at all, we promote to (b) which has stronger
theoretical grounding (DPPs are the standard "diverse subset" tool in ML).

The full loss:

> L = L_ce_rate + λ · L_div

with λ found by a small sweep on 1 epoch.

## 4. Where it plugs into the network

Recall the tutorial-6 arch we already have:

```
Conv2d(1,12,5) → MaxPool2d → Leaky₁
Conv2d(12,64,5) → MaxPool2d → Leaky₂
Flatten → Linear(1024,10) → Leaky_out
```

We attach the diversity term to **Leaky₂'s spike output over T = 50 steps**.
Why the second hidden layer:

- Deep enough that the representation is task-relevant (not raw pixels).
- Shallow enough that gradient signal from the diversity term reaches earlier
  layers reasonably.
- Standard choice for representation-level regularizers.

**What we compute per batch:**

1. Get Leaky₂ spikes: shape `[T, B, C, H, W]` after conv/pool.
2. Flatten to per-example vectors: shape `[B, T · C · H · W]`.
3. Smooth over time (the T dim): convolve with a fixed 5-step exponential
   window. This is what turns discrete spikes into the smooth traces the mCI
   kernel expects.
4. Compute pairwise similarities: `K = X @ X.T`, shape `[B, B]`.
5. Reduce to a scalar via option (a) above.

Nothing about the forward pass changes. We just tap Leaky₂'s output tensor —
which snnTorch already exposes because `init_hidden=True` — and add one loss
term.

### 4.1 The kernel is feature-agnostic (why this design lasts into Neuron Games)

Steps 1–3 above produce a **feature vector per example**: one row of `X`,
shape `[B, D]`. Steps 4–5 (kernel matrix, off-diagonal reduction) only look
at `X`. They do not care what `D` represents. In particular, `X` can be:

- **Smoothed hidden spike trains** — what the pilot uses (steps 1–3 above).
- **Per-neuron plasticity signatures `{e_ij}`** — what the proposal's §5 calls
  the eventual feature. Each row of `X` would then be a neuron's incoming
  eligibility-trace vector rather than an example's spike trace.
- **Anything else with a meaningful notion of "two of these should differ"** —
  hidden membrane trajectories, per-synapse update magnitudes, etc.

Swapping the feature does not require changing the kernel module or the
diversity loss. It only requires a different feature-extractor upstream.
That's why Phase B1 (below) builds the kernel module as feature-agnostic,
even though the pilot only exercises the spike-train pathway.

**Explicit deferral of plasticity signatures.** Surrogate-gradient BP does
not natively compute eligibility traces `{e_ij}` — those are a three-factor
learning rule construct. Rather than fake them with a hand-rolled trace
approximation (which introduces its own hyperparameters and is a real change
to `train.py`), we defer plasticity-signature diversity to the Neuron Games
learning-rule work. When that lands, only the feature extractor changes; the
kernel module, the loss, and the reporting shape from this PR carry over
unchanged.

## 5. Related work — what others do with diversity in SNNs

Compact per-paper descriptions. This is a thinner literature than the energy
side because diversity-in-SNNs is a niche topic.

### 5.1 Foundational — Park et al. 2013

- **Park, Chen, Rao, Príncipe 2013, "Kernel methods on spike train space:
  A tutorial"** ([arxiv 1302.5964](https://arxiv.org/abs/1302.5964)).
  Defines a family of positive-definite kernels on spike trains (memoryless
  cross-intensity, nonlinear cross-intensity, Schoenberg). Establishes that
  the resulting kernel matrices are valid for use in kernel methods (SVM,
  kernel PCA, etc.). Does **not** apply this to SNN training regularization
  — the tutorial's motivating applications are neuroscience data analysis
  (decoding, clustering). Our use of Park 2013 is one step beyond the paper's
  own scope: we're using the kernels *inside* the training loop.

### 5.2 DPPs as diversity regularizers in ANNs (not spiking)

- **Kulesza & Taskar 2012, "Determinantal Point Processes for Machine
  Learning."** The DPP framework: sample or score diverse subsets via
  det(K) on a similarity kernel K. Widely used for coreset selection,
  summarization, exploration. Provides the theoretical grounding for
  option (b) above.
- **Zou et al. 2016, "Priors for Diversity in Generative Latent Variable
  Models."** DPP prior on latent variables to force disentangled codes.
  Standard-ANN, not spiking. Design pattern we mirror: DPP over a hidden-
  layer representation, added as a term in the objective.

### 5.3 Sparsity / activation regularization in SNNs

Diversity is closely related to — but distinct from — the sparsity
regularizers already common in SNN work. Sparsity says "few spikes";
diversity says "different spikes across examples." A network can be sparse
and non-diverse (few spikes, always the same ones), or dense and diverse.

- **Pellegrini et al. 2021, "Low-activity supervised convolutional spiking
  neural networks applied to speech commands recognition."** L1 penalty on
  spike counts. **Not diversity**, but the closest neighbor in the SNN loss-
  regularization space.
- **Rathi et al. 2020, "Enabling Deep Spiking Neural Networks with Hybrid
  Conversion and Spike Timing Dependent Backpropagation."** Threshold and
  activation regularization for training stability. Again, not diversity.

### 5.4 Diversity-in-representations (non-SNN, standard ANN)

- **Cogswell et al. 2016, "Reducing overfitting in deep networks by
  decorrelating representations" (ICLR).** Adds a penalty on the off-diagonal
  of the feature covariance matrix. **Directly analogous** to our option (a)
  above — an off-diagonal penalty on a pairwise-similarity matrix — but on
  ANN features rather than spike-train kernels. Good sanity check that the
  general shape of the loss is known to work.
- **Bardes et al. 2022, VICReg.** Variance-invariance-covariance
  regularization for self-supervised learning. Another
  covariance-penalization design in standard ANNs.

### 5.5 The gap our work sits in

**No paper we surveyed applies Park-2013-style spike-train kernels as a
training-time diversity regularizer for surrogate-gradient BP SNNs.** The
components exist separately: the kernels (Park 2013), the DPP framework
(Kulesza & Taskar), off-diagonal feature penalties (Cogswell). Combining
them for SNN training is what the proposal asks us to pilot.

## 6. Pilot experiment design

Small enough to run in an afternoon on one GPU, big enough to distinguish
"real effect" from "noise."

- **Model:** unchanged tutorial-6 arch.
- **Data:** MNIST, unchanged splits.
- **Training budget:** 5 epochs per run. Longer runs deferred until pilot
  results say they're worth it.
- **Conditions (3 total):**
  1. **Baseline** — no diversity term, λ = 0.
  2. **Diversity, λ = 0 sanity check** — same code path as (3) but λ = 0.
     Must reproduce (1) numerically. Guards against accidental gradient
     leakage from the diversity term at λ = 0.
  3. **Diversity, λ = λ_0** — pilot λ from a small 1-epoch sweep.
- **Seeds:** 3 per condition, so 9 runs total.
- **Per-run outputs:** all fields from the energy protocol (train_only,
  validation, checkpoint_io, inference energy; contamination flag; ac_ops;
  samples/J) plus:
  - `diversity_term` — mean L_div across training batches.
  - `spike_sparsity` — mean fraction of Leaky₂ neurons spiking per timestep.
- **Reporting:** mean ± std across seeds for accuracy, train energy, inference
  energy, spike sparsity, diversity term. Report the **overhead of the
  diversity term** as `train_only_energy(condition 3) − train_only_energy
  (condition 1)`, matched on seed.

## 7. Phased implementation plan

Each phase is independently mergeable and leaves the code in a working state.

### Phase B0 — dependency check

Confirm PR #1's window split (Phase A1) has landed on `main`, or coordinate
so this branch rebases on it before pilot experiments run. Diversity overhead
is meaningless without the `train_only` window separated from `validation`.

### Phase B1 — pluggable kernel module

**Files:** new `neuron_baseline/kernels.py`, `tests/test_kernels.py`.

Build a small kernel library behind a common interface:
`kernel_matrix(features, kind, **params)`. Input `[B, D]` feature tensor,
output `[B, B]` kernel matrix. **Feature-agnostic** (§4.1): the module does
not know whether the features are smoothed spike trains, eligibility
vectors, or anything else — that's the caller's problem.

Implement five kernels behind the same interface:

- **`count`** — feature is a scalar spike count per example; `K[i,j] = n_i · n_j`.
  Timing-blind rate baseline. Cheapest possible sanity check.
- **`binned`** — feature is a per-bin spike-count vector; `K[i,j] = r_i · r_j`.
  Coarse-timing baseline.
- **`mci`** — memoryless cross-intensity. Feature is a smoothed spike train
  (default: exponential window, τ = 5 steps); `K[i,j] = ⟨f_i, f_j⟩`.
  The workhorse. What the pilot uses.
- **`vanrossum`** — mCI restricted to a decaying-exponential smoothing
  window. Free once mCI exists (one config change). Commonly cited as its
  own kernel in the neuroscience literature.
- **`schoenberg`** — mCI-derived distance wrapped in a Gaussian:
  `K[i,j] = exp(-‖f_i - f_j‖² / 2σ²)`. Universal kernel (dense in
  continuous functions), one nonlinearity on top of mCI.

Sanity tests apply uniformly to all five:

- Symmetric: `K[i,j] == K[j,i]`.
- PSD: eigenvalues of `K` ≥ −tolerance (numerical fuzz allowed).
- `K[i,i] > 0` for non-empty inputs.
- Two identical feature vectors → `K[i,j]` close to `K[i,i]`.
- Two orthogonal feature vectors → `K[i,j] ≈ 0` for linear kernels,
  or `≈ exp(-‖f_i‖²/σ²)` for Schoenberg.
- Differentiability: gradient flows to `features` via `.backward()`.

**Out of scope for this phase (deferred as future work):**

- Higher-order / spike-timing-correlation kernels — bigger implementation,
  meaningful theoretical addition, worth doing when the pilot motivates it.
- Victor–Purpura distance — not positive-definite, cannot serve as a loss
  term. Included in the plan only as reference.

Why implement five now rather than only mCI: the incremental cost is small
(the four non-workhorse kernels are ≤ 10 lines each behind the shared
interface), and the Neuron Games work (§5 in the proposal) will want to
swap between them anyway. Building the pluggable module once, in this PR,
means later work is a config change, not a refactor.

No training integration in this phase. This phase ships a tested, isolated
module and nothing else.

### Phase B2 — diversity loss + wiring into `train.py`

**Files:** `neuron_baseline/train.py`, `neuron_baseline/config.py` (add
`diversity_lambda: float = 0.0`, `diversity_layer: str = "leaky2"`).

- `pairwise_diversity_loss(K) = (K.sum() - K.diag().sum()) / (B * (B-1))`.
- Tap Leaky₂ output during forward pass; store the T-step spike history.
- If `diversity_lambda > 0`, compute `L_div` and add `lambda * L_div` to
  the training loss.
- If `diversity_lambda == 0`, the kernel isn't computed (guaranteed no
  overhead, guaranteed identical output to baseline).
- Log `diversity_term` per batch to `results.json`.

Smoke test: `python -m neuron_baseline smoke --diversity-lambda 1e-3`
completes without error and gradient magnitude for Leaky₂ spike output is
non-zero.

### Phase B3 — λ sweep (small)

**Files:** new `scripts/lambda_sweep.py`.

- Run 1 epoch at each of `λ ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1}`, single seed.
- Report: val accuracy, mean diversity_term, mean spike sparsity, train_only
  energy relative to λ=0.
- Pick λ_0 for the pilot: the largest λ that keeps val accuracy within 1
  percentage point of the baseline. Rationale: we want to test whether
  diversity *can* be added at low accuracy cost; we don't need λ to hurt
  accuracy for the pilot to be informative.

### Phase B4 — pilot experiment

Depends on B1, B2, B3, and PR #1's energy protocol.

- Run 3 conditions × 3 seeds × 5 epochs = 9 runs. Log every run under
  `runs/pilot-<timestamp>/<condition>/seed-<n>/`.
- Aggregate via `neuron_baseline aggregate` (from PR #1's Phase A4) into
  a single summary table.

### Phase B5 — writeup

**Files:** new `docs/kernel-diversity-results.md`.

- Table: condition × (test_acc, train_only_j, inference_j, samples/J,
  spike_sparsity), mean ± std across seeds.
- Diversity's measured overhead: mean(train_only_j, cond 3) − mean(cond 1),
  with a seed-matched pairwise version.
- Plain-English discussion: did diversity move accuracy? did it change
  sparsity? did it cost energy? Under what conditions would we scale up?

## 8. What we report per run (`results.json`)

New fields alongside PR #1's energy shape:

```
diversity: {
  enabled: bool,
  lambda: float,
  layer: "leaky2",
  kernel: "mci",
  smoothing: {kind: "exp", tau: 5},
  mean_term: float,          # averaged over training batches
  per_batch_terms: [...],    # optional, for debugging
},
spike_stats: {
  leaky2_spike_fraction: float,  # spikes / total possible
}
```

## 9. What we explicitly do not claim

- **Not** that diversity improves accuracy. It might not. The point is to
  measure the effect, not assume it.
- **Not** that diversity reduces energy. The kernel matrix is O(B²) extra
  work per batch; we expect it to *increase* train-time energy. What we're
  after is a defensible measurement of *how much*.
- **Not** the full DPP log-det loss in this PR (option (b) above). Pilot uses
  the simpler pairwise redundancy penalty; DPP-proper deferred.
- **Not** the full "Neuron Games" framework — no Markov game, no metabolic
  barriers, no eligibility-Shapley. That's separate work.

## 10. Non-goals for this PR

- No new architectures beyond the tutorial-6 spiking convnet.
- No new datasets beyond MNIST.
- No integration with William's STDP baseline.
- No claim about mathematical optimality of mCI vs. other Park-2013 kernels.
- No hyperparameter search beyond the λ sweep in B3.

## 11. Open decisions

1. **Layer attachment point:** Leaky₂ is the default. Could also be Leaky₁
   (earlier, richer features) or Leaky_out (later, class-discriminative).
   Default kept at Leaky₂. Revisit after pilot.
2. **Smoothing constant τ:** 5 timesteps default (out of T = 50). Not tuned.
   Fixed for the pilot; sensitivity later if warranted.
3. **Pilot batch size:** kept at 128 (baseline default) so the O(B²) kernel
   cost is 128² = 16k dot products per batch — trivial.
4. **Loss form:** default pairwise (option a). Promote to DPP log-det
   (option b) only if pairwise shows movement.
5. **Kernel choice for the pilot run** *(among the five Phase B1 implements)*:
   default `mci`. `count` and `binned` are useful baselines to check whether
   any timing-aware kernel matters at all; `vanrossum` and `schoenberg` are
   worth revisiting if `mci` shows partial movement.
6. **K over what?** — the pilot uses **batch-wise K** (rows of `X` are
   per-example features; K is `[B, B]`, encourages examples to produce
   different patterns). The proposal's §5 phrasing ("Σ_ℓ log det(K_ℓ + εI)"
   summed over layers) points at **neuron-wise K** (rows of `X` are
   per-neuron features across the batch; K is `[N_ℓ, N_ℓ]`, encourages
   neurons within a layer to specialize). Both are valid, both use the same
   kernel machinery. Neuron-wise is closer to representational
   specialization; batch-wise is closer to decorrelation. Open decision;
   flagged for review before Phase B2 wiring.

## 11.1 Resolved (feature choice)

- **Feature vector for the pilot:** smoothed per-example hidden spike
  trains (from Leaky₂). See §4.1.
- **Plasticity signatures `{e_ij}`:** deferred to the Neuron Games
  learning-rule work. Requires eligibility traces that surrogate-BP does
  not natively compute; not faked with a hand-rolled approximation.
  When Neuron Games lands, only the feature extractor changes.

## 12. Dependencies on PR #1

This branch already includes the `runner.py` refactor (needed because
diversity training goes through the same run_training path). It does **not**
include PR #1's energy window-split. If PR #1 lands first, this branch
rebases and B2's per-batch overhead measurement becomes trivial. If this
branch lands first, B4 pilot runs will report one aggregated `train` energy
window instead of separated `train_only` / `validation` — still usable as a
comparison across conditions (same aggregation everywhere), just less
precise. Recommend PR #1 first.
