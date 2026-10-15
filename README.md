# NeuroComputing: Neuron Games

Research repository for the **Mukhopadhyay CDDM** effort: training spiking
neural networks **without backpropagation**.

## The project

Backpropagation is the standard way to train neural networks, but it needs a
backward pass that carries exact error gradients through every layer and,
for spiking or recurrent networks, back through every time step (BPTT). That
costs memory proportional to the sequence length and is widely considered
biologically implausible.

This project implements and evaluates the proposal

> **"Neuron Games for Spiking Networks: A Constrained State-Based Potential
> Markov Game with Kernel/DPP Diversity, Metabolic Barriers, and
> Eligibility-Shapley Explainability"**
> (`neuron-games/docs/papers/mukhopadhyay-cddm.pdf`)

The goal is a learning rule that **matches or approaches backpropagation's
accuracy while using less training compute**, measured head to head against a
surrogate-gradient BPTT network of the same architecture.

### The approach in brief

- **Neurons are players.** Each neuron's action is the change to its incoming
  synapses. All neurons share one potential
  `Phi = -J + beta*V - mu*E - lambda*R`: task loss `J`, within-layer diversity
  `V`, metabolic and barrier energy `E`, weight regularizer `R`.
- **Three-factor plasticity is the learning rule.** Each synapse keeps a local
  eligibility trace; a broadcast learning signal (the third factor) decides how
  much of it to apply: `dW_ij = -eta * sum_t m_i[t] * e_ij[t]`.
- **No backward pass.** The output error reaches hidden neurons through fixed
  random feedback weights (e-prop style). Every term of `Phi` contributes a
  per-neuron signal that the eligibility trace turns into a weight change.
- **Diversity and homeostasis keep the game stable**: diversity stops neurons
  from all learning the same feature; energy costs and barriers stop them from
  firing at maximum rate to capture credit.
- **Eligibility-Shapley credit** gives each neuron a tractable share of the
  improvement in `Phi`, used both to scale its learning signal during training
  and to explain the trained network afterwards.

The current status, the full pipeline layout, and who is finishing which part
are in **[`Architecture_Work_14-10-2026.md`](Architecture_Work_14-10-2026.md)**.

## Repository structure

```
NeuroComputing/
├── README.md                          this file
├── Architecture_Work_14-10-2026.md    pipeline architecture, status, work split
├── LICENSE
├── neuron-games/                      the Neuron Games method (main project)
│   ├── README.md                      proposal summary, commands, open questions
│   ├── docs/
│   │   ├── papers/mukhopadhyay-cddm.pdf   the proposal
│   │   └── implementation-plan.md         original implementation plan
│   ├── src/neuron_games/
│   │   ├── config.py                  conditions (ablation rows) and all hyperparameters
│   │   ├── model.py                   784-128-10 LIF network; records eligibility traces
│   │   ├── encoding.py                image -> spike-train encodings
│   │   ├── state.py                   game state, reward, previous actions
│   │   ├── learning_rule.py           three-factor rule
│   │   ├── game.py                    one adaptation window (proposal section 11)
│   │   ├── train.py / eval.py / runner.py   training loop, evaluation, results.json
│   │   ├── potential/                 terms of Phi: task J, diversity V, energy E, regularizer R
│   │   ├── modulator/                 random feedback broadcast, Eligibility-Shapley credit
│   │   └── explain/                   post-hoc attribution and explainability metrics
│   ├── scripts/                       ablation driver and results aggregation
│   ├── tests/                         pytest suite
│   └── data/, notebooks/, results/    datasets (git-ignored), exploration, outputs
└── snn-surrogate-baseline/            snnTorch Tutorial 6 surrogate-gradient BPTT baseline
                                       (conv SNN, energy meter, spike-train kernels);
                                       neuron-games imports its data loaders, seeding,
                                       energy meter, plots and kernel library
```

Run outputs go to `neuron-games/runs/` and `snn-surrogate-baseline/runs/`
(both git-ignored).

## Getting started

Requirements: Python 3.12 with `torch`, `torchvision`, `matplotlib`, `pytest`
(`snntorch` only for two cross-check tests, which skip without it).
`snn-surrogate-baseline/README.md` has an exact pinned environment.

```bash
cd neuron-games
python -m pytest tests -v                                                           # test suite
PYTHONPATH=src python -m neuron_games train --condition three_factor --device cpu   # no-backprop training
PYTHONPATH=src python -m neuron_games train --condition bp_anchor --device cpu      # BPTT reference
```

`--device` defaults to `cuda` and never silently falls back; pass `cpu` or
`mps` explicitly elsewhere. MNIST downloads to `neuron-games/data/raw/` on the
first run. More commands are in `neuron-games/README.md`.

## Design decisions

Decisions taken while building the no-backprop pipeline, with the reasoning
behind each. Change them only after discussing with the group; other parts of
the code are written against them.

### Learning rule and credit assignment

1. **Every term of `Phi` is trained through the eligibility trace; nothing
   runs a backward pass through the network.**
   For our LIF cell, the factorized eligibility trace `e_ij[t] = h_i[t] * eps_j[t]`
   is exactly the local derivative `ds_i[t]/dW_ij`, and `eps_j[t]` alone is
   `du_i[t]/dW_ij` (verified in `tests/test_learning_rule.py` to float64
   precision). So any term that depends on a neuron's spikes or membrane only
   has to supply a per-neuron signal, and the same rule applies it. This keeps
   the method strictly three-factor, as the proposal requires (section 7.3).
   Where a term's signal is computed with autograd, autograd only acts on the
   already-recorded spike, membrane and weight tensors, never through the
   network's dynamics or between layers.

2. **Hidden layers learn from fixed random feedback.**
   The output layer's error is broadcast to hidden neurons through a random
   matrix `B` drawn once and never trained (e-prop with random feedback, the
   proposal's reference [4]). Using the transpose of the output weights instead
   would require weight transport, the main non-local ingredient of
   backpropagation. `B` uses its own random seed, so changing it never changes
   the weight initialization.

3. **The output error is computed in closed form at the output neurons.**
   `dJ/ds_out[t]` for the cross-entropy losses is written out explicitly
   (`potential/task.py`) and checked against autograd in the tests. It is a
   quantity available locally at the output layer, which is what the third
   factor is supposed to be.

4. **No bias terms in the three-factor rows.**
   Every trainable parameter is then a synapse `w_ij` with an eligibility
   trace, which keeps the rule uniform. The config rejects `bias=True` for the
   three-factor rule.

### Optimization

5. **SGD, not Adam, for every three-factor condition.**
   Adam rescales each weight's update by that weight's own gradient history.
   A per-neuron gain from Eligibility-Shapley credit is largely cancelled by
   that rescaling, so the credit modulation would have almost no effect. Plain
   SGD is also the literal form of the proposal's update `dW = eta * int m e dt`
   (eq. 3). The BPTT reference keeps Adam (lr 1e-3).

6. **Default learning rate 2.0.**
   Chosen from a 300-batch MNIST sweep over {0.1, 0.5, 2, 5, 10, 20, 50}:
   2.0 gave the best validation accuracy; 10 and above diverge. The learning
   signals are true batch-mean derivatives of small spike-based losses, so a
   learning rate this large is expected under plain SGD. It must be retuned once
   the diversity and energy terms are switched on.

### The potential and its terms

7. **One sign convention for all terms.**
   Each term returns plain derivatives of its own value; the composer
   (`potential/potential.py`) applies the coefficient and sign from
   `C = -Phi = J - beta*V + mu*E + lambda*R`. Diversity `V` is a reward, so it
   enters with `-beta`; energy and the regularizer are costs. Keeping signs in
   one place prevents each contributor from getting them wrong independently.

8. **Every term is summed over neurons and time steps and averaged over the batch.**
   The task loss `J` is a batch mean, so this keeps `beta`, `mu` and `lambda`
   independent of batch size and on a comparable scale to `J`.

9. **Diversity is normalized before the DPP log-determinant.**
   The kernel matrix is divided by the mean of its diagonal before
   `log det(K + eps*I)`. Without this, the log-determinant can be increased
   simply by making every neuron fire more, which rewards activity rather than
   diversity. The jitter `eps` is relative for the same reason.

10. **Plasticity-signature diversity uses an exact shortcut instead of
    materializing the traces.**
    For linear kernels, the similarity of two neurons' eligibility traces is
    `sum_{t,b} h_i h_k ||eps[t,b]||^2`. Feature rows `h_i[t,b] * ||eps[t,b]||`
    give the same Gram matrix at the cost of a spike-based feature, instead of
    a `[neurons x batch x time x inputs]` tensor.

11. **Diversity and credit apply to hidden layers only, by default.**
    Output neurons are class-specific by design; pushing them apart or
    reweighting their task signal would interfere with the read-out.

12. **Barriers start as quadratic penalty plus projection; the log-barrier
    uses the Kervadec extension.**
    A pure log-barrier on every membrane value at every time step is stiff and
    produces infinities in float32 as soon as a value leaves the box. The
    quadratic form is numerically safe for the first runs; the log-barrier
    extension (Kervadec et al. 2019) is finite everywhere and keeps the
    proposal's constrained-game interpretation (eq. 17).

### Eligibility-Shapley credit

13. **Credit uses each neuron's previous update, not the current one.**
    `phi_i = <G_i, dW_i(previous window)>`, with `G` the current sensitivity of
    `Phi`. Using the current update would make the gain depend on the very
    update it scales, and reduces `phi_i` to the squared size of the neuron's
    gradient. With the previous update, `phi_i` is each neuron's first-order
    share of the actual improvement in `Phi`, and the credits sum to the
    first-order reward estimate (the Shapley efficiency property).

14. **Gains are rescaled to average 1 and computed on standardized credit.**
    The proposal's eq. 23 normalizes gains to sum to 1; we multiply by the
    layer size so the mean gain is 1. Uniform credit then reproduces plain
    random feedback exactly, and the difference from eq. 23 is a constant
    absorbed into the global signal. Credit is divided by its standard deviation
    before the softplus because raw values are tiny (a gradient times an
    update), which would otherwise make every gain about 1.

15. **The first window gets uniform credit.**
    There is no previous action yet, so no neuron can be credited.

### Experiment setup

16. **One adaptation window is one mini-batch presented for T = 25 steps.**
    Weights are updated once per window, after the rollout, matching the
    proposal's windowed algorithm (section 11). The traces are computed step by
    step, so online updates within a window remain possible later.

17. **The no-backprop rule and the BPTT reference train the identical network.**
    Same LIF cell, same 784-128-10 architecture, same encoding, loss, data split
    and seeds, so a difference in results is attributable to the learning rule.

18. **Shared measurement code is imported, not copied.**
    `neuron-games` imports the data loaders, seeding, NVML energy meter, plots
    and kernel library from `snn-surrogate-baseline`, so both pipelines are
    measured by the same machinery.

19. **Ablation conditions are presets, and every term can be toggled
    individually.**
    `--condition` selects a row of the ablation grid (proposal section 12.3);
    any term can also be switched on alone on top of `three_factor`. This lets
    each part be developed and tested before the others are finished.

## Log

- 2026-09-07 - Repo initialized; proposal read and summarized.
- 2026-09-23 - Reorganized into workspace-of-projects layout. Added
  `snn-surrogate-baseline/` (Tutorial 6 BP baseline on RTX 5080).
- 2026-10-08 - `neuron-games`: vendored LIF cell, 784-128-10 network, BPTT reference row.
- 2026-10-14 - `main` set up from the `joohyun` branch. Three-factor core of
  Neuron Games implemented (trains on MNIST without backprop, 91.8% test after
  3 epochs); remaining components assigned in `Architecture_Work_14-10-2026.md`.
