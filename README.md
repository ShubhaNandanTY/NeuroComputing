# haag-mukhopadhyay-cddm

Workspace repo for the **Mukhopadhyay CDDM** research effort.

## Proposal

**"Neuron Games for Spiking Networks: A Constrained State-Based Potential
Markov Game with Kernel/DPP Diversity, Metabolic Barriers, and Eligibility-Shapley
Explainability"** — `neuron-games/docs/papers/mukhopadhyay-cddm.pdf`.

The proposal specifies a protocol (tasks, baselines, ablations, metrics) but
contains no experiments. This repo is where we implement and evaluate it.

## Layout

Each sub-project is self-contained: its own README, its own environment,
its own results.

| Path                        | Owner   | Purpose |
| --------------------------- | ------- | ------- |
| `neuron-games/`             | shared  | Proposal PDF, research notes, formal write-ups, unfinished framework decisions. See `neuron-games/README.md` for the full proposal summary. |
| `snn-surrogate-baseline/`   | Deyvik  | snnTorch Tutorial 6 surrogate-gradient backprop baseline on MNIST. The **BP reference** the proposal's three-factor rule and Eligibility-Shapley alternatives get measured against. |
| _(future)_ `stdp-baseline/` | William | Spike-timing-dependent plasticity baseline / alternative learning rule (in progress separately). |

**Current work plan:** [`Architecture_Work_14-10-2026.md`](Architecture_Work_14-10-2026.md)
— the no-backprop Neuron Games pipeline, its file layout, and who owns which part.

## Working here

There is no top-level environment — each sub-project brings its own.
Start from the sub-project's README:

- `snn-surrogate-baseline/README.md` — exact setup, smoke-test, training,
  checkpoint-resume, and evaluation commands.

## Log

- 2026-09-07 — Repo initialized; proposal read and summarized.
- 2026-09-23 — Reorganized into workspace-of-projects layout. Added
  `snn-surrogate-baseline/` (Tutorial 6 BP baseline on RTX 5080).
- 2026-10-14 — `main` set up from the `joohyun` branch. Neuron Games
  three-factor core implemented in `neuron-games/`; remaining components
  assigned in `Architecture_Work_14-10-2026.md`.
