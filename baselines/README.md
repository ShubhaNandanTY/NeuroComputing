# baselines

Reference models for the Neuron Games comparison. Backprop baselines
(CNN, ResNet, UNet, Transformer) share one training loop so their numbers
are comparable; non-backprop baselines (Hopfield, ART, SOM) come later.

Data split is shared with `snn-surrogate-baseline/` (see `common/data.py`).

## Layout

| Path | Purpose |
| --- | --- |
| `run.py` | Entry point: pick model, seed, config |
| `common/data.py` | MNIST loaders (copied from `snn-surrogate-baseline`) |
| `common/train.py` | Shared train/eval loop for backprop models |
| `models/` | One file per architecture |
| `results/` | One folder per run (gitignored) |

## Setup

```
cd baselines
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
