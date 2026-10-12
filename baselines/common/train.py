"""Shared train/eval loop for all backprop baselines (CNN, ResNet, UNet, Transformer)."""
import time
from pathlib import Path

import torch
from torch import nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()

    total_correct = 0
    total_examples = 0
    loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_correct += (preds == labels).sum().item()
        total_examples += batch_size
        loss_sum += batch_size * loss.item()

    return {"accuracy": (total_correct / total_examples), "loss": (loss_sum / total_examples), "n": total_examples}


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> tuple[float, float]:
    model.train()

    total_correct = 0
    total_examples = 0
    loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # per batch: move to device -> forward -> loss -> zero_grad -> backward -> step
        logits = model(images)
        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_correct += (preds == labels).sum().item()
        total_examples += batch_size
        loss_sum += batch_size * loss.item()

    return ((loss_sum / total_examples), (total_correct / total_examples))


def train(model, loaders, epochs, lr, device, run_dir: Path) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = {"epoch_train_loss": [], "epoch_train_acc": [],
               "epoch_val_loss": [], "epoch_val_acc": [], "epoch_time_s": []}
    best_val_acc = -1.0
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        t0 = time.perf_counter()
        train_loss,train_acc=train_one_epoch(model=model,
                                             loader=loaders["train"],
                                             loss_fn=loss_fn,
                                             optimizer=optimizer,
                                             device=device)
        val=evaluate(model=model,
                    loader=loaders["val"],
                    loss_fn=loss_fn,
                    device=device)
        dt = time.perf_counter() - t0
        val_loss=val["loss"]
        val_acc=val["accuracy"]

        # log metrics
        print(f"[epoch {epoch+1}/{epochs}] "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                      f"time={dt:.1f}s")
        
        history["epoch_time_s"].append(dt)
        history["epoch_train_loss"].append(train_loss)
        history["epoch_train_acc"].append(train_acc)
        history["epoch_val_loss"].append(val_loss)
        history["epoch_val_acc"].append(val_acc)

        if best_val_acc < val_acc:
            torch.save(model.state_dict(), run_dir / "best.pt")
            best_val_acc = val_acc
        

    return {"history": history, "best_val_acc": best_val_acc}