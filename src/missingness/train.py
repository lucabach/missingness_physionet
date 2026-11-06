"""Train one model on one input variant and evaluate it on every test set.

    python -m missingness.train configs/experiment/sepsis_gru.yaml seed=1 inputs=pattern

Writes to outputs/<run_name>/:
    config.yaml        fully resolved config (reproduce the run from this alone)
    normalizer.json    training statistics
    model.pt           best checkpoint (by validation AUROC)
    metrics.json       AUROC/AUPRC with 95% CIs per test set
    predictions/       per-step predictions per test set (parquet)
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn

from missingness.config import load_config, save_config
from missingness.data.preprocessing import Normalizer
from missingness.eval.metrics import auroc
from missingness.experiment import load_data, make_loader, predict, score_predictions, to_device
from missingness.models import build_model
from missingness.paths import OUTPUT_DIR
from missingness.utils import get_device, set_seed


def run_name(cfg: dict) -> str:
    return cfg.get("run_name") or (
        f"{cfg['data']['name']}_{cfg['model']['name']}_{cfg['inputs']}_seed{cfg['seed']}"
    )


def train(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = get_device()
    task = cfg["data"]["task"]
    tcfg = cfg["train"]
    out_dir = OUTPUT_DIR / run_name(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    sets = load_data(cfg)
    norm = Normalizer.fit(sets["train"])
    (out_dir / "normalizer.json").write_text(json.dumps(norm.to_dict()))
    n_vars, n_static = len(norm.mean), len(norm.static_mean)

    model = build_model(cfg["model"], n_vars, n_static, cfg["inputs"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    train_loader = make_loader(sets["train"], norm, task, tcfg["batch_size"], shuffle=True)

    best, patience = -1.0, 0
    for epoch in range(tcfg["epochs"]):
        model.train()
        t0, total = time.time(), 0.0
        for batch in train_loader:
            batch = to_device(batch, device)
            loss = loss_fn(model(batch), batch["labels"])
            loss = (loss * batch["label_mask"]).sum() / batch["label_mask"].sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            total += loss.item()
        val = predict(model, sets["val"], norm, task, device)
        val_auroc = auroc(val["label"], val["score"])
        print(
            f"epoch {epoch:3d}  loss {total / len(train_loader):.4f}  "
            f"val_auroc {val_auroc:.4f}  ({time.time() - t0:.0f}s)"
        )
        if val_auroc > best:
            best, patience = val_auroc, 0
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            patience += 1
            if patience >= tcfg["patience"]:
                break

    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    metrics = {"val_auroc": best}
    (out_dir / "predictions").mkdir(exist_ok=True)
    for name, records in sets.items():
        if not name.startswith("test"):
            continue
        pred = predict(model, records, norm, task, device)
        pred.to_parquet(out_dir / "predictions" / f"{name}.parquet", index=False)
        metrics[name] = score_predictions(pred, task, n_boot=cfg["eval"]["n_boot"])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"done -> {out_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("experiment", help="path to configs/experiment/*.yaml")
    parser.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    args = parser.parse_args()
    train(load_config(args.experiment, args.overrides))


if __name__ == "__main__":
    main()
