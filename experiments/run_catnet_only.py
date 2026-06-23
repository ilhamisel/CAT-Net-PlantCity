"""Dedicated CAT-Net training script.

Used to restart only CAT-Net (the final fruit-32 model) with reduced batch size
and fewer dataloader workers, after the original 6-model run got stuck mid-CAT-Net
training (likely VRAM contention from the prior ViT-B/16 run plus background GPU
usage from other apps). Other 5 fruit checkpoints are already saved.
"""
import json
import sys

import torch

import config

# Override for headroom + safer dataloader behaviour
config.BATCH_SIZE = 16
config.NUM_WORKERS = 2

from engine import train_one  # imports after override so dataset uses new BATCH


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  CROP={config.CROP}  classes={config.NUM_CLASSES}  "
          f"batch={config.BATCH_SIZE}  workers={config.NUM_WORKERS}")
    res = train_one("catnet", device=device, split_mode="clean")
    out = config.LOG_DIR / f"{config.NAMESPACE}catnet_only_summary.json"
    with open(out, "w") as f:
        json.dump({
            "model": "catnet",
            "params_M": res["num_params"] / 1e6,
            "accuracy": res["final_metrics"]["accuracy"],
            "f1_macro": res["final_metrics"]["f1_macro"],
            "auc_ovr": res["final_metrics"]["auc_ovr_macro"],
            "epochs_trained": res["epochs_trained"],
        }, f, indent=2)
    print(f"Written {out}")


if __name__ == "__main__":
    main()
