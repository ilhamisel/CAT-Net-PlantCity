"""Train all models sequentially and dump consolidated summary."""
import json
import sys
from pathlib import Path

import torch

import config
from engine import train_one


def main(models=None, split_modes=("clean", "standard")):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = models or config.MODELS
    print(f"Device: {device}, models: {models}, splits: {split_modes}")
    summary = []
    for split_mode in split_modes:
        for name in models:
            print(f"\n========== TRAINING {name} [{split_mode}] ==========")
            try:
                res = train_one(name, device=device, split_mode=split_mode)
                summary.append({
                    "model": name,
                    "split": split_mode,
                    "params_M": res["num_params"] / 1e6,
                    "accuracy": res["final_metrics"]["accuracy"],
                    "precision_macro": res["final_metrics"]["precision_macro"],
                    "recall_macro": res["final_metrics"]["recall_macro"],
                    "f1_macro": res["final_metrics"]["f1_macro"],
                    "auc_ovr_macro": res["final_metrics"]["auc_ovr_macro"],
                    "epochs_trained": res["epochs_trained"],
                })
            except Exception as e:
                print(f"ERROR training {name}: {e}")
                import traceback; traceback.print_exc()
                summary.append({"model": name, "split": split_mode, "error": str(e)})
            torch.cuda.empty_cache()

    out = config.LOG_DIR / f"{config.NAMESPACE}summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary written to", out)
    for row in summary:
        print(row)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # allow: python run_all.py clean catnet resnet50
        args = sys.argv[1:]
        if args[0] in ("clean", "standard", "both"):
            mode_arg = args[0]
            modes = ("clean", "standard") if mode_arg == "both" else (mode_arg,)
            model_args = args[1:] if len(args) > 1 else None
            main(model_args, split_modes=modes)
        else:
            main(args)
    else:
        main()
