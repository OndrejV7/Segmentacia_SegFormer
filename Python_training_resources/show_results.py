import json
from pathlib import Path

DATA_DIR = Path(__file__).parent

for fname in ["training_history_p1.json", "training_history_p1_rnd.json",
              "training_history_p2.json", "training_history_p2_v2.json",
              "training_history_p2_rnd.json",
              "training_history_single_rnd.json",
              "training_history_single_v2.json",
              "training_history_sweep_ce25_dice75.json",
              "training_history_sweep_ce30_dice70.json",
              "training_history_sweep_ce35_dice65.json",
              "training_history_sweep_ce40_dice60.json",
              "training_history_sweep_ce45_dice55.json",
              "training_history_v3.json",
              "training_history_v3_ablation.json"]:
    path = DATA_DIR / fname
    if not path.exists():
        continue
    with open(path) as f:
        d = json.load(f)
    best = max(d["history"], key=lambda x: x["mean_iou"])
    print(f"=== {fname} ===")
    print(f"  Best epoch : {best['epoch']}  (of {len(d['history'])})")
    print(f"  mIoU       : {best['mean_iou']*100:.2f}%")
    for name, iou in zip(d["class_names"], best["class_ious"]):
        print(f"  {name:<18} {iou*100:.1f}%")
    print()
