import json
from pathlib import Path

DATA_DIR = Path(__file__).parent

for fname in ["training_history_p1.json", "training_history_p2.json", "training_history_p2_v2.json"]:
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
