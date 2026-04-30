"""
Test set evaluation -- nacita test set z splits/test_files.json.

Runs inference with the best trained model, computes full per-class metrics
from a confusion matrix and saves:
  predictions/          -- predicted masks as PNG (class IDs 0-4)
  metrics.json          -- full metrics dict
  metrics.csv           -- per-class table

Usage:
    python evaluate_test.py
    python evaluate_test.py --model net_segformer_b2_v3.pth
    python evaluate_test.py --model net_segformer_b2_v3.pth --test-split splits/test_files.json
"""

import argparse
import json
import csv
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent
TEST_SPLIT  = DATA_DIR / "splits" / "test_v3_files.json"
IMAGE_SIZE  = 512
NUM_CLASSES = 5
BATCH_SIZE  = 8

CLASS_NAMES = ["Drevo", "Kora", "Nezdrava_hrca", "Okolie", "Prasklina"]
MASK_REMAP  = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)

DEFAULT_MODEL = DATA_DIR / "net_segformer_b2_v3.pth"
# Vystupy zavisia od modelu + voliteľneho suffixu (napr. test split tag),
# aby sa nepreplsali pri behu na inom modeli alebo testu.
# Napr. net_segformer_b2_v3.pth + suffix='_3trunks' -> predictions_v3_3trunks/
def output_paths(model_path: Path, suffix: str = ""):
    stem = model_path.stem.replace("net_segformer_b2_", "") + suffix
    return (
        DATA_DIR / f"predictions_{stem}",
        DATA_DIR / f"metrics_{stem}.json",
        DATA_DIR / f"metrics_{stem}.csv",
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ──────────────────────────────────────────────────────────────
def get_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def load_test_split(split_path: Path):
    """Nacita test split JSON, vrati zoznam absolutnych ciest (img, msk)."""
    if not split_path.exists():
        raise FileNotFoundError(
            f"Test split {split_path} neexistuje. Spusti 'python make_splits.py'."
        )
    with open(split_path) as f:
        data = json.load(f)
    imgs, msks = [], []
    for img_rel, msk_rel in data["pairs"]:
        imgs.append(DATA_DIR / img_rel)
        msks.append(DATA_DIR / msk_rel)
    return imgs, msks


def unique_stem(img_path: Path) -> str:
    """
    Vytvori unikatny key z ciesty -- '{trunk}_{stem}'. Test set obsahuje
    snimky z 5 roznych trunkov a viacere maju rovnake mena (001.tif),
    takze samotny stem nestaci.
    """
    # struktura: <trunk>/images/<stem>.tif
    trunk = img_path.parent.parent.name
    return f"{trunk}_{img_path.stem}"


def metrics_from_cm(cm):
    """Derive per-class and aggregate metrics from a confusion matrix."""
    C = cm.shape[0]
    results = {}

    ious, dices, precisions, recalls, f1s = [], [], [], [], []
    per_class = {}

    for c in range(C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp      # predicted c, actually other
        fn = cm[c, :].sum() - tp      # actually c, predicted other
        tn = cm.sum() - tp - fp - fn

        iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        dice      = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1        = dice   # F1 == Dice for binary per-class

        per_class[CLASS_NAMES[c]] = {
            "iou":       round(float(iou), 4),
            "dice":      round(float(dice), 4),
            "precision": round(float(precision), 4),
            "recall":    round(float(recall), 4),
            "f1":        round(float(f1), 4),
            "support":   int(cm[c, :].sum()),   # GT pixels of this class
        }
        ious.append(iou);  dices.append(dice)
        precisions.append(precision);  recalls.append(recall);  f1s.append(f1)

    total_px   = int(cm.sum())
    correct_px = int(np.diag(cm).sum())

    results["per_class"]        = per_class
    results["mean_iou"]         = round(float(np.nanmean(ious)), 4)
    results["mean_dice"]        = round(float(np.nanmean(dices)), 4)
    results["mean_precision"]   = round(float(np.nanmean(precisions)), 4)
    results["mean_recall"]      = round(float(np.nanmean(recalls)), 4)
    results["mean_f1"]          = round(float(np.nanmean(f1s)), 4)
    results["pixel_accuracy"]   = round(correct_px / total_px, 4)
    results["total_pixels"]     = total_px
    results["confusion_matrix"] = cm.tolist()
    return results


# ── Main ──────────────────────────────────────────────────────────────────
def main(model_path: Path, test_split: Path, output_suffix: str = ""):
    pred_dir, metrics_json, metrics_csv = output_paths(model_path, output_suffix)

    print(f"Model      : {model_path.name}")
    print(f"Device     : {DEVICE}")
    print(f"Test split : {test_split.relative_to(DATA_DIR)}")
    print(f"Outputs    : {pred_dir.name}/, {metrics_json.name}, {metrics_csv.name}")

    img_paths, msk_paths = load_test_split(test_split)
    print(f"Images     : {len(img_paths)}\n")

    if not img_paths:
        print(f"ERROR: no image/mask pairs found in {test_split}")
        return

    pred_dir.mkdir(exist_ok=True)

    # ── Load model ──
    model = smp.create_model(
        arch="segformer", encoder_name="mit_b2",
        encoder_weights=None, in_channels=3, classes=NUM_CLASSES,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    transform = get_transform()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for img_path, msk_path in tqdm(zip(img_paths, msk_paths), total=len(img_paths), desc="Inference"):
        img_np  = np.array(Image.open(img_path))
        msk_raw = np.array(Image.open(msk_path))

        orig_h, orig_w = msk_raw.shape[:2]

        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)

        msk_gt = MASK_REMAP[msk_raw]

        out    = transform(image=img_np, mask=msk_gt)
        tensor = out["image"].unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                logits = model(tensor)

        # Upsample back to original resolution for metric computation
        logits_up = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Save prediction (unique stem -- viacere trunky mozu mat rovnake mena)
        Image.fromarray(pred).save(pred_dir / (unique_stem(img_path) + ".png"))

        # Accumulate confusion matrix
        mask_flat = msk_gt.ravel() if msk_gt.shape == pred.shape else \
            np.array(Image.fromarray(msk_gt).resize((orig_w, orig_h), Image.NEAREST)).ravel()
        pred_flat = pred.ravel()
        np.add.at(cm, (mask_flat, pred_flat), 1)

    # ── Metrics ──
    results = metrics_from_cm(cm)
    results["model"]      = model_path.name
    results["test_split"] = str(test_split.relative_to(DATA_DIR).as_posix())
    results["n_images"]   = len(img_paths)

    with open(metrics_json, "w") as f:
        json.dump(results, f, indent=2)

    # ── CSV ──
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Class", "IoU", "Dice/F1", "Precision", "Recall", "Support (px)"])
        for name in CLASS_NAMES:
            m = results["per_class"][name]
            w.writerow([name,
                        f"{m['iou']*100:.2f}%", f"{m['dice']*100:.2f}%",
                        f"{m['precision']*100:.2f}%", f"{m['recall']*100:.2f}%",
                        m["support"]])
        w.writerow([])
        w.writerow(["Mean (macro)", f"{results['mean_iou']*100:.2f}%",
                    f"{results['mean_dice']*100:.2f}%",
                    f"{results['mean_precision']*100:.2f}%",
                    f"{results['mean_recall']*100:.2f}%", ""])
        w.writerow(["Pixel accuracy", f"{results['pixel_accuracy']*100:.2f}%", "", "", "", ""])

    # ── Print ──
    print(f"\n{'='*64}")
    print(f"  {'Class':<18} {'IoU':>7} {'Dice':>7} {'Prec':>7} {'Recall':>7}")
    print(f"  {'-'*54}")
    for name in CLASS_NAMES:
        m = results["per_class"][name]
        print(f"  {name:<18} {m['iou']*100:>6.1f}% {m['dice']*100:>6.1f}%"
              f" {m['precision']*100:>6.1f}% {m['recall']*100:>6.1f}%")
    print(f"  {'-'*54}")
    print(f"  {'Mean (macro)':<18} {results['mean_iou']*100:>6.1f}%"
          f" {results['mean_dice']*100:>6.1f}%"
          f" {results['mean_precision']*100:>6.1f}%"
          f" {results['mean_recall']*100:>6.1f}%")
    print(f"  Pixel accuracy: {results['pixel_accuracy']*100:.2f}%")
    print(f"{'='*64}")
    print(f"\nSaved: {metrics_json}")
    print(f"Saved: {metrics_csv}")
    print(f"Saved: {pred_dir}/  ({len(img_paths)} prediction masks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--test-split",    type=Path, default=TEST_SPLIT)
    parser.add_argument("--output-suffix", type=str,  default="",
                        help="Pripony k vystupnym subovrom (napr. '_3trunks')")
    args = parser.parse_args()
    main(args.model, args.test_split, args.output_suffix)
