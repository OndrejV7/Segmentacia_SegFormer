"""
SegFormer-B2  --  Hyperparameter sweep: CE / Dice ratio
========================================================

Spusti za sebou 5 nezavislych treningov s roznym pomerom CE : Dice
v kombinovanej DiceCE strate. Vsetky ostatne parametre su identicke
s `train_segformer_single_rnd.py` (kanonicky baseline).

Pomery (CE_WEIGHT, DICE_WEIGHT):
    1) 0.25 / 0.75
    2) 0.30 / 0.70
    3) 0.35 / 0.65    <-- baseline
    4) 0.40 / 0.60
    5) 0.45 / 0.55

Vystupy (per run):
    net_segformer_b2_sweep_ce{XX}_dice{YY}.pth
    training_history_sweep_ce{XX}_dice{YY}.json

Sumar:
    sweep_ce_dice_summary.json   -- best mIoU + per-class IoU pre kazdy pomer

Spustenie:
    python sweep_ce_dice_ratio.py
    -- bez argumentov, bez interakcie
    -- 5 x ~30-50 min  ->  ocakavany cas ~3-4 hod na RTX-class GPU
"""

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════
# Sweep configuration
# ═══════════════════════════════════════════════════════════════════════
RATIOS = [
    (0.25, 0.75),
    (0.30, 0.70),
    (0.35, 0.65),    # baseline
    (0.40, 0.60),
    (0.45, 0.55),
]

# ═══════════════════════════════════════════════════════════════════════
# Konfiguracia tréningu (identicka s train_segformer_single_rnd.py)
# ═══════════════════════════════════════════════════════════════════════
DATA_DIR    = Path(__file__).parent
IMAGE_SIZE  = 512
BATCH_SIZE  = 8
NUM_CLASSES = 5
SEED        = 42
SPLIT_SEED  = 0
VAL_RATIO   = 0.20

CLASS_NAMES = ["Drevo", "Kora", "Nezdrava_hrca", "Okolie", "Prasklina"]

ALL_TRUNKS = [
    "kmen1", "kmen2", "kmen3", "kmen4", "kmen5",
    "kmen6", "kmen7", "kmen8", "kmen9", "kmen10",
    "Dub_1", "Dub_2", "Dub_3b", "Dub_4", "Dub_5",
    "Dub_6", "Dub_7", "Dub_8", "Dub_9", "Dub_10",
    "Dub_praskliny_a", "Dub_praskliny_b",
]

OVERSAMPLE_PRASKLINA = 6
OVERSAMPLE_NEZDRAVA  = 3

NUM_EPOCHS  = 100
LR          = 6e-5
PATIENCE    = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_REMAP = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)

SUMMARY_PATH = DATA_DIR / "sweep_ce_dice_summary.json"


# ═══════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════
class DiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1.0):
        super().__init__()
        self.weight = weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        C = inputs.size(1)
        probas     = F.softmax(inputs, dim=1)
        targets_oh = F.one_hot(targets, C).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probas * targets_oh).sum(dims)
        card  = probas.sum(dims) + targets_oh.sum(dims)
        dice  = (2.0 * inter + self.smooth) / (card + self.smooth)
        if self.weight is not None:
            dice = dice * self.weight
            return 1.0 - dice.sum() / self.weight.sum()
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self, alpha=None, ce_w=0.35, dice_w=0.65):
        super().__init__()
        self.ce     = nn.CrossEntropyLoss(weight=alpha)
        self.dice   = DiceLoss(weight=alpha)
        self.ce_w   = ce_w
        self.dice_w = dice_w

    def forward(self, inputs, targets):
        return self.ce_w * self.ce(inputs, targets) + \
               self.dice_w * self.dice(inputs, targets)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def collect_all_pairs():
    imgs, msks = [], []
    for trunk in ALL_TRUNKS:
        img_dir = DATA_DIR / trunk / "images"
        msk_dir = DATA_DIR / trunk / "masks"
        for img_path in sorted(img_dir.glob("*.tif")):
            msk_path = msk_dir / (img_path.stem + ".png")
            if msk_path.exists():
                imgs.append(img_path)
                msks.append(msk_path)
    return imgs, msks


def random_split(imgs, msks, val_ratio, seed):
    n   = len(imgs)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_val     = round(val_ratio * n)
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]
    return ([imgs[i] for i in train_idx], [msks[i] for i in train_idx],
            [imgs[i] for i in val_idx],   [msks[i] for i in val_idx])


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════
class WoodLogDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

        print(f"  Pre-loading {len(image_paths)} images into RAM...", end=" ", flush=True)
        self.images        = []
        self.masks_raw     = []
        self.has_prasklina = []
        self.has_nezdrava  = []
        for ip, mp in zip(image_paths, mask_paths):
            img = np.array(Image.open(ip))
            msk = np.array(Image.open(mp))
            self.images.append(img)
            self.masks_raw.append(msk)
            self.has_prasklina.append(5 in msk)
            self.has_nezdrava.append(4 in msk)
        print("done.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        msk = self.masks_raw[idx].copy()
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        msk = MASK_REMAP[msk]
        if self.transform:
            out = self.transform(image=img, mask=msk)
            img = out["image"]
            msk = out["mask"]
        return img.float(), msk.long()


def get_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=0, fill=0, fill_mask=3, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ═══════════════════════════════════════════════════════════════════════
# Metrics / loops
# ═══════════════════════════════════════════════════════════════════════
def compute_iou(preds, targets, num_classes):
    ious = []
    for c in range(num_classes):
        inter = ((preds == c) & (targets == c)).sum().item()
        union = ((preds == c) | (targets == c)).sum().item()
        ious.append(inter / union if union > 0 else float("nan"))
    return ious


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for imgs, msks in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        msks = msks.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda"):
            loss = criterion(model(imgs), msks)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_ious = [[] for _ in range(NUM_CLASSES)]
    for imgs, msks in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        msks = msks.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out  = model(imgs)
            loss = criterion(out, msks)
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        for c, iou in enumerate(compute_iou(preds.cpu(), msks.cpu(), NUM_CLASSES)):
            if not np.isnan(iou):
                all_ious[c].append(iou)
    class_ious = [np.mean(v) if v else 0.0 for v in all_ious]
    return total_loss / len(loader), class_ious, float(np.nanmean(class_ious))


# ═══════════════════════════════════════════════════════════════════════
# Train one config
# ═══════════════════════════════════════════════════════════════════════
def train_one_run(ce_w, dice_w, train_ds, val_ds, alpha, run_idx, total_runs):
    """
    Spusti jeden tréning s danym pomerom CE:Dice. Vrati dict s vysledkami.
    """
    tag = f"ce{int(ce_w*100):02d}_dice{int(dice_w*100):02d}"
    save_path    = DATA_DIR / f"net_segformer_b2_sweep_{tag}.pth"
    history_path = DATA_DIR / f"training_history_sweep_{tag}.json"

    print("\n" + "=" * 100)
    print(f"  [{run_idx}/{total_runs}]  CE = {ce_w:.2f}   Dice = {dice_w:.2f}")
    print(f"  Save   : {save_path.name}")
    print(f"  History: {history_path.name}")
    print("=" * 100)

    # cisty seed pred kazdym tréningom -> reprodukovatelnost
    seed_everything(SEED)

    # WeightedRandomSampler -- jeho generator pouziva globalny RNG, takze
    # po seed_everything() je sampling identický medzi behmi
    weights = []
    for i in range(len(train_ds)):
        if train_ds.has_prasklina[i]:
            weights.append(OVERSAMPLE_PRASKLINA)
        elif train_ds.has_nezdrava[i]:
            weights.append(OVERSAMPLE_NEZDRAVA)
        else:
            weights.append(1.0)

    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Model -- novy pre kazdy beh, fresh ImageNet weights
    model = smp.create_model(
        arch="segformer", encoder_name="mit_b2",
        encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES,
    ).to(DEVICE)

    criterion = DiceCELoss(alpha=alpha, ce_w=ce_w, dice_w=dice_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler("cuda")

    best_miou         = 0.0
    best_class_ious   = None
    best_epoch        = 0
    pat_counter       = 0
    history           = []
    col_w             = max(len(n) for n in CLASS_NAMES) + 2

    header = (f"{'Ep':>4} | {'TrLoss':>8} | {'VaLoss':>8} | {'mIoU':>7} | "
              + " | ".join(f"{n:>{col_w}}" for n in CLASS_NAMES) + " | LR")
    print(header)
    print("-" * len(header))

    t_run_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        va_loss, class_ious, miou = validate(model, val_loader, criterion)
        scheduler.step(miou)

        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        record = {
            "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
            "mean_iou": miou, "class_ious": class_ious, "lr": lr_now,
        }
        history.append(record)
        with open(history_path, "w") as f:
            json.dump({
                "class_names": CLASS_NAMES,
                "split": "random_80_20", "split_seed": SPLIT_SEED,
                "ce_weight": ce_w, "dice_weight": dice_w,
                "history": history,
            }, f, indent=2)

        iou_str = " | ".join(f"{v*100:>{col_w}.1f}%" for v in class_ious)
        tag_best = " << best" if miou > best_miou else ""
        print(f"{epoch:>4} | {tr_loss:>8.4f} | {va_loss:>8.4f} | {miou*100:>6.2f}% | "
              f"{iou_str} | {lr_now:.2e} | {elapsed:.0f}s{tag_best}")

        if miou > best_miou:
            best_miou       = miou
            best_class_ious = list(class_ious)
            best_epoch      = epoch
            pat_counter     = 0
            torch.save(model.state_dict(), save_path)
        else:
            pat_counter += 1

        if pat_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    elapsed_run = time.time() - t_run_start
    print(f"\n  Run done. Best mIoU: {best_miou*100:.2f}% (ep {best_epoch}/{len(history)})")
    print(f"  Time: {elapsed_run/60:.1f} min")

    # Cleanup pred dalsim behom
    del model, optimizer, scheduler, scaler, criterion
    del train_loader, val_loader, sampler
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "ce_weight": ce_w,
        "dice_weight": dice_w,
        "best_miou": best_miou,
        "best_epoch": best_epoch,
        "best_class_ious": best_class_ious,
        "total_epochs": len(history),
        "elapsed_min": elapsed_run / 60.0,
        "weights_file": save_path.name,
        "history_file": history_path.name,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main sweep loop
# ═══════════════════════════════════════════════════════════════════════
def main():
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()} -- "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Device  : {DEVICE}")
    print(f"Sweep   : {len(RATIOS)} configs (CE:Dice)")
    for i, (a, b) in enumerate(RATIOS, 1):
        print(f"          [{i}] {a:.2f} : {b:.2f}")

    # Datasety pripravime IBA RAZ -- to setri cca 30-60 sekund per beh
    print("\n" + "=" * 100)
    print("  Preparing data once (shared across all 5 runs)")
    print("=" * 100)

    seed_everything(SEED)
    all_imgs, all_msks = collect_all_pairs()
    train_imgs, train_msks, val_imgs, val_msks = random_split(
        all_imgs, all_msks, VAL_RATIO, SPLIT_SEED
    )
    print(f"Train: {len(train_imgs)}  |  Val: {len(val_imgs)}")

    train_ds = WoodLogDataset(train_imgs, train_msks, get_train_transform())
    val_ds   = WoodLogDataset(val_imgs,   val_msks,   get_val_transform())

    # Class weights -- spocitane raz, identicke pre vsetky behy
    print("\nComputing class pixel frequencies...")
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for msk_raw in train_ds.masks_raw:
        msk_py = MASK_REMAP[msk_raw]
        for c in range(NUM_CLASSES):
            pixel_counts[c] += int((msk_py == c).sum())

    freq  = pixel_counts / pixel_counts.sum()
    alpha = torch.tensor(np.median(freq) / freq, dtype=torch.float32).to(DEVICE)
    alpha = torch.clamp(alpha, max=20.0)

    print(f"{'Class':<18} {'Freq %':>8} {'Weight':>8}")
    print("-" * 36)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<16} {freq[i]*100:>7.3f}% {alpha[i].item():>8.2f}")

    # ── Sweep ──
    results        = []
    t_total_start  = time.time()

    for run_idx, (ce_w, dice_w) in enumerate(RATIOS, 1):
        result = train_one_run(ce_w, dice_w, train_ds, val_ds, alpha,
                               run_idx, len(RATIOS))
        results.append(result)

        # Priebezne ukladaj summary -- ak nieco padne, mame aspon ciastocne
        with open(SUMMARY_PATH, "w") as f:
            json.dump({
                "class_names": CLASS_NAMES,
                "results": results,
            }, f, indent=2)

    elapsed_total = time.time() - t_total_start

    # ── Final summary ──
    print("\n" + "=" * 100)
    print(f"  SWEEP DONE -- total time: {elapsed_total/60:.1f} min")
    print("=" * 100)
    header = (f"  {'CE':>5} | {'Dice':>5} | {'mIoU':>7} | {'BestEp':>6} | "
              + " | ".join(f"{n:>14}" for n in CLASS_NAMES))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        ious_str = " | ".join(f"{v*100:>13.1f}%" for v in r["best_class_ious"])
        print(f"  {r['ce_weight']:>5.2f} | {r['dice_weight']:>5.2f} | "
              f"{r['best_miou']*100:>6.2f}% | {r['best_epoch']:>6} | {ious_str}")

    best = max(results, key=lambda r: r["best_miou"])
    print(f"\n  BEST CONFIG: CE={best['ce_weight']:.2f} / Dice={best['dice_weight']:.2f} "
          f"  ->  mIoU = {best['best_miou']*100:.2f}%")
    print(f"  Weights: {best['weights_file']}")
    print(f"\n  Summary saved to: {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
