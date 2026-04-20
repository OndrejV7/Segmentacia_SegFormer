"""
SegFormer-B2  --  Phase 1 Training  [v2: differential LR]
Loss: Weighted CrossEntropy (class-frequency inverse weighting)
Optimizer: AdamW with split LR -- encoder 6e-5, decoder+head 6e-4

Dataset split (trunk-level, NOT random slice split):
  Train : 17 trunks  (kmen1-7,9 + kmen3 + Dub_1-8,10 + Dub_3b,5)  -- 1155 images
  Val   : kmen8, kmen10, Dub_9                                       --  192 images
  Test  : hrce_mixed  (not touched here)

Class mapping (alphabetical, 0-indexed):
  MATLAB mask value -> Python class ID
  1 Okolie        -> 3
  2 Kora          -> 1
  3 Drevo         -> 0
  4 Nezdrava_hrca -> 2
  5 Prasklina     -> 4

Outputs:
  net_segformer_b2_p1_v2.pth     -- best model weights
  training_history_p1_v2.json    -- per-epoch metrics
"""

import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import time

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
DATA_DIR   = Path(__file__).parent
IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_CLASSES = 5
SEED = 42

CLASS_NAMES = ["Drevo", "Kora", "Nezdrava_hrca", "Okolie", "Prasklina"]

# trunk-level split
ALL_TRUNKS = [
    "kmen1", "kmen2", "kmen3", "kmen4", "kmen5",
    "kmen6", "kmen7", "kmen8", "kmen9", "kmen10",
    "Dub_1", "Dub_2", "Dub_3b", "Dub_4", "Dub_5",
    "Dub_6", "Dub_7", "Dub_8", "Dub_9", "Dub_10",
]
VAL_TRUNKS   = {"kmen8", "kmen10", "Dub_9"}
TRAIN_TRUNKS = [t for t in ALL_TRUNKS if t not in VAL_TRUNKS]

# WeightedRandomSampler multipliers for rare classes
OVERSAMPLE_PRASKLINA = 6
OVERSAMPLE_NEZDRAVA  = 3

# Phase 1 training
NUM_EPOCHS = 100
LR_ENCODER = 6e-5   # pretrained MiT-B2 encoder -- conservative
LR_DECODER = 6e-4   # randomly-init decoder + head -- 10x higher
PATIENCE   = 20

SAVE_PATH    = DATA_DIR / "net_segformer_b2_p1_v2.pth"
HISTORY_PATH = DATA_DIR / "training_history_p1_v2.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MATLAB pixel IDs (1-5) -> Python class IDs (0-4, alphabetical)
# [unused, Okolie->3, Kora->1, Drevo->0, Nezdrava->2, Prasklina->4]
MASK_REMAP = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)


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


def collect_pairs(trunks):
    """Return sorted lists of (image_path, mask_path) for given trunk names."""
    imgs, msks = [], []
    for trunk in trunks:
        img_dir = DATA_DIR / trunk / "images"
        msk_dir = DATA_DIR / trunk / "masks"
        for img_path in sorted(img_dir.glob("*.tif")):
            msk_path = msk_dir / (img_path.stem + ".png")
            if msk_path.exists():
                imgs.append(img_path)
                msks.append(msk_path)
    return imgs, msks


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
        self.masks_raw     = []   # MATLAB IDs 1-5
        self.has_prasklina = []   # MATLAB value 5
        self.has_nezdrava  = []   # MATLAB value 4

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

        # grayscale -> RGB
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # remap MATLAB IDs to Python class IDs *before* augmentation
        # so that mask_value=3 in Rotate correctly fills with Okolie
        msk = MASK_REMAP[msk]

        if self.transform:
            out = self.transform(image=img, mask=msk)
            img = out["image"]
            msk = out["mask"]

        return img.float(), msk.long()


# ═══════════════════════════════════════════════════════════════════════
# Augmentation
# ═══════════════════════════════════════════════════════════════════════
def get_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=0, value=0, mask_value=3, p=1.0),
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
# Metrics
# ═══════════════════════════════════════════════════════════════════════
def compute_iou(preds, targets, num_classes):
    ious = []
    for c in range(num_classes):
        inter = ((preds == c) & (targets == c)).sum().item()
        union = ((preds == c) | (targets == c)).sum().item()
        ious.append(inter / union if union > 0 else float("nan"))
    return ious


# ═══════════════════════════════════════════════════════════════════════
# Train / Val loops
# ═══════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for imgs, msks in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        msks = msks.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out  = model(imgs)
            loss = criterion(out, msks)
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
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    seed_everything(SEED)

    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()} -- {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Device  : {DEVICE}")

    # ── Data ──
    train_imgs, train_msks = collect_pairs(TRAIN_TRUNKS)
    val_imgs,   val_msks   = collect_pairs(list(VAL_TRUNKS))

    print(f"\nTrain trunks ({len(TRAIN_TRUNKS)}): {', '.join(TRAIN_TRUNKS)}")
    print(f"Val   trunks ({len(VAL_TRUNKS)}):   {', '.join(sorted(VAL_TRUNKS))}")
    print(f"Train images : {len(train_imgs)}")
    print(f"Val   images : {len(val_imgs)}")

    train_ds = WoodLogDataset(train_imgs, train_msks, get_train_transform())
    val_ds   = WoodLogDataset(val_imgs,   val_msks,   get_val_transform())

    # Weighted sampler -- oversample rare-class images
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
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    n_prask = sum(train_ds.has_prasklina)
    n_hrca  = sum(train_ds.has_nezdrava)
    print(f"\nTrain images with Prasklina    : {n_prask} ({n_prask/len(train_ds)*100:.1f}%)")
    print(f"Train images with Nezdrava_hrca: {n_hrca}  ({n_hrca/len(train_ds)*100:.1f}%)")

    # ── Class weights (inverse frequency) ──
    print("\nComputing class pixel frequencies on training set...")
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

    # ── Model ──
    model = smp.create_model(
        arch="segformer",
        encoder_name="mit_b2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)
    print(f"\nModel: SegFormer-B2, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"LR: encoder={LR_ENCODER:.1e}  decoder+head={LR_DECODER:.1e}")

    criterion = nn.CrossEntropyLoss(weight=alpha)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),          "lr": LR_ENCODER},
        {"params": model.decoder.parameters(),          "lr": LR_DECODER},
        {"params": model.segmentation_head.parameters(),"lr": LR_DECODER},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ──
    best_miou     = 0.0
    pat_counter   = 0
    history       = []
    col_w         = max(len(n) for n in CLASS_NAMES) + 2

    header = (f"{'Ep':>4} | {'TrLoss':>8} | {'VaLoss':>8} | {'mIoU':>7} | "
              + " | ".join(f"{n:>{col_w}}" for n in CLASS_NAMES)
              + " | LR(enc/dec)")
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        va_loss, class_ious, miou = validate(model, val_loader, criterion)
        scheduler.step(miou)

        lr_enc  = optimizer.param_groups[0]["lr"]
        lr_dec  = optimizer.param_groups[1]["lr"]
        elapsed = time.time() - t0

        record = {
            "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
            "mean_iou": miou, "class_ious": class_ious,
            "lr_encoder": lr_enc, "lr_decoder": lr_dec,
        }
        history.append(record)
        with open(HISTORY_PATH, "w") as f:
            json.dump({"class_names": CLASS_NAMES, "val_trunks": sorted(VAL_TRUNKS),
                       "train_trunks": TRAIN_TRUNKS, "history": history}, f, indent=2)

        iou_str = " | ".join(f"{v*100:>{col_w}.1f}%" for v in class_ious)
        tag = " << best" if miou > best_miou else ""
        print(f"{epoch:>4} | {tr_loss:>8.4f} | {va_loss:>8.4f} | {miou*100:>6.2f}% | "
              f"{iou_str} | enc={lr_enc:.1e} dec={lr_dec:.1e} | {elapsed:.0f}s{tag}")

        if miou > best_miou:
            best_miou   = miou
            pat_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            pat_counter += 1

        if pat_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
            break

    print(f"\nPhase 1 done. Best mIoU: {best_miou*100:.2f}%")
    print(f"Weights : {SAVE_PATH}")
    print(f"History : {HISTORY_PATH}")


if __name__ == "__main__":
    main()
