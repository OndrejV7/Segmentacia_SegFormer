"""
SegFormer-B2  --  Phase 2 Training
Loss: 35% Weighted CrossEntropy + 65% Weighted Dice  (refinement)

Loads Phase 1 weights (net_segformer_b2_p1.pth) and fine-tunes.
Same trunk-level train/val split as Phase 1.

Run after Phase 1 completes:
    python train_segformer_phase2.py

Outputs:
  net_segformer_b2_p2.pth     -- best Phase 2 model weights
  training_history_p2.json    -- per-epoch metrics
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

ALL_TRUNKS = [
    "kmen1", "kmen2", "kmen3", "kmen4", "kmen5",
    "kmen6", "kmen7", "kmen8", "kmen9", "kmen10",
    "Dub_1", "Dub_2", "Dub_3b", "Dub_4", "Dub_5",
    "Dub_6", "Dub_7", "Dub_8", "Dub_9", "Dub_10",
    "Dub_praskliny_a", "Dub_praskliny_b",
]
VAL_TRUNKS   = {"kmen8", "kmen10", "Dub_9"}
TRAIN_TRUNKS = [t for t in ALL_TRUNKS if t not in VAL_TRUNKS]

OVERSAMPLE_PRASKLINA = 6
OVERSAMPLE_NEZDRAVA  = 3

# Phase 2 config
NUM_EPOCHS  = 80
LR          = 3e-5      # optimum nájdené z warmup experimentu (v2 peak ep4)
PATIENCE    = 20
CE_WEIGHT   = 0.35
DICE_WEIGHT = 0.65

PHASE1_PATH  = DATA_DIR / "net_segformer_b2_p1.pth"
SAVE_PATH    = DATA_DIR / "net_segformer_b2_p2.pth"
HISTORY_PATH = DATA_DIR / "training_history_p2.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MASK_REMAP = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════
# Loss: Weighted Dice + Weighted CE
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
    def __init__(self, alpha=None, ce_w=0.4, dice_w=0.6):
        super().__init__()
        self.ce      = nn.CrossEntropyLoss(weight=alpha)
        self.dice    = DiceLoss(weight=alpha)
        self.ce_w    = ce_w
        self.dice_w  = dice_w

    def forward(self, inputs, targets):
        return self.ce_w * self.ce(inputs, targets) + self.dice_w * self.dice(inputs, targets)


# ═══════════════════════════════════════════════════════════════════════
# Dataset (identical to Phase 1)
# ═══════════════════════════════════════════════════════════════════════
def collect_pairs(trunks):
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

    if not PHASE1_PATH.exists():
        print(f"\nERROR: Phase 1 weights not found at {PHASE1_PATH}")
        print("Run train_segformer_phase1.py first.")
        return

    # ── Data ──
    train_imgs, train_msks = collect_pairs(TRAIN_TRUNKS)
    val_imgs,   val_msks   = collect_pairs(list(VAL_TRUNKS))

    print(f"\nTrain: {len(train_imgs)} images  |  Val: {len(val_imgs)} images")

    train_ds = WoodLogDataset(train_imgs, train_msks, get_train_transform())
    val_ds   = WoodLogDataset(val_imgs,   val_msks,   get_val_transform())

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

    # ── Class weights ──
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

    # ── Model: load Phase 1 weights ──
    model = smp.create_model(
        arch="segformer",
        encoder_name="mit_b2",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)

    model.load_state_dict(torch.load(PHASE1_PATH, map_location=DEVICE, weights_only=True))
    print(f"\nLoaded Phase 1 weights from {PHASE1_PATH}")
    print(f"Model: SegFormer-B2, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    criterion = DiceCELoss(alpha=alpha, ce_w=CE_WEIGHT, dice_w=DICE_WEIGHT)
    print(f"Loss: {CE_WEIGHT:.0%} weighted CE + {DICE_WEIGHT:.0%} weighted Dice")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-8
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ──
    best_miou   = 0.0
    pat_counter = 0
    history     = []
    col_w       = max(len(n) for n in CLASS_NAMES) + 2

    header = (f"{'Ep':>4} | {'TrLoss':>8} | {'VaLoss':>8} | {'mIoU':>7} | "
              + " | ".join(f"{n:>{col_w}}" for n in CLASS_NAMES)
              + " | LR")
    print(f"\n{'='*len(header)}")
    print(f"  Phase 2: {CE_WEIGHT:.0%} CE + {DICE_WEIGHT:.0%} Dice  |  LR={LR}")
    print(f"{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")

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
        with open(HISTORY_PATH, "w") as f:
            json.dump({"class_names": CLASS_NAMES, "val_trunks": sorted(VAL_TRUNKS),
                       "train_trunks": TRAIN_TRUNKS, "history": history}, f, indent=2)

        iou_str = " | ".join(f"{v*100:>{col_w}.1f}%" for v in class_ious)
        tag = " << best" if miou > best_miou else ""
        print(f"{epoch:>4} | {tr_loss:>8.4f} | {va_loss:>8.4f} | {miou*100:>6.2f}% | "
              f"{iou_str} | {lr_now:.2e} | {elapsed:.0f}s{tag}")

        if miou > best_miou:
            best_miou   = miou
            pat_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            pat_counter += 1

        if pat_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
            break

    print(f"\nPhase 2 done. Best mIoU: {best_miou*100:.2f}%")
    print(f"Weights : {SAVE_PATH}")
    print(f"History : {HISTORY_PATH}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    main()
