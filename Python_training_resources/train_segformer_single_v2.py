"""
SegFormer-B2  --  Single-phase Training v2  [rnd: random 80/20 split]
Loss: 35% Weighted CrossEntropy + 65% Weighted Dice

Zmeny oproti train_segformer_single_rnd.py:
  1) CosineAnnealingLR(T_max=80, eta_min=1e-7)  -- hladka schedule
     namiesto schodoviteho ReduceLROnPlateau
  2) Differential learning rate:
       encoder (MiT-B2)      : LR * 0.5  (ImageNet features -- pomalsie)
       decoder (SegFormer hd): LR * 1.0  (nove parametre  -- plnou rychlostou)

Outputs:
  net_segformer_b2_single_v2.pth      -- best model weights
  training_history_single_v2.json     -- per-epoch metrics
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
    "Dub_praskliny_a",
]

OVERSAMPLE_PRASKLINA = 6
OVERSAMPLE_NEZDRAVA  = 3

NUM_EPOCHS    = 80          # fixna dlzka pre cosine schedule
LR            = 6e-5        # "base" LR = decoder LR
ENCODER_LR_MULT = 0.5       # encoder dostane LR * 0.5
WEIGHT_DECAY  = 1e-4
PATIENCE      = 20
CE_WEIGHT     = 0.35
DICE_WEIGHT   = 0.65
ETA_MIN       = 1e-7

SAVE_PATH    = DATA_DIR / "net_segformer_b2_single_v2.pth"
HISTORY_PATH = DATA_DIR / "training_history_single_v2.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MASK_REMAP = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)


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
# Optimizer: differential LR (encoder slower, decoder faster)
# ═══════════════════════════════════════════════════════════════════════
def build_optimizer(model, base_lr, encoder_mult, weight_decay):
    encoder_params, decoder_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(p)
        else:
            decoder_params.append(p)
    param_groups = [
        {"params": encoder_params, "lr": base_lr * encoder_mult, "name": "encoder"},
        {"params": decoder_params, "lr": base_lr,               "name": "decoder"},
    ]
    print(f"  Encoder params: {sum(p.numel() for p in encoder_params)/1e6:.1f}M  "
          f"-> LR = {base_lr * encoder_mult:.2e}")
    print(f"  Decoder params: {sum(p.numel() for p in decoder_params)/1e6:.1f}M  "
          f"-> LR = {base_lr:.2e}")
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    seed_everything(SEED)

    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()} -- "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Device  : {DEVICE}")
    print(f"Split   : random 80/20  (seed={SPLIT_SEED})")
    print(f"Loss    : {CE_WEIGHT:.0%} CE + {DICE_WEIGHT:.0%} Dice")
    print(f"Schedule: CosineAnnealingLR  T_max={NUM_EPOCHS}  eta_min={ETA_MIN}")
    print(f"LR      : decoder={LR}, encoder={LR*ENCODER_LR_MULT}")

    all_imgs, all_msks = collect_all_pairs()
    train_imgs, train_msks, val_imgs, val_msks = random_split(
        all_imgs, all_msks, VAL_RATIO, SPLIT_SEED
    )
    print(f"Train: {len(train_imgs)}  |  Val: {len(val_imgs)}")

    train_ds = WoodLogDataset(train_imgs, train_msks, get_train_transform())
    val_ds   = WoodLogDataset(val_imgs,   val_msks,   get_val_transform())

    weights = []
    for i in range(len(train_ds)):
        if train_ds.has_prasklina[i]:   weights.append(OVERSAMPLE_PRASKLINA)
        elif train_ds.has_nezdrava[i]:  weights.append(OVERSAMPLE_NEZDRAVA)
        else:                           weights.append(1.0)

    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ── Class weights ──
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

    # ── Model ──
    model = smp.create_model(
        arch="segformer", encoder_name="mit_b2",
        encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES,
    ).to(DEVICE)
    print(f"\nModel: SegFormer-B2, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    criterion = DiceCELoss(alpha=alpha, ce_w=CE_WEIGHT, dice_w=DICE_WEIGHT)

    print("\nBuilding optimizer with differential LR...")
    optimizer = build_optimizer(model, base_lr=LR,
                                encoder_mult=ENCODER_LR_MULT,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ──
    best_miou   = 0.0
    pat_counter = 0
    history     = []
    col_w       = max(len(n) for n in CLASS_NAMES) + 2

    header = (f"{'Ep':>4} | {'TrLoss':>8} | {'VaLoss':>8} | {'mIoU':>7} | "
              + " | ".join(f"{n:>{col_w}}" for n in CLASS_NAMES)
              + " | LR(dec)")
    print(f"\n{'='*len(header)}")
    print(f"  Single-phase v2 : cosine + diff-LR")
    print(f"{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        va_loss, class_ious, miou = validate(model, val_loader, criterion)
        scheduler.step()

        lr_dec  = optimizer.param_groups[1]["lr"]   # decoder group
        lr_enc  = optimizer.param_groups[0]["lr"]   # encoder group
        elapsed = time.time() - t0

        record = {
            "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
            "mean_iou": miou, "class_ious": class_ious,
            "lr_encoder": lr_enc, "lr_decoder": lr_dec,
        }
        history.append(record)
        with open(HISTORY_PATH, "w") as f:
            json.dump({
                "class_names": CLASS_NAMES,
                "split": "random_80_20", "split_seed": SPLIT_SEED,
                "loss": f"{CE_WEIGHT} CE + {DICE_WEIGHT} Dice",
                "schedule": f"CosineAnnealingLR T_max={NUM_EPOCHS} eta_min={ETA_MIN}",
                "encoder_lr_mult": ENCODER_LR_MULT,
                "history": history,
            }, f, indent=2)

        iou_str = " | ".join(f"{v*100:>{col_w}.1f}%" for v in class_ious)
        tag = " << best" if miou > best_miou else ""
        print(f"{epoch:>4} | {tr_loss:>8.4f} | {va_loss:>8.4f} | {miou*100:>6.2f}% | "
              f"{iou_str} | {lr_dec:.2e} | {elapsed:.0f}s{tag}")

        if miou > best_miou:
            best_miou   = miou
            pat_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            pat_counter += 1

        if pat_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
            break

    print(f"\nSingle-phase v2 done. Best mIoU: {best_miou*100:.2f}%")
    print(f"Weights : {SAVE_PATH}")
    print(f"History : {HISTORY_PATH}")


if __name__ == "__main__":
    main()
