"""
SegFormer-B2  --  Training v3 + Focal-Tversky Loss
====================================================
Identicke s train_segformer_v3.py okrem loss funkcie:
  - Dice loss nahradeny Focal-Tversky loss s PER-CLASS alpha/beta
  - Cielom je zvysit RECALL na Prasklinu a Nezdravu_hrca
    (FN su penalizovane viac nez FP)

Per-class nastavenie:
  Trieda          alpha   beta   beta/alpha
  Drevo            0.5    0.5     1.0  (= Dice)
  Kora             0.5    0.5     1.0  (= Dice)
  Nezdrava_hrca    0.3    0.7     2.3  (mierny recall boost)
  Okolie           0.5    0.5     1.0  (= Dice)
  Prasklina        0.15   0.85    5.7  (silny recall boost)

Focal exponent gamma=1.33 -- zameriava sa na hard examples.

Reference:
  - Salehi et al. 2017 "Tversky loss for image segmentation"
  - Abraham & Khan 2019 "Novel Focal Tversky loss" (arXiv:1810.07842)

Outputs:
  net_segformer_b2_v3_focaltversky.pth
  training_history_v3_focaltversky.json
"""

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
# Configuration
# ═══════════════════════════════════════════════════════════════════════
DATA_DIR    = Path(__file__).parent
SPLITS_DIR  = DATA_DIR / "splits"
IMAGE_SIZE  = 512
BATCH_SIZE  = 8
NUM_CLASSES = 5
SEED        = 42

CLASS_NAMES = ["Drevo", "Kora", "Nezdrava_hrca", "Okolie", "Prasklina"]

OVERSAMPLE_PRASKLINA = 6
OVERSAMPLE_NEZDRAVA  = 3

NUM_EPOCHS  = 200
LR          = 6e-5
PATIENCE    = 25
CE_WEIGHT   = 0.30
FT_WEIGHT   = 0.70    # Focal-Tversky weight (instead of DICE_WEIGHT)

# ── Per-class Tversky parametre ──
# Drevo, Kora, Nezdrava_hrca, Okolie, Prasklina
ALPHA_PER_CLASS = [0.5, 0.5, 0.30, 0.5, 0.15]   # FP penalty
BETA_PER_CLASS  = [0.5, 0.5, 0.70, 0.5, 0.85]   # FN penalty (recall boost na Nezdravu+Prasklinu)
GAMMA           = 1.33                          # focal exponent

SAVE_PATH    = DATA_DIR / "net_segformer_b2_v3_focaltversky.pth"
HISTORY_PATH = DATA_DIR / "training_history_v3_focaltversky.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_REMAP = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════
class FocalTverskyLoss(nn.Module):
    """
    Focal-Tversky loss with per-class alpha/beta.
        TI(c) = TP / (TP + alpha(c)*FP + beta(c)*FN)
        FT(c) = (1 - TI(c)) ** gamma
        loss  = sum_c ( weight(c) * FT(c) ) / sum_c weight(c)

    Args:
        weight (Tensor [C]) -- per-class skalar (typ. inverse-freq alpha
                               z hlavneho trening loopu)
        alpha  (Tensor [C]) -- per-class FP penalty
        beta   (Tensor [C]) -- per-class FN penalty
        gamma  (float)      -- focal exponent
        smooth (float)      -- numericka stabilita
    """
    def __init__(self, weight=None, alpha=None, beta=None,
                 gamma=1.33, smooth=1.0):
        super().__init__()
        self.weight = weight
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        C = inputs.size(1)
        probas     = F.softmax(inputs, dim=1)
        targets_oh = F.one_hot(targets, C).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        TP = (probas * targets_oh).sum(dims)              # [C]
        FP = (probas * (1 - targets_oh)).sum(dims)        # [C]
        FN = ((1 - probas) * targets_oh).sum(dims)        # [C]

        TI = (TP + self.smooth) / \
             (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal = (1.0 - TI) ** self.gamma                  # [C]

        if self.weight is not None:
            focal = focal * self.weight
            return focal.sum() / self.weight.sum()
        return focal.mean()


class CEFocalTverskyLoss(nn.Module):
    def __init__(self, alpha_freq=None, alpha_tv=None, beta_tv=None,
                 ce_w=0.30, ft_w=0.70, gamma=1.33):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=alpha_freq)
        self.ft = FocalTverskyLoss(weight=alpha_freq,
                                   alpha=alpha_tv, beta=beta_tv,
                                   gamma=gamma)
        self.ce_w = ce_w
        self.ft_w = ft_w

    def forward(self, inputs, targets):
        return self.ce_w * self.ce(inputs, targets) + \
               self.ft_w * self.ft(inputs, targets)


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


def load_split(name):
    path = SPLITS_DIR / f"{name}_files.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file {path} neexistuje.")
    with open(path) as f:
        data = json.load(f)
    imgs, msks = [], []
    for img_rel, msk_rel in data["pairs"]:
        imgs.append(DATA_DIR / img_rel)
        msks.append(DATA_DIR / msk_rel)
    return imgs, msks


# ═══════════════════════════════════════════════════════════════════════
# Dataset (identicky s v3)
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


def compute_recall(preds, targets, num_classes):
    """Per-class recall pre logging (sledujeme ako sa Prask/Nez vyvija)."""
    recalls = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else float("nan"))
    return recalls


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
    all_ious    = [[] for _ in range(NUM_CLASSES)]
    all_recalls = [[] for _ in range(NUM_CLASSES)]
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
        for c, r in enumerate(compute_recall(preds.cpu(), msks.cpu(), NUM_CLASSES)):
            if not np.isnan(r):
                all_recalls[c].append(r)
    class_ious    = [np.mean(v) if v else 0.0 for v in all_ious]
    class_recalls = [np.mean(v) if v else 0.0 for v in all_recalls]
    return total_loss / len(loader), class_ious, class_recalls, \
           float(np.nanmean(class_ious))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    seed_everything(SEED)

    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()} -- "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Device  : {DEVICE}")
    print(f"Splits  : hybrid (full data) -- splits/{{train,val}}_files.json")
    print(f"Loss    : {CE_WEIGHT:.0%} CE + {FT_WEIGHT:.0%} Focal-Tversky (gamma={GAMMA})")
    print(f"Epochs  : {NUM_EPOCHS}, patience={PATIENCE}")

    print(f"\nPer-class Tversky alpha/beta:")
    print(f"  {'Trieda':<18} {'alpha':>6} {'beta':>6} {'beta/alpha':>10}")
    for i, name in enumerate(CLASS_NAMES):
        a, b = ALPHA_PER_CLASS[i], BETA_PER_CLASS[i]
        ratio = b / a if a > 0 else float("inf")
        print(f"  {name:<18} {a:>6.2f} {b:>6.2f} {ratio:>10.2f}")

    train_imgs, train_msks = load_split("train")
    val_imgs,   val_msks   = load_split("val")
    print(f"\nTrain: {len(train_imgs)}  |  Val: {len(val_imgs)}")

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

    print("\nComputing class pixel frequencies (full train)...")
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for msk_raw in train_ds.masks_raw:
        msk_py = MASK_REMAP[msk_raw]
        for c in range(NUM_CLASSES):
            pixel_counts[c] += int((msk_py == c).sum())

    freq       = pixel_counts / pixel_counts.sum()
    alpha_freq = torch.tensor(np.median(freq) / freq, dtype=torch.float32).to(DEVICE)
    alpha_freq = torch.clamp(alpha_freq, max=20.0)

    print(f"{'Class':<18} {'Freq %':>8} {'Weight':>8}")
    print("-" * 36)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<16} {freq[i]*100:>7.3f}% {alpha_freq[i].item():>8.2f}")

    # Per-class Tversky tensors
    alpha_tv = torch.tensor(ALPHA_PER_CLASS, dtype=torch.float32).to(DEVICE)
    beta_tv  = torch.tensor(BETA_PER_CLASS,  dtype=torch.float32).to(DEVICE)

    model = smp.create_model(
        arch="segformer", encoder_name="mit_b2",
        encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES,
    ).to(DEVICE)
    print(f"\nModel: SegFormer-B2, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    criterion = CEFocalTverskyLoss(
        alpha_freq=alpha_freq,         # inverse-freq class weights (vstupne)
        alpha_tv=alpha_tv, beta_tv=beta_tv,
        ce_w=CE_WEIGHT, ft_w=FT_WEIGHT, gamma=GAMMA,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler("cuda")

    best_miou   = 0.0
    pat_counter = 0
    history     = []
    col_w       = max(len(n) for n in CLASS_NAMES) + 2

    header = (f"{'Ep':>4} | {'TrLoss':>8} | {'VaLoss':>8} | {'mIoU':>7} | "
              + " | ".join(f"{n:>{col_w}}" for n in CLASS_NAMES)
              + " | LR")
    print(f"\n{'='*len(header)}")
    print(f"  v3 + FOCAL-TVERSKY: {CE_WEIGHT:.0%} CE + {FT_WEIGHT:.0%} FT (gamma={GAMMA})")
    print(f"{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        va_loss, class_ious, class_recalls, miou = validate(model, val_loader, criterion)
        scheduler.step(miou)

        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        record = {
            "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
            "mean_iou": miou, "class_ious": class_ious,
            "class_recalls": class_recalls, "lr": lr_now,
        }
        history.append(record)
        with open(HISTORY_PATH, "w") as f:
            json.dump({
                "class_names": CLASS_NAMES,
                "loss": f"{CE_WEIGHT} CE + {FT_WEIGHT} Focal-Tversky",
                "alpha_per_class": ALPHA_PER_CLASS,
                "beta_per_class":  BETA_PER_CLASS,
                "gamma": GAMMA,
                "history": history,
            }, f, indent=2)

        iou_str = " | ".join(f"{v*100:>{col_w}.1f}%" for v in class_ious)
        tag = " << best" if miou > best_miou else ""
        print(f"{epoch:>4} | {tr_loss:>8.4f} | {va_loss:>8.4f} | {miou*100:>6.2f}% | "
              f"{iou_str} | {lr_now:.2e} | {elapsed:.0f}s{tag}")
        # Per-epoch recall pre Prask + Nez (kluvocy signal)
        print(f"     Recalls: Nez={class_recalls[2]*100:.1f}%  Prask={class_recalls[4]*100:.1f}%")

        if miou > best_miou:
            best_miou   = miou
            pat_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            pat_counter += 1

        if pat_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
            break

    print(f"\nv3 Focal-Tversky done. Best mIoU: {best_miou*100:.2f}%")
    print(f"Weights : {SAVE_PATH}")
    print(f"History : {HISTORY_PATH}")


if __name__ == "__main__":
    main()
