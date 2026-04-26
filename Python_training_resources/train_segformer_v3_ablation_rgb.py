"""
SegFormer-B2  --  Training v3 ABLATION RGB (adjacent-slice encoding)
=====================================================================
Identicke s train_segformer_v3_ablation.py okrem:
  - Vstup je 3-kanalovy obrazok kde:
      R = rez (n-1)
      G = rez (n)       <-- centralny, tomu zodpoveda mask
      B = rez (n+1)
  - Trenuje sa na splits/ablation_rgb_{train,val}_files.json
    (z kazdeho kmena vyhodeny prvy a posledny rez)
  - in_channels=3 -- ImageNet pretrained vahy fungovuju natívne
    (kluvocy rozdiel oproti 5-kanal 2.5D experimentu)

Reference (cite-friendly):
  - Roth et al. 2014, "A New 2.5D Representation for Lymph Node Detection"
  - Christ et al. 2016, "Liver and lesion segmentation in CT with FCN"
  - Avesta et al. 2023, "Comparing 3D, 2.5D, and 2D Brain Auto-Segmentation"

Outputs:
  net_segformer_b2_v3_ablation_rgb.pth
  training_history_v3_ablation_rgb.json
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
DICE_WEIGHT = 0.70

TRAIN_SPLIT_NAME = "ablation_rgb_train"
VAL_SPLIT_NAME   = "ablation_rgb_val"
SAVE_PATH        = DATA_DIR / "net_segformer_b2_v3_ablation_rgb.pth"
HISTORY_PATH     = DATA_DIR / "training_history_v3_ablation_rgb.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_REMAP = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════
# Loss (identicke s v3)
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
    def __init__(self, alpha=None, ce_w=0.30, dice_w=0.70):
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


def load_split(name):
    path = SPLITS_DIR / f"{name}_files.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file {path} neexistuje. Spusti najprv "
            f"'python make_splits_ablation_rgb.py'."
        )
    with open(path) as f:
        data = json.load(f)
    imgs, msks = [], []
    for img_rel, msk_rel in data["pairs"]:
        imgs.append(DATA_DIR / img_rel)
        msks.append(DATA_DIR / msk_rel)
    return imgs, msks


# ═══════════════════════════════════════════════════════════════════════
# RGB Adjacent-Slice Dataset
# ═══════════════════════════════════════════════════════════════════════
class WoodLogDatasetRGB(Dataset):
    """
    Dataset ktory pre kazdy split image vytvara 3-kanalovy obrazok:
      R = predchadzajuci rez (n-1)
      G = centralny rez (n)
      B = nasledujuci rez (n+1)
    Anotacia je pre centralny rez.

    Predpoklada ze split images neobsahuju prvy ani posledny rez kmena
    (zabezpeci to make_splits_ablation_rgb.py).
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

        # Identify all referenced trunks
        trunks = sorted(set(p.parent.parent.name for p in image_paths))

        # Pre-load ALL slices from these trunks
        print(f"  Pre-loading slices from {len(trunks)} trunks...",
              end=" ", flush=True)
        self.trunk_data = {}    # trunk -> list of (stem, np.uint8 image)
        self.trunk_index = {}   # (trunk, stem) -> index
        total = 0
        for trunk in trunks:
            img_dir = DATA_DIR / trunk / "images"
            slices = []
            for img_path in sorted(img_dir.glob("*.tif")):
                stem = img_path.stem
                img = np.array(Image.open(img_path))
                slices.append((stem, img))
            self.trunk_data[trunk] = slices
            for i, (stem, _) in enumerate(slices):
                self.trunk_index[(trunk, stem)] = i
            total += len(slices)
        print(f"{total} rezov.")

        # Pre-load masks for split images
        print(f"  Pre-loading {len(image_paths)} masiek...", end=" ", flush=True)
        self.masks_raw     = []
        self.has_prasklina = []
        self.has_nezdrava  = []
        for mp in mask_paths:
            msk = np.array(Image.open(mp))
            self.masks_raw.append(msk)
            self.has_prasklina.append(5 in msk)
            self.has_nezdrava.append(4 in msk)
        print("done.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        trunk = img_path.parent.parent.name
        stem  = img_path.stem
        center_idx = self.trunk_index[(trunk, stem)]
        slices_list = self.trunk_data[trunk]

        # Predpokladame ze split image NIE JE prvy ani posledny
        # (make_splits_ablation_rgb.py to zabezpecil).
        # Pre istotu klampujeme.
        n = len(slices_list)
        i_prev = max(0, center_idx - 1)
        i_curr = center_idx
        i_next = min(n - 1, center_idx + 1)

        s_prev = slices_list[i_prev][1]
        s_curr = slices_list[i_curr][1]
        s_next = slices_list[i_next][1]

        # Ensure 2D grayscale
        for s_arr in (s_prev, s_curr, s_next):
            if s_arr.ndim == 3:
                pass  # ok, ale neocakavame
        if s_prev.ndim == 3: s_prev = s_prev[..., 0]
        if s_curr.ndim == 3: s_curr = s_curr[..., 0]
        if s_next.ndim == 3: s_next = s_next[..., 0]

        # Stack as RGB: (H, W, 3) with channel order [R=prev, G=curr, B=next]
        rgb = np.stack([s_prev, s_curr, s_next], axis=-1)  # uint8 typically

        msk = MASK_REMAP[self.masks_raw[idx]]

        if self.transform:
            out = self.transform(image=rgb, mask=msk)
            rgb = out["image"]
            msk = out["mask"]

        return rgb.float(), msk.long()


def get_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=0, fill=0, fill_mask=3, p=1.0),
        # ImageNet stats -- pretrained encoder ich ocakava (input je uint8 0-255)
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
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    seed_everything(SEED)

    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()} -- "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Device  : {DEVICE}")
    print(f"Splits  : ablation RGB (adjacent-slice encoding, edges dropped)")
    print(f"Loss    : {CE_WEIGHT:.0%} CE + {DICE_WEIGHT:.0%} Dice")
    print(f"Epochs  : {NUM_EPOCHS}, patience={PATIENCE}")
    print(f"Mode    : RGB adjacent-slice encoding  (R=n-1, G=n, B=n+1)")

    train_imgs, train_msks = load_split(TRAIN_SPLIT_NAME)
    val_imgs,   val_msks   = load_split(VAL_SPLIT_NAME)
    print(f"Train: {len(train_imgs)}  |  Val: {len(val_imgs)}")

    train_ds = WoodLogDatasetRGB(train_imgs, train_msks, transform=get_train_transform())
    val_ds   = WoodLogDatasetRGB(val_imgs,   val_msks,   transform=get_val_transform())

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

    print("\nComputing class pixel frequencies (ablation RGB train only)...")
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

    # Model -- in_channels=3 (DEFAULT, ImageNet pretrained funguje native)
    model = smp.create_model(
        arch="segformer", encoder_name="mit_b2",
        encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES,
    ).to(DEVICE)
    print(f"\nModel: SegFormer-B2 (in_channels=3, RGB adjacent-slice), "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    criterion = DiceCELoss(alpha=alpha, ce_w=CE_WEIGHT, dice_w=DICE_WEIGHT)
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
    print(f"  v3 ABLATION RGB: adjacent-slice (R=n-1, G=n, B=n+1), {CE_WEIGHT:.0%} CE + {DICE_WEIGHT:.0%} Dice")
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
            json.dump({
                "class_names": CLASS_NAMES,
                "split_source": "splits/ablation_rgb_{train,val}_files.json",
                "mode": "RGB adjacent-slice (R=n-1, G=n, B=n+1)",
                "loss": f"{CE_WEIGHT} CE + {DICE_WEIGHT} Dice",
                "history": history,
            }, f, indent=2)

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

    print(f"\nv3 ablation RGB done. Best mIoU: {best_miou*100:.2f}%")
    print(f"Weights : {SAVE_PATH}")
    print(f"History : {HISTORY_PATH}")


if __name__ == "__main__":
    main()
