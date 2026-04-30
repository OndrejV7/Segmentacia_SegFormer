"""
Augmentation visual check
Saves N random training samples as a stacked TIFF.

Each page (3 panels, left to right):
  1. Original image  + coloured mask overlay
  2. Augmented image + coloured mask overlay
  3. Coloured mask alone  (border colour = fill_mask verification)

Run:
    python check_augmentation.py
Output:
    augmentation_check.tif
"""

import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent
IMAGE_SIZE  = 512
N_SAMPLES   = 20
SEED        = 123
OUTPUT_PATH = DATA_DIR / "augmentation_check.tif"
OVERLAY_A   = 0.45   # mask opacity in overlay panels

CLASS_NAMES = ["Drevo", "Kora", "Nezdrava_hrca", "Okolie", "Prasklina"]

#  RGB colours per class
CLASS_COLORS = np.array([
    [160, 100,  40],   # 0 Drevo        brown
    [ 80,  40,  10],   # 1 Kora         dark brown
    [220,  50,  50],   # 2 Nezdrava     red
    [200, 200, 200],   # 3 Okolie       light grey
    [255, 220,   0],   # 4 Prasklina    yellow
], dtype=np.uint8)

MASK_REMAP = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)

ALL_TRUNKS = [
    "kmen1", "kmen2", "kmen3", "kmen4", "kmen5",
    "kmen6", "kmen7", "kmen8", "kmen9", "kmen10",
    "Dub_1", "Dub_2", "Dub_3b", "Dub_4", "Dub_5",
    "Dub_6", "Dub_7", "Dub_8", "Dub_9", "Dub_10",
    "Dub_praskliny_a", "Dub_praskliny_b",
]
VAL_TRUNKS   = {"kmen8", "kmen10", "Dub_9"}
TRAIN_TRUNKS = [t for t in ALL_TRUNKS if t not in VAL_TRUNKS]

# ── Augmentation (no Normalize / ToTensor -- visualisation only) ────────
import albumentations as A

def get_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=0, fill=0, fill_mask=3, p=1.0),
    ])


# ── Helpers ──────────────────────────────────────────────────────────────
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


def colorize(mask: np.ndarray) -> np.ndarray:
    """Integer mask (H,W) -> RGB (H,W,3)."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in enumerate(CLASS_COLORS):
        out[mask == c] = col
    return out


def blend(img_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(img_rgb * (1 - alpha) + mask_rgb * alpha, 0, 255).astype(np.uint8)


def add_label(img_pil: Image.Image, text: str) -> Image.Image:
    """Burn a small text label into the top-left corner."""
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([0, 0, img_pil.width, 18], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return img_pil


def make_legend(width: int, height: int = 28) -> Image.Image:
    """Horizontal colour legend strip."""
    leg = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(leg)
    n  = len(CLASS_NAMES)
    sw = width // n
    for i, (name, col) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        x0 = i * sw
        draw.rectangle([x0, 2, x0 + 18, height - 2], fill=tuple(col))
        draw.text((x0 + 22, 6), name, fill=(255, 255, 255))
    return leg


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    imgs, msks = collect_pairs(TRAIN_TRUNKS)
    print(f"Found {len(imgs)} training pairs. Sampling {N_SAMPLES}...")

    transform = get_train_transform()
    indices   = random.sample(range(len(imgs)), N_SAMPLES)
    pages     = []

    for i, idx in enumerate(indices):
        img_np  = np.array(Image.open(imgs[idx]))
        msk_raw = np.array(Image.open(msks[idx]))

        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)

        msk_py = MASK_REMAP[msk_raw]

        # ── Resize original for left panel ──
        img_res = np.array(Image.fromarray(img_np).resize((IMAGE_SIZE, IMAGE_SIZE)))
        msk_res = np.array(
            Image.fromarray(msk_py).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        )

        # ── Augment ──
        out     = transform(image=img_np.copy(), mask=msk_py.copy())
        img_aug = out["image"]
        msk_aug = out["mask"]

        # ── Panels ──
        panel_orig = blend(img_res,  colorize(msk_res), OVERLAY_A)
        panel_aug  = blend(img_aug,  colorize(msk_aug), OVERLAY_A)
        panel_msk  = colorize(msk_aug)

        row = np.concatenate([panel_orig, panel_aug, panel_msk], axis=1)
        row_pil = Image.fromarray(row)

        label = f"{imgs[idx].parent.parent.name}/{imgs[idx].name}"
        add_label(row_pil, label)

        # Legend at the bottom
        legend = make_legend(row_pil.width)
        page   = Image.new("RGB", (row_pil.width, row_pil.height + legend.height))
        page.paste(row_pil, (0, 0))
        page.paste(legend,  (0, row_pil.height))
        pages.append(page)

        # Class pixel % in augmented mask
        total = msk_aug.size
        stats = "  ".join(
            f"{CLASS_NAMES[c][0]}:{(msk_aug==c).sum()/total*100:.0f}%"
            for c in range(5)
        )
        print(f"  [{i+1:2d}/{N_SAMPLES}] {label}  |  {stats}")

    pages[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=pages[1:],
        compression="tiff_lzw",
    )
    print(f"\nSaved -> {OUTPUT_PATH}")
    print(f"Panel layout per page:  [orig+mask | aug+mask | mask]  ({IMAGE_SIZE*3} x {IMAGE_SIZE+28} px)")
    print(f"Check: rotated border should be GREY (Okolie), not BROWN (Drevo).")


if __name__ == "__main__":
    main()
