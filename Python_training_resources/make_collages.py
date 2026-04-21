"""
Visual collages for test set evaluation.

Each collage (PNG + one page in stacked TIFF):
  Left   : input image  (original resolution, resized to IMAGE_SIZE)
  Middle : predicted mask (colourised overlay)
  Right  : ground truth mask (colourised overlay)

Requires evaluate_test.py to be run first (predictions/ folder must exist).

Usage:
    python make_collages.py
    python make_collages.py --max 50
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# ── Config ───────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent
TEST_TRUNK  = "hrce_mixed"
PRED_DIR    = DATA_DIR / "predictions"
COLLAGE_DIR = DATA_DIR / "collages"
TIFF_OUT    = DATA_DIR / "collages_all.tif"
IMAGE_SIZE  = 512
OVERLAY_A   = 0.45

CLASS_NAMES = ["Drevo", "Kora", "Nezdrava_hrca", "Okolie", "Prasklina"]
CLASS_COLORS = np.array([
    [160, 100,  40],   # 0 Drevo
    [ 80,  40,  10],   # 1 Kora
    [220,  50,  50],   # 2 Nezdrava_hrca
    [200, 200, 200],   # 3 Okolie
    [255, 220,   0],   # 4 Prasklina
], dtype=np.uint8)

MASK_REMAP = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)
LEGEND_H   = 28


# ── Helpers ──────────────────────────────────────────────────────────────
def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in enumerate(CLASS_COLORS):
        out[mask == c] = col
    return out


def blend(img: np.ndarray, mask_rgb: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(img * (1 - alpha) + mask_rgb * alpha, 0, 255).astype(np.uint8)


def resize_img(arr: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize((size, size)))


def resize_mask(arr: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize((size, size), Image.NEAREST))


def make_legend(width: int) -> Image.Image:
    leg = Image.new("RGB", (width, LEGEND_H), (30, 30, 30))
    draw = ImageDraw.Draw(leg)
    sw = width // len(CLASS_NAMES)
    for i, (name, col) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        x0 = i * sw
        draw.rectangle([x0 + 2, 4, x0 + 16, LEGEND_H - 4], fill=tuple(col))
        draw.text((x0 + 20, 7), name, fill=(255, 255, 255))
    return leg


def add_header(img_pil: Image.Image, texts: list[str]) -> Image.Image:
    """Add panel title bar above the image."""
    header_h = 20
    panel_w  = img_pil.width // len(texts)
    header   = Image.new("RGB", (img_pil.width, header_h), (50, 50, 50))
    draw     = ImageDraw.Draw(header)
    for i, text in enumerate(texts):
        draw.text((i * panel_w + 6, 3), text, fill=(220, 220, 220))
    out = Image.new("RGB", (img_pil.width, img_pil.height + header_h))
    out.paste(header, (0, 0))
    out.paste(img_pil, (0, header_h))
    return out


# ── Main ──────────────────────────────────────────────────────────────────
def main(max_images: int):
    img_dir = DATA_DIR / TEST_TRUNK / "images"
    msk_dir = DATA_DIR / TEST_TRUNK / "masks"

    img_paths = sorted(img_dir.glob("*.tif"))
    if max_images:
        img_paths = img_paths[:max_images]

    COLLAGE_DIR.mkdir(exist_ok=True)
    pages = []

    for img_path in img_paths:
        msk_path  = msk_dir / (img_path.stem + ".png")
        pred_path = PRED_DIR / (img_path.stem + ".png")

        if not msk_path.exists() or not pred_path.exists():
            print(f"  SKIP {img_path.name}  (missing mask or prediction)")
            continue

        img_np   = np.array(Image.open(img_path))
        msk_raw  = np.array(Image.open(msk_path))
        pred_raw = np.array(Image.open(pred_path))

        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)

        msk_gt = MASK_REMAP[msk_raw]

        img_r  = resize_img(img_np,  IMAGE_SIZE)
        gt_r   = resize_mask(msk_gt, IMAGE_SIZE)
        pred_r = resize_mask(pred_raw, IMAGE_SIZE)

        panel_img  = img_r
        panel_pred = blend(img_r, colorize(pred_r), OVERLAY_A)
        panel_gt   = blend(img_r, colorize(gt_r),   OVERLAY_A)

        row     = np.concatenate([panel_img, panel_pred, panel_gt], axis=1)
        row_pil = Image.fromarray(row)
        row_pil = add_header(row_pil, ["Vstupný obrázok", "Predikcia modelu", "Ground truth"])

        legend = make_legend(row_pil.width)
        page   = Image.new("RGB", (row_pil.width, row_pil.height + LEGEND_H))
        page.paste(row_pil, (0, 0))
        page.paste(legend,  (0, row_pil.height))

        out_png = COLLAGE_DIR / (img_path.stem + ".png")
        page.save(out_png)
        pages.append(page)

    if not pages:
        print("No collages created. Run evaluate_test.py first.")
        return

    # Convert to clean images (removes encoder config that causes PIL TIFF errors)
    clean = [Image.fromarray(np.array(p)) for p in pages]
    clean[0].save(TIFF_OUT, save_all=True, append_images=clean[1:])

    print(f"Collages : {COLLAGE_DIR}/  ({len(pages)} PNG files)")
    print(f"TIFF     : {TIFF_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0, help="max images (0 = all)")
    args = parser.parse_args()
    main(args.max)
