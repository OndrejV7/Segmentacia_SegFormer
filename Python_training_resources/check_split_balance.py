"""
Porovnanie zastúpenia vzácnych tried v train/val sete
pre oba typy splitov: trunk-level vs random 80/20
"""
import numpy as np
from pathlib import Path
from PIL import Image

DATA_DIR   = Path(__file__).parent
MASK_REMAP = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)
CLASS_NAMES = ["Drevo", "Kora", "Nezdrava_hrca", "Okolie", "Prasklina"]

ALL_TRUNKS = [
    "kmen1","kmen2","kmen3","kmen4","kmen5",
    "kmen6","kmen7","kmen8","kmen9","kmen10",
    "Dub_1","Dub_2","Dub_3b","Dub_4","Dub_5",
    "Dub_6","Dub_7","Dub_8","Dub_9","Dub_10",
    "Dub_praskliny_a",
]
VAL_TRUNKS_TL = {"kmen8", "kmen10", "Dub_9"}


def collect_all():
    imgs, msks, trunks = [], [], []
    for trunk in ALL_TRUNKS:
        for img_path in sorted((DATA_DIR / trunk / "images").glob("*.tif")):
            msk_path = DATA_DIR / trunk / "masks" / (img_path.stem + ".png")
            if msk_path.exists():
                imgs.append(img_path); msks.append(msk_path); trunks.append(trunk)
    return imgs, msks, trunks


def analyze(msk_paths, label):
    px   = np.zeros(5, dtype=np.int64)
    n_nezdrava = 0
    n_prasklina = 0
    for mp in msk_paths:
        msk = MASK_REMAP[np.array(Image.open(mp))]
        for c in range(5):
            px[c] += (msk == c).sum()
        if (msk == 2).any(): n_nezdrava  += 1
        if (msk == 4).any(): n_prasklina += 1
    total = px.sum()
    print(f"\n── {label}  ({len(msk_paths)} snímok) ──")
    print(f"  {'Trieda':<18} {'Pixely':>12}  {'%':>7}  {'Snímky s triedou':>18}")
    print(f"  {'-'*58}")
    for c, name in enumerate(CLASS_NAMES):
        n_img = n_nezdrava if c == 2 else (n_prasklina if c == 4 else "–")
        pct = px[c] / total * 100
        img_str = f"{n_img} ({n_img/len(msk_paths)*100:.1f}%)" if isinstance(n_img, int) else n_img
        print(f"  {name:<18} {px[c]:>12,}  {pct:>6.3f}%  {img_str:>18}")


def main():
    imgs, msks, trunks = collect_all()
    print(f"Celkový dataset: {len(imgs)} snímok zo {len(ALL_TRUNKS)} kmeňov")

    # ── Trunk-level split ──
    tl_train_m = [m for m, t in zip(msks, trunks) if t not in VAL_TRUNKS_TL]
    tl_val_m   = [m for m, t in zip(msks, trunks) if t in VAL_TRUNKS_TL]
    analyze(tl_train_m, "TRUNK-LEVEL  Train")
    analyze(tl_val_m,   "TRUNK-LEVEL  Val  ")

    # ── Random 80/20 split ──
    n = len(imgs)
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    n_val = round(0.20 * n)
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]
    rnd_train_m = [msks[i] for i in train_idx]
    rnd_val_m   = [msks[i] for i in val_idx]
    analyze(rnd_train_m, "RANDOM 80/20  Train")
    analyze(rnd_val_m,   "RANDOM 80/20  Val  ")


if __name__ == "__main__":
    main()
