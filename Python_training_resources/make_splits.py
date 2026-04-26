"""
Definuje train / val / test split pre dataset SegFormer-B2.

Test set (fixny, trunk-level):
    - kmen4    (cely)            64 snimok
    - kmen9    (cely)            64 snimok
    - Dub_3b   (cely)            64 snimok
    - hrce_mixed[:15]            15 snimok (prvych 15 podla mena)
    - Dub_praskliny_a[:15]       15 snimok (prvych 15 podla mena)
    -------------------------------------
    SPOLU                        222 snimok (cca 16% datasetu)

Train + Val:
    Zvysok (1175 snimok) sa nahodne rozdeli 80/20 -- bez ohladu na trunk
    (within-trunk leakage je akceptovatelny, lebo TEST je trunk-level OK).

Vystupy (JSON s relativnymi cestami od DATA_DIR):
    splits/test_files.json
    splits/train_files.json
    splits/val_files.json
    splits/split_stats.json

Spustenie:
    python make_splits.py
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image

DATA_DIR    = Path(__file__).parent
SPLITS_DIR  = DATA_DIR / "splits"
SPLITS_DIR.mkdir(exist_ok=True)

MASK_REMAP  = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)
CLASS_NAMES = ["Drevo", "Kora", "Nezdrava_hrca", "Okolie", "Prasklina"]

ALL_TRUNKS = [
    "kmen1", "kmen2", "kmen3", "kmen4", "kmen5",
    "kmen6", "kmen7", "kmen8", "kmen9", "kmen10",
    "Dub_1", "Dub_2", "Dub_3b", "Dub_4", "Dub_5",
    "Dub_6", "Dub_7", "Dub_8", "Dub_9", "Dub_10",
    "Dub_praskliny_a", "hrce_mixed",
]

# ── Test set definicia ─────────────────────────────────────────
TEST_FULL_TRUNKS = ["kmen4", "kmen9", "Dub_3b"]    # cele trunky idu do testu
TEST_PARTIAL = {                                    # prvych N snimok
    "hrce_mixed":      15,
    "Dub_praskliny_a": 15,
}

# ── Train/Val rozdelenie ───────────────────────────────────────
VAL_RATIO  = 0.20
SPLIT_SEED = 0


def collect_pairs(trunk):
    """Pre dany trunk vrati zoradeny zoznam (img_path, msk_path) parov."""
    img_dir = DATA_DIR / trunk / "images"
    msk_dir = DATA_DIR / trunk / "masks"
    pairs = []
    for img_path in sorted(img_dir.glob("*.tif")):
        msk_path = msk_dir / (img_path.stem + ".png")
        if msk_path.exists():
            pairs.append((img_path, msk_path))
    return pairs


def to_relative(pairs):
    """Konvertuje (img,msk) Path tuples na relativne string cesty."""
    return [(str(i.relative_to(DATA_DIR).as_posix()),
             str(m.relative_to(DATA_DIR).as_posix())) for i, m in pairs]


def class_stats(msk_paths_rel):
    """Spocita pixelove zastupenie a #snimok obsahujucich kazdu triedu."""
    px        = np.zeros(5, dtype=np.int64)
    n_per_cls = np.zeros(5, dtype=np.int64)
    for _, mp in msk_paths_rel:
        msk = MASK_REMAP[np.array(Image.open(DATA_DIR / mp))]
        for c in range(5):
            cnt = int((msk == c).sum())
            px[c] += cnt
            if cnt > 0:
                n_per_cls[c] += 1
    return px, n_per_cls


def print_stats(label, pairs, total_n=None):
    px, n_per_cls = class_stats(pairs)
    total_px = px.sum()
    print(f"\n── {label}  (n={len(pairs)}"
          + (f", {len(pairs)/total_n*100:.1f}% datasetu)" if total_n else ")"))
    print(f"  {'Trieda':<18} {'Pixely':>14} {'%':>8} {'#snimok':>10}")
    print(f"  {'-'*52}")
    for c, name in enumerate(CLASS_NAMES):
        pct = px[c] / total_px * 100 if total_px > 0 else 0
        print(f"  {name:<18} {px[c]:>14,} {pct:>7.3f}% {n_per_cls[c]:>10}")
    return px, n_per_cls


def main():
    print("=" * 70)
    print("  Building train / val / test splits")
    print("=" * 70)

    # ── 1) Zber vsetkych parov ──
    all_pairs_by_trunk = {t: collect_pairs(t) for t in ALL_TRUNKS}
    total_n = sum(len(v) for v in all_pairs_by_trunk.values())
    print(f"\nTotal pairs: {total_n} v {len(ALL_TRUNKS)} trunkoch")
    for t in ALL_TRUNKS:
        print(f"  {t:<20} {len(all_pairs_by_trunk[t]):>4}")

    # ── 2) Test set ──
    test_pairs = []
    for t in TEST_FULL_TRUNKS:
        test_pairs.extend(all_pairs_by_trunk[t])
    for t, n_take in TEST_PARTIAL.items():
        test_pairs.extend(all_pairs_by_trunk[t][:n_take])

    # ── 3) Pool pre train+val: vsetko ostatne ──
    test_set = set(test_pairs)
    pool = []
    for t in ALL_TRUNKS:
        for p in all_pairs_by_trunk[t]:
            if p not in test_set:
                pool.append(p)

    # ── 4) Random 80/20 train/val ──
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(pool))
    n_val = round(VAL_RATIO * len(pool))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]
    train_pairs = [pool[i] for i in train_idx]
    val_pairs   = [pool[i] for i in val_idx]

    print(f"\nSplit summary:")
    print(f"  Test  : {len(test_pairs):>5}  ({len(test_pairs)/total_n*100:.1f}%)")
    print(f"  Train : {len(train_pairs):>5}  ({len(train_pairs)/total_n*100:.1f}%)")
    print(f"  Val   : {len(val_pairs):>5}  ({len(val_pairs)/total_n*100:.1f}%)")
    print(f"  Spolu : {len(test_pairs)+len(train_pairs)+len(val_pairs):>5}")

    # ── 5) Konverzia na relativne cesty ──
    test_rel  = to_relative(test_pairs)
    train_rel = to_relative(train_pairs)
    val_rel   = to_relative(val_pairs)

    # ── 6) Statistika ──
    print("\n" + "=" * 70)
    print("  Per-class statistika")
    print("=" * 70)
    px_test,  n_test  = print_stats("TEST",  test_rel,  total_n)
    px_train, n_train = print_stats("TRAIN", train_rel, total_n)
    px_val,   n_val_  = print_stats("VAL",   val_rel,   total_n)
    px_all = px_test + px_train + px_val

    print("\n── Pomer (test % / dataset %) ──")
    print(f"  {'Trieda':<18} {'Test%':>8} {'Train%':>8} {'Val%':>8} {'Pomer test':>12}")
    for c, name in enumerate(CLASS_NAMES):
        ds_pct = px_all[c] / px_all.sum() * 100
        t_pct  = px_test[c] / px_test.sum() * 100 if px_test.sum() > 0 else 0
        tr_pct = px_train[c] / px_train.sum() * 100
        v_pct  = px_val[c] / px_val.sum() * 100 if px_val.sum() > 0 else 0
        ratio  = t_pct / ds_pct if ds_pct > 0 else 0
        flag   = " (!)" if (ratio < 0.5 or ratio > 2.0) else ""
        print(f"  {name:<18} {t_pct:>7.3f}% {tr_pct:>7.3f}% {v_pct:>7.3f}% {ratio:>10.2f}x{flag}")

    # ── 7) Ulozenie ──
    with open(SPLITS_DIR / "test_files.json", "w") as f:
        json.dump({"description": "Test split: kmen4+kmen9+Dub_3b complete + first 15 of hrce_mixed and Dub_praskliny_a",
                   "n": len(test_rel), "pairs": test_rel}, f, indent=2)
    with open(SPLITS_DIR / "train_files.json", "w") as f:
        json.dump({"description": f"Train split: random 80% of remaining pool (seed={SPLIT_SEED})",
                   "n": len(train_rel), "pairs": train_rel}, f, indent=2)
    with open(SPLITS_DIR / "val_files.json", "w") as f:
        json.dump({"description": f"Val split: random 20% of remaining pool (seed={SPLIT_SEED})",
                   "n": len(val_rel), "pairs": val_rel}, f, indent=2)
    with open(SPLITS_DIR / "split_stats.json", "w") as f:
        json.dump({
            "class_names": CLASS_NAMES,
            "test_full_trunks": TEST_FULL_TRUNKS,
            "test_partial":     TEST_PARTIAL,
            "val_ratio": VAL_RATIO,
            "split_seed": SPLIT_SEED,
            "n_test": len(test_rel),
            "n_train": len(train_rel),
            "n_val":   len(val_rel),
            "test_pixels":  px_test.tolist(),
            "train_pixels": px_train.tolist(),
            "val_pixels":   px_val.tolist(),
            "test_imgs_per_class":  n_test.tolist(),
            "train_imgs_per_class": n_train.tolist(),
            "val_imgs_per_class":   n_val_.tolist(),
        }, f, indent=2)

    print(f"\nSplits ulozene do:")
    print(f"  {SPLITS_DIR / 'test_files.json'}")
    print(f"  {SPLITS_DIR / 'train_files.json'}")
    print(f"  {SPLITS_DIR / 'val_files.json'}")
    print(f"  {SPLITS_DIR / 'split_stats.json'}")


if __name__ == "__main__":
    main()
