"""
Ablation split pre RGB-encoding 2.5D experiment.

Identicky s make_splits_ablation.py okrem:
  - Z kazdeho kmena vyhadzujeme PRVY a POSLEDNY rez (nemaju oboch susedov
    pre RGB encoding (n-1, n, n+1))
  - Pool je tym padom o 17 * 2 = 34 snimok mensi (1155 -> 1121)

Vystupy:
    splits/ablation_rgb_train_files.json
    splits/ablation_rgb_val_files.json
    splits/ablation_rgb_split_stats.json

Test set sa nemodifikuje cez tento skript -- pouziva sa
make_test_3trunks_rgb.py (vyhodenie prvy/posledny len pre 3-trunks test).
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image

DATA_DIR    = Path(__file__).parent
SPLITS_DIR  = DATA_DIR / "splits"

MASK_REMAP  = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)
CLASS_NAMES = ["Drevo", "Kora", "Nezdrava_hrca", "Okolie", "Prasklina"]

# Iba povodne kmen+Dub trunky, bez Dub_praskliny_a a hrce_mixed
ABLATION_TRUNKS = [
    "kmen1", "kmen2", "kmen3", "kmen4", "kmen5",
    "kmen6", "kmen7", "kmen8", "kmen9", "kmen10",
    "Dub_1", "Dub_2", "Dub_3b", "Dub_4", "Dub_5",
    "Dub_6", "Dub_7", "Dub_8", "Dub_9", "Dub_10",
]
TEST_FULL_TRUNKS = ["kmen4", "kmen9", "Dub_3b"]

VAL_RATIO  = 0.20
SPLIT_SEED = 0


def collect_pairs_drop_edges(trunk):
    """Vrati zoradeny zoznam parov BEZ prveho a posledneho rezu."""
    img_dir = DATA_DIR / trunk / "images"
    msk_dir = DATA_DIR / trunk / "masks"
    pairs = []
    for img_path in sorted(img_dir.glob("*.tif")):
        msk_path = msk_dir / (img_path.stem + ".png")
        if msk_path.exists():
            pairs.append((img_path, msk_path))
    if len(pairs) <= 2:
        return []
    return pairs[1:-1]                   # vyhod prvy a posledny


def to_relative(pairs):
    return [(str(i.relative_to(DATA_DIR).as_posix()),
             str(m.relative_to(DATA_DIR).as_posix())) for i, m in pairs]


def class_stats(pairs_rel):
    px        = np.zeros(5, dtype=np.int64)
    n_per_cls = np.zeros(5, dtype=np.int64)
    for _, mp in pairs_rel:
        msk = MASK_REMAP[np.array(Image.open(DATA_DIR / mp))]
        for c in range(5):
            cnt = int((msk == c).sum())
            px[c] += cnt
            if cnt > 0:
                n_per_cls[c] += 1
    return px, n_per_cls


def print_stats(label, pairs_rel):
    px, n_per_cls = class_stats(pairs_rel)
    total_px = px.sum()
    print(f"\n── {label}  (n={len(pairs_rel)}) ──")
    print(f"  {'Trieda':<18} {'Pixely':>14} {'%':>8} {'#snimok':>10}")
    print(f"  {'-'*52}")
    for c, name in enumerate(CLASS_NAMES):
        pct = px[c] / total_px * 100 if total_px > 0 else 0
        print(f"  {name:<18} {px[c]:>14,} {pct:>7.3f}% {n_per_cls[c]:>10}")
    return px, n_per_cls


def main():
    print("=" * 70)
    print("  Building ABLATION RGB train / val splits")
    print("  (vyhadzujeme prvy a posledny rez kazdeho kmena)")
    print("=" * 70)

    # Pre kazdy trunk: zber parov BEZ okrajov
    all_pairs_by_trunk = {t: collect_pairs_drop_edges(t) for t in ABLATION_TRUNKS}
    total_n_dropped = sum(len(v) for v in all_pairs_by_trunk.values())
    print(f"\n20 povodnych trunkov, prvy/posledny rez vyhodeny -> {total_n_dropped} parov")
    for t in ABLATION_TRUNKS:
        # full count
        img_dir = DATA_DIR / t / "images"
        full = len(list(img_dir.glob("*.tif")))
        kept = len(all_pairs_by_trunk[t])
        print(f"  {t:<12} full={full:>3}  kept={kept:>3}  (-{full-kept})")

    # Test trunky
    test_pairs = []
    for t in TEST_FULL_TRUNKS:
        if t in all_pairs_by_trunk:
            test_pairs.extend(all_pairs_by_trunk[t])
    print(f"\nTest trunky (kmen4+kmen9+Dub_3b) parov bez okrajov: {len(test_pairs)}")
    print("  -- tieto idu do testu, do pool-u sa nepocitaju")

    # Pool train+val: vsetko zo zoznamu okrem test trunkov
    test_set = set(test_pairs)
    pool = []
    for t in ABLATION_TRUNKS:
        for p in all_pairs_by_trunk[t]:
            if p not in test_set:
                pool.append(p)

    print(f"Pool po odpoctu testu: {len(pool)} snimok")

    # Random 80/20
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(pool))
    n_val = round(VAL_RATIO * len(pool))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]
    train_pairs = [pool[i] for i in train_idx]
    val_pairs   = [pool[i] for i in val_idx]

    print(f"\nAblation RGB split summary:")
    print(f"  Train : {len(train_pairs):>5}")
    print(f"  Val   : {len(val_pairs):>5}")

    train_rel = to_relative(train_pairs)
    val_rel   = to_relative(val_pairs)

    print("\n" + "=" * 70)
    print("  Per-class statistika")
    print("=" * 70)
    px_train, n_train = print_stats("ABLATION RGB TRAIN", train_rel)
    px_val,   n_val_  = print_stats("ABLATION RGB VAL",   val_rel)

    SPLITS_DIR.mkdir(exist_ok=True)
    with open(SPLITS_DIR / "ablation_rgb_train_files.json", "w") as f:
        json.dump({"description": f"Ablation RGB train: kmen1-10 + Dub_1-10 (no test trunks), "
                                  f"first/last slice of each trunk dropped. seed={SPLIT_SEED}",
                   "n": len(train_rel), "pairs": train_rel}, f, indent=2)
    with open(SPLITS_DIR / "ablation_rgb_val_files.json", "w") as f:
        json.dump({"description": f"Ablation RGB val. seed={SPLIT_SEED}",
                   "n": len(val_rel), "pairs": val_rel}, f, indent=2)
    with open(SPLITS_DIR / "ablation_rgb_split_stats.json", "w") as f:
        json.dump({
            "class_names": CLASS_NAMES,
            "ablation_trunks": ABLATION_TRUNKS,
            "test_full_trunks": TEST_FULL_TRUNKS,
            "edges_dropped": True,
            "val_ratio": VAL_RATIO,
            "split_seed": SPLIT_SEED,
            "n_train": len(train_rel),
            "n_val":   len(val_rel),
            "train_pixels": px_train.tolist(),
            "val_pixels":   px_val.tolist(),
            "train_imgs_per_class": n_train.tolist(),
            "val_imgs_per_class":   n_val_.tolist(),
        }, f, indent=2)

    print(f"\nUlozene:")
    print(f"  splits/ablation_rgb_train_files.json")
    print(f"  splits/ablation_rgb_val_files.json")
    print(f"  splits/ablation_rgb_split_stats.json")


if __name__ == "__main__":
    main()
