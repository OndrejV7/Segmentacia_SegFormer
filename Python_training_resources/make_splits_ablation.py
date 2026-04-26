"""
Ablation split: vyhodi z train+val pool-u snimky z hrce_mixed a Dub_praskliny_a.
Test set zostava identicky s v3 (splits/test_files.json) -- tak vieme cisto
porovnat aky prinos malo pridanie tychto 114 snimok do trainu.

Train+Val pool sa redukuje na povodne 20 kmenov (kmen1-10 + Dub_1-10) minus
3 trunky v teste (kmen4, kmen9, Dub_3b).
  = 1347 - 192 (test) = 1155 snimok
  Train (80%) ~= 924
  Val   (20%) ~= 231

Vystupy:
    splits/ablation_train_files.json
    splits/ablation_val_files.json
    splits/ablation_split_stats.json

Test set sa NEMODIFIKUJE -- test_files.json zostava nedotknuty.

Spustenie:
    python make_splits_ablation.py
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image

DATA_DIR    = Path(__file__).parent
SPLITS_DIR  = DATA_DIR / "splits"

MASK_REMAP  = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)
CLASS_NAMES = ["Drevo", "Kora", "Nezdrava_hrca", "Okolie", "Prasklina"]

# IBA POVODNE TRUNKY -- bez Dub_praskliny_a a hrce_mixed
ABLATION_TRUNKS = [
    "kmen1", "kmen2", "kmen3", "kmen4", "kmen5",
    "kmen6", "kmen7", "kmen8", "kmen9", "kmen10",
    "Dub_1", "Dub_2", "Dub_3b", "Dub_4", "Dub_5",
    "Dub_6", "Dub_7", "Dub_8", "Dub_9", "Dub_10",
]

# Test set (identicky s v3 make_splits.py) -- nech vie skript filtrovat
TEST_FULL_TRUNKS = ["kmen4", "kmen9", "Dub_3b"]
TEST_PARTIAL = {
    "hrce_mixed":      15,
    "Dub_praskliny_a": 15,
}

VAL_RATIO  = 0.20
SPLIT_SEED = 0       # rovnaky seed ako v3, aby sampling bol porovnatelny


def collect_pairs(trunk):
    img_dir = DATA_DIR / trunk / "images"
    msk_dir = DATA_DIR / trunk / "masks"
    pairs = []
    for img_path in sorted(img_dir.glob("*.tif")):
        msk_path = msk_dir / (img_path.stem + ".png")
        if msk_path.exists():
            pairs.append((img_path, msk_path))
    return pairs


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


def print_stats(label, pairs_rel, total_n=None):
    px, n_per_cls = class_stats(pairs_rel)
    total_px = px.sum()
    print(f"\n── {label}  (n={len(pairs_rel)}"
          + (f", {len(pairs_rel)/total_n*100:.1f}% pool-u)" if total_n else ")"))
    print(f"  {'Trieda':<18} {'Pixely':>14} {'%':>8} {'#snimok':>10}")
    print(f"  {'-'*52}")
    for c, name in enumerate(CLASS_NAMES):
        pct = px[c] / total_px * 100 if total_px > 0 else 0
        print(f"  {name:<18} {px[c]:>14,} {pct:>7.3f}% {n_per_cls[c]:>10}")
    return px, n_per_cls


def main():
    print("=" * 70)
    print("  Building ABLATION train / val splits")
    print("  (test set zostava identicky s splits/test_files.json)")
    print("=" * 70)

    # ── 1) Zber parov LEN z povodnych 20 trunkov ──
    all_pairs_by_trunk = {t: collect_pairs(t) for t in ABLATION_TRUNKS}
    total_n = sum(len(v) for v in all_pairs_by_trunk.values())
    print(f"\n20 povodnych trunkov: {total_n} parov")

    # ── 2) Spocitaj test set indexy (aby sme ich vyfiltrovali z poolu) ──
    test_pairs = []
    for t in TEST_FULL_TRUNKS:
        if t in all_pairs_by_trunk:
            test_pairs.extend(all_pairs_by_trunk[t])
    # Partial test trunky (hrce_mixed, Dub_praskliny_a) -- nie su v
    # ABLATION_TRUNKS, takze test obsahuje tieto snimky ale train+val NIE.
    # To je presne to co chceme.

    # ── 3) Pool train+val: vsetko zo zoznamu okrem test ──
    test_set = set(test_pairs)
    pool = []
    for t in ABLATION_TRUNKS:
        for p in all_pairs_by_trunk[t]:
            if p not in test_set:
                pool.append(p)

    print(f"\nTest trunky v pool-e (kmen4, kmen9, Dub_3b): "
          f"{len(test_pairs)} snimok bude vyfiltrovanych")
    print(f"Pool po odpoctu testu: {len(pool)} snimok")

    # ── 4) Random 80/20 ──
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(pool))
    n_val = round(VAL_RATIO * len(pool))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]
    train_pairs = [pool[i] for i in train_idx]
    val_pairs   = [pool[i] for i in val_idx]

    print(f"\nAblation split summary:")
    print(f"  Test  : 222 (nedotknute, nacita sa z splits/test_files.json)")
    print(f"  Train : {len(train_pairs):>5}")
    print(f"  Val   : {len(val_pairs):>5}")
    print(f"  Spolu (train+val): {len(train_pairs)+len(val_pairs)}")

    # ── 5) Konverzia + ulozenie ──
    train_rel = to_relative(train_pairs)
    val_rel   = to_relative(val_pairs)

    # ── 6) Stats ──
    print("\n" + "=" * 70)
    print("  Per-class statistika")
    print("=" * 70)
    px_train, n_train = print_stats("ABLATION TRAIN", train_rel, len(pool))
    px_val,   n_val_  = print_stats("ABLATION VAL",   val_rel,   len(pool))

    # Porovnanie s v3 splitom
    v3_train_path = SPLITS_DIR / "train_files.json"
    v3_val_path   = SPLITS_DIR / "val_files.json"
    if v3_train_path.exists() and v3_val_path.exists():
        with open(v3_train_path) as f: v3_train = json.load(f)
        with open(v3_val_path)   as f: v3_val   = json.load(f)
        print(f"\n── Porovnanie s v3 splitom ──")
        print(f"  Train: v3={v3_train['n']}, ablation={len(train_rel)}, "
              f"diff={v3_train['n']-len(train_rel)} stratenych")
        print(f"  Val  : v3={v3_val['n']},   ablation={len(val_rel)}, "
              f"diff={v3_val['n']-len(val_rel)} stratenych")

    # ── 7) Save ──
    SPLITS_DIR.mkdir(exist_ok=True)
    with open(SPLITS_DIR / "ablation_train_files.json", "w") as f:
        json.dump({"description": f"Ablation train: only kmen1-10 + Dub_1-10 (no Dub_praskliny_a, no hrce_mixed). seed={SPLIT_SEED}",
                   "n": len(train_rel), "pairs": train_rel}, f, indent=2)
    with open(SPLITS_DIR / "ablation_val_files.json", "w") as f:
        json.dump({"description": f"Ablation val: only kmen1-10 + Dub_1-10. seed={SPLIT_SEED}",
                   "n": len(val_rel), "pairs": val_rel}, f, indent=2)
    with open(SPLITS_DIR / "ablation_split_stats.json", "w") as f:
        json.dump({
            "class_names": CLASS_NAMES,
            "ablation_trunks": ABLATION_TRUNKS,
            "test_full_trunks": TEST_FULL_TRUNKS,
            "test_partial":     TEST_PARTIAL,
            "val_ratio": VAL_RATIO,
            "split_seed": SPLIT_SEED,
            "n_train": len(train_rel),
            "n_val":   len(val_rel),
            "train_pixels": px_train.tolist(),
            "val_pixels":   px_val.tolist(),
            "train_imgs_per_class": n_train.tolist(),
            "val_imgs_per_class":   n_val_.tolist(),
        }, f, indent=2)

    print(f"\nSplits ulozene do:")
    print(f"  {SPLITS_DIR / 'ablation_train_files.json'}")
    print(f"  {SPLITS_DIR / 'ablation_val_files.json'}")
    print(f"  {SPLITS_DIR / 'ablation_split_stats.json'}")
    print(f"\n(Test set nedotknuty -- {SPLITS_DIR / 'test_files.json'})")


if __name__ == "__main__":
    main()
