"""
make_splits_v3p5 -- Midpoint split pre data-scaling studiu (50% v4 priorastku).

Test set: identicky s v3/v4 (222 snimok). Test split file SA NEVYTVARA
(pouziva sa existujuci test_v4_files.json pri evaluacii).

Train+Val pool obsahuje:
    - 17 trunkov (kmen+Dub minus test, plne)        1155 snimok
    - 17 imgs Dub_praskliny_a (po test 15: imgs 15-31)  17 (z 35 v pool-e)
    - 25 imgs Dub_praskliny_b (imgs 0-24)               25 (z 50)
    - 40 imgs hrce_mixed (po test 15: imgs 15-54)       40 (z 79)
    -------------------------------------
    SPOLU pool                                       1237 snimok

To je presne 50% pridanych dat oproti v3_ablation:
    v3_ablation pool   1155 (0% pridanych extra dat)
    v3p5 pool          1237 (+82 = 50% pridanych)
    v4 pool            1319 (+164 = 100% pridanych)

Vystupy:
    splits/train_v3p5_files.json
    splits/val_v3p5_files.json
    splits/split_stats_v3p5.json

Spustenie:
    python make_splits_v3p5.py
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
    "Dub_praskliny_a", "Dub_praskliny_b", "hrce_mixed",
]

# ── Test set definicia (zhodne s v4) ──────────────────────────
TEST_FULL_TRUNKS = ["kmen4", "kmen9", "Dub_3b"]    # cele trunky idu do testu
TEST_PARTIAL = {                                    # prvych N snimok
    "hrce_mixed":      15,
    "Dub_praskliny_a": 15,
}

# ── Train+val pool zuzenie pre v3p5 (50% v4 priorastku) ───────
# Pre tieto trunky sa po odstraneni test imgs (TEST_PARTIAL) berie len
# prvych N snimok do pool-u; zvysok sa zahodi (NIE je ani v teste).
TRAIN_PARTIAL = {
    "Dub_praskliny_a": 17,    # po test 15 imgs zostane 35, berieme prvych 17
    "Dub_praskliny_b": 25,    # 50 total, no test, berieme prvych 25
    "hrce_mixed":      40,    # po test 15 imgs zostane 79, berieme prvych 40
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
    print(f"\n-- {label}  (n={len(pairs)}"
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

    # ── 3) Pool pre train+val: limit per trunk podla TRAIN_PARTIAL ──
    test_set = set(test_pairs)
    pool = []
    for t in ALL_TRUNKS:
        # Vsetky non-test imgs tohto trunku (zachovane poradie)
        non_test = [p for p in all_pairs_by_trunk[t] if p not in test_set]
        # Pre trunky v TRAIN_PARTIAL berieme len prvych N
        if t in TRAIN_PARTIAL:
            non_test = non_test[:TRAIN_PARTIAL[t]]
        pool.extend(non_test)

    print(f"\nTRAIN_PARTIAL filter aplikovany na: {list(TRAIN_PARTIAL.keys())}")
    for t in TRAIN_PARTIAL:
        full_pool = sum(1 for p in all_pairs_by_trunk[t] if p not in test_set)
        print(f"  {t:<20}: pool {full_pool} -> selected {TRAIN_PARTIAL[t]}")

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

    print("\n-- Pomer (test % / dataset %) --")
    print(f"  {'Trieda':<18} {'Test%':>8} {'Train%':>8} {'Val%':>8} {'Pomer test':>12}")
    for c, name in enumerate(CLASS_NAMES):
        ds_pct = px_all[c] / px_all.sum() * 100
        t_pct  = px_test[c] / px_test.sum() * 100 if px_test.sum() > 0 else 0
        tr_pct = px_train[c] / px_train.sum() * 100
        v_pct  = px_val[c] / px_val.sum() * 100 if px_val.sum() > 0 else 0
        ratio  = t_pct / ds_pct if ds_pct > 0 else 0
        flag   = " (!)" if (ratio < 0.5 or ratio > 2.0) else ""
        print(f"  {name:<18} {t_pct:>7.3f}% {tr_pct:>7.3f}% {v_pct:>7.3f}% {ratio:>10.2f}x{flag}")

    # ── 7) Ulozenie (v3p5 suffix -- midpoint study) ──
    # Test sa NEVYTVARA (zhodny s test_v4_files.json)
    with open(SPLITS_DIR / "train_v3p5_files.json", "w") as f:
        json.dump({"description": f"v3p5 train split: 50% data scaling midpoint (TRAIN_PARTIAL: 17+25+40), seed={SPLIT_SEED}",
                   "train_partial": TRAIN_PARTIAL,
                   "n": len(train_rel), "pairs": train_rel}, f, indent=2)
    with open(SPLITS_DIR / "val_v3p5_files.json", "w") as f:
        json.dump({"description": f"v3p5 val split: 50% data scaling midpoint, seed={SPLIT_SEED}",
                   "train_partial": TRAIN_PARTIAL,
                   "n": len(val_rel), "pairs": val_rel}, f, indent=2)
    with open(SPLITS_DIR / "split_stats_v3p5.json", "w") as f:
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
    print(f"  {SPLITS_DIR / 'train_v3p5_files.json'}")
    print(f"  {SPLITS_DIR / 'val_v3p5_files.json'}")
    print(f"  {SPLITS_DIR / 'split_stats_v3p5.json'}")
    print(f"  (test = test_v4_files.json, nezmenený)")


if __name__ == "__main__":
    main()
