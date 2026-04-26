"""
Vytvori specialny test split obsahujuci LEN 3 cele held-out trunky:
  kmen4 + kmen9 + Dub_3b  =  192 snimok

Pouzitie: porovnanie 2D vs 2.5D modelov na CT-spatialne koherentnom
testu (bez Dub_praskliny_a a hrce_mixed[:15], kde 2.5D context nedava
zmysel).

Vystup:
    splits/test_3trunks_files.json
    splits/test_3trunks_stats.json
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

TEST_TRUNKS = ["kmen4", "kmen9", "Dub_3b"]


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


def main():
    print("=" * 70)
    print(f"  Building 3-trunks test split: {' + '.join(TEST_TRUNKS)}")
    print("=" * 70)

    test_pairs = []
    for t in TEST_TRUNKS:
        pairs = collect_pairs(t)
        print(f"  {t:<10} -> {len(pairs)} parov")
        test_pairs.extend(pairs)

    test_rel = to_relative(test_pairs)

    # ── Stats ──
    px, n_per_cls = class_stats(test_rel)
    total_px = px.sum()
    print(f"\n── 3-TRUNKS TEST  (n={len(test_rel)}) ──")
    print(f"  {'Trieda':<18} {'Pixely':>14} {'%':>8} {'#snimok':>10}")
    print(f"  {'-'*52}")
    for c, name in enumerate(CLASS_NAMES):
        pct = px[c] / total_px * 100 if total_px > 0 else 0
        print(f"  {name:<18} {px[c]:>14,} {pct:>7.3f}% {n_per_cls[c]:>10}")

    # ── Save ──
    out_json = SPLITS_DIR / "test_3trunks_files.json"
    with open(out_json, "w") as f:
        json.dump({
            "description": "Special test: only kmen4+kmen9+Dub_3b (3 fully held-out trunks). "
                           "Used for fair 2D vs 2.5D comparison -- no Dub_praskliny_a or "
                           "hrce_mixed where 2.5D context is meaningless or noisy.",
            "trunks": TEST_TRUNKS,
            "n": len(test_rel),
            "pairs": test_rel,
        }, f, indent=2)

    out_stats = SPLITS_DIR / "test_3trunks_stats.json"
    with open(out_stats, "w") as f:
        json.dump({
            "class_names": CLASS_NAMES,
            "trunks": TEST_TRUNKS,
            "n": len(test_rel),
            "pixels":              px.tolist(),
            "imgs_per_class":      n_per_cls.tolist(),
            "pixel_pct_per_class": (px / total_px * 100).tolist(),
        }, f, indent=2)

    print(f"\nSplits ulozene:")
    print(f"  {out_json}")
    print(f"  {out_stats}")


if __name__ == "__main__":
    main()
