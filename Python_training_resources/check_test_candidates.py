"""
Diagnostika kandidatov na test set.
Pre kazdy zvoleny kmen vypise:
  - pocet snimok
  - rozdelenie pixelov medzi triedy (Drevo, Kora, Nezdrava, Okolie, Prasklina)
  - pocet snimok kde sa nachadza Nezdrava_hrca a Prasklina
A porovna to s priemerom celeho datasetu.
"""
import numpy as np
from pathlib import Path
from PIL import Image

DATA_DIR    = Path(__file__).parent
MASK_REMAP  = np.array([0, 3, 1, 0, 2, 4], dtype=np.uint8)
CLASS_NAMES = ["Drevo", "Kora", "Nezdrava_hrca", "Okolie", "Prasklina"]

ALL_TRUNKS = [
    "kmen1", "kmen2", "kmen3", "kmen4", "kmen5",
    "kmen6", "kmen7", "kmen8", "kmen9", "kmen10",
    "Dub_1", "Dub_2", "Dub_3b", "Dub_4", "Dub_5",
    "Dub_6", "Dub_7", "Dub_8", "Dub_9", "Dub_10",
    "Dub_praskliny_a",
]

# Kandidati na test set
TEST_CANDIDATES = ["kmen9", "Dub_3b"]


def stats_for_trunk(trunk):
    msk_dir = DATA_DIR / trunk / "masks"
    masks = sorted(msk_dir.glob("*.png"))
    px        = np.zeros(5, dtype=np.int64)
    n_nez     = 0
    n_prask   = 0
    for mp in masks:
        msk = MASK_REMAP[np.array(Image.open(mp))]
        for c in range(5):
            px[c] += (msk == c).sum()
        if (msk == 2).any(): n_nez   += 1
        if (msk == 4).any(): n_prask += 1
    return {
        "n": len(masks), "px": px,
        "n_nez": n_nez, "n_prask": n_prask,
    }


def main():
    print("Analyzing 20 trunkov ...\n")
    all_stats = {}
    for t in ALL_TRUNKS:
        all_stats[t] = stats_for_trunk(t)

    # Celkovy dataset (priemer)
    total_n     = sum(s["n"] for s in all_stats.values())
    total_px    = np.sum([s["px"] for s in all_stats.values()], axis=0)
    total_nez   = sum(s["n_nez"] for s in all_stats.values())
    total_prask = sum(s["n_prask"] for s in all_stats.values())

    print(f"{'Kmen':<12} {'#img':>5} | "
          + " ".join(f"{n:>10}" for n in CLASS_NAMES)
          + f" | {'#Nez':>5} {'#Prask':>7}")
    print("-" * 110)

    for t in ALL_TRUNKS:
        s = all_stats[t]
        tot_t = s["px"].sum()
        pct_str = " ".join(f"{s['px'][c]/tot_t*100:>9.2f}%" for c in range(5))
        marker = "  <-- TEST" if t in TEST_CANDIDATES else ""
        print(f"{t:<12} {s['n']:>5} | {pct_str} | "
              f"{s['n_nez']:>5} {s['n_prask']:>7}{marker}")

    print("-" * 110)
    pct_total = " ".join(f"{total_px[c]/total_px.sum()*100:>9.2f}%" for c in range(5))
    print(f"{'CELY DATASET':<12} {total_n:>5} | {pct_total} | "
          f"{total_nez:>5} {total_prask:>7}")

    # Test set sumar
    print("\n" + "=" * 110)
    print(f"TEST SET = {' + '.join(TEST_CANDIDATES)}")
    print("=" * 110)

    test_n     = sum(all_stats[t]["n"] for t in TEST_CANDIDATES)
    test_px    = np.sum([all_stats[t]["px"] for t in TEST_CANDIDATES], axis=0)
    test_nez   = sum(all_stats[t]["n_nez"] for t in TEST_CANDIDATES)
    test_prask = sum(all_stats[t]["n_prask"] for t in TEST_CANDIDATES)

    print(f"\n  Snimok       : {test_n} z {total_n} ({test_n/total_n*100:.1f}%)")
    print(f"  Snimky s Nez : {test_nez} z {total_nez} ({test_nez/total_nez*100:.1f}%)")
    print(f"  Snimky s Prask: {test_prask} z {total_prask} ({test_prask/total_prask*100:.1f}%)")

    print(f"\n  {'Trieda':<18} {'Test %':>10} {'Dataset %':>12} {'Pomer':>8}")
    print(f"  {'-'*52}")
    for c, name in enumerate(CLASS_NAMES):
        test_pct = test_px[c] / test_px.sum() * 100
        ds_pct   = total_px[c] / total_px.sum() * 100
        ratio    = test_pct / ds_pct if ds_pct > 0 else float("inf")
        flag     = " (!)" if (ratio < 0.5 or ratio > 2.0) else ""
        print(f"  {name:<18} {test_pct:>9.3f}% {ds_pct:>11.3f}% {ratio:>7.2f}x{flag}")

    # ── Ranking: ktore kmene maju najviac Praskliny ──
    print("\n" + "=" * 110)
    print("RANKING TRUNKOV PODLA PRASKLINY")
    print("=" * 110)

    # 1) Podla poctu snimok s Prasklinou
    by_count = sorted(ALL_TRUNKS,
                      key=lambda t: all_stats[t]["n_prask"], reverse=True)
    print(f"\n  Podla poctu snimok s Prasklinou:")
    print(f"  {'Rank':>4}  {'Kmen':<12} {'#img':>5}  {'#snimok s Prask':>17}  {'%':>6}")
    print(f"  {'-'*54}")
    for rank, t in enumerate(by_count, 1):
        s = all_stats[t]
        pct_imgs = s["n_prask"] / s["n"] * 100 if s["n"] > 0 else 0
        marker = "  <-- top 3" if rank <= 3 else ""
        print(f"  {rank:>4}  {t:<12} {s['n']:>5}  {s['n_prask']:>17}  {pct_imgs:>5.1f}%{marker}")

    # 2) Podla pixeloveho podielu Praskliny
    def prask_pct(t):
        px = all_stats[t]["px"]
        return px[4] / px.sum() * 100 if px.sum() > 0 else 0

    by_pixels = sorted(ALL_TRUNKS, key=prask_pct, reverse=True)
    print(f"\n  Podla pixeloveho podielu Praskliny:")
    print(f"  {'Rank':>4}  {'Kmen':<12} {'Prasklina px':>13}  {'%':>7}")
    print(f"  {'-'*46}")
    for rank, t in enumerate(by_pixels, 1):
        s = all_stats[t]
        marker = "  <-- top 3" if rank <= 3 else ""
        print(f"  {rank:>4}  {t:<12} {s['px'][4]:>13,}  {prask_pct(t):>6.3f}%{marker}")

    # 3) Podla poctu snimok s Nezdravou_hrcou (pre stratifikaciu)
    by_nez = sorted(ALL_TRUNKS,
                    key=lambda t: all_stats[t]["n_nez"], reverse=True)
    print(f"\n  Podla poctu snimok s Nezdravou_hrcou:")
    print(f"  {'Rank':>4}  {'Kmen':<12} {'#img':>5}  {'#snimok s Nez':>15}  {'%':>6}")
    print(f"  {'-'*52}")
    for rank, t in enumerate(by_nez, 1):
        s = all_stats[t]
        pct_imgs = s["n_nez"] / s["n"] * 100 if s["n"] > 0 else 0
        marker = "  <-- top 3" if rank <= 3 else ""
        print(f"  {rank:>4}  {t:<12} {s['n']:>5}  {s['n_nez']:>15}  {pct_imgs:>5.1f}%{marker}")


if __name__ == "__main__":
    main()
