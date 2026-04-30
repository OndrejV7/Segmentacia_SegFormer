"""
Wrapper na spustenie export_report_word.py s vsetkymi metrics subormi.
Generuje vedecky-stylovy report.docx so 6 sekciami:
  1. Uvod
  2. Metodika (dataset, splity, architektura, hyperparametre, loss, trening, metriky)
  3. Vysledky (hlavny model, data ablation, 2.5D experimenty, Focal-Tversky)
  4. Diskusia (Pareto trade-off, prakticke implikacie, obmedzenia)
  5. Zaver
  6. Pouzita literatura

Spusti cez VS Code Run/Debug tlacidlo.
"""
import sys
from types import SimpleNamespace
from export_report_word import main, DATA_DIR
from pathlib import Path

args = SimpleNamespace(
    main                       = DATA_DIR / "metrics_v3.json",
    ablation                   = DATA_DIR / "metrics_v3_ablation.json",
    v3_3trunks                 = DATA_DIR / "metrics_v3_3trunks.json",
    ablation_3trunks           = DATA_DIR / "metrics_v3_ablation_3trunks.json",
    ablation_25d_3trunks       = DATA_DIR / "metrics_v3_ablation_25d_3trunks.json",
    ablation_3trunks_rgb       = DATA_DIR / "metrics_v3_ablation_3trunks_rgb.json",
    ablation_25d_3trunks_rgb   = DATA_DIR / "metrics_v3_ablation_25d_3trunks_rgb.json",
    ablation_rgb_3trunks       = DATA_DIR / "metrics_v3_ablation_rgb_3trunks.json",
    ft_v1                      = DATA_DIR / "metrics_v3_focaltversky.json",
    ft_v2                      = DATA_DIR / "metrics_v3_focaltversky_v2.json",
    v4_main                    = DATA_DIR / "metrics_v4.json",
    v4_ft_v1                   = DATA_DIR / "metrics_v4_focaltversky.json",
    v4_ft_v2                   = DATA_DIR / "metrics_v4_focaltversky_v2.json",
    out                        = DATA_DIR / "report.docx",
)

# Fallback ak metrics_v3.json nie je k dispozicii
if not args.main.exists():
    fallback = DATA_DIR / "metrics.json"
    if fallback.exists():
        print(f"NOTE: pouzivam {fallback.name} (stary nazov)")
        args.main = fallback

main(args)
