"""
Wrapper na spustenie export_report_word.py s defaultnymi argumentmi.
Generuje report.docx so vsetkymi sekciami:
  1. Intro / metriky
  2. Hlavny model (v3, full test)
  3. Ablation studia dat (v3 vs v3_ablation, full test)
  4. Experiment 2.5D (3-trunks porovnanie v3 vs v3_ablation vs v3_ablation_25d)
  5. Zaver

Spusti cez VS Code Run/Debug tlacidlo.
"""
import sys
sys.argv = ["export_report_word.py"]
from export_report_word import main, DATA_DIR
from pathlib import Path

main_path     = DATA_DIR / "metrics_v3.json"
ablation_path = DATA_DIR / "metrics_v3_ablation.json"
out_path      = DATA_DIR / "report.docx"

# 3-trunks specialny test (sekcia 5: 192 imgs s extra trunkami)
comp_3trunks = {
    "v3":              DATA_DIR / "metrics_v3_3trunks.json",
    "v3_ablation":     DATA_DIR / "metrics_v3_ablation_3trunks.json",
    "v3_ablation_25d": DATA_DIR / "metrics_v3_ablation_25d_3trunks.json",
}

# 3-trunks RGB test (sekcia 5.4: 186 imgs bez okrajov, fer porovnanie 2D vs 2.5D vs RGB)
comp_3trunks_rgb = {
    "v3_ablation":     DATA_DIR / "metrics_v3_ablation_3trunks_rgb.json",
    "v3_ablation_25d": DATA_DIR / "metrics_v3_ablation_25d_3trunks_rgb.json",
    "v3_ablation_rgb": DATA_DIR / "metrics_v3_ablation_rgb_3trunks.json",
}

# Fallback ak metrics_v3.json mal stary nazov
if not main_path.exists():
    fallback = DATA_DIR / "metrics.json"
    if fallback.exists():
        print(f"NOTE: pouzivam {fallback.name} (stary nazov pred refaktoringom)")
        main_path = fallback
    else:
        print(f"ERROR: ani {main_path.name} ani metrics.json neexistuje.")
        sys.exit(1)

main(main_path,
     ablation_path if ablation_path.exists() else None,
     comp_3trunks,
     comp_3trunks_rgb,
     out_path)
