"""
Wrapper na spustenie export_report_data_scaling.py.
Generuje samostatny Word report o data scaling studii (v3_ablation/v3p5/v4).
"""
from types import SimpleNamespace
from export_report_data_scaling import main, DATA_DIR

args = SimpleNamespace(
    abl_3t   = DATA_DIR / "metrics_v3_ablation_3trunks.json",
    mid_3t   = DATA_DIR / "metrics_v3p5_3trunks.json",
    full_3t  = DATA_DIR / "metrics_v4_3trunks.json",
    abl_full = DATA_DIR / "metrics_v3_ablation.json",
    mid_full = DATA_DIR / "metrics_v3p5.json",
    full_full= DATA_DIR / "metrics_v4.json",
    out      = DATA_DIR / "report_data_scaling.docx",
)
main(args)
