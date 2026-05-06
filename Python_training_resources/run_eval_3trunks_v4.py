"""
Wrapper: v4 model na 3-trunks teste (192 imgs).
Pre data-scaling porovnanie: v3_ablation vs v3 vs v4 na rovnakom teste.

Vystupy: predictions_v4_3trunks/, metrics_v4_3trunks.json
"""
from evaluate_test import main, DATA_DIR

main(
    model_path    = DATA_DIR / "net_segformer_b2_v4.pth",
    test_split    = DATA_DIR / "splits" / "test_3trunks_files.json",
    output_suffix = "_3trunks",
)
