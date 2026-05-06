"""
Wrapper: v3p5 model na 3-trunks teste (192 imgs).
Pre data-scaling porovnanie: v3_ablation vs v3p5 vs v4 na rovnakom teste.
"""
from evaluate_test import main, DATA_DIR

main(
    model_path    = DATA_DIR / "net_segformer_b2_v3p5.pth",
    test_split    = DATA_DIR / "splits" / "test_3trunks_files.json",
    output_suffix = "_3trunks",
)
