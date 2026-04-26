"""
Wrapper: 2D ablation model na 3-trunks RGB teste (186 imgs, bez okrajov).
Pre fair porovnanie 2D vs RGB-encoding na rovnakom test sete.
Vystup: metrics_v3_ablation_3trunks_rgb.json
"""
from evaluate_test import main, DATA_DIR

main(
    model_path    = DATA_DIR / "net_segformer_b2_v3_ablation.pth",
    test_split    = DATA_DIR / "splits" / "test_3trunks_rgb_files.json",
    output_suffix = "_3trunks_rgb",
)
