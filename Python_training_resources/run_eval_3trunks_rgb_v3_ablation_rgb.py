"""
Wrapper: RGB adjacent-slice model na 3-trunks RGB teste (186 imgs).
Vystupy: predictions_v3_ablation_rgb_3trunks/, metrics_v3_ablation_rgb_3trunks.json
"""
from evaluate_test_rgb import main, DATA_DIR

main(
    model_path    = DATA_DIR / "net_segformer_b2_v3_ablation_rgb.pth",
    test_split    = DATA_DIR / "splits" / "test_3trunks_rgb_files.json",
    output_suffix = "_3trunks",
)
