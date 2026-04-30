"""
Wrapper: v4 model (Dice baseline na rozsirenom datasete) na test_v4 (= test_v3, 222 imgs).
Vystupy: predictions_v4/, metrics_v4.json, metrics_v4.csv
"""
from evaluate_test import main, DATA_DIR

main(
    model_path    = DATA_DIR / "net_segformer_b2_v4.pth",
    test_split    = DATA_DIR / "splits" / "test_v4_files.json",
    output_suffix = "",
)
