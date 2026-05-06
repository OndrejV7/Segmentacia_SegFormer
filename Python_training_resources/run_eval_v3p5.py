"""
Wrapper: v3p5 model na full teste (222 imgs, identicky s v3/v4).
Vystupy: predictions_v3p5/, metrics_v3p5.json
"""
from evaluate_test import main, DATA_DIR

main(
    model_path    = DATA_DIR / "net_segformer_b2_v3p5.pth",
    test_split    = DATA_DIR / "splits" / "test_v4_files.json",  # = test_v3 obsahom
    output_suffix = "",
)
