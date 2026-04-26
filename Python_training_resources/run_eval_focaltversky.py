"""
Wrapper: Focal-Tversky model na full teste (222 imgs).
Pre porovnanie s v3 -- rovnaky test set ako v3.

Vystupy: predictions_v3_focaltversky/, metrics_v3_focaltversky.json
"""
from evaluate_test import main, DATA_DIR, TEST_SPLIT

main(
    model_path    = DATA_DIR / "net_segformer_b2_v3_focaltversky.pth",
    test_split    = TEST_SPLIT,                          # = splits/test_files.json (222 imgs)
    output_suffix = "",
)
