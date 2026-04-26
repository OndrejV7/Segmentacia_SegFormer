"""
Wrapper: Focal-Tversky V2 (aggressive: beta_prask=0.90, beta_nez=0.80)
na full teste (222 imgs).

Vystupy: predictions_v3_focaltversky_v2/, metrics_v3_focaltversky_v2.json
"""
from evaluate_test import main, DATA_DIR, TEST_SPLIT

main(
    model_path    = DATA_DIR / "net_segformer_b2_v3_focaltversky_v2.pth",
    test_split    = TEST_SPLIT,
    output_suffix = "",
)
