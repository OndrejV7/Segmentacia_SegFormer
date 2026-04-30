"""
Wrapper: v4 Focal-Tversky v1 (β_prask=0.85) na test_v4.
"""
from evaluate_test import main, DATA_DIR

main(
    model_path    = DATA_DIR / "net_segformer_b2_v4_focaltversky.pth",
    test_split    = DATA_DIR / "splits" / "test_v4_files.json",
    output_suffix = "",
)
