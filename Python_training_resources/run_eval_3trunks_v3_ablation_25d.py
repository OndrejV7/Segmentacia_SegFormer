"""
Wrapper: 2.5D model 'v3_ablation_25d' na 3-trunks teste.
Toto je hlavne porovnanie -- meriame ci 2.5D context pomohol.
Vystupy: predictions_v3_ablation_25d_3trunks/, metrics_v3_ablation_25d_3trunks.json
"""
from evaluate_test_25d import main, DATA_DIR

main(
    model_path    = DATA_DIR / "net_segformer_b2_v3_ablation_25d.pth",
    test_split    = DATA_DIR / "splits" / "test_3trunks_files.json",
    output_suffix = "_3trunks",
)
