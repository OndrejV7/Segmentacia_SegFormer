"""
Wrapper: 2D model 'v3_ablation' (bez Dub_praskliny_a a hrce_mixed v traini)
na 3-trunks teste. Toto je matching-train 2D baseline pre 2.5D porovnanie.
Vystupy: predictions_v3_ablation_3trunks/, metrics_v3_ablation_3trunks.json
"""
from evaluate_test import main, DATA_DIR

main(
    model_path    = DATA_DIR / "net_segformer_b2_v3_ablation.pth",
    test_split    = DATA_DIR / "splits" / "test_3trunks_files.json",
    output_suffix = "_3trunks",
)
