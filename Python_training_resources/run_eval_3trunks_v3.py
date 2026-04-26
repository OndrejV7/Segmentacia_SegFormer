"""
Wrapper: 2D model 'v3' (najlepsi standardny) na 3-trunks teste.
Vystupy: predictions_v3_3trunks/, metrics_v3_3trunks.json
"""
from evaluate_test import main, DATA_DIR
from pathlib import Path

main(
    model_path    = DATA_DIR / "net_segformer_b2_v3.pth",
    test_split    = DATA_DIR / "splits" / "test_3trunks_files.json",
    output_suffix = "_3trunks",
)
