"""
Wrapper na spustenie evaluate_test.py s ablation modelom
bez potreby pisat CLI argumenty.
Spusti cez VS Code Run/Debug tlacidlo.
"""
import sys
sys.argv = [
    "evaluate_test.py",
    "--model", "net_segformer_b2_v3_ablation.pth",
]
from evaluate_test import main, DEFAULT_MODEL, TEST_SPLIT
from pathlib import Path
import argparse

# Re-parse s nasimi argv
parser = argparse.ArgumentParser()
parser.add_argument("--model",      type=Path, default=DEFAULT_MODEL)
parser.add_argument("--test-split", type=Path, default=TEST_SPLIT)
args = parser.parse_args()
main(args.model, args.test_split)
