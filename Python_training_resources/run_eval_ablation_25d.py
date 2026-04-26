"""
Wrapper na spustenie evaluate_test_25d.py s ablation 2.5D modelom.
Spusti cez VS Code Run/Debug tlacidlo.
"""
import sys
sys.argv = ["evaluate_test_25d.py"]
from evaluate_test_25d import main, DEFAULT_MODEL, TEST_SPLIT

main(DEFAULT_MODEL, TEST_SPLIT)
