"""train.py — Standalone training script.

Usage:
    python train.py --universe combined --config ltc_config.toml
"""

import argparse

from main import cmd_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LTC engine")
    parser.add_argument("--universe", default="combined", choices=["fi", "equity", "combined"])
    parser.add_argument("--config", default="ltc_config.toml")
    cmd_train(parser.parse_args())
