"""evaluate.py — Evaluate on test set and print metrics.

Usage:
    python evaluate.py --universe combined --checkpoint checkpoints/best_val_sharpe.pt
"""
import argparse

from main import cmd_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LTC engine on test set")
    parser.add_argument("--universe",   default="combined", choices=["fi","equity","combined"])
    parser.add_argument("--config",     default="ltc_config.toml")
    parser.add_argument("--checkpoint", default="checkpoints/best_val_sharpe.pt")
    cmd_eval(parser.parse_args())
