"""
Main entry point for the dataset builder pipeline.
"""
import argparse
import logging
import os
import sys
import yaml
import random
import numpy as np
from datetime import datetime
from yaml import dump as yaml_dump

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Import the real pipeline orchestrator
from pipeline import run_pipeline

DEFAULT_CONFIG = "config/dataset_config.yaml"
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def setup_logging(log_level: str = "INFO"):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_filename}")


def load_config(config_path: str):
    # Resolve config path:
    # 1) if absolute, use it
    # 2) if relative and exists from cwd, use cwd-relative path
    # 3) otherwise try relative to the script directory (packaged path)
    if os.path.isabs(config_path):
        resolved = config_path
    else:
        cwd_candidate = os.path.abspath(config_path)
        if os.path.exists(cwd_candidate):
            resolved = cwd_candidate
        else:
            resolved = os.path.join(SCRIPT_DIR, config_path)
    if not os.path.exists(resolved):
        logging.error(f"Config file not found: {resolved}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to parse config: {e}")
            sys.exit(1)
    logging.info(f"Loaded config from {resolved}")
    return config


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info("Set PyTorch random seed.")
    except ImportError:
        logging.info("PyTorch not installed; skipping torch seed setup.")
    logging.info(f"Random seed set to {seed}")


def validate_config_schema(config):
    required_keys = ["random_seed", "image_rules", "class_targets", "split_ratios"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        logging.error(f"Config missing required keys: {missing}")
        sys.exit(1)
    logging.info("Config schema validation passed.")


def log_full_config(config):
    try:
        config_str = yaml_dump(config, default_flow_style=False, sort_keys=False)
        logging.info("Full configuration:\n" + config_str)
    except Exception as e:
        logging.warning(f"Could not pretty-print config: {e}")


def launch_pipeline(config_path: str, dry_run: bool = False):
    # Call the real orchestrator from pipeline.py
    try:
        # import here to avoid circular imports at module import time
        from pipeline import run_pipeline
        run_pipeline(config_path, dry_run=dry_run, append=False)
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Deepfake Dataset Builder")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG, help='Path to config YAML file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--dry-run', action='store_true', help='Run pipeline in dry-run mode (no changes made)')
    parser.add_argument('--append', action='store_true', help='Append/merge new artifacts into existing artifacts instead of skipping')
    parser.add_argument('--artifacts-dir', type=str, default=None, help='Override artifacts directory for this run')
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config)
    validate_config_schema(config)
    if args.dry_run:
        config["dry_run"] = True
        logging.info("Dry run mode enabled.")
    if args.append:
        config["append"] = True
        logging.info("Append/merge mode enabled.")
    log_full_config(config)
    set_random_seed(config.get('random_seed', 42))
    try:
        # Pass the config path and dry_run/artifacts override to the orchestrator
        from pipeline import run_pipeline
        run_pipeline(args.config, dry_run=args.dry_run, append=args.append, artifacts_dir_override=args.artifacts_dir)
    except KeyboardInterrupt:
        logging.warning("Pipeline interrupted by user (Ctrl+C). Exiting gracefully.")
        sys.exit(130)

if __name__ == "__main__":
    main()
