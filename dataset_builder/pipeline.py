"""
Production-grade, deterministic dataset builder pipeline orchestrator.
Integrates all pipeline modules in strict order, enforcing config, logging, determinism, and artifact flow.
"""
import os
import sys
import time
import shutil
import random
import logging
from pathlib import Path
from typing import Dict, Any
import yaml
import numpy as np

# Import all pipeline modules (assume they exist and are robust)
from modules.indexer import index_dataset
from modules.validator import validate_images
from modules.deduplicator import deduplicate_images
from modules.sampler import sample_dataset
from modules.splitter import split_dataset
from modules.exporter import export_dataset
from modules.audit_dataset import audit_dataset

# --- Helper utilities ---
def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("dataset_pipeline")
    logger.setLevel(logging.INFO)
    # Remove all handlers to prevent duplication (idempotent)
    handlers = list(logger.handlers)
    for h in handlers:
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def validate_artifact(path: Path, required_fields: set, min_rows: int = 1, stage: str = ""):
    import csv
    if not path.exists():
        raise FileNotFoundError(f"{stage}: Artifact not found: {path}")
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not required_fields.issubset(fields):
            raise ValueError(f"{stage}: Artifact schema missing fields: {required_fields - fields}")
        row_count = sum(1 for _ in reader)
        if row_count < min_rows:
            raise ValueError(f"{stage}: Artifact has too few rows: {row_count}")
    return True

_SEED_SET = False
def set_global_seed(seed: int, logger=None):
    global _SEED_SET
    if not _SEED_SET:
        random.seed(seed)
        np.random.seed(seed)
        _SEED_SET = True
        if logger:
            logger.info(f"Global random seed set to {seed}")

# --- Main Orchestrator ---
def run_pipeline(config_path: str, dry_run: bool = False):
    start_time = time.time()
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dataset_version = config.get('dataset_version', 'v1')
    artifacts_dir = Path(config.get('artifacts_dir', f'artifacts/{dataset_version}')).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifacts_dir / 'pipeline.log'
    logger = setup_logger(log_path)
    logger.info(f"Pipeline started. Artifacts dir: {artifacts_dir}")
    set_global_seed(config.get('random_seed', 42), logger)
    # Stage-specific required fields
    base_fields = {'path','class_label','dataset_source','quality_score','md5_hash','sha256','width','height'}
    split_fields = base_fields | {'split','cluster_id'}
    export_fields = split_fields | {'export_path'}
    # Stage 1: Indexing
    index_csv = artifacts_dir / 'index.csv'
    logger.info("[1/8] Dataset Indexing...")
    if not index_csv.exists() or dry_run:
        project_root = Path(config.get('project_root', '.'))
        source_dirs = config.get('source_dirs', [])
        class_map = config.get('class_map', {})
        index_dataset(source_dirs, str(index_csv), logger, dry_run=dry_run, project_root=project_root, class_map=class_map)
    if not dry_run:
        validate_artifact(index_csv, base_fields, stage="Indexing")
    # Stage 2: Validation
    validated_csv = artifacts_dir / 'validated_index.csv'
    logger.info("[2/8] Image Validation...")
    if not validated_csv.exists() or dry_run:
        validate_images(str(index_csv), str(validated_csv), config, logger, dry_run=dry_run)
    if not dry_run:
        validate_artifact(validated_csv, base_fields, stage="Validation")
    # Stage 3: Deduplication
    deduped_csv = artifacts_dir / 'deduped_index.csv'
    logger.info("[3/8] Deduplication...")
    if not deduped_csv.exists() or dry_run:
        deduplicate_images(str(validated_csv), str(deduped_csv), config, logger, dry_run=dry_run)
    if not dry_run:
        validate_artifact(deduped_csv, base_fields, stage="Deduplication")
    # Stage 4 & 5: Quality Filtering and Sampling (combined in sample_dataset)
    sampled_csv = artifacts_dir / 'sampled_index.csv'
    logger.info("[4/8] Quality Filtering and Class Balancing...")
    if not sampled_csv.exists() or dry_run:
        sample_dataset(str(deduped_csv), str(sampled_csv), config, logger, dry_run=dry_run)
    if not dry_run:
        validate_artifact(sampled_csv, base_fields, stage="Sampling")
    # Stage 5: Cluster-Based Split
    split_csv = artifacts_dir / 'split_index.csv'
    split_report_csv = artifacts_dir / 'split_report.csv'
    logger.info("[5/8] Cluster-Based Train/Val/Test Split...")
    if not split_csv.exists() or dry_run:
        split_dataset(str(sampled_csv), str(split_csv), str(split_report_csv), config, logger, dry_run=dry_run)
    if not dry_run:
        validate_artifact(split_csv, split_fields, stage="Split")
    # Stage 6: Export and Packaging
    export_csv = artifacts_dir / 'export_index.csv'
    logger.info("[6/8] Export and Packaging...")
    export_root = config.get('export_root', str(artifacts_dir / 'exported_dataset'))
    if not export_csv.exists() or dry_run:
        export_dataset(str(split_csv), export_root, config, logger, dry_run=dry_run)
        # After export, ensure export_index.csv is present in artifacts_dir
        export_root_index = Path(export_root) / 'export_index.csv'
        if not dry_run and export_root_index.exists():
            try:
                # Copy or overwrite to artifacts_dir
                shutil.copy2(str(export_root_index), str(export_csv))
            except Exception as e:
                logger.error(f"Failed to copy export_index.csv to artifacts_dir: {e}")
                raise
    if not dry_run:
        if not export_csv.exists():
            logger.error(f"Export artifact missing: {export_csv}")
            raise FileNotFoundError(f"Export artifact missing: {export_csv}")
        validate_artifact(export_csv, export_fields, stage="Export")
    # Stage 7: Audit
    audit_json = artifacts_dir / 'audit_report.json'
    logger.info("[7/8] Dataset Integrity Audit...")
    audit_dataset(str(export_csv), str(audit_json), config, logger, dry_run=dry_run)
    # Strict mode enforcement
    if not dry_run and config.get('strict_mode', False):
        import json
        with open(audit_json, 'r') as f:
            audit_report = json.load(f)
        if audit_report.get('verdict') == 'FAIL':
            logger.error("Audit verdict is FAIL and strict_mode is enabled. Stopping pipeline.")
            raise RuntimeError("Pipeline failed strict audit.")
    # Final summary
    logger.info("\n=== FINAL PIPELINE SUMMARY ===")
    logger.info(f"Artifacts written to: {artifacts_dir}")
    logger.info(f"Audit report: {audit_json}")
    logger.info(f"Total runtime: {round(time.time() - start_time, 2)} seconds.")
    print(f"\nPipeline complete. See {artifacts_dir}/pipeline.log for details.")
