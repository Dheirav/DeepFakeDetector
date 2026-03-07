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
def run_pipeline(config_path: str, dry_run: bool = False, append: bool = False, artifacts_dir_override: str = None):
    start_time = time.time()
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dataset_version = config.get('dataset_version', 'v1')
    # Allow CLI override of artifacts dir for per-dataset runs
    if artifacts_dir_override:
        artifacts_dir = Path(artifacts_dir_override).resolve()
    else:
        artifacts_dir = Path(config.get('artifacts_dir', f'artifacts/{dataset_version}')).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # If append mode, write the new run into a temporary artifacts dir and merge later
    if append:
        timestamp = int(time.time())
        artifacts_dir_new = artifacts_dir.parent / f"{artifacts_dir.name}_tmp_{timestamp}"
        artifacts_dir_target = artifacts_dir_new.resolve()
    else:
        artifacts_dir_target = artifacts_dir
    artifacts_dir_target.mkdir(parents=True, exist_ok=True)
    log_path = artifacts_dir_target / 'pipeline.log'
    logger = setup_logger(log_path)
    logger.info(f"Pipeline started. Artifacts dir: {artifacts_dir_target}")
    set_global_seed(config.get('random_seed', 42), logger)
    # Stage-specific required fields
    base_fields = {'path','class_label','dataset_source','quality_score','md5_hash','sha256','width','height'}
    split_fields = base_fields | {'split','cluster_id'}
    export_fields = split_fields | {'export_path'}
    # Stage 1: Indexing
    index_csv = artifacts_dir_target / 'index.csv'
    logger.info("[1/8] Dataset Indexing...")
    if not index_csv.exists() or dry_run:
        project_root = Path(config.get('project_root', '.'))
        source_dirs = config.get('source_dirs', [])
        class_map = config.get('class_map', {})
        index_dataset(source_dirs, str(index_csv), logger, dry_run=dry_run, project_root=project_root, class_map=class_map)
    if not dry_run:
        validate_artifact(index_csv, base_fields, stage="Indexing")
    # Stage 2: Validation
    validated_csv = artifacts_dir_target / 'validated_index.csv'
    logger.info("[2/8] Image Validation...")
    if not validated_csv.exists() or dry_run:
        validate_images(str(index_csv), str(validated_csv), config, logger, dry_run=dry_run)
    if not dry_run:
        validate_artifact(validated_csv, base_fields, stage="Validation")
    # Stage 3: Deduplication
    deduped_csv = artifacts_dir_target / 'deduped_index.csv'
    logger.info("[3/8] Deduplication...")
    if not deduped_csv.exists() or dry_run:
        deduplicate_images(str(validated_csv), str(deduped_csv), config, logger, dry_run=dry_run)
    if not dry_run:
        validate_artifact(deduped_csv, base_fields, stage="Deduplication")
    # Stage 4 & 5: Quality Filtering and Sampling (combined in sample_dataset)
    sampled_csv = artifacts_dir_target / 'sampled_index.csv'
    logger.info("[4/8] Quality Filtering and Class Balancing...")
    if not sampled_csv.exists() or dry_run:
        sample_dataset(str(deduped_csv), str(sampled_csv), config, logger, dry_run=dry_run)
    if not dry_run:
        validate_artifact(sampled_csv, base_fields, stage="Sampling")
    # Stage 5: Cluster-Based Split
    split_csv = artifacts_dir_target / 'split_index.csv'
    split_report_csv = artifacts_dir_target / 'split_report.csv'
    logger.info("[5/8] Cluster-Based Train/Val/Test Split...")
    if not split_csv.exists() or dry_run:
        split_dataset(str(sampled_csv), str(split_csv), str(split_report_csv), config, logger, dry_run=dry_run)
    if not dry_run:
        validate_artifact(split_csv, split_fields, stage="Split")
    # Stage 6: Export and Packaging
    export_csv = artifacts_dir_target / 'export_index.csv'
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
    audit_json = artifacts_dir_target / 'audit_report.json'
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
    logger.info(f"Artifacts written to: {artifacts_dir_target}")
    logger.info(f"Audit report: {audit_json}")
    logger.info(f"Total runtime: {round(time.time() - start_time, 2)} seconds.")
    print(f"\nPipeline complete. See {artifacts_dir_target}/pipeline.log for details.")


def _merge_artifacts(existing_dir: Path, new_dir: Path, config: Dict[str, Any], logger: logging.Logger):
    """Merge CSV artifacts from new_dir into existing_dir safely, deduplicating on md5+sha256."""
    import csv
    from shutil import copy2

    csv_names = ['index.csv', 'validated_index.csv', 'deduped_index.csv', 'sampled_index.csv', 'split_index.csv', 'export_index.csv']

    def merge_csv(existing_path: Path, new_path: Path, out_path: Path):
        # read header and rows, dedupe by md5_hash + sha256
        existing_rows = []
        seen = set()
        headers = None
        if existing_path.exists():
            with open(existing_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                for r in reader:
                    k = (r.get('md5_hash',''), r.get('sha256',''))
                    if k in seen:
                        continue
                    seen.add(k)
                    existing_rows.append(r)
        # add new rows
        if new_path.exists():
            with open(new_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                if headers is None:
                    headers = reader.fieldnames
                for r in reader:
                    k = (r.get('md5_hash',''), r.get('sha256',''))
                    if k in seen:
                        continue
                    seen.add(k)
                    existing_rows.append(r)
        # write merged
        if headers is None:
            # nothing to write
            return
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)

    # Merge CSVs
    for name in csv_names:
        existing_path = existing_dir / name
        new_path = new_dir / name
        out_path = existing_dir / name
        if not new_path.exists():
            logger.info(f"No new artifact to merge for {name}")
            continue
        logger.info(f"Merging {name}: new={new_path} into existing={existing_path}")
        merge_csv(existing_path, new_path, out_path)

    # Merge exported files listed in export_index.csv: copy any new exported files into existing export_root
    export_csv_new = new_dir / 'export_index.csv'
    if export_csv_new.exists():
        new_export_paths = []
        with open(export_csv_new, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                p = r.get('export_path')
                if p:
                    new_export_paths.append(p)
        export_root = config.get('export_root', str(existing_dir / 'exported_dataset'))
        for p in new_export_paths:
            src = Path(p)
            # if src is absolute or relative outside, try locate in new_dir parent export_root
            if not src.exists():
                candidate = new_dir.parent / src.name
                if candidate.exists():
                    src = candidate
            dest_root = Path(export_root)
            dest_root.mkdir(parents=True, exist_ok=True)
            dest = dest_root / src.name
            try:
                if src.exists() and not dest.exists():
                    copy2(str(src), str(dest))
            except Exception as e:
                logger.warning(f"Failed to copy exported file {src} -> {dest}: {e}")

    # If append mode, merge new artifacts into existing artifacts_dir
    if append:
        logger.info("Append mode: merging new artifacts into existing artifacts dir")
        try:
            _merge_artifacts(artifacts_dir, artifacts_dir_target, config, logger)
        except Exception as e:
            logger.error(f"Failed to merge artifacts: {e}")
            raise
        finally:
            # cleanup temporary artifacts dir
            try:
                shutil.rmtree(artifacts_dir_target)
            except Exception:
                pass
        logger.info(f"Merged artifacts into: {artifacts_dir}")
        print(f"Merged artifacts into: {artifacts_dir}")
