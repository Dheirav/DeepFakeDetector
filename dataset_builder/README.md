# Deepfake Dataset Builder Pipeline

This module provides a production-grade, deterministic, and auditable pipeline for building machine learning datasets for deepfake detection and related tasks.

## Features
- Modular, robust, and deterministic pipeline stages
- Config-driven and reproducible artifact flow
- Strong error handling and compliance-grade validation
- Dry-run and strict mode support
- Structured logging and professional reporting

## Pipeline Stages
1. **Indexing**: Scans and indexes all dataset files.
2. **Validation**: Validates image integrity and metadata.
3. **Deduplication**: Removes duplicate or near-duplicate files.
4. **Quality Filtering**: Filters low-quality or outlier samples.
5. **Sampling & Class Balancing**: Ensures balanced class distribution.
6. **Cluster-Based Split**: Splits data into train/val/test sets by cluster.
7. **Export & Packaging**: Exports the final dataset and index.
8. **Audit**: Runs integrity and compliance checks.

## Usage

### Command Line
Run the pipeline from the `dataset_builder` directory:

```bash
python main.py --config path/to/config.yaml [--dry-run] [--log-level INFO]
```

- `--config`: Path to the YAML config file (see below).
- `--dry-run`: Simulate all stages without writing artifacts.
- `--log-level`: Set logging verbosity (DEBUG, INFO, WARNING, ERROR).

### Example
```bash
python main.py --config config/dataset_config.yaml --log-level INFO
```

### Configuration
The pipeline is fully driven by a YAML config file. Example keys:

```yaml
random_seed: 42
artifacts_dir: artifacts/v1
export_root: artifacts/v1/exported_dataset
strict_mode: true
image_rules:
  min_width: 256
  min_height: 256
class_targets:
  - real
  - ai_edited
  - ai_generated
split_ratios:
  train: 0.7
  val: 0.15
  test: 0.15
```

## Artifacts
All intermediate and final artifacts are written to the `artifacts_dir` specified in the config:
- `index.csv`: Raw dataset index
- `validated_index.csv`: Validated index
- `deduped_index.csv`: Deduplicated index
- `filtered_index.csv`: Quality-filtered index
- `sampled_index.csv`: Class-balanced sample
- `split_index.csv`: Train/val/test split
- `export_index.csv`: Final exported index
- `audit_report.json`: Audit report
- `pipeline.log`: Full pipeline log

## Logging & Reporting
- All logs are written to `artifacts_dir/pipeline.log` and the console.
- Audit and summary reports are generated at the end of the run.

## Extending the Pipeline
Each stage is modular and can be extended or replaced. See the `modules/` directory for details.

## Troubleshooting
- Use `--dry-run` to test config and pipeline logic without writing files.
- Enable `strict_mode` in config to enforce audit compliance.
- Check `pipeline.log` for detailed error messages and diagnostics.
