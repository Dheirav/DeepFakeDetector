# Data Scripts

This directory contains scripts for dataset cleaning, splitting, and statistics:

- `clean_dataset.py`: Removes corrupted or unreadable images from each class folder.
- `split_data.py`: Splits images in each class into training and validation sets.
- `dataset_stats.py`: Reports the number of images per class and overall dataset statistics.

**Usage:**
- Run each script with `python scriptname.py` and follow any CLI prompts or arguments.
- Ensure your data is organized in `data/real/`, `data/ai_generated/`, and `data/ai_edited/` before running these scripts.
