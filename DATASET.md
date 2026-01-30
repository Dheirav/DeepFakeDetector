
# Multi Level Deepfake Dataset Builder

This project builds a research grade dataset pipeline for deepfake detection.

Goals:
- Build a dataset with three classes: real, AI generated, AI edited
- Enforce class balance
- Deduplicate images across all sources
- Filter low quality images
- Sample exact image quotas per dataset
- Create deterministic train, validation, and test splits
- Produce reproducible manifests and logs

Code quality requirements:
- Modular architecture
- Clear logging
- Strong error handling
- Deterministic outputs
- No shortcuts or brittle assumptions

This project prioritizes correctness over speed.
