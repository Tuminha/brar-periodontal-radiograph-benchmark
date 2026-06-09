# Pre-Model Audit

Date: 2026-06-07

## Runtime

- Python executable: `/Users/francisco/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`
- Python version: `3.12.13 (main, Mar  3 2026, 15:35:03) [Clang 21.1.4 ]`
- Platform: `macOS-26.4.1-arm64-arm-64bit`
- Pillow: `12.2.0`
- NumPy: `2.3.5`
- pandas: `2.2.3`
- scikit-learn: `not available (ModuleNotFoundError)`
- torch: `not available (ModuleNotFoundError)`
- torchvision: `not available (ModuleNotFoundError)`

## Near-Duplicate Audit

- Images fingerprinted: 988
- Exact duplicate SHA-256 groups from manifest: 0
- Low-priority hash-only candidates logged: 1226
- High-priority near-duplicate candidates: 0
- Medium-priority near-duplicate candidates: 0
- Candidate split-risk rows: 0
- Candidate split-risk rows crossing fold group: 0

Interpretation: exact duplicates remain absent. Low-priority hash-only similarities are expected in panoramic radiographs because the images share a common global silhouette. Any high/medium near-duplicate candidate crossing fold groups should be manually reviewed before final image-model evaluation.

## Split Balance Summary

| Split | N range | Mean age range | Gender=1 proportion range | Mean aspect-ratio range |
| --- | ---: | ---: | ---: | ---: |
| train | 592-594 | 39.946-41.024 | 0.4696-0.5093 | 2.0777-2.0839 |
| val | 197-198 | 39.788-42.041 | 0.4467-0.5354 | 2.0772-2.0914 |
| test | 197-198 | 39.788-42.041 | 0.4467-0.5354 | 2.0772-2.0914 |

Interpretation: the current repeated stratified folds are balanced by severity and do not show severe age, gender, or image-geometry imbalance.

## Generated Files

- `data/processed/image_fingerprints.csv`
- `reports/near_duplicate_candidates.csv`
- `reports/near_duplicate_nearest_neighbors.csv`
- `reports/near_duplicate_split_risk.csv`
- `reports/split_balance_audit.csv`
- `reports/pre_model_audit_summary.json`
- `reports/environment_report.md`
