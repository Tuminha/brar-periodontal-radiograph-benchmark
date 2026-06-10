# BRAR Periodontal Radiograph Benchmark

Created: 2026-06-04

This workspace contains the feasibility pack and early modeling outputs for a publishable BRAR panoramic radiograph machine-learning project. Sprint 1 verified the dataset, linkage, leakage constraints, literature gap, and GO/NO-GO decision. Sprint 2 has now produced guarded split audits, negative controls, frozen image baselines, publication-strength analysis, and embedding sensitivity models.

## Decision

Proceed with BRAR, but with revised scope:

- feasible: public image-level benchmark for predicting released `Level` severity from panoramic radiographs;
- not yet feasible: tooth-level multimodal benchmark, because the inspected public ZIP does not include the tooth-level annotation tables described by the article/codebook.

See `reports/GO_NO_GO.md`.

## Key Local Files

- `data/raw/BRAR-anchored_multimodal_dataset.zip` - Figshare full ZIP, file id `58062268`.
- `data/extracted/meta_data.csv` - released metadata table.
- `data/extracted/level_1/`, `level_2/`, `level_3/` - released panoramic JPG files by grade folder.
- `scripts/build_feasibility_reports.py` - reproducible local audit script.
- `scripts/01_build_manifest.py` - builds the modeling manifest with image paths, labels, dimensions, and hashes.
- `scripts/02_make_splits.py` - creates repeated stratified 5-fold train/validation/test split manifests.
- `scripts/03_render_dataset_split_report.py` - renders the static dataset/split HTML report.
- `scripts/04_audit_splits_and_near_duplicates.py` - audits near-duplicate image risk and split balance beyond class labels.
- `scripts/05_run_negative_control_baselines.py` - runs non-image negative-control and sensitivity baselines.
- `scripts/06_render_guardrail_report.py` - renders the static pre-model guardrail HTML report.
- `scripts/07_setup_training_environment.py` - records the local image-training environment and lock file.
- `scripts/08_train_frozen_image_baseline.py` - extracts frozen image embeddings and trains the image-only baseline.
- `scripts/09_evaluate_image_baseline.py` - evaluates raw and validation-temperature-scaled image baseline predictions.
- `scripts/10_render_model_report.py` - renders the static image-baseline HTML report.
- `scripts/11_analyze_publication_strength.py` - summarizes image-vs-metadata strength, binary task signals, subgroup checks, and confident errors.
- `scripts/12_train_embedding_sensitivity_models.py` - trains cached-embedding image-plus-metadata sensitivity models.
- `scripts/13_analyze_severe_grade_binary_ordinal.py` - analyzes validation-thresholded severe-grade binary tasks and ordinal three-class behavior across saved models.
- `scripts/14_train_tile_efficientnet_baseline.py` - trains the tile-based frozen EfficientNet-B0 image-only baseline.
- `scripts/15_publication_uncertainty_and_error_audit.py` - builds image-level uncertainty intervals, paired comparisons, subgroup checks, and tile-model error-audit contact sheets from saved predictions.
- `data/processed/brar_manifest.csv` - model-ready manifest generated from released metadata and images.
- `data/processed/splits/brar_repeated_5fold_splits.csv` - long-format repeated split manifest.
- `data/processed/image_fingerprints.csv` - perceptual image fingerprints used for near-duplicate auditing.
- `data/processed/negative_controls/negative_control_predictions.csv` - predictions from non-image guardrail baselines.
- `data/processed/image_baseline/frozen_efficientnet_b0_384x192_predictions.csv` - image-only frozen baseline predictions.
- `data/processed/image_baseline/embeddings_efficientnet_b0_384x192.npz` - cached frozen image embeddings.
- `data/processed/image_baseline/frozen_resnet50_384x192_predictions.csv` - predeclared stronger frozen image-only encoder check.
- `data/processed/image_baseline/tile_efficientnet_b0_384_meanmax_predictions.csv` - tile-based image-only EfficientNet-B0 predictions.
- `data/processed/image_baseline/tile_embeddings_efficientnet_b0_384_4tiles.npz` - cached four-tile EfficientNet-B0 embeddings.
- `data/processed/embedding_sensitivity/efficientnet_b0_384x192_sensitivity_predictions.csv` - cached-embedding sensitivity predictions.
- `reports/file_inventory.csv` - file inventory for raw and extracted data.
- `reports/data_linkage_audit.md` - checksum, file structure, linkage, and image-readability audit.
- `reports/leakage_audit.md` - predictor/outcome policy and codebook-variable leakage rules.
- `reports/GO_NO_GO.md` - final feasibility recommendation.
- `reports/dataset_split_report.html` - static checkpoint report for reviewing data and split behavior before training.
- `reports/pre_model_audit.md` - near-duplicate, split-balance, and runtime audit.
- `reports/negative_control_summary.md` - non-image baseline results.
- `reports/guardrail_report.html` - browser-viewable guardrail report.
- `reports/training_environment_report.md` - local training environment report.
- `reports/image_baseline_summary.md` - aggregate image-baseline metrics.
- `reports/image_baseline_report.html` - browser-viewable image-baseline report.
- `reports/publication_strength_summary.md` - current publication-strength interpretation and next analysis priorities.
- `reports/publication_strength_report.html` - browser-viewable publication-strength report.
- `reports/embedding_sensitivity_summary.md` - image-plus-metadata sensitivity model summary.
- `reports/embedding_sensitivity_report.html` - browser-viewable embedding-sensitivity report.
- `reports/image_baseline_resnet50_384x192_summary.md` - ResNet50 frozen image-only summary.
- `reports/image_baseline_resnet50_384x192_report.html` - browser-viewable ResNet50 image-baseline report.
- `reports/image_baseline_tile_efficientnet_b0_384_meanmax_summary.md` - tile-based EfficientNet-B0 image-only summary.
- `reports/image_baseline_tile_efficientnet_b0_384_meanmax_report.html` - browser-viewable tile-based image-baseline report.
- `reports/severe_grade_binary_ordinal_summary.md` - severe-grade binary and ordinal analysis summary.
- `reports/severe_grade_binary_ordinal_report.html` - browser-viewable severe-grade binary/ordinal report.
- `reports/publication_uncertainty_summary.md` - manuscript-oriented uncertainty, paired-comparison, subgroup, and error-audit summary.
- `reports/publication_uncertainty_report.html` - browser-viewable publication uncertainty and error-audit report.
- `reports/publication_ready_model_table.csv` - primary manuscript-ready model table with CV summaries and image-level bootstrap intervals.
- `reports/publication_paired_interval_table.csv` - paired image-level bootstrap deltas for tile EfficientNet-B0 versus key comparators.
- `reports/tile_model_confident_errors.csv` - ranked tile-model error audit with metadata and severe false-negative flags.
- `reports/error_audit_contact_sheets/` - visual contact sheets for top confident errors, severe false negatives, and two-grade errors.

## Verified Dataset Facts

- Figshare file id: `58062268`.
- Expected and observed MD5: `4df0368a88f23f403958e6b371057f11`.
- Extracted JPG images: 988.
- Metadata rows: 988.
- Grade counts: Level 1 = 170, Level 2 = 560, Level 3 = 258.
- Linkage key: anonymized image filename; no separate `patient_id` column in the inspected public CSV.
- Tooth-level table: absent in the inspected public ZIP.

## Rebuild Reports

From this folder:

```bash
python3 scripts/build_feasibility_reports.py
```

The script regenerates the file inventory, image-readability sample, data-linkage audit, and leakage audit.

## Build Modeling Manifest And Splits

From this folder:

```bash
python3 scripts/01_build_manifest.py
python3 scripts/02_make_splits.py
python3 scripts/03_render_dataset_split_report.py
```

Generated outputs:

- `data/processed/brar_manifest.csv`
- `data/processed/brar_manifest_summary.json`
- `data/processed/splits/brar_fold_assignments.csv`
- `data/processed/splits/brar_repeated_5fold_splits.csv`
- `data/processed/splits/brar_split_summary.csv`
- `data/processed/splits/brar_split_summary.json`
- `reports/dataset_split_report.html`

Default split design:

- 3 repeats with seeds `20260606`, `20260607`, and `20260608`.
- 5 stratified folds per repeat.
- For each fold, the current fold is test, the next fold is validation, and the remaining three folds are training.
- Exact duplicate image hashes, if any appear later, are grouped so they cannot cross folds.

## Next Modeling Checkpoint

Before the first image model, follow `reports/strategy_loophole_audit.md`.

Revised sequence:

1. Lock a reproducible local ML environment.
2. Run near-duplicate and split-balance audits beyond class labels.
3. Run negative-control baselines:
   - majority class;
   - stratified random;
   - age/sex metadata;
   - image geometry/file size;
   - downstream dental-status upper-bound.
4. Then train the first frozen image-encoder baseline.

The primary model should be image-only. Metadata and downstream dental-status variables should remain separate sensitivity analyses.

Current guardrail results:

- No exact SHA-256 duplicate images.
- No high/medium near-duplicate candidates crossing fold groups after stricter low-resolution correlation checking.
- Image geometry/file-size control is weak: test macro-F1 `0.3044`.
- Administrative filename/index control is weak: test macro-F1 `0.1929`.
- Age/sex metadata baseline is strong: test macro-F1 `0.4850`.
- Downstream plus age/sex upper-bound is strongest: test macro-F1 `0.5063`.

Implication: the future image-only model must be judged against the age/sex baseline, not just the majority-class baseline.

## First Image Baseline Result

Environment:

- Local venv: `.venv`
- Encoder: `torchvision` EfficientNet-B0 with ImageNet weights.
- Device used: `mps`.
- Preprocessing: RGB conversion, aspect-fit black padding to `384x192`, ImageNet normalization.
- Classifier: class-balanced multinomial logistic regression on frozen image embeddings.
- Splits: all 15 repeated fold evaluations.

Test aggregate metrics:

| Probability mode | Macro-F1 | Balanced accuracy | Accuracy | Log loss | ECE | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 0.4928 | 0.5037 | 0.5273 | 1.4013 | 0.2144 | 0.6804 |
| Temperature scaled | 0.4928 | 0.5037 | 0.5273 | 0.9630 | 0.0930 | 0.6829 |

Interpretation:

- The frozen image baseline slightly exceeds the age/sex baseline on macro-F1 (`0.4928` vs `0.4850`).
- The margin is small, so this is not yet a strong image-only win.
- Temperature scaling improves calibration metrics but does not change classification metrics.
- The next modeling step should focus on improving image signal while preserving the same leakage guardrails.

## Current Modeling Interpretation

As of 2026-06-09:

- Tile EfficientNet-B0 mean/max pooling is now the best current image-only frozen baseline: test macro-F1 `0.5329`, balanced accuracy `0.5399`, temperature-scaled ECE `0.0736`.
- Whole-image EfficientNet-B0 remains the simple image-only reference: test macro-F1 `0.4928`, balanced accuracy `0.5037`, temperature-scaled ECE `0.0930`.
- Age/sex metadata is a strong guardrail baseline: test macro-F1 `0.4850`, balanced accuracy `0.5445`.
- Image plus age/sex sensitivity is only modestly better than image-only: test macro-F1 `0.4981`, balanced accuracy `0.5082`.
- ResNet50 frozen image-only did not improve the benchmark: test macro-F1 `0.4813`, balanced accuracy `0.4798`.
- The severe-vs-lower endpoint is strongest overall with image plus age/sex by balanced accuracy `0.6371`, but the best image-only severe model is tile EfficientNet-B0 with balanced accuracy `0.6309` and AUROC `0.6965`.
- The publication analysis pack adds image-level bootstrap intervals and paired comparisons. Tile EfficientNet-B0 beats whole-image EfficientNet-B0 on image-level macro-F1, but balanced accuracy, QWK, and severe-grade deltas versus whole-image EfficientNet overlap no-difference intervals. Tile EfficientNet-B0 shows clearer image-level advantages over age/sex for macro-F1, QWK, severe balanced accuracy, and severe AUROC.
- The tile error audit currently flags 698 images with at least one tile-model test error or severe-threshold error across repeated test appearances. The highest-priority review cases are severe Level 3 images repeatedly predicted as Level 1 or Level 2.

Implication: this is now stronger than the first whole-image baseline, but it still should not be framed as a high-performance clinical model. The strongest publication direction is a reproducible, leakage-aware, calibrated BRAR benchmark with explicit metadata-confounding analysis, tile-based image-only benchmarking, and a focused severe-grade secondary endpoint.
