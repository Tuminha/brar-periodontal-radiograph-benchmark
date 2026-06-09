# Negative-Control Baseline Summary

Date: 2026-06-07

## Purpose

These are non-image baselines. They estimate how much BRAR severity can be predicted before any image pixels are used. They are guardrails against overstating an image model if metadata, image geometry, file order, or downstream dental-status variables already carry substantial signal.

## Test-Set Aggregate Metrics

| Model | Feature set | Macro-F1 mean | Balanced accuracy mean | Accuracy mean | Log loss mean | ECE mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| multinomial_logistic | downstream_plus_age_sex | 0.5063 | 0.5681 | 0.5398 | 0.9625 | 0.0781 |
| multinomial_logistic | age_sex | 0.4850 | 0.5445 | 0.5152 | 0.9750 | 0.0736 |
| multinomial_logistic | age_sex_geometry | 0.4571 | 0.5199 | 0.4831 | 0.9984 | 0.0730 |
| multinomial_logistic | downstream_status | 0.3559 | 0.4787 | 0.3532 | 1.0573 | 0.1488 |
| stratified_random | class_prior | 0.3282 | 0.3294 | 0.4177 | 0.9753 | 0.1491 |
| multinomial_logistic | image_geometry_file | 0.3044 | 0.3474 | 0.3528 | 1.0979 | 0.0838 |
| majority_class | class_prior | 0.2412 | 0.3333 | 0.5668 | 0.9753 | 0.0019 |
| multinomial_logistic | admin_index | 0.1929 | 0.2831 | 0.2129 | 1.0991 | 0.1640 |

## Interpretation Rules

- `majority_class` is the minimum baseline. It should have high raw accuracy because Level 2 is common, but low macro-F1 and balanced accuracy.
- `stratified_random` estimates random-label performance under the train class prevalence.
- `age_sex` is a metadata sensitivity baseline, not the primary model.
- `image_geometry_file` and `admin_index` are negative controls. Strong performance here would suggest acquisition/file-order confounding.
- `downstream_status` and `downstream_plus_age_sex` are upper-bound sensitivity models and are not deployment-ready.

## Generated Files

- `data/processed/negative_controls/negative_control_predictions.csv`
- `reports/negative_control_metrics.csv`
- `reports/negative_control_model_selection.csv`
- `reports/negative_control_confusion_matrices.csv`
- `reports/negative_control_summary.json`
