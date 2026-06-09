# Severe-Grade Binary And Ordinal Analysis

Date: 2026-06-08

## Recommendation

**Use tile EfficientNet-B0 as the new leading image-only benchmark; keep metadata/downstream models as sensitivity analyses.**

The severe-grade endpoint is defined as BRAR Level 3 versus Levels 1-2. Thresholds are selected on validation folds only, then evaluated on held-out test folds. Metadata and downstream models remain guardrails or sensitivity analyses, not primary image models.

Best severe-grade model by balanced accuracy: `image_plus_age_sex`. Best image-only severe-grade model: `image_tile_efficientnet_b0_meanmax`. Best ordinal three-class model by quadratic weighted kappa: `image_tile_efficientnet_b0_meanmax`.

## Severe-Grade Test Summary

| model_id | kind | folds | balanced_accuracy_mean | sensitivity_mean | specificity_mean | f1_mean | auroc_mean | ece_10_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_plus_age_sex | metadata_sensitivity | 15 | 0.6371 | 0.6367 | 0.6374 | 0.4771 | 0.6906 | 0.0831 |
| image_plus_downstream_age_sex_upper_bound | upper_bound | 15 | 0.6365 | 0.6497 | 0.6233 | 0.4775 | 0.6872 | 0.0812 |
| image_tile_efficientnet_b0_meanmax | image_only | 15 | 0.6309 | 0.5663 | 0.6954 | 0.4661 | 0.6965 | 0.0761 |
| image_efficientnet_b0 | image_only | 15 | 0.6280 | 0.6304 | 0.6256 | 0.4662 | 0.6911 | 0.0895 |
| image_resnet50 | image_only | 15 | 0.6099 | 0.4919 | 0.7279 | 0.4337 | 0.6506 | 0.0753 |
| downstream_plus_age_sex | upper_bound | 15 | 0.5840 | 0.4548 | 0.7132 | 0.3914 | 0.6367 | 0.0889 |
| age_sex | metadata_guardrail | 15 | 0.5681 | 0.3060 | 0.8301 | 0.3252 | 0.5855 | 0.1024 |
| majority_class | baseline | 15 | 0.5000 | 1.0000 | 0.0000 | 0.4141 | 0.5000 | 0.0025 |

## Ordinal Three-Class Summary

| model_id | kind | folds | macro_f1_mean | ordinal_mae_mean | two_grade_error_rate_mean | quadratic_weighted_kappa_mean | severe_auroc_from_prob_3_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| image_tile_efficientnet_b0_meanmax | image_only | 15 | 0.5329 | 0.4649 | 0.0364 | 0.3824 | 0.6965 |
| image_plus_age_sex | metadata_sensitivity | 15 | 0.4981 | 0.5044 | 0.0378 | 0.3520 | 0.6906 |
| image_plus_downstream_age_sex_upper_bound | upper_bound | 15 | 0.4942 | 0.5085 | 0.0368 | 0.3514 | 0.6872 |
| image_efficientnet_b0 | image_only | 15 | 0.4928 | 0.5132 | 0.0405 | 0.3419 | 0.6911 |
| downstream_plus_age_sex | upper_bound | 15 | 0.5063 | 0.5442 | 0.0840 | 0.3101 | 0.6367 |
| image_resnet50 | image_only | 15 | 0.4813 | 0.5010 | 0.0445 | 0.2863 | 0.6506 |
| age_sex | metadata_guardrail | 15 | 0.4850 | 0.5816 | 0.0968 | 0.2605 | 0.5855 |
| majority_class | baseline | 15 | 0.2412 | 0.4332 | 0.0000 | 0.0000 | 0.5000 |

## Key Paired Deltas

Positive deltas favor the left model, except for log loss, ECE, ordinal MAE, and two-grade error rate.

| left_model | right_model | metric | paired_folds | mean_delta | sd_delta | left_better_folds |
| --- | --- | --- | --- | --- | --- | --- |
| image_efficientnet_b0 | age_sex | balanced_accuracy | 15 | 0.0599 | 0.0453 | 14 |
| image_efficientnet_b0 | age_sex | sensitivity | 15 | 0.3244 | 0.2678 | 13 |
| image_efficientnet_b0 | age_sex | specificity | 15 | -0.2046 | 0.2520 | 3 |
| image_efficientnet_b0 | age_sex | auroc | 15 | 0.1056 | 0.0584 | 15 |
| image_tile_efficientnet_b0_meanmax | image_efficientnet_b0 | balanced_accuracy | 15 | 0.0029 | 0.0520 | 9 |
| image_tile_efficientnet_b0_meanmax | image_efficientnet_b0 | sensitivity | 15 | -0.0641 | 0.1883 | 6 |
| image_tile_efficientnet_b0_meanmax | image_efficientnet_b0 | specificity | 15 | 0.0699 | 0.1896 | 9 |
| image_tile_efficientnet_b0_meanmax | image_efficientnet_b0 | auroc | 15 | 0.0055 | 0.0376 | 8 |
| image_tile_efficientnet_b0_meanmax | age_sex | balanced_accuracy | 15 | 0.0628 | 0.0376 | 15 |
| image_tile_efficientnet_b0_meanmax | age_sex | sensitivity | 15 | 0.2603 | 0.1974 | 14 |
| image_tile_efficientnet_b0_meanmax | age_sex | specificity | 15 | -0.1347 | 0.2086 | 2 |
| image_tile_efficientnet_b0_meanmax | age_sex | auroc | 15 | 0.1111 | 0.0477 | 15 |
| image_plus_age_sex | age_sex | balanced_accuracy | 15 | 0.0690 | 0.0328 | 15 |
| image_plus_age_sex | age_sex | sensitivity | 15 | 0.3307 | 0.2412 | 14 |
| image_plus_age_sex | age_sex | specificity | 15 | -0.1927 | 0.2212 | 1 |
| image_plus_age_sex | age_sex | auroc | 15 | 0.1051 | 0.0601 | 15 |
