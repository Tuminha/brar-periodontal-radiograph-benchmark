# Publication Strength Summary

Date: 2026-06-08

## Recommendation

**Proceed, but strengthen before manuscript drafting.** The image-only baseline beats age/sex, but the margin is small.

The current EfficientNet-B0 frozen image baseline is methodologically clean, but its three-class macro-F1 margin over age/sex is small: `0.0078`. This makes the strongest near-term article angle a leakage-aware calibrated benchmark with explicit metadata guardrails, not a claim of clinical-grade image AI.

## Primary Model Comparison

| model_id | folds | macro_f1_mean | macro_f1_sd | balanced_accuracy_mean | ece_10_mean | quadratic_weighted_kappa_mean | ovr_auroc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| downstream_plus_age_sex | 15 | 0.5063 | 0.0277 | 0.5681 | 0.0781 | 0.3101 | 0.7212 |
| image_efficientnet_b0_temperature_scaled | 15 | 0.4928 | 0.0448 | 0.5037 | 0.0930 | 0.3419 | 0.6829 |
| age_sex | 15 | 0.4850 | 0.0317 | 0.5445 | 0.0736 | 0.2605 | 0.6968 |
| image_geometry_file | 15 | 0.3044 | 0.0348 | 0.3474 | 0.0838 | -0.0116 | 0.5313 |
| majority_class | 15 | 0.2412 | 0.0004 | 0.3333 | 0.0019 | 0.0000 | 0.5000 |
| stratified_random | 15 | 0.2412 | 0.0004 | 0.3333 | 0.0019 | 0.0000 | 0.5000 |
| admin_index | 15 | 0.1929 | 0.0465 | 0.2831 | 0.1640 | -0.0519 | 0.5293 |

## Paired Deltas

Positive deltas favor the left model, except for log loss and ECE where negative deltas favor the left model.

| left_model | right_model | metric | mean_delta | sd_delta | left_better_folds |
| --- | --- | --- | --- | --- | --- |
| image_efficientnet_b0_temperature_scaled | age_sex | macro_f1 | 0.0078 | 0.0532 | 8 |
| image_efficientnet_b0_temperature_scaled | age_sex | balanced_accuracy | -0.0409 | 0.0594 | 5 |
| image_efficientnet_b0_temperature_scaled | age_sex | log_loss | -0.0120 | 0.0630 | 10 |
| image_efficientnet_b0_temperature_scaled | age_sex | ece_10 | 0.0195 | 0.0412 | 5 |
| image_efficientnet_b0_temperature_scaled | downstream_plus_age_sex | macro_f1 | -0.0135 | 0.0552 | 7 |
| image_efficientnet_b0_temperature_scaled | downstream_plus_age_sex | balanced_accuracy | -0.0644 | 0.0609 | 3 |
| image_efficientnet_b0_temperature_scaled | downstream_plus_age_sex | log_loss | 0.0005 | 0.0813 | 9 |
| image_efficientnet_b0_temperature_scaled | downstream_plus_age_sex | ece_10 | 0.0149 | 0.0488 | 5 |

## Binary Task Signals

These are derived from the same multiclass probabilities with thresholds selected on the validation fold only.

| task | model_id | folds | balanced_accuracy_mean | f1_mean | auroc_mean | ece_10_mean |
| --- | --- | --- | --- | --- | --- | --- |
| level_1_vs_higher | age_sex | 15 | 0.7504 | 0.8100 | 0.8387 | 0.0920 |
| level_1_vs_higher | image_efficientnet_b0_temperature_scaled | 15 | 0.6955 | 0.7677 | 0.7679 | 0.0602 |
| level_3_vs_lower | image_efficientnet_b0_temperature_scaled | 15 | 0.6280 | 0.4662 | 0.6911 | 0.0676 |
| level_3_vs_lower | age_sex | 15 | 0.5681 | 0.3252 | 0.5855 | 0.0986 |

## Next Analysis To Prioritize

1. Train an `image_plus_age_sex` frozen-embedding sensitivity model to test whether images add incremental signal to demographics.
2. Run one stronger image-only encoder, preferably ResNet50 or ConvNeXt-Tiny, using the same split manifest and no test-set model shopping.
3. Add repeat-aware confidence intervals and subgroup tables to support a transparent benchmark manuscript.
4. Treat binary severe/non-severe performance as a secondary analysis only if it is clearly more stable than the three-class task.
