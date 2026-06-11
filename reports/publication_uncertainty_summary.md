# Publication Uncertainty And Error Audit

Date: 2026-06-09

## Recommendation

**Proceed toward a benchmark manuscript outline, with the tile EfficientNet-B0 model as the primary frozen-feature image-only benchmark.**

The new analysis keeps the current claim boundary: this is a reproducible, leakage-aware, calibrated BRAR benchmark with metadata guardrails and a severe-grade secondary endpoint, not a clinical-grade deployment model.

Uncertainty uses image-level out-of-fold bootstrap intervals after averaging each image's three test appearances. Severe-grade decisions use validation-selected thresholds only, then majority vote across the three held-out test appearances.

Important framing caveats:

- The inspected public full ZIP contains 988 linked images/metadata rows, although the source data descriptor describes a richer 1,104-patient cohort. The 988-image benchmark size reflects the inspected public release, not an investigator-applied exclusion.
- The released `Level` target is a BRAR-derived age-dependent grade, not an independent clinical periodontitis-stage diagnosis.
- The image-only models are frozen ImageNet feature extractors plus logistic regression. They are reproducible benchmark floors, not a ceiling for fine-tuned dental-radiograph models.
- Macro-F1 is the primary three-class manuscript metric. Other paired metrics and subgroup rows are secondary or exploratory.

## Primary Table

| model_label | kind | image_count | cv_macro_f1_mean | oof_macro_f1 | oof_macro_f1_low | oof_macro_f1_high | cv_balanced_accuracy_mean | oof_severe_balanced_accuracy | oof_severe_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tile EfficientNet-B0 | image_only | 988 | 0.5329 | 0.5481 | 0.5136 | 0.5832 | 0.5399 | 0.6371 | 0.7099 |
| Whole-image EfficientNet-B0 | image_only | 988 | 0.4928 | 0.5043 | 0.4694 | 0.5376 | 0.5037 | 0.6395 | 0.7032 |
| Whole-image ResNet50 | image_only | 988 | 0.4813 | 0.5035 | 0.4699 | 0.5377 | 0.4798 | 0.6336 | 0.6630 |
| Age/sex guardrail | metadata_guardrail | 988 | 0.4850 | 0.4838 | 0.4534 | 0.5129 | 0.5445 | 0.5746 | 0.5836 |
| Image plus age/sex | metadata_sensitivity | 988 | 0.4981 | 0.5151 | 0.4806 | 0.5486 | 0.5082 | 0.6432 | 0.7041 |
| Image plus downstream upper bound | upper_bound | 988 | 0.4942 | 0.5096 | 0.4755 | 0.5426 | 0.5047 | 0.6471 | 0.6996 |
| Downstream plus age/sex upper bound | upper_bound | 988 | 0.5063 | 0.5161 | 0.4851 | 0.5450 | 0.5681 | 0.5790 | 0.6367 |
| Majority class | baseline | 988 | 0.2412 | 0.2412 | 0.2412 | 0.2412 | 0.3333 | 0.5000 | 0.4972 |

Key reference points:

- Tile EfficientNet-B0 CV macro-F1 `0.5329` and balanced accuracy `0.5399`.
- Whole-image EfficientNet-B0 CV macro-F1 `0.4928` and balanced accuracy `0.5037`.
- Age/sex guardrail CV macro-F1 `0.4850` and balanced accuracy `0.5445`, which is numerically slightly higher than the tile model's CV balanced accuracy.
- Tile severe-grade CV balanced accuracy `0.6309` and severe AUROC `0.6965`.

## Paired Image-Level Intervals

Positive deltas favor the tile model except for ordinal MAE and ECE, where lower is better. These are interval estimates, not p-values.

| left_label | right_label | metric | delta_left_minus_right | delta_low | delta_high | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Tile EfficientNet-B0 | Whole-image EfficientNet-B0 | macro_f1 | 0.0438 | 0.0032 | 0.0839 | left_interval_better |
| Tile EfficientNet-B0 | Whole-image EfficientNet-B0 | balanced_accuracy | 0.0404 | -0.0020 | 0.0818 | overlaps_no_difference |
| Tile EfficientNet-B0 | Whole-image EfficientNet-B0 | quadratic_weighted_kappa | 0.0332 | -0.0344 | 0.0992 | overlaps_no_difference |
| Tile EfficientNet-B0 | Whole-image EfficientNet-B0 | severe_balanced_accuracy | -0.0025 | -0.0444 | 0.0387 | overlaps_no_difference |
| Tile EfficientNet-B0 | Whole-image EfficientNet-B0 | severe_auroc | 0.0067 | -0.0328 | 0.0455 | overlaps_no_difference |
| Tile EfficientNet-B0 | Age/sex guardrail | macro_f1 | 0.0643 | 0.0236 | 0.1035 | left_interval_better |
| Tile EfficientNet-B0 | Age/sex guardrail | balanced_accuracy | 0.0077 | -0.0339 | 0.0479 | overlaps_no_difference |
| Tile EfficientNet-B0 | Age/sex guardrail | quadratic_weighted_kappa | 0.1442 | 0.0731 | 0.2142 | left_interval_better |
| Tile EfficientNet-B0 | Age/sex guardrail | severe_balanced_accuracy | 0.0625 | 0.0212 | 0.1037 | left_interval_better |
| Tile EfficientNet-B0 | Age/sex guardrail | severe_auroc | 0.1263 | 0.0762 | 0.1741 | left_interval_better |
| Tile EfficientNet-B0 | Whole-image ResNet50 | macro_f1 | 0.0445 | 0.0010 | 0.0874 | left_interval_better |
| Tile EfficientNet-B0 | Whole-image ResNet50 | balanced_accuracy | 0.0507 | 0.0073 | 0.0936 | left_interval_better |
| Tile EfficientNet-B0 | Whole-image ResNet50 | quadratic_weighted_kappa | 0.0855 | 0.0123 | 0.1578 | left_interval_better |
| Tile EfficientNet-B0 | Whole-image ResNet50 | severe_balanced_accuracy | 0.0035 | -0.0381 | 0.0439 | overlaps_no_difference |
| Tile EfficientNet-B0 | Whole-image ResNet50 | severe_auroc | 0.0469 | 0.0035 | 0.0895 | left_interval_better |
| Tile EfficientNet-B0 | Image plus age/sex | macro_f1 | 0.0330 | -0.0060 | 0.0725 | overlaps_no_difference |
| Tile EfficientNet-B0 | Image plus age/sex | balanced_accuracy | 0.0295 | -0.0106 | 0.0709 | overlaps_no_difference |
| Tile EfficientNet-B0 | Image plus age/sex | quadratic_weighted_kappa | 0.0167 | -0.0464 | 0.0813 | overlaps_no_difference |
| Tile EfficientNet-B0 | Image plus age/sex | severe_balanced_accuracy | -0.0061 | -0.0473 | 0.0347 | overlaps_no_difference |
| Tile EfficientNet-B0 | Image plus age/sex | severe_auroc | 0.0058 | -0.0326 | 0.0431 | overlaps_no_difference |

## Subgroup Checks

Subgroup rows use image-level out-of-fold probabilities and skip cells with fewer than the configured minimum sample size.

| model_label | subgroup_type | subgroup | n | macro_f1 | balanced_accuracy | severe_balanced_accuracy | severe_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Age/sex guardrail | age_band | age_35_50 | 311 | 0.3539 | 0.4185 | 0.5131 | 0.5026 |
| Whole-image EfficientNet-B0 | age_band | age_35_50 | 311 | 0.4720 | 0.5169 | 0.6519 | 0.7437 |
| Tile EfficientNet-B0 | age_band | age_35_50 | 311 | 0.4878 | 0.4974 | 0.6772 | 0.7557 |
| Age/sex guardrail | age_band | age_<35 | 427 | 0.3562 | 0.4342 | 0.5000 | 0.5750 |
| Whole-image EfficientNet-B0 | age_band | age_<35 | 427 | 0.4864 | 0.4823 | 0.6382 | 0.6806 |
| Tile EfficientNet-B0 | age_band | age_<35 | 427 | 0.5010 | 0.5002 | 0.6100 | 0.6739 |
| Age/sex guardrail | age_band | age_>50 | 250 | 0.4497 | 0.4671 | 0.6863 | 0.7107 |
| Whole-image EfficientNet-B0 | age_band | age_>50 | 250 | 0.3640 | 0.3661 | 0.6000 | 0.6561 |
| Tile EfficientNet-B0 | age_band | age_>50 | 250 | 0.4098 | 0.4140 | 0.6059 | 0.7007 |
| Age/sex guardrail | aspect_ratio_tertile | aspect_high | 228 | 0.5166 | 0.5855 | 0.5707 | 0.6449 |
| Whole-image EfficientNet-B0 | aspect_ratio_tertile | aspect_high | 228 | 0.5045 | 0.5049 | 0.6588 | 0.7019 |
| Tile EfficientNet-B0 | aspect_ratio_tertile | aspect_high | 228 | 0.5589 | 0.5584 | 0.6402 | 0.7383 |
| Age/sex guardrail | aspect_ratio_tertile | aspect_low | 333 | 0.4759 | 0.5321 | 0.5425 | 0.5241 |
| Whole-image EfficientNet-B0 | aspect_ratio_tertile | aspect_low | 333 | 0.5021 | 0.5043 | 0.6235 | 0.7070 |
| Tile EfficientNet-B0 | aspect_ratio_tertile | aspect_low | 333 | 0.5087 | 0.5069 | 0.6224 | 0.6698 |
| Age/sex guardrail | aspect_ratio_tertile | aspect_mid | 427 | 0.4584 | 0.5351 | 0.6082 | 0.5995 |
| Whole-image EfficientNet-B0 | aspect_ratio_tertile | aspect_mid | 427 | 0.5050 | 0.5170 | 0.6410 | 0.6960 |
| Tile EfficientNet-B0 | aspect_ratio_tertile | aspect_mid | 427 | 0.5740 | 0.5833 | 0.6462 | 0.7267 |
| Age/sex guardrail | file_size_tertile | file_size_large | 329 | 0.4901 | 0.5413 | 0.6045 | 0.5681 |
| Whole-image EfficientNet-B0 | file_size_tertile | file_size_large | 329 | 0.5060 | 0.5076 | 0.6715 | 0.7446 |
| Tile EfficientNet-B0 | file_size_tertile | file_size_large | 329 | 0.5358 | 0.5306 | 0.6558 | 0.7297 |
| Age/sex guardrail | file_size_tertile | file_size_mid | 329 | 0.4698 | 0.5584 | 0.5834 | 0.6452 |
| Whole-image EfficientNet-B0 | file_size_tertile | file_size_mid | 329 | 0.5078 | 0.5278 | 0.6390 | 0.7283 |
| Tile EfficientNet-B0 | file_size_tertile | file_size_mid | 329 | 0.5346 | 0.5460 | 0.6678 | 0.7495 |
| Age/sex guardrail | file_size_tertile | file_size_small | 330 | 0.4834 | 0.5411 | 0.5434 | 0.5427 |
| Whole-image EfficientNet-B0 | file_size_tertile | file_size_small | 330 | 0.4966 | 0.4956 | 0.6020 | 0.6440 |
| Tile EfficientNet-B0 | file_size_tertile | file_size_small | 330 | 0.5732 | 0.5854 | 0.5942 | 0.6591 |
| Age/sex guardrail | gender | gender_0 | 503 | 0.3704 | 0.5093 | 0.6369 | 0.6811 |
| Whole-image EfficientNet-B0 | gender | gender_0 | 503 | 0.5431 | 0.5466 | 0.6939 | 0.7824 |
| Tile EfficientNet-B0 | gender | gender_0 | 503 | 0.5849 | 0.5857 | 0.6973 | 0.7912 |

## Tile Error Audit

Tile-model error rows: `698`.

Contact sheets:

- Top confident errors: `reports/error_audit_contact_sheets/top_confident_tile_errors.png`
- Severe false negatives: `reports/error_audit_contact_sheets/severe_false_negatives.png`
- Two-grade errors: `reports/error_audit_contact_sheets/two_grade_errors.png`

Top error cases:

| file_name | true_label_mode | wrong_predictions | most_common_wrong_pred | mean_wrong_confidence | severe_false_negative_count | two_grade_error_count | age | gender | number_of_missing_teeth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| patient_image_000509_04857e03.jpg | 3 | 3 | 1 | 0.6438 | 3 | 3 | 19 | 0 | 0 |
| patient_image_000607_35725f57.jpg | 3 | 3 | 1 | 0.6209 | 3 | 3 | 28 | 0 | 0 |
| patient_image_000071_7ff3fd0c.jpg | 3 | 3 | 1 | 0.5655 | 3 | 3 | 24 | 1 | 0 |
| patient_image_000002_4bb31c71.jpg | 3 | 3 | 1 | 0.5578 | 3 | 3 | 18 | 1 | 4 |
| patient_image_000013_306b0a74.jpg | 3 | 3 | 1 | 0.5158 | 3 | 3 | 20 | 1 | 0 |
| patient_image_000549_9432a4fa.jpg | 3 | 3 | 1 | 0.4858 | 3 | 3 | 24 | 0 | 0 |
| patient_image_000147_0f29426c.jpg | 3 | 3 | 1 | 0.4212 | 3 | 3 | 28 | 1 | 0 |
| patient_image_000660_d4b75c89.jpg | 3 | 3 | 1 | 0.4019 | 3 | 3 | 32 | 0 | 0 |
| patient_image_000813_ad646fe1.jpg | 3 | 3 | 1 | 0.5141 | 3 | 2 | 43 | 0 | 0 |
| patient_image_000642_c3ea74ba.jpg | 3 | 3 | 1 | 0.4008 | 3 | 2 | 31 | 0 | 1 |
| patient_image_000134_1642341c.jpg | 3 | 3 | 2 | 0.5837 | 3 | 1 | 27 | 1 | 0 |
| patient_image_000191_b8777160.jpg | 3 | 3 | 2 | 0.5696 | 3 | 1 | 30 | 1 | 0 |
| patient_image_000053_78f8d04f.jpg | 3 | 3 | 2 | 0.5694 | 3 | 1 | 23 | 1 | 0 |
| patient_image_000156_f850dcfc.jpg | 3 | 3 | 1 | 0.5105 | 2 | 3 | 28 | 1 | 0 |
| patient_image_000714_2a0bebba.jpg | 3 | 3 | 2 | 0.5087 | 3 | 1 | 35 | 0 | 1 |

## Method Notes

- Bootstrap iterations: `5000`.
- Confidence level: `0.95`.
- Temperature scaling is fitted on validation folds only.
- Severe-grade thresholds are fitted on validation folds only.
- Metadata and downstream-status models remain guardrails or sensitivity analyses, not primary deployable models.
- Image-level out-of-fold estimates average each image's three repeated test appearances, so they are repeat-aware benchmark summaries rather than single-shot deployment estimates.
- Primary three-class manuscript metric: macro-F1.
- Secondary/exploratory metrics: balanced accuracy, QWK, ordinal MAE, ECE, severe balanced accuracy, severe AUROC, paired deltas beyond the primary comparison, and subgroup rows.
