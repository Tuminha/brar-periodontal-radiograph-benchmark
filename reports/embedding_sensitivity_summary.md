# Embedding Sensitivity Summary

Date: 2026-06-08

## Test Aggregate Metrics

| Feature set | Macro-F1 | Balanced accuracy | Accuracy | Log loss | ECE | AUROC | QWK |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| image_embedding_plus_age_sex | 0.4981 | 0.5082 | 0.5334 | 0.9480 | 0.0801 | 0.6890 | 0.3520 |
| image_embedding_plus_downstream_age_sex_upper_bound | 0.4942 | 0.5047 | 0.5283 | 0.9499 | 0.0743 | 0.6880 | 0.3514 |

## Interpretation

The image-plus-age/sex sensitivity model does not yet provide a strong incremental-value result.

The downstream-status feature set is an upper-bound sensitivity analysis only and should not be described as deployment-ready.
