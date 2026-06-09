# Image Baseline Summary

Date: 2026-06-07

## Test Aggregate Metrics

| Mode | Macro-F1 | Balanced accuracy | Accuracy | Log loss | ECE | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.4813 | 0.4798 | 0.5435 | 1.3375 | 0.2037 | 0.6683 |
| temperature_scaled | 0.4813 | 0.4798 | 0.5435 | 0.9573 | 0.0766 | 0.6707 |

## Interpretation

Compare image-only macro-F1 against the age/sex negative-control baseline (`0.4850`) and downstream-plus-age/sex upper-bound (`0.5063`). The image-only model is compelling only if it meaningfully exceeds age/sex without relying on metadata.
