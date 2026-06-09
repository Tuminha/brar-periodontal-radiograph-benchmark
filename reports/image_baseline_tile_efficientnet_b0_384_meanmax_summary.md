# Image Baseline Summary

Date: 2026-06-07

## Test Aggregate Metrics

| Mode | Macro-F1 | Balanced accuracy | Accuracy | Log loss | ECE | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.5329 | 0.5399 | 0.5715 | 1.1750 | 0.1986 | 0.7094 |
| temperature_scaled | 0.5329 | 0.5399 | 0.5715 | 0.9038 | 0.0736 | 0.7119 |

## Interpretation

Compare image-only macro-F1 against the age/sex negative-control baseline (`0.4850`) and downstream-plus-age/sex upper-bound (`0.5063`). The image-only model is compelling only if it meaningfully exceeds age/sex without relying on metadata.
