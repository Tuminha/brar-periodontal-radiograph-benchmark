# Image Baseline Summary

Date: 2026-06-07

## Test Aggregate Metrics

| Mode | Macro-F1 | Balanced accuracy | Accuracy | Log loss | ECE | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.4850 | 0.5368 | 0.4932 | 0.9719 | 0.0799 | 0.7015 |
| temperature_scaled | 0.4850 | 0.5368 | 0.4932 | 0.9707 | 0.0655 | 0.7015 |

## Interpretation

Compare image-only macro-F1 against the age/sex negative-control baseline (`0.4850`) and downstream-plus-age/sex upper-bound (`0.5063`). The image-only model is compelling only if it meaningfully exceeds age/sex without relying on metadata.
