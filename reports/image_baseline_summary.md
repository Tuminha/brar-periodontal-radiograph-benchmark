# Image Baseline Summary

Date: 2026-06-07

## Test Aggregate Metrics

| Mode | Macro-F1 | Balanced accuracy | Accuracy | Log loss | ECE | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.4928 | 0.5037 | 0.5273 | 1.4013 | 0.2144 | 0.6804 |
| temperature_scaled | 0.4928 | 0.5037 | 0.5273 | 0.9630 | 0.0930 | 0.6829 |

## Interpretation

Compare image-only macro-F1 against the age/sex negative-control baseline (`0.4850`) and downstream-plus-age/sex upper-bound (`0.5063`). The image-only model is compelling only if it meaningfully exceeds age/sex without relying on metadata.
