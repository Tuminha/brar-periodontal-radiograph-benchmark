# Leakage Audit

Date: 2026-06-04

## Planned Primary Task

Predict BRAR severity `Level` from panoramic radiograph images using patient-level splitting. In the released data, one image appears to correspond to one anonymized patient filename, so patient-level and image-level splitting are equivalent if filenames remain unique. The plan's `severity_grade` target maps to the released `Level` column.

## Variable Policy

| variable | role | primary_model_use | reason |
| --- | --- | --- | --- |
| File name | identifier/linkage | path only | Used only to load the image; not a tabular predictor. |
| Age | metadata sensitivity | exclude from image-only primary model | Age appears in the BRAR-derived grading formula, so use only in labelled sensitivity analyses. |
| Gender | metadata sensitivity | exclude from image-only primary model | Allowed for sensitivity/subgroup analyses, but primary benchmark should be image-only. |
| Bone resorption | outcome-derived | forbidden | Direct bone-loss measure used to derive severity. |
| Bone resorption Age | outcome-derived | forbidden | Age-normalized bone-resorption value used to derive `Level`. |
| Level | outcome | target only | Released severity class to be predicted; treat as the public equivalent of planned `severity_grade`. |
| Number of missing teeth | downstream sensitivity | exclude | Likely consequence/correlate of periodontal history; use only in upper-bound sensitivity. |
| Implant | downstream sensitivity | exclude | Treatment/restoration status can reflect prior disease and access to care. |
| Residual root | downstream sensitivity | exclude | Disease/treatment consequence; not a clean upstream predictor. |
| Functional tooth logarithm | downstream sensitivity | exclude | Functional dentition summary may encode disease consequences. |

## Primary Model Feature Rule

The primary benchmark should use only image pixels and the target `Level`. No tabular metadata should enter the primary model.

## Codebook Variable Policy

The richer codebook/protocol fields are not present in the inspected public ZIP, but they remain important if additional annotation tables are obtained later.

| codebook/planned variable | status in inspected ZIP | model policy |
| --- | --- | --- |
| `bl_mm` | absent | outcome-derived; forbidden as predictor |
| `rl_mm` | absent | outcome-derived; forbidden as predictor |
| `bl_rl_ratio` | absent | outcome-derived; forbidden as predictor |
| `max_bl_rl_ratio` | absent | outcome-derived; forbidden as predictor |
| `brar` | represented by released `Bone resorption Age` | outcome-derived; forbidden as predictor |
| `missing_tooth` | represented by `Number of missing teeth` | downstream sensitivity only |
| `implant` | represented by `Implant` | downstream sensitivity only |
| `residual_root` | represented by `Residual root` | downstream sensitivity only |
| `functional_pair` | represented by `Functional tooth logarithm` | downstream sensitivity only |

## Sensitivity Model Rules

- Age/gender sensitivity: allowed only as explicitly labelled metadata sensitivity analysis.
- Upper-bound downstream sensitivity: `Number of missing teeth`, `Implant`, `Residual root`, and `Functional tooth logarithm` may be used only to estimate how much downstream dental-status information inflates performance.
- Forbidden predictors in any predictive model for `Level`: `Bone resorption`, `Bone resorption Age`, and any direct transformation of them.

## Current Risk Assessment

- Low leakage risk for an image-only benchmark if splits are based on unique filenames and no duplicate/repeated patient images exist.
- Moderate conceptual leakage risk for metadata models because `Age` is part of the BRAR definition and dental-status variables may be downstream of disease.
- Major limitation: released CSV does not include tooth-level rows, `patient_id`, explicit image-quality ratings, CEJ/apex landmarks, `bl_mm`, `rl_mm`, or `bl_rl_ratio` despite the codebook describing those fields.
