# GO/NO-GO Decision

Date: 2026-06-04

## Decision

**GO - revise scope.**

Proceed with BRAR as the first new article candidate, but frame it as a public, image-level, leakage-aware and calibrated benchmark. Do not frame sprint 2 as a tooth-level multimodal benchmark unless the missing tooth-level annotation tables are obtained.

## Why This Is Feasible

- The Figshare full ZIP downloaded successfully from file id `58062268`.
- MD5 verification passed: `4df0368a88f23f403958e6b371057f11`.
- The extracted public release contains 988 readable JPG panoramic radiographs.
- `meta_data.csv` contains 988 rows and each row links to exactly one JPG image.
- Folder labels (`level_1`, `level_2`, `level_3`) match the CSV `Level` target for every image.
- Grade counts are imbalanced but usable: Level 1 = 170, Level 2 = 560, Level 3 = 258.
- The literature screen found recent model-development papers, but a clear gap remains for public-data reproducibility, leakage controls, calibration, macro metrics, and disciplined claims.

## Scope Revision Required

The Scientific Data article and codebook describe a richer multimodal release with patient-level and tooth-level annotations for 1,104 patients. The inspected public ZIP contains:

- one patient/image-level metadata file: `data/extracted/meta_data.csv`;
- 988 panoramic JPG images;
- no separate `patient_id` column;
- no tooth-level annotation table;
- no explicit CEJ/apex landmark table;
- no released `bl_mm`, `rl_mm`, `bl_rl_ratio`, `max_bl_rl_ratio`, or tooth-level `brar` columns.

The image filename should therefore be treated as the anonymized linkage key for sprint 2. Patient-level and image-level splitting are equivalent only because every released filename is unique and maps to one metadata row.

## Approved Primary Article Direction

**Working title:** "A leakage-aware, calibrated baseline benchmark for BRAR periodontal severity grading on public panoramic radiographs."

**Primary task:** predict released BRAR `Level` severity grade from panoramic radiograph pixels.

**Primary model:** image-only baseline.

**Primary split:** patient/image-level stratified split using unique filenames, with a saved split manifest before augmentation or preprocessing.

**Primary metrics:**

- macro-F1;
- balanced accuracy;
- per-class sensitivity/specificity or recall/precision;
- AUROC if implemented one-vs-rest with appropriate caveats;
- calibration metrics such as ECE and Brier score;
- confusion matrix and reliability plot;
- confidence intervals by bootstrap on the held-out test set.

## Predictor Policy

Forbidden in any model predicting `Level`:

- `Bone resorption`;
- `Bone resorption Age`;
- `bl_mm`;
- `rl_mm`;
- `bl_rl_ratio`;
- `max_bl_rl_ratio`;
- `brar`;
- any direct transformation of those fields.

Allowed only as labelled sensitivity/subgroup analyses:

- `Age`;
- `Gender`.

Allowed only as non-deployment-ready upper-bound/downstream sensitivity:

- `Number of missing teeth` / `missing_tooth`;
- `Implant` / `implant`;
- `Residual root` / `residual_root`;
- `Functional tooth logarithm` / `functional_pair`.

## Minimum Sprint 2 Analysis Plan

1. Create immutable split manifests with filename, target `Level`, and split.
2. Implement image-only baseline with no tabular metadata.
3. Use class-balanced loss or sampler, and report macro metrics.
4. Add calibration evaluation and post-hoc calibration on validation set only.
5. Run subgroup checks by age band and gender, clearly labelled exploratory.
6. Run sensitivity models:
   - age/sex metadata baseline;
   - image plus age/sex;
   - downstream tooth-status upper-bound model, explicitly non-deployment-ready.
7. Write the paper around transparency, leakage prevention, calibration, and public reproducibility, not clinical deployment.

## Stop Conditions

Switch away from BRAR, likely to the NHANES missingness-as-signal project, if any of these happen:

- full-text literature review finds an already published BRAR benchmark with public split manifests, calibration, leakage audit, and the same target;
- image quality or labels prove unreliable after deeper visual/manual audit;
- journal scope requires external validation that cannot be supported with BRAR alone;
- the article cannot be framed honestly without tooth-level annotations.

## DenPAR Decision

DenPAR stays out of sprint 1 and should not be merged into the BRAR work yet. Revisit it only after the BRAR baseline is built and only if DenPAR supports a meaningful external or modality-shift analysis. It is a periapical radiograph dataset, so it is not a simple external validation set for panoramic BRAR grading.

## Acceptance Status

- ZIP checksum verified: PASS.
- Expected public archive files present: PASS for ZIP, codebook, and protocol from Figshare; FAIL/limited for tooth-level tables in the full ZIP.
- Image-to-metadata linkage: PASS.
- Unique released linkage keys: PASS for filenames; separate `patient_id` absent.
- Sample image readability: PASS.
- Primary predictor leakage audit: PASS.
- Literature gap table with DOI/PMID/source URL identifiers: PASS.

## Final Recommendation

Start sprint 2 on BRAR. The first publishable target should be a careful methods-and-benchmark article, not a high-performance model paper. The value is in making the public BRAR release usable, reproducible, leakage-aware, calibrated, and honestly bounded.
