# Data Linkage Audit

Date: 2026-06-04

## Archive Verification

- Raw ZIP: `data/raw/BRAR-anchored_multimodal_dataset.zip`
- Expected MD5: `4df0368a88f23f403958e6b371057f11`
- Observed MD5: `4df0368a88f23f403958e6b371057f11`
- Checksum status: `PASS`

## Released File Structure

- Extracted image files: 988
- Metadata rows: 988
- Metadata file: `data/extracted/meta_data.csv`
- Top-level extracted folders: `level_1`, `level_2`, `level_3`
- Important discrepancy: the article/codebook describe richer patient/tooth-level variables and 1,104 patients, but the released ZIP inspected here contains 988 images and one patient/image-level metadata CSV.
- Patient-level annotation table: present as `meta_data.csv`, with one row per released image/anonymized patient filename.
- Tooth-level annotation table: not present in the inspected public full ZIP.
- Separate `patient_id` column: not present; the anonymized image filename is the only linkage key in the released CSV.

### File Counts By Extension

| extension | count |
| --- | --- |
| csv | 1 |
| jpg | 988 |
| zip | 1 |

### File Counts By Folder

| folder | count |
| --- | --- |
| data/extracted | 1 |
| data/extracted/level_1 | 170 |
| data/extracted/level_2 | 560 |
| data/extracted/level_3 | 258 |
| data/raw | 1 |

## Metadata Columns

| column | missing | unique | first_values |
| --- | --- | --- | --- |
| File name | 0 | 988 | patient_image_000001_07f48740.jpg, patient_image_000002_4bb31c71.jpg, patient_image_000003_2b2e7506.jpg, patient_image_000004_ba49da4c.jpg, patient_image_000005_89a2810f.jpg, patient_image_000006_4aa37c68.jpg, patient_image_000007_833861b5.jpg, patient_image_000008_fdfc183f.jpg |
| Age | 0 | 58 | 18, 19, 20, 21, 22, 23, 24, 25 |
| Gender | 0 | 2 | 1, 0 |
| Bone resorption | 0 | 748 | 0, 0.223684211, 0.270834291, 0.182891628, 0.217948718, 0.183477706, 0.159300892, 0.141423139 |
| Bone resorption Age | 0 | 831 | 0, 1.242690058, 1.425443638, 0.962587517, 1.147098516, 0.917388532, 0.796504462, 0.707115696 |
| Level | 0 | 3 | 1, 3, 2 |
| Number of missing teeth | 0 | 24 | 0, 4, 1, 3, 2, 8, 5, 6 |
| Implant | 0 | 10 | 0, 1, 3, 2, 4, 6, 5, 12 |
| Residual root | 0 | 10 | 0, 1, 2, 5, 4, 3, 6, 8 |
| Functional tooth logarithm | 0 | 16 | 14, 12, 13, 10, 11, 9, 6, 1 |

## Linkage Checks

- Unique metadata filenames: 988
- Duplicate metadata filenames: 0
- Metadata rows without matching image: 0
- Images without matching metadata row: 0
- Images where folder grade differs from CSV `Level`: 0
- Filename-to-patient linkage: `PASS` for image loading because each anonymized filename maps to one metadata row and one image path. `patient_id` itself is not released as a separate column.

### Grade Counts

| Grade | CSV rows | Image files |
|---|---:|---:|
| 1 | 170 | 170 |
| 2 | 560 | 560 |
| 3 | 258 | 258 |

## Image Readability Sample

`sips` was used to read pixel dimensions from three images per released grade folder.

| relative_path | grade_folder | sips_exit_code | pixel_width | pixel_height |
| --- | --- | --- | --- | --- |
| data/extracted/level_1/patient_image_000001_07f48740.jpg | 1 | 0 | 2976 | 1536 |
| data/extracted/level_1/patient_image_000005_89a2810f.jpg | 1 | 0 | 2976 | 1536 |
| data/extracted/level_1/patient_image_000006_4aa37c68.jpg | 1 | 0 | 2795 | 1316 |
| data/extracted/level_2/patient_image_000004_ba49da4c.jpg | 2 | 0 | 2976 | 1536 |
| data/extracted/level_2/patient_image_000008_fdfc183f.jpg | 2 | 0 | 2796 | 1316 |
| data/extracted/level_2/patient_image_000009_ea22106e.jpg | 2 | 0 | 2786 | 1316 |
| data/extracted/level_3/patient_image_000002_4bb31c71.jpg | 3 | 0 | 2786 | 1316 |
| data/extracted/level_3/patient_image_000003_2b2e7506.jpg | 3 | 0 | 2796 | 1316 |
| data/extracted/level_3/patient_image_000007_833861b5.jpg | 3 | 0 | 2775 | 1480 |

## Linkage Verdict

The released BRAR ZIP is usable for a patient/image-level image-classification benchmark: all 988 metadata rows link to exactly one JPG image, and folder labels match CSV `Level`. It is not usable for the tooth-level analyses described by the richer codebook unless additional unreleased annotation tables are obtained.
