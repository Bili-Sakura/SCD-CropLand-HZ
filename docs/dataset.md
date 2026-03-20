# JL1-dataset

> https://www.jl1mall.com/contest/match/info?id=1645664411716952066
> https://www.jl1mall.com/resrepo/

We use the same split in paper 'Cross-Difference Semantic Consistency Network for Semantic Change Detection'.

## JL1_second (ChangeMamba SCD format)

The `JL1_second` dataset is reformatted for ChangeMamba semantic change detection.

### Structure

```
datasets/JL1_second/
├── train/
│   ├── T1/          # Pre-change images (00001.png, ...)
│   ├── T2/          # Post-change images
│   ├── GT_T1/       # Land-cover map T1 (single-channel)
│   ├── GT_T2/       # Land-cover map T2
│   └── GT_CD/       # Binary change map (0/255)
├── val/
│   └── ...          # Same structure as train (incl. GT_*)
├── test/
│   ├── T1/          # Hold-out images only (no GT)
│   └── T2/
├── train.txt        # Sample names (no extension)
├── val.txt          # Sample names (with .png)
└── test.txt
```

### Split semantics (important)

- **Basenames are not aligned across splits.** The same stem (e.g. `00001`) in `train/T1/` vs `val/T1/` or `test/T1/` refers to **different images**. Treat the full path under each split as the sample ID.
- **`val/` vs `test/`:** Validation uses the labeled subset (same folder layout as train). **`test/` is the competition hold-out** from `jl1_cropland_competition_2023/test` (T1/T2 only, **no** labels). Do not expect `val` and `test` metrics to be comparable sample-for-sample.

See also `datasets/JL1_second/README.md`.

### Statistics

| Split | Samples | Image size | Disk (approx) |
|-------|---------|------------|---------------|
| train | 4,050   | 256×256×3  | ~390 MB (T1)  |
| val   | 1,950   | 256×256×3  | ~377 MB       |
| test  | 2,000   | 256×256×3  | T1+T2 only (~2× volume of val T1) |
| **Total** | **8,000** | | *(see above)* |

### Semantic classes (GT_T1 / GT_T2)

| Index | Class         | Color (R,G,B) |
|-------|---------------|----------------|
| 0     | background    | (0, 0, 0)      |
| 1     | cropland      | (255, 255, 0)  |
| 2     | road          | (128, 128, 128)|
| 3     | forest-grass  | (0, 180, 0)    |
| 4     | building      | (255, 0, 0)    |
| 5     | other         | (0, 0, 255)    |

**GT_CD:** 0 = no change, 255 = change

### Pixel distribution (train split)

| Class | GT_T1 (%) | GT_T2 (%) |
|-------|-----------|-----------|
| 0 background   | 72.61 | 72.61 |
| 1 cropland     | 13.26 | 14.13 |
| 2 road         |  7.36 |  4.40 |
| 3 forest-grass |  2.14 |  0.01 |
| 4 building     |  0.41 |  1.39 |
| 5 other        |  4.22 |  7.46 |

Change pixels make up ~27.4% of all pixels; background (no observed cropland-change) is ~72.6%.

### Label source

Ground-truth semantic maps are derived from the **jl1_cropland_competition_2023** change labels (0-8)
per `datasets/jl1_cropland_competition_2023/readme.txt`:

| Change label | GT_T1 class | GT_T2 class |
|:---:|:---:|:---:|
| 0 | background | background |
| 1 | cropland | road |
| 2 | cropland | forest-grass |
| 3 | cropland | building |
| 4 | cropland | other |
| 5 | road | cropland |
| 6 | forest-grass | cropland |
| 7 | building | cropland |
| 8 | other | cropland |

### Rebuild / Reformat

```bash
# Rebuild GT_T1/GT_T2/GT_CD from competition change labels
python scripts/rebuild_gt_from_change_labels.py

# Original reformatting from raw JL1_second (images only, labels obsolete)
python scripts/reformat_jl1_second_to_scd.py --root datasets/JL1_second

# Competition hold-out test → datasets/JL1_second/test (T1/T2 PNG, no GT)
python scripts/build_jl1_second_holdout_test.py
```

```bibtex
@article{wangCrossDifferenceSemanticConsistency2024,
  title = {Cross-{{Difference Semantic Consistency Network}} for {{Semantic Change Detection}}},
  author = {Wang, Qi and Jing, Wei and Chi, Kaichen and Yuan, Yuan},
  year = 2024,
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume = {62},
  pages = {1--12},
  issn = {1558-0644},
  doi = {10.1109/TGRS.2024.3386334},
  keywords = {Cross-difference,Data mining,Data models,deep learning,Feature extraction,remote sensing image,Self-supervised learning,semantic change detection (SCD),semantic consistency,Semantics,Solid modeling,Task analysis}
}
```
