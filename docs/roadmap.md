# Flagship BCD model — training roadmap (three stages)

This document outlines a **three-stage** schedule for the flagship **binary change detection (BCD)** backbone: high-volume pretraining on JL1 at 256px, then **joint** multi-dataset training at **512px** with a **deterministic tiling** rule, followed by a final Stage 3 run on `datasets/cropland_bcd_collections/input_quick` (merged former Stage 3 + Stage 4 behavior).

Stage 2 pools **CLCD**, **HRSCD**, **FPCD**, and **Hi-CNA** (as each is available in-repo); native chip sizes differ, so each scene is sliced into **512×512** crops following the same rules.

### At a glance

| | Stage 1 | Stage 2 | Stage 3 |
|---|---------|---------|---------|
| Resolution | 256×256 | 512×512 | 512×512 |
| Dataset source | JL1 | CLCD, HRSCD, FPCD, Hi-CNA | HZ (`input_quick`) |
| Patch number | ~6,000 | 126,573 | ~460 |
| BS | 32 | 8 | 8 |
| Steps | 40,000 | 50,000 | 12,000 |

Patch counts are approximate: Stage 2 total matches the [joint pool](#stage-2-patch-count-512512) row; Stage 3 depends on the `input_quick` manifest. Batch sizes match `scripts/train_flagship_bcd_multistage.sh` defaults.

### Native geometry (public cropland change datasets)

Values below match common paper summaries (GSD, chip size at release, pair count). **Stage 1** trains at native JL1 patch size (256px). **Stage 2** standardizes all listed sources to **512×512** using the [cropping rules](#stage-2-cropping-rules-full-coverage-minimal-overlap) below (**no padding** to fake extra area; **no random** crops).

| Dataset | Ground resolution (m) | Native image size (px) | Image pairs | Bands | Semantic labels |
|---------|----------------------|-------------------------|-------------|-------|-----------------|
| CLCD (Liu et al., 2022) | 0.5–2 | 512×512 | 600 | RGB | No |
| HRSCD (Daudt et al., 2019) | 0.5 | 10,000×10,000 | 291 | RGB | Yes |
| FPCD (Tundia et al., 2023) | 1 | 1,024×768 | 694 | RGB | Yes |
| Hi-CNA (proposed) | 0.8 | 512×512 | 6,797 | RGB + NIR | Yes |

---

## Stage 1 — JL1 (pretrain)

| Item | Setting |
|------|---------|
| Data | Full JL1 **training** split (~6k samples; adjust if your reformatted layout differs) |
| GPU | **NVIDIA RTX 4090 (24GB)** — flagship config uses this as the reference machine for Stage 1 |
| Input size | 256×256 |
| Augmentation | **No** random crop (full-patch training) |
| Steps | 40,000 |
| Batch size | **8** (`configs/flagship_bcd_stage1_jl1_vmamba_base.yaml`) |

**Goal:** Learn robust BCD cues on a large, fixed-crop cropland competition–style set before scaling resolution.

---

## Stage 2 cropping rules (full coverage, minimal overlap)

Patch size **P = 512**. For each image (and its **aligned** change / semantic label), use the **same** crop boxes for all channels.

1. **Use every pixel.** The union of all crops from one pair must cover the full **W × H** canvas. Do not drop edge strips; do not pad with synthetic pixels.
2. **Minimal overlap per axis.** Along one axis of length **L** (width or height):
   - Let **n = ⌈L / P⌉** (number of **P**-wide windows needed to cover **L**).
   - If **n = 1**, use a single window starting at **0** (only valid if **L = P**; if **L < P**, upsampling or a different stage-2 policy is required—out of scope here).
   - If **n ≥ 2**, place **n** windows of length **P** so the first starts at **0**, the last ends at **L**, and overlap is as small as possible: use **equally spaced** top-left coordinates from **0** to **L − P** inclusive (**n** positions). Concretely, index **i ∈ {0,…,n−1}**:
     - **t_i = round(i · (L − P) / (n − 1))**, with **t_0 = 0** and **t_{n−1} = L − P** fixed after rounding so the last window is exactly **[L−P, L)**.
   - Equivalent recipe for code: e.g. **`numpy.linspace(0, L - P, n, dtype=int)`** then set the last value to **`L - P`** if rounding drifted.
3. **2D grid.** Apply the 1D rule independently for **x** and **y**. The crops are the **Cartesian product** of the x-starts and y-starts, each window **P×P**.
4. **No randomness.** Same filenames always yield the same crop indices (deterministic training).

**Already P×P (CLCD, Hi-CNA):** **n_x = n_y = 1** → exactly **one** crop per pair; **no** overlap.

**FPCD (1,024 × 768):** **n_x = 2**, **n_y = 2** → **four** crops. Width: starts **0, 512** (no overlap). Height: **L = 768**, **n = 2** → starts **0** and **768 − 512 = 256**; the two rows overlap by **256** pixels in **y ∈ [256, 512)**. Crop top-left corners **(0,0), (512,0), (0,256), (512,256)**.

**HRSCD (10,000 × 10,000):** **n_x = n_y = ⌈10,000 / 512⌉ = 20**; spacing along each axis uses **(10,000 − 512) / 19** between consecutive starts (~**499.37** px). Overlap per axis is the minimum needed so all **10,000** rows/columns are covered.

### Stage 2 patch count (512×512)

Per pair, the number of training patches is **n_x · n_y** with **n_x = ⌈W / 512⌉**, **n_y = ⌈H / 512⌉** (same as in the tiling rules). Multiply by the number of **image pairs** you actually include from each dataset (below uses the **pair counts** from the [native geometry](#native-geometry-public-cropland-change-datasets) table; your on-disk splits may differ).

| Dataset | Native **W × H** | **n_x** | **n_y** | Patches **per pair** | Pairs (table) | **Patches (total)** |
|---------|------------------|--------|--------|----------------------|---------------|---------------------|
| CLCD | 512 × 512 | 1 | 1 | **1** | 600 | **600** |
| Hi-CNA | 512 × 512 | 1 | 1 | **1** | 6,797 | **6,797** |
| FPCD | 1,024 × 768 | ⌈1,024/512⌉ = **2** | ⌈768/512⌉ = **2** | **4** | 694 | **2,776** |
| HRSCD | 10,000 × 10,000 | ⌈10,000/512⌉ = **20** | **20** | **400** | 291 | **116,400** |
| **Joint pool (all four)** | | | | | **8,382** pairs | **126,573** |

**Check:** 600 + 6,797 + 2,776 + 116,400 = **126,573** patches.

If you **exclude** test (or any split) from training, recompute the **Pairs** column from your file lists and scale the totals accordingly.

---

## Stage 2 — Multi-dataset joint fine-tune (512px)

| Item | Setting |
|------|---------|
| Data | **CLCD** + **HRSCD** + **FPCD** + **Hi-CNA**, trained **together** (mixed batches or interleaved sampling—implement in the dataloader). Include CLCD **train / val / test** as in the earlier plan unless you reserve splits for evaluation. HRSCD / FPCD / Hi-CNA: wire paths and splits when available. |
| Preprocess | **512×512** crops per [Stage 2 cropping rules](#stage-2-cropping-rules-full-coverage-minimal-overlap) (full canvas coverage; minimal overlap when **W** or **H** is not a multiple of 512). |
| Input size | **512×512** |
| Augmentation | **No** random crop (fixed grid only) |
| Steps | **50,000** |
| Batch size | **2** |

**Per-source native geometry (before 512 tiling):** see the table in [Native geometry](#native-geometry-public-cropland-change-datasets).

**Hi-CNA:** Native **RGB + NIR**; decide whether Stage 2 uses **4-channel** inputs or drops NIR to match a 3-channel backbone.

**Goal:** Adapt Stage 1 weights to 512px while mixing cropland change domains under a single, reproducible tiling rule.

*Note:* For unbiased benchmarks, hold out designated test splits and exclude them from this joint pool, and document the exception.

---

## Stage 3 — Input-quick refinement

| Item | Setting |
|------|---------|
| Data | `datasets/cropland_bcd_collections/input_quick` |
| Purpose | Final targeted adaptation on full input_quick train+val after Stage 2 |
| Init checkpoint | Best (or final) checkpoint from Stage 2 |
| Preprocess | Same as Stage 2: **512×512** crops per [Stage 2 cropping rules](#stage-2-cropping-rules-full-coverage-minimal-overlap) |
| Input size | **512×512** |
| Eval list | Full input_quick train+val manifest (intentional overlap-aware monitoring) |
| Steps / batch | **12,000 / 2** (merged former Stage 3 + Stage 4) |

**Goal:** Single final-domain pass tailored to the `input_quick` distribution while using all available input_quick train+val samples.

---

## Practical notes

- **Checkpointing:** Each stage should start from the best (or final) checkpoint of the previous stage unless an ablation calls for a fresh init.
- **Configs:** When YAML configs exist per stage, link or name them here (e.g. JL1 256 / multi-dataset 512 joint) so runs are reproducible.
- **JL1 sample count:** `docs/dataset.md` lists 4,050 train patches for `JL1_second`; if your pipeline uses **train ∪ val** (~6k) or a `cropland_bcd_collections` export, keep the table above in sync with the actual dataloader.

This roadmap is a living document: replace remaining **TBD** paths and sampling weights as HRSCD, FPCD, and Hi-CNA are added to `make_data_loader` and training scripts.
