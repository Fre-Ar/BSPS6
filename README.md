# BSPS6 — Setup and Usage

End-to-end instructions for setting up the project from a fresh machine and
running experiments. Tested on macOS (Apple Silicon, MPS) and Windows with an NVIDIA GPU (CUDA).

The benchmark itself, the design rationale, and the analyses are described
in [`docs/preregistration.md`](docs/preregistration.md). This file is the
operational how-to.

---

## 1. Get the code

```sh
git clone https://github.com/Fre-Ar/BSPS6.git
cd BSPS6
```

---

## 2. Python environment

Python **3.10+** is required.

### 2a. Create a virtual environment

**macOS / Linux:**

```sh
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2b. Install PyTorch with the right device backend

PyTorch needs the build that matches your hardware.

**macOS (Apple Silicon / MPS) or CPU-only:**

```sh
pip install torch
```

**Windows with NVIDIA GPU (CUDA 12.1):**

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu132
```

For other CUDA versions, see the wheel selector at <https://pytorch.org/get-started/locally/>.

### 2c. Install the rest of the dependencies

```sh
pip install -r requirements.txt
```


This is everything you need to train, evaluate, and analyze runs from the
already-preprocessed `*.nc` files.

If you also intend to **regenerate the preprocessed NetCDFs from raw
sources** (§3 + §4), install the preprocessing-only deps on top:

```sh
pip install -r requirements-preprocess.txt
```

**Windows caveat:** `healpy` (needed only for the Planck CMB preprocessor)
has no PyPI wheels for Windows because it wraps the HEALPix C++ library
and CFITSIO. `pip install -r requirements-preprocess.txt` will fail on
Windows trying to build healpy from source. Two ways around it:

1. **Conda-forge install (recommended if you want to preprocess on
   Windows):** create the env with conda, get healpy from conda-forge,
   then pip-install everything else:

   ```sh
   conda install -c conda-forge healpy
   pip install -r requirements.txt
   pip install opencv-python    # the only other preprocess-only dep
   ```

2. **Preprocess elsewhere, copy the NetCDFs:** run §3 + §4 on a macOS or
   Linux machine, then copy `src/datasets/files/*_512x1024.nc` and
   `src/datasets/files/*_511x1023_held_out.nc` to your Windows box.
   Training, tests, and analysis on Windows then only need the core
   `requirements.txt` — no healpy install required.


Verify your install picked up the right device:

```sh
PYTHONPATH=src python -c "from utils.device import device_display_name; print(device_display_name())"
```

Expected output: `cuda (<GPU name>)` on Windows with NVIDIA, or
`mps (Apple Silicon / Metal)` on Mac, or `cpu` otherwise.

---

## 3. Raw datasets

The benchmark uses five spherical signals. Each comes from a different
public source; the preprocessor turns each one into a standardized 512×1024
NetCDF (training file) and a 511×1023 NetCDF (held-out file) under
`src/datasets/files/`.

**Total raw-data download is roughly 2.5–3 GB.**

Create the directory if it isn't there yet:

```sh
mkdir -p src/datasets/files
```

### 3a. ETOPO1 (NOAA elevation)

Download the ice-surface .grd file from the NOAA archive:

* <https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/netcdf/ETOPO1_Ice_g_gmt4.grd.gz>

Decompress and place at `src/datasets/files/ETOPO1_Ice_g_gmt4.grd`.

### 3b. ERA5 (ECMWF 2m temperature)

The pipeline ships a snapshot of ERA5 2m temperature for 2023-06-15 12:00
UTC. The simplest path is to download it via the Copernicus CDS:

1. Register a free account at <https://cds.climate.copernicus.eu/> and
   accept the ERA5 license.
2. Install the CDS API key (`~/.cdsapirc` on Linux/Mac,
   `%USERPROFILE%\.cdsapirc` on Windows).
3. Use the CDS web UI to download "ERA5 hourly data on single levels" for
   `2m_temperature`, single hour `2023-06-15 12:00`, NetCDF format,
   geographical area "Whole available region". Or `pip install cdsapi`
   and use the Python client (the loader doesn't depend on it; we only
   need the resulting `.nc` file).
   client (already in `requirements.txt`).
4. Save the resulting `.nc` file as
   `src/datasets/files/ERA5_t2m_2023_06_15_1200.nc`.

### 3c. Planck CMB (ESA Planck Legacy Archive)

Download the SMICA full-mission temperature map (HEALPix Nside=2048 FITS,
~2 GB):

* <https://pla.esac.esa.int/pla-sl/data-action?MAP.MAP_ID=COM_CMB_IQU-smica_2048_R3.00_full.fits>

Place at `src/datasets/files/COM_CMB_IQU-smica_2048_R3.00_full.fits`.

### 3d. HDRI (Poly Haven, two equirectangular panoramas)

Download the 2k `.exr` files (free under CC0):

* Sky: <https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky> →
  download "2K EXR" → `src/datasets/files/kloofendal_48d_partly_cloudy_puresky_2k.exr`
* Urban: <https://polyhaven.com/a/shanghai_bund> → download "2K EXR" →
  `src/datasets/files/shanghai_bund_2k.exr`

---

## 4. Preprocess the datasets

Preprocessing needs the extra dependencies from §2c (`requirements-preprocess.txt`).
You only need to run this step on whichever machine generates the `.nc`
files; you can then copy the resulting NetCDFs to any other machine and
skip §4 there entirely.

Run the preprocessor for all five datasets in one shot:

```sh
PYTHONPATH=src python -m datasets.preprocess
```


Each invocation produces **two** NetCDF files per dataset, both written to
`src/datasets/files/`:

* `*_512x1024.nc` — the training grid.
* `*_511x1023_held_out.nc` — the half-pixel-offset grid used for held-out
  PSNR (preregistration §3.4). Both files are sampled from the same base
  source and share the same `target_min` / `target_max` normalization
  reference, so every sub-sample lies within [-1, 1].

The training file is ~5 MB per scalar dataset, ~15 MB per RGB dataset.

---

## 5. Run the tests

The test suite is fast and catches almost all setup mistakes:

```sh
PYTHONPATH=src python tests/test_architectures.py
PYTHONPATH=src python tests/test_run_grid.py
PYTHONPATH=src python tests/test_runs_csv.py
PYTHONPATH=src python tests/test_datasets.py
PYTHONPATH=src python tests/test_analysis.py
PYTHONPATH=src python tests/test_configs_smoke.py
```

All six should print `All ... tests passed.`. `test_datasets.py` and the
per-dataset block of `test_configs_smoke.py` require the preprocessed
NetCDFs from §4 to be present; the others run standalone.

**Always run the smoke test before launching the full grid on a new
machine.** `test_configs_smoke.py` instantiates every (activation × PE)
cell + every SH `L_max` value the launcher will use, runs forward +
5 Adam steps on a tiny batch of coordinates that includes both polar
caps and the longitude seam, and verifies that predictions, losses,
gradients, and post-step parameters are all finite. It also runs one real
end-to-end check per preprocessed dataset. Total wall-clock is ~30–90 s
on CPU (no GPU needed). This catches the silent-NaN class of bug — e.g.,
an activation that divides by zero in a corner of the input space — before
you burn 75 GPU-hours on a grid that produces `nan` PSNRs.

---

## 6. Run the experiments

Per [`docs/preregistration.md`](docs/preregistration.md) §3.5, the locked
grid is 90 runs at 1 seed (75 main + 6 SH post-saturation + 9 SH
pre-saturation).

### 6a. The whole grid

```sh
python src/run_grid.py
```

Resume policy: a cell with `status='completed'` in `results/runs.csv` is
skipped on subsequent invocations. If a run crashes or OOMs, the launcher
records the failure status and you can re-run to retry only the failed
cells.

### 6b. A subset

The launcher supports `--filter_pe`, `--filter_act`, `--filter_dataset`, and a coarser `--grid` switch.

**Pick by PE** (here: only the SH and FKAN cells):

```sh
python src/run_grid.py --filter_pe sh,fkan
```

**Pick by activation:**

```sh
python src/run_grid.py --filter_act relu,scaled_sine
```

**Pick by dataset:**

```sh
python src/run_grid.py --filter_dataset etopo1,cmb
```

**Pick the SH ablation only** (the 6 post-saturation + 9 pre-saturation
runs, no main grid):

```sh
python src/run_grid.py --grid sh_ablation
```

**Combine filters** (set intersection). Run just the SH cells on CMB:

```sh
python src/run_grid.py --filter_pe sh --filter_dataset cmb
```

**Plan the run without executing**:

```sh
python src/run_grid.py --filter_pe sh --filter_dataset cmb --dry_run
```

**`--grid` choices:**

| `--grid` value         | Cells included                                          |
|------------------------|----------------------------------------------------------|
| `main`                 | 75 main-grid cells                                       |
| `sh_post_saturation`   | 6 cells (3 activations × 2 low-bandwidth datasets)       |
| `sh_pre_saturation`    | 9 cells (3 activations × 3 high-bandwidth datasets)      |
| `sh_ablation`          | 15 cells (both SH sub-grids together)                    |
| `all` (default)        | 90 cells (main + sh_ablation)                            |

### 6c. Copying results back from a borrowed machine

After the borrowed-machine run finishes, copy back **two** things to your
laptop:

1. `results/runs.csv` — the per-cell PSNR / parameter-count / timing /
   status table. Append-only on the source machine; you can either replace
   yours wholesale (if your local CSV is empty) or merge by appending the
   new rows.

2. `logs/grid/` — the per-cell TensorBoard logs and best-model checkpoints
   (one `<cell_key>__<dataset>__seedN/ckpt/best_model_*.ckpt` per cell).
   These are needed if you want to re-evaluate or visualize the trained
   models later.

Both are gitignored, so a plain `scp -r` or `rsync` works:

```sh
scp -r friend@host:~/BSPS6/results/runs.csv ./results/
scp -r friend@host:~/BSPS6/logs/grid/         ./logs/
```

---

## 7. Analyze the results

Run all four pre-committed analyses (preregistration §2):

```sh
PYTHONPATH=src python src/analysis/run_analysis.py \
    --runs_csv results/runs.csv \
    --output_dir results/analysis
```

Outputs (under `results/analysis/`):

* `summary.json` — machine-readable summary statistics for each analysis.
* `summary.md` — human-readable top-level summary.
* `tables/variance_decomposition.md` — η² per factor + bootstrap CIs.
* `tables/polar_penalty_contrast.md` — median Δ(angular − cartesian) + CI.
* `tables/characterization_correlations.md` — Spearman ρ per PE × feature.
* `tables/sh_lmax_ablation.md` — median Δ for post- and pre-saturation regimes.
* `figures/*.png` — one figure per analysis (skip with `--skip_figures`).

The analyses are descriptive — every analysis produces a number and a CI,
which gets reported regardless of direction or magnitude.

---

## 8. Where files live

| Path                                              | Contents                                            | Tracked in git? |
|---------------------------------------------------|-----------------------------------------------------|-----------------|
| `src/datasets/files/*_Ice_g_gmt4.grd` etc.        | Raw source files (you download these in §3).        | No              |
| `src/datasets/files/*_512x1024.nc`                | Pre-processed training NetCDFs (§4 output).          | No              |
| `src/datasets/files/*_511x1023_held_out.nc`       | Pre-processed held-out NetCDFs (§4 output).          | No              |
| `results/runs.csv`                                 | One row per training run (PSNR, params, timing).    | No              |
| `results/param_counts.md`                          | `src/tabulate_params.py` output.                | No              |
| `results/analysis/summary.json` / `summary.md`     | Top-level analysis outputs (§7).                    | No              |
| `results/analysis/tables/*.md`                     | Per-analysis tables (§7).                           | No              |
| `results/analysis/figures/*.png`                   | Per-analysis figures (§7).                          | No              |
| `logs/grid/<cell_key>__<dataset>__seedN/`         | Per-cell TensorBoard logs.                          | No              |
| `logs/grid/<cell_key>__<dataset>__seedN/ckpt/`    | Best-checkpoint `*.ckpt` files (one per cell).      | No              |
| `docs/preregistration.md`                          | Pre-committed design + analyses (the contract).     | Yes             |
| `src/`, `tests/`                       | Code.                                                | Yes             |

---

## 9. Cross-platform notes

The codebase auto-detects the device via `src/utils/device.py`:

* CUDA (Windows / Linux with NVIDIA) — pin_memory on, 4 DataLoader workers,
  `torch.set_float32_matmul_precision('high')`, GPU peak-memory tracked in
  the `peak_gpu_mem_mb` column of `runs.csv`.
* MPS (Apple Silicon) — pin_memory off, 0 DataLoader workers (fork overhead
  hurts on macOS), peak-memory column left blank (MPS doesn't expose
  `max_memory_allocated`).
* CPU — same conservative defaults as MPS; suitable for smoke tests, not
  the full grid.

The Lightning Trainer uses the matching `accelerator` automatically. No
per-platform code paths exist in `main.py` or the analysis scripts.

A run launched on macOS and a run launched on Windows will produce
bit-different model weights (different RNG streams / kernel choices) but
the same `runs.csv` schema. You can merge runs.csv rows from both
platforms; the analysis pipeline treats them uniformly.

---

## 10. Troubleshooting

* **A test hangs silently on Windows** (most often `test_datasets.py`
  inside `test_sh_orthonormality_on_gauss_legendre`, or any test that
  follows a `torch` import and then calls `numpy.dot` / `@` / LAPACK) —
  this is a known OpenBLAS thread-pool deadlock in the PyPI NumPy wheels
  for Windows. NumPy's OpenBLAS and PyTorch's bundled OpenMP runtime
  fight over thread pinning. Pin NumPy's BLAS to a single thread before
  launching Python:

  ```powershell
  $env:OPENBLAS_NUM_THREADS = '1'
  $env:MKL_NUM_THREADS      = '1'        # harmless if NumPy isn't MKL-linked
  python tests/test_datasets.py
  ```

  Or switch to a conda-forge NumPy (`conda install -c conda-forge numpy`),
  which ships a differently-configured OpenBLAS. The training pipeline
  itself doesn't seem to trigger this in practice (all matmuls go through
  torch's own BLAS), but the test files use NumPy directly.
* **`pip install -r requirements-preprocess.txt` fails on Windows
  while building healpy** (errors mentioning `cfitsio`, `chealpix`, or
  `Microsoft Visual C++`) — healpy has no Windows pip wheels. Use
  `conda install -c conda-forge healpy` instead, or skip the
  preprocessing step entirely on Windows and copy the preprocessed
  NetCDFs over from a macOS / Linux machine (see §2c).
* **`src/run_grid.py: error: unrecognized arguments: --omega 1024`** —
  your local copy of `src/config/opts.py` is stale relative to
  `src/config/architectures.py`. Re-pull the repo and clear
  `src/config/__pycache__/`.
* **CUDA out of memory on the SH cells (~2.3 GB coord tensor)** —
  reduce `--batch_size` from 8192 to 4096 on the affected cell, or run
  `src/run_grid.py --filter_pe sh --filter_dataset <one_dataset>`
  separately. Document any per-cell mitigation in `runs.csv`'s
  `mitigation_note` column.
* **`held_out_psnr` values look much worse than `reconstruction_psnr`** —
  expected. The held-out grid samples positions the model never saw during
  training; the gap quantifies generalization.
* **Spearman ρ test in §2.3 returns NaN** — n=5 datasets is the absolute
  floor for the test. Check that all 5 main-grid SH cells (for example)
  have a `held_out_psnr` value in `runs.csv`.

For anything else, the test suite is the first stop:
`PYTHONPATH=src python tests/test_<area>.py` will usually pinpoint the
problem.
