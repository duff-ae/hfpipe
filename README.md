# HF Afterglow & Type-1 Luminosity Correction Pipeline

This repository implements a complete reconstruction chain for HFET (or ...) luminosity data used in BRIL.
It performs:

1. **Afterglow LSQ recovery** (FFT + Conjugate Gradient).  
2. **Type-1 spillover modelling** (polynomial fits).  
3. **Type-1 subtraction**, producing final per-BX luminosities.

The pipeline operates on per-fill HFET HD5 files and produces corrected HD5 outputs and optional debug plots.

---

## Quick Start

### 1. Run the pipeline on one or more fills

```bash
python -m hfcli.run_pipeline --config configs/analysis_25.yaml --fills 10709
```

The `--fills` argument supports:

```
10709
10709 10710 10711
10709-10730
10709,10711-10715
```

If omitted, the pipeline automatically discovers all fills under `io.input_dir`.

---

## Input structure

The pipeline expects HFET HD5 files under:

```
<io.input_dir>/<fill>/<fill>_*.hd5
```

and an active BX mask:

```
activeBXMask_fill<fill>.npy
```

All paths are defined in the YAML config.

---

## YAML Config (minimal example)

```yaml
io:
  input_dir: "/cephfs/brilshare/.../hf_origin/hfet/25"
  input_pattern: "{fill}/{fill}_*.hd5"

  output_dir: "/cephfs/brilshare/.../hf_reprocessed/hfet/25_v2"
  output_pattern: "{fill}/{fill}.hd5"

  node: "hfetlumi"
  active_mask_pattern: "/cephfs/.../activeBXMask_fill{fill}.npy"
  type1_dir: "/eos/.../hf_type_plots"

steps:
  restore_rates: true
  compute_type1: true
  apply_type1: true

type1:
  offsets: [1, 2, 3, 4]
  sbil_min: 0.1
  make_plots: true
  debug: true
  debug_after_apply: true
```

---

## Output

### 1. Corrected HFET HD5

```
<output_dir>/<fill>/<fill>.hd5
```

Contains:

- `bxraw` — fully corrected μ (afterglow + pedestal + Type-1)  
- `bx` — per-BX instantaneous luminosity  
- `avg` — SBIL  
- all HFET metadata columns  

### 2. Type-1 coefficients

```
<type1_dir>/type1_coeffs_fill<fill>.h5
```

### 3. Debug plots (optional)

```
<type1_dir>/<fill>/before/type1_1.png
<type1_dir>/<fill>/after/type1_1.png
```

---

## Methods (short overview)

### Afterglow LSQ Solver
- Inverts the HF afterglow convolution using FFT-accelerated Conjugate Gradient.
- Applies dynamic pedestal subtraction (mod4 pedestals).
- Produces corrected per-BX μ for each luminosity section.

### Type-1 Spillover Correction

For each offset `t`:

```
frac = bxraw[i+offset] / bxraw[i]
```

- Offset 1 → quadratic fit.  
- Offsets ≥2 → linear fits.  

Correction formula:

```
bxraw[j] -= y * (p0 + p1*y + p2*y²)
```

where `y = bxraw[i]`.

---

## Contact

Alexey Shevelev — BRIL / CMS Luminosity
ChatGPT — optimization & debugging
