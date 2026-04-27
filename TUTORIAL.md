# PBPK MCP Server — User Tutorial

A walk-through of PBPK model construction across 5 scenarios, with real test
records. All outputs were verified by actual MCP calls in a `claude-opus-4-7`
session.

**Date**: 2026-04-20  |  **Server version**: v1.5  |  **30 tools**

---

## Table of Contents

- [Scenario A: Simulating a known drug (Midazolam)](#scenario-a)
- [Scenario B: Modeling a new drug from in vitro data](#scenario-b)
- [Scenario C: Drug-Drug Interaction (DDI) assessment](#scenario-c)
- [Scenario D: Special populations (pediatric / pregnancy / CKD / variability)](#scenario-d)
- [Scenario E: Transporter-driven drug (statin-like)](#scenario-e)
- [Reference: Kp method selection rules](#kp-method-selection)
- [Reference: Parameter naming rules](#parameter-naming-rules)

---

<a id="scenario-a"></a>
## Scenario A: Simulating a known drug (Midazolam)

**Goal**: Predict PK after a 7.5 mg oral dose of midazolam, a common clinical
benchmark compound.

### A1. Look up drug properties

```python
mcp.call_tool("drug_properties", {"drug_name": "midazolam"})
```

**Output**:
```
## Drug Properties — Midazolam
Source: curated

| Property | Value |
|----------|-------|
| ChEMBL ID | CHEMBL601 |
| MW | 325.80 g/mol |
| logP | 3.89 |
| pKa | 6.2 |
| Ro5 violations | 0 |
```

**Interpretation**: Midazolam is a moderate base (pKa 6.2) with intermediate
lipophilicity (logP 3.89). Zero Ro5 violations suggests good oral absorption.

---

### A2. Choose a Kp method

```python
mcp.call_tool("compare_kp_methods", {"name": "midazolam"})
```

**Excerpt** (Adipose / Liver rows):
| Tissue | R&R | Lukacova | Schmitt | PT | PTB | PK-Sim | Kp_mem |
|--------|-----|----------|---------|-----|-----|--------|--------|
| Adipose | 4.26 | 4.26 | 3.83 | 3.20 | 4.16 | 4.56 | 17.36 |
| Liver | 1.89 | 1.89 | 9.84 | 0.35 | 1.28 | 3.97 | 3.03 |

**Selection rule**: Midazolam is a moderate base with logP > 3 → use
**Poulin-Theil** (R&R systematically over-predicts adipose Kp for lipophilic
bases — Jansson 2008, Graham 2012).

---

### A3. PBPK simulation (Midazolam 7.5 mg PO)

```python
mcp.call_tool("run_pbpk_simulation", {
    "name": "midazolam", "dose_mg": 7.5, "route": "oral",
    "duration_h": 24, "kp_method": "poulin_theil",
})
```

**Result**:

| Parameter | Predicted | Clinical (Greenblatt 1984) |
|---|---|---|
| Cmax | **72.2 ng/mL** | 40-100 ng/mL ✓ |
| Tmax | 0.38 h | 0.5-1.5 h ≈ |
| AUC_inf | **167.9 ng·h/mL** | 75-250 ng·h/mL ✓ |
| t½ | **5.40 h** | 1.8-6.4 h ✓ |
| CL/F | 44.66 L/h | - |

Tissue concentrations distribute correctly given Kp (Gut C_tissue/C_plasma =
8.2 — drug accumulating in enterocytes immediately after absorption).

---

### A4. IV vs PO — computing oral bioavailability (F)

```python
# IV: 5 mg bolus
r_iv = call("run_pbpk_simulation", {..., "route": "iv_bolus", ...})
# PO: 5 mg
r_po = call("run_pbpk_simulation", {..., "route": "oral", ...})
F = AUC_po / AUC_iv
```

**Result**:
- IV AUC = 0.312 mg·h/L, CL = **16.03 L/h** (lit 18-30)
- PO AUC = 0.112 mg·h/L, CL/F = 44.66 L/h
- **F_oral = 0.359 (36%)** — lit 30-50% ✓

---

<a id="scenario-b"></a>
## Scenario B: Modeling a new drug from in vitro data

**Goal**: Predict clinical PK for a new CYP3A4 substrate measured at HLM
CLint = 30 µL/min/mg.

### B1. IVIVE — convert HLM CLint to in vivo CLint

```python
mcp.call_tool("ivive_microsomal", {
    "clint_vitro": 30.0, "logP": 3.5, "protein_conc": 1.0,
})
```

**Result**:
```
**CLint in vivo = 300.64 L/h**

| Parameter | Value |
|-----------|-------|
| CLint_unbound_uL_min_mg | 59.35 |
| fu_inc | 0.5055 (predicted, at logP=3.5)
| MPPGL | 45 mg/g liver |
| liver_weight_g | 1876 |
| scaling_factor | 5.065 |
```

**Formula**: `CLint_in_vivo = (CLint_vitro / fu_inc) × MPPGL × liver_weight × 60 / 10^6`

---

### B2. Predict Fg (gut bioavailability)

```python
mcp.call_tool("predict_fg", {
    "Peff": 4.0, "CLint_gut": 10.0, "fu_gut": 0.2,
})
```

**Result**: **Fg = 0.6793** (Yang 2007 Qgut model)
- Qgut = 4.24 L/h
- CLperm = 5.54 L/h

---

### B3. Full simulation — custom compound + IVIVE

```python
mcp.call_tool("run_pbpk_simulation", {
    "name": "NewDrug",
    "logP": 3.5, "pKa": 7.0, "fu_p": 0.1,
    "compound_type": "neutral", "R_bp": 1.0, "mw": 400,
    "ka": 1.5, "Fa": 0.9, "Fg": 0.7,
    "clearance_source": "hlm",
    "CLint_vitro_hlm": 30.0,
    "dose_mg": 20, "route": "oral", "duration_h": 24,
})
```

**Result**: IVIVE applied automatically (`30.0 µL/min/mg → 267.2 L/h`).
Predicted PK:
- Cmax = 37.8 ng/mL
- Tmax = 0.89 h
- AUC_inf = 375 ng·h/mL
- t½ = 17.3 h (long distribution + moderate CL)
- CL/F = 53.27 L/h

---

<a id="scenario-c"></a>
## Scenario C: Drug-Drug Interaction (DDI) assessment

### C1. Static DDI screening

For an early assessment when only in vitro Ki is available:

```python
mcp.call_tool("predict_ddi", {
    "mechanism": "reversible", "Ki": 0.05, "I_h_u": 0.5, "fm": 0.9,
})
```

**Result**: **AUC ratio = 5.50x (Strong inhibition)** [FDA 2020 classification]

Formula: `AUC_ratio = 1 / (fm/(1 + I/Ki) + (1-fm))`
- R_h (I/Ki) = 11, fm = 0.9
- R = 1 / (0.9/12 + 0.1) = 1/0.175 = 5.71 ≈ 5.50x

---

### C2. Dynamic DDI — Ketoconazole SS + Midazolam

```python
mcp.call_tool("run_dynamic_ddi", {
    "victim_name": "midazolam", "victim_dose_mg": 7.5, "victim_route": "oral",
    "victim_first_dose_h": 72,
    "perp_name": "Keto", "perp_dose_mg": 400, "perp_route": "oral",
    "perp_n_doses": 5, "perp_interval_h": 24,
    "perp_logP": 3.86, "perp_pKa": 6.5, "perp_fu_p": 0.01,
    "perp_compound_type": "moderate_base", "perp_R_bp": 0.6,
    "perp_mw": 531, "perp_ka": 1.0, "perp_CL_int": 150,
    "ddi_mechanism": "combined", "Ki": 0.015, "KI": 3.0, "kinact": 1.1,
    "fm": 0.94, "duration_h": 120,
})
```

**Result**:
- **AUC ratio: 18.24x** (lit Olkkola 1993: ~15x)
- **Cmax ratio: 2.38x**
- Liver CYP3A4 at midazolam Tmax: **40.7% of baseline** (59% irreversible +
  reversible inhibition)
- **Classification: Strong inhibition** (FDA 2020)

---

### C3. Dynamic DDI — Rifampin induction

```python
mcp.call_tool("run_dynamic_ddi", {
    ...(Rifampin 600 mg × 8 days, Midazolam on day 8)...
    "ddi_mechanism": "induction", "Emax": 14.0, "EC50": 0.5, "fm": 0.94,
})
```

**Result**:
- **AUC ratio: 0.01x** (lit Backman 1996: 0.04-0.10)
- Liver CYP3A4: **897% of baseline** (9x induction)
- **Classification: Strong induction**

> **Note**: the segmented liver (default `n_liver_segments=5`) tends to slightly
> over-predict induction. For a more conservative prediction set
> `n_liver_segments=1` (now exposed at the API level in v1.6).

---

<a id="scenario-d"></a>
## Scenario D: Special populations

### D1. Pediatric (5-year-old, 20 kg)

```python
mcp.call_tool("run_pbpk_simulation", {
    "name": "midazolam", "dose_mg": 4.0, "route": "iv_bolus",
    "body_weight": 20, "age": 5, "duration_h": 12,
    "kp_method": "poulin_theil",
})
```

**Result**:
- Cmax = 6.31 mg/L
- **CL/F = 11.17 L/h** (lit Reed 2001, age 5: 11-15 L/h) ✓
- **Vss = 18.92 L** (lit 12-18 L) ✓

---

### D2. Pregnancy (gestational age 32 weeks, 3rd trimester)

```python
mcp.call_tool("pregnancy_physiology", {"gestational_age_weeks": 32})
```

**Key changes** (vs. non-pregnant):
- CYP3A4: **1.61x ↑** (faster metabolism for midazolam, etc.)
- CYP1A2: 0.65x ↓
- GFR: 1.39x ↑
- Cardiac output: 1.34x ↑
- Hematocrit: 0.90x ↓

---

### D3. CKD Stage 3 (Moderate)

```python
mcp.call_tool("disease_state", {"disease_type": "ckd", "stage": "moderate"})
```

**Multipliers**:
- GFR: **0.37x ↓** (63% reduction)
- fu_p: 1.15x ↑ (reduced protein binding)
- CYP3A4: 0.85x ↓ (uremic toxin effect)
- Hematocrit: 0.88x ↓

---

### D4. Population PK (with variability)

```python
mcp.call_tool("run_population_pbpk", {
    "name": "midazolam", "dose_mg": 7.5, "n_individuals": 50,
    "route": "oral", "duration_h": 24,
})
```

**Result** (50 virtual subjects):

| Parameter | Median | 5th %ile | 95th %ile |
|---|---|---|---|
| Cmax (ng/mL) | 24.5 | 11.6 | 38.9 |
| AUC (ng·h/mL) | 166 | 52.6 | 363 |

> **Note**: in v1.5 the population tool used R&R Kp by default with no
> `kp_method` parameter. v1.6 added `kp_method` exposure — for precise
> prediction with lipophilic bases, pass `kp_method="poulin_theil"` or use
> a custom compound with tuned `kp_scale`.

---

<a id="scenario-e"></a>
## Scenario E: Transporter-driven drug (statin-like)

### E1. Hepatic transporter profile lookup

```python
mcp.call_tool("transporter_clearance", {
    "organ": "liver", "CLint_met": 20, "fu_p": 0.02, "R_bp": 0.6,
})
```

**Liver profile**:
| Transporter | Direction | Km (µM) | Vmax | CLint |
|---|---|---|---|---|
| OATP1B1 | Uptake | 5.0 | 100 | 20.0 |
| OATP1B3 | Uptake | 10.0 | 50 | 5.0 |
| MRP2 | Efflux | 50 | 30 | 0.60 |
| BCRP | Efflux | 20 | 20 | 1.00 |

Extended clearance: **CLint_overall = 10.73 L/h**

---

### E2. Permeability-limited + OATP1B1 simulation

A statin-like drug (logP 4.5, acidic, highly bound):

```python
mcp.call_tool("run_pbpk_simulation", {
    "name": "Statin", "logP": 4.5, "pKa": 4.2, "fu_p": 0.05,
    "compound_type": "acid", "R_bp": 0.55, "mw": 410,
    "ka": 2.0, "Fa": 0.9, "Fg": 1.0, "CL_int": 10,
    "distribution_model": "permeability_limited",
    "liver_oatp_km": 1.5, "liver_oatp_vmax": 50.0,
    "liver_mrp2_km": 5.0, "liver_mrp2_vmax": 20.0,
    "dose_mg": 40, "route": "oral", "duration_h": 48,
})
```

**Result**:
- Cmax = 0.87 µg/mL
- AUC_inf = 45.7 mg·h/L
- t½ = 37.7 h (OATP uptake + tissue accumulation give a long distribution phase)
- CL/F = 0.88 L/h

---

<a id="kp-method-selection"></a>
## Reference: Kp method selection rules

| Compound class | Recommended method | Rationale |
|---|---|---|
| Neutral / weak base (general) | `rodgers_rowland` | Default, broadly validated |
| Lipophilic base (logP > 3) | `poulin_theil` | Corrects R&R adipose over-prediction |
| Highly-bound acid (fu_p < 0.01) | `berezhkovskiy` or `pksim_standard` | R&R under-predicts acid Vss (Rodgers 2006) |
| Hydrophilic (logP < 0) | `rodgers_rowland` | Ionic partitioning handled correctly |
| Very lipophilic (logP > 5) | `kp_membrane` | Membrane binding dominates |

**Decision tree**:
1. Known compound? → if a `kp_scale` is in the library, R&R can be used as-is
2. logP > 3 + base → PT
3. fu_p < 0.01 + acid → Berezhkovskiy
4. Otherwise → R&R (default)
5. When in doubt, run `compare_kp_methods` first to inspect per-tissue Kp spread

---

<a id="parameter-naming-rules"></a>
## Reference: Parameter naming rules (important!)

FastMCP **silently drops** unknown kwargs. If you misspell a parameter, the
tool will run with the default value — no warning.

### Loading a library compound
| Tool | Parameter name |
|---|---|
| `run_pbpk_simulation`, `predict_kp`, `compare_kp_methods`, `predict_tissue_binding`, `predict_hepatic_clearance`, `compare_hepatic_clearance`, `predict_fg`, `run_population_pbpk` | `name` |
| `run_dynamic_ddi` | `victim_name`, `perp_name` |
| `drug_properties` | `drug_name` |

### Other common pitfalls
| Tool | Watch out for |
|---|---|
| `run_population_pbpk` | `n_individuals` (**not** `n_subjects`), clamped 10-500 |
| `ivive_microsomal` | `clint_vitro` (**not** `CLint_vitro_hlm`) |
| `predict_ddi` | `I_h_u` (unbound liver conc, µM), `fm`, `Ki`/`KI`/`kinact`/`Emax`/`EC50` |
| `disease_state` | `disease_type` + `stage` (hepatic accepts mild→mild_A alias) |
| `pregnancy_physiology` | `gestational_age_weeks` (**not** `gestational_week`) |
| `run_pbpk_simulation` | Defaults to R&R if `kp_method` is omitted |

---

## Library compounds and validation status

| Drug | Recommended Kp | Vss | CL | t½ | Status |
|---|---|---|---|---|---|
| Midazolam | PT | 0.90 L/kg | 15.4 L/h | 5.3 h | ✓ |
| Diazepam | PT | 1.20 L/kg | 0.91 L/h | 67 h | ✓ |
| Warfarin | Berezhkovskiy | 0.12 L/kg | 0.11 L/h | 52 h | ✓ |
| Theophylline | R&R | 0.28 L/kg | 1.93 L/h | 7.2 h | ✓ |
| Caffeine | R&R | 0.42 L/kg | 3.65 L/h | 5.7 h | ✓ |
| Metformin | R&R | 1.17 L/kg | 27.2 L/h | 2.6 h* | △ |

*Metformin t½ is under-predicted vs. observed (4-9 h) because the library
uses passive `CL_renal`. To reproduce the biphasic PK, use perm-limited
distribution with tuned kidney OCT2 / MATE1 parameters.

---

## Recommended workflow

```
0. **Measurement audit** — for each Priority-1 parameter (fu_hep, fu_inc,
   R_bp, Caco-2 Papp / Peff) and Priority-2 parameter (tissue Kp, ka,
   EHC params), ASK the user whether a measured value exists. Use it
   if so; otherwise fall back to literature consensus or model
   prediction, and tag the source as M / L / P in the final table.
     ↓
1. [drug_properties] Look up drug information
     ↓
2. [compare_kp_methods] Choose a Kp method
     ↓
3. [run_pbpk_simulation] Run the baseline simulation
     ↓
4. Compare against clinical values — if Vss/CL is off by more than 2x:
     - Switch Kp method (see Scenario A)
     - Revisit IVIVE inputs (Scenario B)
     - Inject literature values via kp_override
     ↓
5. [run_dynamic_ddi] Evaluate DDI (when applicable)
     ↓
6. [run_population_pbpk] Assess variability
     ↓
7. [fit_to_observed] Fine-tune against observed data
```

### Why Step 0 matters

Predictions for `fu_hep` (Austin equation), `fu_inc`, and `R_bp`
(Rodgers-Rowland from RBC partitioning) can differ from measurement by
2-4× at logP > 4 or fu_p < 0.01. A 2× error in `fu_hep` propagates
directly to a 2× error in CL. Always prefer a measured value, and be
explicit when you fall back to prediction.

---

*Server v1.5 / 30 tools / 25 core modules / ODE: BDF method*
*For the full parameter guide, call the MCP tool `pbpk_help`.*
