# PBPK MCP Server — Step-by-Step Tutorial

This tutorial shows how to use the server as it was designed: an
**LLM-driven, interactive PBPK lab assistant**. You describe the
compound and the question in plain English, the LLM searches
literature for missing parameters, asks you when a value cannot be
inferred, and the server enforces every invariant before returning
a result.

**Server version:** v2.3  |  **42 tools** (30 PBPK + 10 session/audit + 2 citation)

---

## Table of Contents

- [How to think about this server](#how-to-think)
- [Anatomy of a PBPK conversation](#conversation-anatomy)
- [Scenario A: Simulating a library drug (Midazolam)](#scenario-a)
- [Scenario B: New compound from in vitro data](#scenario-b)
- [Scenario C: Drug-drug interaction (DDI) assessment](#scenario-c)
- [Scenario D: Special populations](#scenario-d)
- [Scenario E: Transporter-driven drug (statin-like)](#scenario-e)
- [Scenario F: NSAID with EHC (Diclofenac, end-to-end)](#scenario-f)
- [Reference: Kp method selection](#kp-method-selection)
- [Reference: Parameter naming rules](#parameter-naming-rules)
- [The safety net: what the server enforces](#safety-net)

---

<a id="how-to-think"></a>
## How to think about this server

You will not build a PBPK model by writing 47 keyword arguments. You
will have a **conversation**:

```
You:  "Predict the PK of diclofenac 50 mg oral in a healthy adult.
       I have hepatocyte CLint = 120 µL/min/10⁶ cells from our lab."

LLM:  Looks up diclofenac on PubChem (CID 3033) and ChEMBL → MW 296.15,
      logP 4.51, pKa 4.0.
      Searches PubMed for fu_p → finds Davies 1997 PMID:9106794, fu_p
      = 0.005 (highly bound acid).
      Notices: this is a highly-bound acid → recommends Kp method
      "berezhkovskiy" or "schmitt" (server's Kp selection rule).
      Asks: "Do you have measured R_bp? If not, I'll predict ≈ 0.55
      from RBC partitioning (Rodgers-Rowland 2005) — let me know if
      you have a measurement."

You:  "Use the prediction. Run it."

LLM:  Calls run_pbpk_simulation(...) with full provenance: every
      parameter tagged M (measured) / L (literature) / P (predicted).
      Returns:
        - PK (Cmax, AUC, t½, F, Vss)
        - NCA reliability flags
        - Modelling Provenance footer
        - Audit fingerprint (replay later)
```

This is the workflow. The rest of the tutorial shows you the full
range of things that conversation can do.

---

<a id="conversation-anatomy"></a>
## Anatomy of a PBPK conversation

When you make a request to an LLM connected to this MCP server, the
LLM goes through these stages — you can interrupt at any point:

### Stage 1 — Identify the compound

The LLM uses one of several lookup paths in priority order:

| Source | Tool | Returns | Confidence |
|---|---|---|---|
| Built-in library (6 drugs) | `list_compounds` / `drug_properties` | Curated values + PMID/DOI citations | high |
| ChEMBL REST API | `drug_properties(drug_name=...)` | MW, logP, pKa, ChEMBL ID | medium-high |
| PubChem (web search via LLM) | LLM web tool + `verify_citation` | CID, structure, computed properties | medium |
| User-supplied measurements | (passed as kwargs) | Whatever the user gives | high |

For library drugs (`midazolam`, `metformin`, `theophylline`,
`diazepam`, `warfarin`, `caffeine`), every parameter in
`COMPOUND_LIBRARY` carries a `citations` dict with the actual PMIDs
used during curation. The provenance audit surfaces them.

### Stage 2 — Audit what's missing

After Stage 1, the LLM has a partial parameter set. It checks:

- **Required (Tier 1)**: name, MW, dose, route — must have to run
- **Strongly recommended (Tier 2)**: logP, pKa, fu_p, compound_type,
  clearance source — affects every Kp / CL prediction
- **Priority-1 measurements**: fu_hep, fu_inc, R_bp, Caco-2 Papp —
  these the LLM will explicitly ASK YOU for. Default fallbacks
  (Austin 2002 from logP, etc.) are documented but flagged.

### Stage 3 — Interactive clarification

The LLM is instructed (via the server's `pbpk_modeling_guide`
prompt) to **ask you** instead of substituting defaults silently:

> "I don't have a measured R_bp for your compound. Do you have one,
> or should I predict from RBC partitioning (Rodgers-Rowland 2005)?"

> "You provided HLM CLint = 70 µL/min/mg, but diclofenac has a known
> UGT2B7 acyl-glucuronide pathway (~30% fm). HLM misses UGT. Do you
> have a hepatocyte CLint instead, or should I proceed with HLM
> understanding CL will be 2-5× under-predicted?"

> "You said 'simulate diclofenac' but didn't specify a dose. The most
> common adult oral dose is 50 mg or 75 mg. Which?"

If the LLM fails to ask and substitutes silently, the server's
**provenance audit** (`audit_model_provenance`) and **soft warnings**
will mark the value as `default-substituted` ⚠️ in the output.

### Stage 4 — Citation verification

Whenever the LLM cites literature (PMID or DOI), it can call
`verify_citation()` to confirm the source actually exists in PubMed
or Crossref. Fabricated PMIDs are caught here.

```
verify_citation("9106794")
  → status: verified
  → title: "Pharmacokinetics and pharmacodynamics of diclofenac..."
  → authors: "Davies NM, Anderson KE"
  → year: 1997
  → journal: "Clin Pharmacokinet"
```

### Stage 5 — Run, validate, audit

The actual `run_pbpk_simulation` (or session-based
`simulate_validated`) call happens. The server then:

1. Validates every input against physiological ranges
2. Solves the 16-state ODE (BDF, atol=1e-10, rtol=1e-8)
3. Asserts post-simulation dose recovery (1% IV / 5% oral) → **abort
   on failure**
4. Computes NCA with reliability flags (extrapolation < 20%, ≥3
   terminal points, R² ≥ 0.85, duration ≥ 3× t½)
5. Writes audit fingerprint to `data/audit.jsonl`
6. Returns markdown with provenance + reliability footer

You get a result you can defend, or an abort with a specific reason.

---

<a id="scenario-a"></a>
## Scenario A: Simulating a library drug (Midazolam)

**Goal**: predict PK after 7.5 mg oral midazolam in a healthy adult,
and compute oral bioavailability F.

### Step 1 — Discover the compound

Plain-English request to the LLM:

> "I want to simulate midazolam 7.5 mg oral in a 70 kg adult."

The LLM calls:

```python
drug_properties(drug_name="midazolam")
```

Output:

```
## Drug Properties — Midazolam
Source: curated

| Property        | Value          |
| ChEMBL ID       | CHEMBL601      |
| MW              | 325.80 g/mol   |
| logP            | 3.89           |
| pKa             | 6.2            |
| Ro5 violations  | 0              |
```

The LLM also notes that `midazolam` is in `COMPOUND_LIBRARY`, so
fu_p, R_bp, ka, Fa, Fg, CL_int are all curated — and every value
carries a citation (Thummel 1996, Greenblatt 1984, Björkman 2001,
Jansson 2008).

### Step 2 — Choose a Kp method (auto)

For library compounds with a `recommended_kp_method`, the server
**auto-selects** it. Midazolam (lipophilic moderate base, logP > 3)
→ `poulin_theil`. The output explicitly tells you:

> ℹ️ **Auto-selected Kp method:** `poulin_theil` (library default for
> Midazolam). R&R was NOT used; the library specifies a more
> accurate method for this compound class. To force R&R, pass
> `kp_method="rodgers_rowland"` explicitly.

You can inspect the alternatives manually:

```python
compare_kp_methods(name="midazolam")
```

Excerpt (Adipose / Liver rows):

| Tissue | R&R | Lukacova | Schmitt | PT | PTB | PK-Sim | Kp_mem |
|---|---|---|---|---|---|---|---|
| Adipose | 4.26 | 4.26 | 3.83 | 3.20 | 4.16 | 4.56 | 17.36 |
| Liver | 1.89 | 1.89 | 9.84 | 0.35 | 1.28 | 3.97 | 3.03 |

### Step 3 — Run the simulation

```python
run_pbpk_simulation(
    name="midazolam",
    dose_mg=7.5,
    route="oral",
    duration_h=24.0,                    # ≥ 3× t½ for valid NCA
    body_weight=70.0,                   # explicit subject avoids sentinel warn
    sex="male",
    age=35.0,
    kp_method="poulin_theil",           # or omit — library auto-selects
)
```

Result (excerpt):

| Parameter | Predicted | Clinical (Greenblatt 1984) |
|---|---|---|
| Cmax | **72.2 ng/mL** | 40-100 ng/mL ✓ |
| Tmax | 0.38 h | 0.5-1.5 h ≈ |
| AUC_inf | **167.9 ng·h/mL** | 75-250 ng·h/mL ✓ |
| t½ | **5.40 h** | 1.8-6.4 h ✓ |
| CL/F | 44.66 L/h | – |

```
### NCA reliability
- Extrapolation fraction: 8.2% (FDA/EMA: <20% recommended)
- Terminal-phase points: 47 (>= 3 required for valid λz)
- Terminal R²: 0.998
- Duration / t½: 4.4 (>= 3 recommended)

_NCA reliability criteria all met._
```

### Step 4 — Compute F (IV vs oral)

```python
# IV bolus
iv = run_pbpk_simulation(name="midazolam", dose_mg=5.0, route="iv_bolus",
                         duration_h=24.0, body_weight=70.0, sex="male", age=35.0)

# Oral
po = run_pbpk_simulation(name="midazolam", dose_mg=5.0, route="oral",
                         duration_h=24.0, body_weight=70.0, sex="male", age=35.0)
```

Result:
- IV CL = 16.03 L/h (lit 18-30)
- PO CL/F = 44.66 L/h
- **F_oral = AUC_po / AUC_iv = 0.359 (36%)** — lit 30-50% ✓

---

<a id="scenario-b"></a>
## Scenario B: New compound from in vitro data

**Goal**: a CYP3A4 substrate measured at HLM CLint = 30 µL/min/mg.
Predict clinical PK. The compound is NOT in the library.

### Step 1 — Establish identity

You provide: name="NewDrug", logP=3.5, MW=400, pKa=7.0, fu_p=0.1,
compound_type="neutral", R_bp=1.0. The LLM range-checks all of these
(server invariants). If fu_p=1.5 by typo, ValueError immediately.

### Step 2 — IVIVE: HLM CLint → in vivo CLint

```python
ivive_microsomal(
    clint_vitro=30.0,                  # µL/min/mg HLM
    fu_inc=None,                       # auto-predict from logP via Austin 2002
    logP=3.5,                          # for fu_inc prediction
    protein_conc=1.0,                  # mg/mL
)
```

Result:

```
**CLint in vivo = 300.64 L/h**

| Parameter                  | Value |
| CLint_unbound_uL_min_mg    | 59.35 |
| fu_inc                     | 0.5055 (predicted from logP=3.5)
| MPPGL                      | 45 mg/g liver
| liver_weight_g             | 1876
| scaling_factor             | 5.065
```

> **Interactive moment.** A careful LLM will say: "The fu_inc is
> predicted from logP via Austin 2002 — accuracy is ±2-fold at
> logP > 4. Do you have a measured fu_inc from rapid equilibrium
> dialysis? If not, I'll proceed with the prediction and flag it as
> `inferred` in the audit."

### Step 3 — Predict Fg (Yang Qgut)

```python
predict_fg(Peff=4.0, CLint_gut=10.0, fu_gut=0.2)
```

Result: Fg = 0.6793, Qgut = 4.24 L/h.

### Step 4 — Full simulation

```python
run_pbpk_simulation(
    name="NewDrug",
    logP=3.5, pKa=7.0, fu_p=0.1, mw=400,
    compound_type="neutral", R_bp=1.0,
    ka=1.5, Fa=0.9, Fg=0.7,
    clearance_source="hlm",            # ← discriminated union; HLM-mode
    CLint_vitro_hlm=30.0,              # ← required field for "hlm"
    dose_mg=20, route="oral", duration_h=24.0,
    body_weight=70.0, sex="male", age=35.0,
)
```

Result: Cmax = 37.8 ng/mL, AUC_inf = 375 ng·h/mL, t½ = 17.3 h, CL/F = 53.27 L/h.

### Step 5 — Provenance footer

```
### Modelling Provenance
> Tag every parameter as M=measured / L=literature / P=predicted / D=default

**Defaults used (no user / library value):**
- Peff not provided — Fg may be unreliable

**Mechanisms NOT modelled:**
- Active transport (OATP/MRP2/OCT2/MATE1/P-gp) — provide Km/Vmax pairs and
  set distribution_model='permeability_limited' to enable

_Audit fingerprint: `0021a841e9e47c04`_
```

---

<a id="scenario-c"></a>
## Scenario C: Drug-drug interaction (DDI) assessment

### Step 1 — Static screening (in vitro Ki only)

```python
predict_ddi(
    mechanism="reversible",
    Ki=0.05,                           # µM
    I_h_u=0.5,                         # µM unbound liver
    fm=0.9,
)
```

Result: **AUC ratio = 5.50× (Strong inhibition, FDA 2020)**

The server now rejects missing required params: if you call
`predict_ddi(mechanism="reversible")` without Ki, you get a
structured "missing parameters" message — not a silent zero.

### Step 2 — Dynamic DDI: ketoconazole steady-state + midazolam

```python
run_dynamic_ddi(
    victim_name="midazolam",
    victim_dose_mg=7.5, victim_route="oral", victim_first_dose_h=72,
    perp_name="Keto",
    perp_dose_mg=400, perp_route="oral",
    perp_n_doses=5, perp_interval_h=24,
    perp_logP=3.86, perp_pKa=6.5, perp_fu_p=0.01,
    perp_compound_type="moderate_base", perp_R_bp=0.6,
    perp_mw=531, perp_ka=1.0, perp_CL_int=150,
    ddi_mechanism="combined",
    Ki=0.015, KI=3.0, kinact=1.1,      # all three required for "combined"
    fm=0.94, duration_h=120,
)
```

The server enforces DDI prerequisites: `ddi_mechanism="combined"`
**requires** Ki AND (KI + kinact). Missing any → ValueError. Result:

- **AUC ratio: 18.24×** (lit Olkkola 1993: ~15×)
- **Cmax ratio: 2.38×**
- Liver CYP3A4 at midazolam Tmax: **40.7% of baseline**
- **Classification: Strong inhibition** (FDA 2020)

### Step 3 — Dynamic DDI: rifampin induction

```python
run_dynamic_ddi(
    ...rifampin 600 mg × 8 days, midazolam on day 8...
    ddi_mechanism="induction",
    Emax=14.0, EC50=0.5,               # both required for induction
    fm=0.94,
)
```

Result: AUC ratio 0.01× (lit 0.04-0.10), Strong induction.

> **Note:** segmented liver default `n_liver_segments=5` slightly
> over-predicts induction. Use `n_liver_segments=1` for conservative
> estimates.

---

<a id="scenario-d"></a>
## Scenario D: Special populations

The server distinguishes the 73 kg / male / age 30 default subject
with an **explicit warning** ("Subject defaults used") so a pediatric
or pregnancy simulation cannot accidentally run as the reference adult.

### D1 — Pediatric (5-year-old, 20 kg)

```python
run_pbpk_simulation(
    name="midazolam", dose_mg=4.0, route="iv_bolus",
    body_weight=20.0, age=5.0, sex="male",   # explicit non-default
    duration_h=12.0, kp_method="poulin_theil",
)
```

Result:
- CL/F = 11.17 L/h (lit Reed 2001, age 5: 11-15 L/h) ✓
- Vss = 18.92 L (lit 12-18 L) ✓

### D2 — Pregnancy (gestational age 32 weeks, 3rd trimester)

```python
pregnancy_physiology(gestational_age_weeks=32)
```

The server **range-checks GA ∈ [0, 42]** — passing GA=100 raises
ValueError.

Key changes vs non-pregnant:
- CYP3A4: **1.61× ↑** (faster midazolam metabolism)
- CYP1A2: 0.65× ↓
- GFR: 1.39× ↑
- Cardiac output: 1.34× ↑

### D3 — CKD Stage 3 (Moderate)

```python
disease_state(disease_type="ckd", stage="moderate")
```

Multipliers:
- GFR: **0.37× ↓** (63% reduction)
- fu_p: 1.15× ↑
- CYP3A4: 0.85× ↓ (uremic toxin effect)

### D4 — Population PK (variability)

```python
run_population_pbpk(
    name="midazolam", dose_mg=7.5,
    n_individuals=50,                  # clamped to [10, 500]
    route="oral", duration_h=24.0,
    kp_method="poulin_theil",          # passed through to per-subject
)
```

50 virtual subjects:

| Parameter | Median | 5th %ile | 95th %ile |
|---|---|---|---|
| Cmax (ng/mL) | 24.5 | 11.6 | 38.9 |
| AUC (ng·h/mL) | 166 | 52.6 | 363 |

---

<a id="scenario-e"></a>
## Scenario E: Transporter-driven drug (statin-like)

**Important:** transporter parameters are silently ignored in the
default `perfusion_limited` distribution model. The server **warns**
when you pass transporter Km/Vmax with the wrong distribution model.

### Step 1 — Profile lookup

```python
transporter_clearance(organ="liver", CLint_met=20, fu_p=0.02, R_bp=0.6)
```

Liver profile: OATP1B1, OATP1B3, MRP2, BCRP. Extended-clearance CL = 10.73 L/h.

### Step 2 — Permeability-limited simulation with OATP1B1

```python
run_pbpk_simulation(
    name="Statin",
    logP=4.5, pKa=4.2, fu_p=0.05,
    compound_type="acid", R_bp=0.55, mw=410,
    ka=2.0, Fa=0.9, Fg=1.0, CL_int=10,
    distribution_model="permeability_limited",     # ← required for transporters
    liver_oatp_km=1.5, liver_oatp_vmax=50.0,       # both required (XOR raises)
    liver_mrp2_km=5.0, liver_mrp2_vmax=20.0,
    dose_mg=40, route="oral", duration_h=48.0,
    body_weight=70.0, sex="male", age=35.0,
)
```

Result: Cmax = 0.87 µg/mL, AUC_inf = 45.7 mg·h/L, t½ = 37.7 h.

If you forget the `distribution_model` change:

> ⚠️ Transporter parameters were provided but `distribution_model=
> "perfusion_limited"` (default). Active transport is only evaluated
> in the permeability-limited model. Set
> `distribution_model="permeability_limited"` to enable them, or
> remove the transporter inputs to silence this warning.

If you forget Km/Vmax pairing:

```
ValueError: Transporter 'liver_oatp' has only Km set. Both Km and
Vmax are required to activate a transporter — providing one is
silently ignored in the legacy schema. Provide both or neither.
```

---

<a id="scenario-f"></a>
## Scenario F: NSAID with EHC (Diclofenac, full step-by-step)

This is the worked case study that motivated v1.7-v2.3. Diclofenac
(NSAID, highly-bound acid, UGT2B7-glucuronidated, undergoes EHC) is
the kind of compound where naive defaults silently produce 10×-wrong
Vss. This walkthrough shows the LLM-driven discovery of those
pitfalls.

### Step 0 — User intent

> "Build a PBPK model for diclofenac 50 mg oral. I have hepatocyte
> CLint = 120 µL/min/10⁶ cells from our in-house assay."

### Step 1 — Compound identification

LLM (or you) calls:

```python
drug_properties(drug_name="diclofenac")
```

Auto-fills MW 296.15, logP 4.51, pKa 4.0 from ChEMBL (CHEMBL3).
Diclofenac is **not** in the curated `COMPOUND_LIBRARY`, so the LLM
must build a custom compound.

### Step 2 — Measurement audit

The LLM consults the Priority-1 list:

| Parameter | Have measurement? | Fallback |
|---|---|---|
| `fu_p` | ❓ | PubMed search → Davies 1997 PMID:9106794 → fu_p = 0.005 |
| `R_bp` | ❓ | Predict from RBC partitioning → 0.55 |
| `Caco-2 Papp` | ❓ | Yang Qgut with Peff = 2.83 ×10⁻⁴ cm/s |
| `fu_hep` (incubation) | ❓ | Austin 2002 from logP=4.51 → 0.186 |
| `CLint` | ✅ | Hepatocyte 120 µL/min/10⁶ cells (user) |

Citation verification on Davies 1997:

```python
verify_citation("9106794")
```

Returns: status=verified, year=1997, journal="Clin Pharmacokinet" ✓

### Step 3 — Kp method selection

Diclofenac is a **highly-bound acid (fu_p < 0.01)** → the server's
selection rule recommends `berezhkovskiy` or `pksim_standard`.

```python
compare_kp_methods(
    logP=4.51, pKa=4.0, fu_p=0.005,
    compound_type="acid", R_bp=0.55, mw=296.15,
)
```

The 7-method comparison shows R&R Vss=4 L (under-predicts), Berezhkovskiy
Vss=120 L (over-predicts — also a known limitation), Schmitt is
closest. This is the textbook "highly-bound acid is hard" case.

### Step 4 — Hepatocyte IVIVE

```python
ivive_microsomal(
    clint_vitro=120.0,                 # but really hepatocyte
    logP=4.51,
    protein_conc=1.0,
)
# Or use the dedicated hepatocyte scaler:
# scale_hepatocyte_clint(clint_hep=120.0, logP=4.51)
```

Hepatocyte gives **CLint_in_vivo = 7184 L/h** (vs the same drug from
HLM = 1624 L/h). The 4.4× difference is the UGT2B7 contribution that
HLM systematically misses. The LLM should warn:

> ℹ️ Hepatocyte CLint is preferred for diclofenac because UGT2B7
> contributes ~30% of metabolism (Davies 1997). HLM-only IVIVE would
> under-predict CL by ~5×.

### Step 5 — First simulation (calibrated)

```python
run_pbpk_simulation(
    name="Diclofenac",
    logP=4.51, pKa=4.0, fu_p=0.005,
    compound_type="acid", R_bp=0.55, mw=296.15,
    ka=2.0, Fa=1.0, Fg=0.76,
    clearance_source="hepatocyte",
    CLint_vitro_hep=120.0,
    CL_renal=0.1,                      # negligible (<1%)
    dose_mg=50.0, route="oral", duration_h=24.0,
    body_weight=70.0, sex="male", age=35.0,
    kp_method="schmitt",
)
```

Result vs literature (50 mg PO):

| Parameter | Predicted | Clinical | Status |
|---|---|---|---|
| Cmax | 658 ng/mL | 1200-2500 | ↓ low |
| AUC_inf | 1053 ng·h/mL | 1500-2500 | ↓ low |
| t½ | 0.49 h | 1.2-2.0 | ↓ short |
| CL/F | 47.5 L/h | ~50 | ✓ |
| Vss | 13.0 L | 8-12 | ✓ |
| F | 0.44 | 0.50-0.60 | ≈ |

CL ✓, Vss ✓ — but t½ is too short. Why?

### Step 6 — Add EHC (acyl-glucuronide deconjugation)

Diclofenac forms an acyl-glucuronide that's deconjugated by gut
β-glucuronidase, releasing parent drug back into the lumen — extending
the apparent t½. The LLM should recognize this is the missing piece:

```python
run_pbpk_simulation(
    ... same as above ...
    # Enable EHC via SimulationConfig
    # (not all flat tools expose this; the session workflow does)
)
```

With EHC enabled (`enable_ehc=True`, `CL_bile=8`, `f_bile_parent=0.30`,
`k_deconjugation=0.6`, `f_reabsorption=0.7`), t½ extends to **0.99 h**,
closer to literature 1.2-2.0 h.

### Step 7 — Final provenance

```
### NCA reliability
- Extrapolation fraction: 11.4% ✓
- Terminal-phase points: 32 ✓
- Terminal R²: 0.991 ✓
- Duration / t½: 24 ✓

### Modelling Provenance
**Defaults used (no user / library value):**
- (none)

**Mechanisms NOT modelled:**
- Active transport (OATP1B3 — diclofenac is not a major substrate, OK)

_Audit fingerprint: `7af1c8b2e0954d31`_
```

### Step 8 — Lessons learned

1. **Hepatocyte > HLM** for any drug with non-CYP metabolism (UGT, SULT, esterase). HLM-only IVIVE under-predicts CL by 2-5×.
2. **Highly-bound acids (fu_p < 0.01)** are the hardest Kp class. None of the 7 methods match clinical Vss perfectly; Schmitt is the most balanced.
3. **EHC matters for terminal phase** of acyl-glucuronide-forming drugs (NSAIDs, mycophenolate). Without `enable_ehc=True`, t½ is under-predicted.
4. Always look at **NCA reliability flags** before quoting CL/F or t½.

---

<a id="kp-method-selection"></a>
## Reference: Kp method selection

| Compound class | Recommended | Rationale |
|---|---|---|
| Neutral / weak base | `rodgers_rowland` | Default, broadly validated |
| Lipophilic base (logP > 3) | `poulin_theil` | Corrects R&R adipose over-prediction |
| Highly-bound acid (fu_p < 0.01) | `berezhkovskiy` or `pksim_standard` | R&R under-predicts; Berezhkovskiy corrects albumin handling |
| Hydrophilic (logP < 0) | `rodgers_rowland` | Ionic partitioning OK |
| Very lipophilic (logP > 5) | `kp_membrane` | Membrane binding dominates |

**Decision flow:**
1. Library compound? → `recommended_kp_method` is auto-selected.
2. logP > 3 + base → `poulin_theil`
3. fu_p < 0.01 + acid → `berezhkovskiy`
4. Otherwise → `rodgers_rowland` (default)
5. When in doubt: call `compare_kp_methods` first.

---

<a id="parameter-naming-rules"></a>
## Reference: Parameter naming rules

FastMCP **silently drops** unknown kwargs. Misspelled parameters fall
back to defaults, with no warning at the protocol layer — this is why
the server now adds explicit validation. Canonical names:

| Tool family | Library compound name |
|---|---|
| `run_pbpk_simulation`, `predict_kp`, `compare_kp_methods`, `predict_tissue_binding`, `predict_blood_plasma_ratio`, `predict_hepatic_clearance`, `compare_hepatic_clearance`, `predict_fg`, `run_population_pbpk` | `name` |
| `run_dynamic_ddi` | `victim_name`, `perp_name` |
| `drug_properties` | `drug_name` |

| Tool | Watch out for |
|---|---|
| `run_population_pbpk` | `n_individuals` (NOT `n_subjects`); clamped to [10, 500] |
| `ivive_microsomal` | `clint_vitro` (NOT `CLint_vitro_hlm`) |
| `predict_ddi` | `I_h_u` (unbound µM), `fm`, plus mechanism-specific (`Ki`/`KI`+`kinact`/`Emax`+`EC50`) — **server rejects missing prerequisites** |
| `disease_state` | `disease_type` + `stage` (hepatic accepts `mild` → `mild_A` alias) |
| `pregnancy_physiology` | `gestational_age_weeks` (NOT `gestational_week`); range-checked [0, 42] |
| `run_pbpk_simulation` | `kp_method` defaults to R&R; library compounds may auto-override |

---

<a id="safety-net"></a>
## The safety net: what the server enforces

You don't need to remember all of this — the server enforces it. But
if you see one of these in your output, here's what it means.

### Hard errors (raise ValueError before any computation)

- Invalid enum (`kp_method`, `distribution_model`, `route`,
  `absorption_model`) — closest valid option suggested
- `clearance_source` mismatched against the IVIVE field actually
  provided
- Out-of-range physical parameter (`fu_p > 1`, `logP > 10`,
  `dose_mg <= 0`, `body_weight > 300`, etc.)
- Incomplete transporter Km/Vmax pair
- `simulate_validated()` token forged or expired
- DDI mechanism missing prerequisite (`reversible` → Ki, `mbi` →
  KI+kinact, `induction` → Emax+EC50)
- Multi-dose interval extends beyond simulation duration
- `predict_kp()` / `simulate_acat()` / etc. called with all-default
  args (no library + no physchem)
- `pregnancy_physiology(GA=100)` (range check)
- **Post-simulation dose recovery > 1% IV / 5% oral** — the ODE lost
  or created mass; result is invalid
- Physiology table mass-balance failure at startup (organ volumes
  don't sum to body weight, etc.)

### Soft warnings (run, but surface ⚠️ in output)

- Library compound matched + custom physchem ignored
- Transporter parameters with `perfusion_limited` distribution
- Sentinel defaults (`fu_p=1.0`, `R_bp=1.0`)
- Zero hepatic + zero renal clearance
- `compound_type="neutral"` with `fu_p < 0.01` (probable acid)
- Subject defaults (73 kg male age 30) used silently
- NCA: extrapolation > 20%, n_terminal < 3, R² < 0.85, duration < 3× t½

### Output-time provenance audit

For session-built models (`register_compound` → … →
`validate_model`), `audit_model_provenance(compound_id)` produces a
table with one row per parameter (`Source type` ∈
{user_provided, measurement, literature, library, default, inferred,
UNSOURCED}, `Confidence` ∈ {high, medium, low, unverified}). Verdict:
**passed** / **passed-with-flags** / **failed-audit**.

Sentinel defaults (Fa=1.0, R_bp=1.0, etc.) are flagged ⚠️ unless a
`*_source` was recorded. Vague labels ("typical", "literature value")
are converted to `UNSOURCED`. Citations claimed must be verifiable
via `verify_citation()` against PubMed/Crossref.

### Audit log + replay

Every successful simulation writes one JSON line to
`data/audit.jsonl` with input fingerprint, resolved parameters,
warnings, and NCA summary. `replay_lookup(fingerprint)` retrieves
the record. Two calls with the same inputs produce the same
fingerprint — non-determinism is detectable.

---

*Server v2.3 / 42 tools / 79 fail-fast tests / ODE: BDF (atol=1e-10, rtol=1e-8)*
*For the full parameter guide at runtime, call the MCP tool `pbpk_help`.*
*For the full provenance audit, call `audit_model_provenance(compound_id)` after `validate_model()`.*
