# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Whole-body PBPK (physiologically-based pharmacokinetic) modeling exposed as an MCP server (FastMCP, stdio transport). Architecture follows PK-Sim / Simcyp: 13 tissue compartments + arterial/venous pools + gut lumen, with dual liver input (hepatic artery + portal vein) and lung between venous→arterial.

## Common commands

```bash
# Activate the bundled venv (Python 3.14)
source .venv/bin/activate

# Run the server standalone (stdio — meant to be spawned by an MCP client)
python server.py

# Smoke test that imports + tool registration succeed
python -c "from server import mcp; print(len(mcp._tool_manager._tools))"   # expect 30

# Quick functional check via library API (no MCP client needed)
python -c "from core.compound import COMPOUND_LIBRARY; \
from core.partition_coeff import predict_kp_all, KpMethod; \
print(predict_kp_all(COMPOUND_LIBRARY['midazolam'], KpMethod.POULIN_THEIL))"

# Reinstall dependencies
pip install mcp numpy scipy matplotlib
```

There is **no test suite** in this repo — validation is done by running reference compounds through `run_pbpk_simulation` and checking Vss/CL/t½ against the table in README.md (Midazolam, Diazepam, Warfarin, Theophylline, Caffeine, Metformin) and DDI cases (Keto+Midaz ~14x, Rifampin+Midaz ~0.034x).

## Architecture (big picture)

The server is a thin shim — `server.py` registers tools from `tools/pbpk_tools.py` and exposes one resource (`pbpk://status`) and two prompts (`pbpk_setup_guide`, `pbpk_modeling_guide`). All scientific logic lives in `core/`.

### Module dependency layers

```
server.py
  └── tools/pbpk_tools.py      ← all 30 @mcp.tool() definitions live here
        └── core/*.py           ← pure-Python scientific modules (no MCP deps)
              └── data/PKSimDB.sqlite  (lazy-loaded by core/pksim_db.py)
```

`tools/pbpk_tools.py` is a single 1850-line file. Adding a new tool means adding another `@mcp.tool()` inside `register_pbpk_tools()`. Heavy deps (`pksim_db`, `pksim_import`, `data_fitting`) are **lazy-imported inside tool bodies** to keep startup fast — preserve that pattern.

### Core ODE engine — `core/pbpk_model.py`

`PBPKModel` builds and integrates the whole-body system with `scipy.integrate.solve_ivp` (BDF). State vector layout (perfusion-limited, 16 base states) is documented at the top of the file — when adding compartments, update the index map and the RHS together. Optional sub-systems extend the state vector:

- ACAT 9-segment GI absorption (`core/acat.py`) — replaces simple lumen depot
- EHC bile/gallbladder cycle (`core/ehc.py`)
- Lymphatic uptake (`core/lymphatic.py`)
- Permeability-limited 3-subcompartment organs (`core/pbpk_model.py`, alternate RHS)
- Transporter ODE (OATP/MRP2/OCT2/MATE1/P-gp, Michaelis-Menten — `core/transporters.py`). **Only active when `distribution_model="permeability_limited"`** — perfusion-limited model silently ignores transporter parameters because it has no vascular/cell separation. See MODEL_DESCRIPTION.md §7.5.

Mass conservation is the primary correctness check. Two historical bugs to be aware of (see CHANGELOG v1.3, v1.4): (1) perfusion-limited LUNG flow must come from `Qco` not from `self._Q` (which excludes LUNG); (2) perm-limited liver portal inflow must include pancreas, otherwise inert drug loses ~70% mass.

### Compound + physiology data

- `core/compound.py` — `CompoundSpec` dataclass + `COMPOUND_LIBRARY` (16 reference drugs). New library entries should set `recommended_kp_method` so `run_pbpk_simulation` can emit a Tip if the user picks a sub-optimal Kp method.
- `core/physiology.py` — ICRP 89 organ volumes, R&R/Schmitt tissue composition tables, GFR (Rhodin allometric), albumin ratios (R&R 2006 Table II).

### Partition coefficient methods (`core/partition_coeff.py`)

7 methods: `rodgers_rowland` (default), `lukacova`, `schmitt`, `poulin_theil`, `berezhkovskiy`, `pksim_standard`, `kp_membrane`. The Kp method selection rules are part of the user-facing contract — they're encoded in three places that must stay in sync: the `pbpk_modeling_guide` prompt in `server.py`, `pbpk_help` tool output, and the per-compound `recommended_kp_method` field. Selection rules:

- Lipophilic base (logP>3, e.g. Midazolam) → `poulin_theil`
- Highly-bound acid (fu_p<0.01, e.g. Warfarin) → `berezhkovskiy` or `pksim_standard`
- Neutral / weak base / hydrophilic → `rodgers_rowland`
- Very lipophilic (logP>5) → `kp_membrane`

### Hepatic clearance + DDI (`core/hepatic_models.py`, `core/ddi_dynamic.py`)

- Static DDI (`predict_ddi`): MSM (FDA/ICH M12) net-effect equation with reversible inhibition, MBI (kinact/KI), and induction (Emax/EC50).
- Dynamic DDI (`run_dynamic_ddi`): segmented liver dispersion model (1–N CSTRs in series, default 5), inlet-driven enzyme pool dynamics, supports independent dosing schedules for victim and perpetrator.
- DDI baseline must use the **same** segmented ODE for both alone-control and combined run, otherwise no-DDI ratio drifts off 1.0 (see v1.4 fix).

### IVIVE / Fg pipeline

`core/ivive.py` (HLM/hepatocyte/rCYP ISEF + Barter ontogeny) → `core/fg_prediction.py` (Qgut + per-CYP gut CLint scaling from liver fm). When user provides liver `fm_per_cyp`, `scale_gut_clint_per_cyp` derives gut CLint per enzyme automatically — keep this auto-derivation working when refactoring.

### PKSimDB integration (`core/pksim_db.py`)

Lazy-loaded SQLite DB (29 MB, GPLv2 — see NOTICE.md). Provides 38K population parameter distributions, 294 ontogeny points, 38 transporters. Don't bundle data into Python imports; query through this module.

## Conventions specific to this codebase

- **MCP parameter naming is load-bearing.** FastMCP silently drops unknown kwargs, so misnamed parameters fall back to defaults with no warning. The canonical names are:
  - `run_pbpk_simulation` uses `name` (not `compound_name`), `kp_method` (R&R default)
  - `run_dynamic_ddi` uses `victim_name`/`perp_name`, `Ki` (not `Ki_uM`), `kp_method`, `n_liver_segments`
  - `predict_ddi` uses `I_h_u` (unbound liver conc), `fm`, `Ki` — and now rejects missing required params + echoes inputs (v1.6)
  - `run_population_pbpk` uses `n_individuals` (not `n_subjects`), clamped 10–500
  - `disease_state` uses `disease_type` + `stage`, with hepatic aliases (mild→mild_A, etc.)

- **Units are mixed but consistent within the model**: amount mg, time h, volume L, flow L/h, concentration mg/L. CL_int is L/h (intrinsic). Don't introduce µM/min/mg unit hops without converting at the boundary.

- **No comments / no docstring sprawl in tool bodies.** Tool docstrings are user-facing (shown in MCP clients) — keep them tight and accurate. Every parameter listed in a docstring should match the function signature exactly.

- **Lazy imports for heavy deps** (`pksim_db`, `pksim_import`, `data_fitting`) — preserve to keep cold start fast.

- **Validation discipline**: when changing the ODE, Kp formulas, or DDI logic, re-run the 6 reference compounds and the 2 DDI cases. CHANGELOG.md documents what "passing" looks like for each version.

## Files to read first when investigating

- `core/pbpk_model.py` — ODE assembly, state vector layout, RHS
- `core/compound.py` — what a `CompoundSpec` carries
- `core/physiology.py` — organ flows/volumes, tissue composition tables
- `tools/pbpk_tools.py` — all MCP-facing parameter contracts
- `MODEL_DESCRIPTION.md` — equations + 32 references for the math
- `TUTORIAL.md` — 5 end-to-end scenarios that exercise most tools
