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

## Modeling workflow rule (MANDATORY)

**Before building any PBPK model, audit which parameters the user has
measured.** Do not silently fall back to predictions. The Priority-1
parameters where measurement vs. prediction can differ by ≥2× are:

- `fu_hep` / `fu_inc` (hepatocyte / HLM unbound fraction) — Austin equation
  prediction can be 2-4× off at logP > 4. Measured value (rapid equilibrium
  dialysis) is strongly preferred.
- `R_bp` (blood:plasma ratio) — drives all Kp_blood values.
- Caco-2 `Papp` or human `Peff` — determines `Fg` via Yang Qgut model.

Priority-2 parameters where measurement supersedes any in silico method:

- Tissue Kp (rat distribution data) — replaces R&R/Schmitt/PT prediction
- `ka` — fit from oral C-t data
- EHC parameters — from bile cannulation studies

**For each parameter, ask the user explicitly.** Mark each value in the
final summary as **M** (measured), **L** (literature consensus), or
**P** (predicted). The `pbpk_modeling_guide` MCP prompt and the
`WELCOME_PROMPT` in `prompts/user_guide.py` enforce this workflow at
runtime — do not regress them when refactoring.

This rule was added after the Diclofenac case study where blindly
applying defaults gave Vss off by 10× (Berezhkovskiy) or 0.5× (R&R)
even though CL was right.

## Server-side safety architecture

The MCP server combines (a) five invariants declared in the system
prompt — Refuse-to-default, Cite-or-abstain, Unit-explicit,
Range-check, Mass-balance — with (b) seven runtime layers that
enforce them. Prompt + runtime is stronger than either alone:

### The five invariants (declared in `server.py::FastMCP(instructions=...)`)

1. **Refuse-to-default** — never silently substitute a default for
   a missing required parameter.
2. **Cite-or-abstain** — every literature value carries a verifiable
   PMID/DOI; cache miss + network error → mark `unverified`.
3. **Unit-explicit** — pass canonical-unit floats or unit-bearing
   strings; pint validators reject incompatible units.
4. **Range-check** — out-of-range values are REJECTED, not clipped.
5. **Mass-balance** — post-simulation dose recovery aborts on >1%
   IV / >5% oral.

### Runtime enforcement (the seven layers):

1. **`core/clearance_spec.py`** — Pydantic discriminated union
   (`DirectClearance | HLMClearance | HepatocyteClearance |
   RecombinantCYPClearance`) keyed by `source`. Required fields per
   variant are enforced at schema level. Each CLint field accepts
   unit-bearing strings ('70 uL/min/mg') via pint validators.
2. **`core/transporter_spec.py`** — `TransporterKwargs.from_legacy_kwargs`
   pairs Km/Vmax at schema level (XOR is an error). Eliminates the
   "I gave km but vmax was None" silent drop.
3. **`core/invariants.py`** — physiological-range hard limits for
   every numeric parameter. Mass-balance checks (organ volumes Σ ≈
   body weight, blood flow Σ ≈ 1.0) run inside `get_physiology`.
4. **`core/units.py`** — pint canonical unit table. `parse_quantity`
   converts user-supplied strings to canonical units; incompatible
   units raise with dimensional mismatch.
5. **`core/citation.py`** — PMID verification via NCBI E-utils, DOI
   via Crossref. Cached to `data/citation_cache.jsonl`. Strict mode
   raises on cache miss + network error — use for publication-grade
   workflows.
6. **`core/session.py`** + **`tools/session_tools.py`** —
   prerequisite-gated workflow. Decomposes the 47-parameter flat
   call into 8 explicit steps. `validate_model()` issues a token;
   `simulate_validated()` only accepts that token. Half-built
   sessions cannot be simulated.
7. **`core/audit.py`** — append-only JSONL at `data/audit.jsonl`
   with input fingerprint, resolved parameters, warnings, and NCA
   summary. `replay_lookup(fingerprint)` for reproducibility.
8. **`core/web_param_search.py`** — anti-hallucination layer for
   web-sourced parameters. Eight verification gates (required_fields,
   citation_exists, topic_match, evidence_class, range_check,
   unit_check, context_match, evidence_binding). Server fetches the
   PubMed abstract or PMC OA full text and confirms the LLM's
   verbatim snippet, value, and unit appear in the retrievable text.
   Defends against (a) fabricated PMIDs, (b) real PMIDs for unrelated
   topics, (c) "right paper, wrong number" — the comparator-drug-value
   failure mode — and (d) citation laundering via
   `upstream_citation_id` collapse. Five confidence states:
   `high` (≥2 INDEPENDENT measurement groups within tolerance),
   `medium` (1 candidate, snippet exact-verified), `low` (loose
   binding), `conflict` (≥2 verified candidates that disagree —
   refuse-to-merge), `none`. Auto-merge of disagreement (geomean of
   0.5 and 50 → 5) is explicitly forbidden. Exposed via the
   `search_parameter_with_citation` MCP tool and the
   `web_search_grounding` MCP prompt that LLM drivers must read
   before issuing a literature search query.

When refactoring, preserve all eight layers. The fail-fast test
suites (`tests/test_silent_fallback.py`, `tests/test_v28_patches.py`,
`tests/test_web_param_search.py`, run via `python -m tests.<name>`)
cover 153 specific failure modes across these layers.

### Workflow signals (codex UX review 2026-04-30)

`validate_model` returns three independent signals — do not collapse them:

- **schema_ok** — all required parameter groups (compound,
  binding, clearance, absorption, structure) are present and pass
  range/discriminated-union validation. This issues the
  `validation_token` so `simulate_validated()` can run.
- **audit_ok** — the provenance audit verdict is `passed` or
  `passed-with-flags`, not `failed-audit`. Failure here means
  citations are missing or silent fallbacks fired.
- **simulation_ready** — both above are true. Token is issued whenever
  schema_ok is true (so the user can iterate on a draft), but
  simulation_ready=false explicitly tells the user that running the
  simulation will produce numbers that are NOT prediction-grade.

`simulate_validated()` will run as long as schema_ok=true, even if
audit_ok=failed-audit, because we want users to inspect intermediate
results during iteration. The audit verdict is preserved in the
output so the user knows the result is not yet citable.

### Mass-balance tolerance note

`simulate_validated()` and the legacy `run_pbpk_simulation()` both
gate on `check_dose_recovery(tolerance=0.01, oral_tolerance=0.05)`.
The 1% IV tolerance assumes BDF integration error compounded with
trapezoidal post-hoc elimination integration. Configurations with
non-trivial CL_renal added on top of CL_int can drift toward the
edge of this tolerance — if a session-built compound fails mass
balance with `rel_err` between 1% and 2%, this is calibration, not a
physical bug. Possible mitigations: tighten the ODE solver atol/rtol
upstream, or accept up to 2% IV drift. The current 1% gate
intentionally fails closed.

### DDI mechanism scope

`predict_ddi(mechanism="reversible")` returns a screening-grade
estimate that EXCLUDES time-dependent/mechanism-based inhibition
(MBI) and induction. Known MBI inhibitors (ketoconazole, ritonavir,
clarithromycin, fluvoxamine, verapamil, itraconazole) typically show
3-5x more inhibition under MBI than under reversible-only Ki. The
tool now emits a Mechanism-scope notice on every reversible/induction/
mbi-only call pointing the user at `mechanism="net"` (combined
inhibition + MBI + induction) or `run_dynamic_ddi` (time-resolved).

For ketoconazole + midazolam, expect:
- Static `mechanism="reversible"` → ~3-4x AUC ratio (lower bound)
- Static `mechanism="net"` → ~8-12x AUC ratio (depends on KI/kinact)
- Dynamic `run_dynamic_ddi` → 14-15x AUC ratio (matches lit 10-15x)

### Provenance audit (output-time, separate layer)

`prompts/provenance_audit.py` adds a second, output-time enforcement
axis. Where input validation rejects malformed inputs, the audit
detects outputs that *look* reasonable but were assembled from
defaults the LLM didn't notice. Two artifacts:

- **`provenance_audit` MCP prompt** — generic auditor template;
  apply to any PBPK model output. Forces per-parameter rows with
  source type / citation / confidence; refuses verdict `passed`
  unless every row has a verifiable source.
- **`audit_model_provenance(compound_id)` MCP tool** — deterministic
  audit from session state. Tags each parameter as
  user_provided / measurement / literature / library / default /
  inferred / UNSOURCED; sentinel detection respects recorded
  sources (Fa=1.0 with `Fa_source='BCS II'` is not flagged).

Verdict labels: `passed` / `passed-with-flags` / `failed-audit`.
The audit is automatically embedded in every `validate_model()`
response, so users see it before simulating.

### Tool surface (42 total)

- **30 legacy PBPK tools** — flat-parameter API kept for
  back-compat. `run_pbpk_simulation` goes through schema validation
  but accepts the same kwargs.
- **10 session tools** — `register_compound`, `add_binding`,
  `add_clearance`, `add_absorption`, `add_transporters`,
  `select_model_structure`, `validate_model`, `simulate_validated`,
  `session_summary`, `audit_model_provenance`. Use these for
  fabrication-resistant workflows.
- **2 citation tools** — `verify_citation`, `verify_citation_list`.
  Call before inserting any PMID/DOI into a Source field.

## Input validation (`core/validation.py`)

`run_pbpk_simulation` rejects or warns on the following silent-failure
patterns. When adding new tools or refactoring, preserve these checks
or factor them through the same module:

**Hard errors (raise ValueError before any computation):**
- Invalid `kp_method` string (suggests closest match — `poulin-theil` →
  `poulin_theil`)
- `clearance_source` mismatched against the IVIVE field provided
  (e.g. `clearance_source="hlm"` with only `CLint_vitro_hep` given)
- Invalid `distribution_model` or `route` enum

**Soft warnings (run simulation, surface in output `⚠️` block):**
- Library compound matched but user supplied custom physicochemical
  params — the library values win, custom values are silently dropped.
  Tell the user to use a non-library `name=` or edit `COMPOUND_LIBRARY`.
- Transporter parameters provided with default `perfusion_limited`
  distribution model (transporters only fire in `permeability_limited`).
- Sentinel defaults (`fu_p=1.0`, `R_bp=1.0`) on a custom compound.
- Zero hepatic + zero renal clearance.
- `compound_type="neutral"` with `fu_p<0.01` (almost certainly an acid).

The split between hard errors and soft warnings is intentional: hard
errors fire when the simulation cannot be physically meaningful;
warnings fire when it runs but probably isn't what the user intended.

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
