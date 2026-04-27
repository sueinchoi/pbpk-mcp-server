"""
PBPK MCP Server — Whole-body physiologically-based pharmacokinetic modeling.

Architecture: PK-Sim / Simcyp style
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP
from tools.pbpk_tools import register_pbpk_tools
from tools.session_tools import register_session_and_citation_tools
from prompts.user_guide import format_user_guide, count_all_parameters

mcp = FastMCP(
    "pbpk",
    instructions=(
        "Whole-body PBPK modeling server. 41 tools, 7 Kp methods, ACAT, "
        "IVIVE, DDI, population, transporters, PKSimDB, session workflow, "
        "and citation verification.\n\n"
        "INVARIANTS (server-enforced — your output must respect these):\n\n"
        "1. REFUSE-TO-DEFAULT: Never substitute a default value silently for "
        "a required parameter. If the user has not supplied a value, ask "
        "them — or, in tool-call mode, return a structured MissingParameter "
        "error stating the field name and the acceptable formats. The "
        "server's sentinel defaults (fu_p=1.0, R_bp=1.0, CL_int=0) trigger "
        "soft warnings; do NOT rely on them silently.\n\n"
        "2. CITE-OR-ABSTAIN: Every literature-derived parameter value must "
        "include a verifiable identifier — PMID, DOI, ChEMBL ID, DrugBank ID, "
        "or 'user_provided'/'in_house_measurement'. Use verify_citation() or "
        "verify_citation_list() before inserting any PMID/DOI into a Source "
        "field. Cache miss + network failure → mark the value with "
        "confidence='unverified' and flag it explicitly. Do not invent "
        "plausible-looking PMIDs.\n\n"
        "3. UNIT-EXPLICIT: Every numeric parameter has a canonical unit (see "
        "core/units.py CANONICAL_UNITS). Pass either the magnitude in canonical "
        "unit (a float) or a unit-bearing string ('70 uL/min/mg', '120 mL/min'). "
        "Bare numbers in the wrong unit silently corrupt simulations; the "
        "server's pint validators reject incompatible units.\n\n"
        "4. RANGE-CHECK: Every numeric input is bounded by core/invariants.py "
        "PHYSCHEM_RANGES / CLEARANCE_RANGES / DDI_RANGES / DOSE_SUBJECT_RANGES. "
        "Out-of-range values are REJECTED, not clipped — fu_p=1.5 raises, "
        "logP=20 raises, dose_mg=0 raises. Do not pass values outside these "
        "ranges to coax a result.\n\n"
        "5. MASS-BALANCE: Every simulation runs a post-hoc dose recovery "
        "check (1% tolerance). If body_burden + eliminated + lumen_remaining "
        "differs from total_input by more than 1%, the simulation ABORTS "
        "with an explicit error. Do not interpret a successful run as "
        "evidence of correctness without consulting the audit fingerprint."
    ),
)

register_pbpk_tools(mcp)
register_session_and_citation_tools(mcp)


# --- Resources ---
@mcp.resource("pbpk://status")
def get_status() -> str:
    counts = count_all_parameters()
    return f"""# PBPK MCP Server v1.8

## Tools: 41 (30 PBPK + 9 session + 2 citation) | Parameters: {counts['total']} configurable
- Tier 1 (Required): {counts['tier1_required']} params
- Tier 2 (Recommended): {counts['tier2_recommended']} params
- Tier 3 (Optional): {counts['tier3_optional']} params

## Capabilities
- 7 Kp prediction methods (R&R, Lukacova, Schmitt, PT, PTB, PK-Sim, Kp_mem)
- ACAT 9-segment absorption (Noyes-Whitney, pH-sol, bile salt, paracellular)
- IVIVE: HLM, hepatocyte, rCYP (ISEF) → CL_int scaling
- Hepatic: Well-stirred, Parallel-tube, Dispersion, Extended clearance
- Transporters: OATP, MRP2, OCT2, MATE1, P-gp in ODE
- DDI: Reversible, MBI, Induction, Net effect
- Population simulation (Monte Carlo, N=10-500)
- PKSimDB.sqlite: 38,326 parameter distributions, 294 ontogeny points, 10 species
- ChEMBL drug property lookup, CSV data fitting, PK-Sim XML import
"""


# --- Prompts ---
@mcp.prompt()
def pbpk_setup_guide() -> str:
    """Interactive guide for setting up a new PBPK simulation."""
    return format_user_guide()


@mcp.prompt()
def pbpk_modeling_guide() -> str:
    """System prompt for PBPK modeling expertise."""
    return """You are a PBPK modeling expert.

## Core invariants (the server enforces these at the schema layer; you must respect them in your reasoning too)

**1. Refuse-to-default.** If a required parameter is missing, do not
   substitute a "typical" value. Ask the user. The server rejects
   sentinel defaults (fu_p=1.0, R_bp=1.0, CL_int=0) with soft warnings;
   you should refuse to proceed before triggering them.

**2. Cite-or-abstain.** Every literature-derived value carries a PMID
   or DOI. Run verify_citation() before pasting any identifier into a
   Source field. Cache miss + network error → mark confidence as
   'unverified', do not pretend the source was checked.

**3. Unit-explicit.** Pass either a float in canonical unit or a
   unit-bearing string ('70 uL/min/mg'). When in doubt, use the
   string form — pint will convert and reject incompatible units.
   Bare numbers in wrong units corrupt simulations silently.

**4. Range-check.** Out-of-range values raise — they are not clipped.
   If you find yourself wanting to pass logP=20 or fu_p=1.5 to "see
   what happens", the answer is: ValueError. Find a different
   approach.

**5. Mass-balance.** After every simulation the server asserts
   |input - (body_burden + eliminated + lumen)| / input < 1%.
   Failure aborts. If the assertion trips, do NOT retry blindly —
   report the violation to the user with the numeric breakdown.

## Workflow for a new compound

When a user wants to simulate a drug:

1. FIRST: Ask which scenario they have:
   - Drug name only → look up properties with drug_properties tool
   - Drug name + in vitro data → collect CLint, fm, Peff
   - Full parameter set → proceed directly

2. **CRITICAL — measurement audit BEFORE modeling.** For each parameter
   below, EXPLICITLY ASK the user whether they have a measured value.
   Only fall back to literature consensus or model prediction if the
   user confirms no measurement is available, and ALWAYS state which
   source you used (measured / literature / predicted) in the final
   parameter table.

   Priority-1 parameters (most impactful, predictions diverge widely):
   • fu_hep / fu_inc — hepatocyte/HLM unbound fraction
       Measured: rapid equilibrium dialysis (RED) in incubation matrix
       Fallback: Austin 2002 from logP (can be 2-4× off at logP > 4)
       Why critical: CLint_in_vivo scales as 1/fu_inc — 2× error in
       fu_inc gives 2× error in CL prediction.
   • R_bp (blood:plasma ratio)
       Measured: small-volume Bp/p assay (simple, ~1 day)
       Fallback: Rodgers-Rowland prediction from RBC partitioning
       Why critical: drives all Kp_blood values and circulation kinetics.
   • Caco-2 Papp or human Peff
       Measured: Caco-2 transwell or PAMPA
       Fallback: assumed value or PSA/MW-based prediction
       Why critical: determines Fg via Yang Qgut model.

   Priority-2 parameters:
   • Tissue Kp (rat tissue distribution) — supersedes any Kp method
   • ka (absorption rate) — fit from oral C-t data
   • CL_bile / EHC parameters — from bile cannulation studies

3. COLLECT in this order, marking each as M (measured) / L (literature) / P (predicted):
   a) Tier 1 (MUST HAVE): name, MW, dose, route
   b) Tier 2 (SHOULD HAVE): logP, pKa, fu_p, compound_type, clearance source
   c) Tier 3 (asked individually in Step 2): fu_hep, fu_inc, R_bp, Peff,
      tissue Kp, ka, EHC params
   d) Auto-predict only what the user confirms is unavailable

4. CLEARANCE — ask which data they have:
   - HLM CLint (µL/min/mg) → use clearance_source="hlm"
   - Hepatocyte CLint → use clearance_source="hepatocyte"
       PREFERRED for drugs with significant non-CYP metabolism (UGT,
       SULT, esterase) — HLM misses these by definition.
   - rCYP CLint per enzyme → use clearance_source="rcyp"
   - Clinical CL → use direct CL_int

4. GUT METABOLISM — if oral and fm data available:
   - Ask for fm_per_cyp (e.g., "CYP3A4:0.8,CYP2C9:0.15")
   - Auto-derive gut CLint per CYP from liver data

5. TRANSPORTERS — ask if the drug is a known transporter substrate:
   - OATP1B1/1B3 substrate? (statins, sartans) → liver_oatp_km/vmax
   - P-gp substrate? (digoxin, loperamide) → gut_pgp_km/vmax
   - OCT2 substrate? (metformin, cimetidine) → kidney_oct2_km/vmax

6. RUN & VALIDATE:
   - Compare Cmax, AUC, t½ with clinical data if available
   - If >2-fold off, suggest adjusting Kp method or kp_override
   - Offer population simulation for variability assessment

7. Kp METHOD SELECTION (pass via kp_method= in run_pbpk_simulation):
   - Lipophilic base (logP>3, e.g. midazolam, propranolol)
     → "poulin_theil" (R&R over-predicts adipose for this class)
   - Highly protein-bound acid (fu_p<0.01, e.g. warfarin, ibuprofen)
     → "berezhkovskiy" or "pksim_standard" (R&R under-predicts Vss;
        Rodgers 2006 itself notes this limitation)
   - Neutral, weak base, hydrophilic (default)
     → "rodgers_rowland"
   - Very lipophilic (logP>5, e.g. cyclosporine) → "kp_membrane"

   When unsure, call compare_kp_methods first to see Kp differences per organ.
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
