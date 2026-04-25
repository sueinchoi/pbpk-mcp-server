"""
PBPK MCP Server — Whole-body physiologically-based pharmacokinetic modeling.

Architecture: PK-Sim / Simcyp style
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP
from tools.pbpk_tools import register_pbpk_tools
from prompts.user_guide import format_user_guide, count_all_parameters

mcp = FastMCP(
    "pbpk",
    instructions=(
        "Whole-body PBPK modeling server with 7 Kp methods, ACAT absorption, "
        "IVIVE pipeline, DDI prediction, population simulation, transporters, "
        "and PKSimDB integration. 30 tools available."
    ),
)

register_pbpk_tools(mcp)


# --- Resources ---
@mcp.resource("pbpk://status")
def get_status() -> str:
    counts = count_all_parameters()
    return f"""# PBPK MCP Server v1.6

## Tools: 30 | Parameters: {counts['total']} configurable
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
    return """You are a PBPK modeling expert. When a user wants to simulate a drug:

1. FIRST: Ask which scenario they have:
   - Drug name only → look up properties with drug_properties tool
   - Drug name + in vitro data → collect CLint, fm, Peff
   - Full parameter set → proceed directly

2. COLLECT in this order:
   a) Tier 1 (MUST HAVE): name, MW, dose, route
   b) Tier 2 (SHOULD HAVE): logP, pKa, fu_p, compound_type, clearance source
   c) Auto-predict what's missing: R_bp, Kp, fu_inc, GFR, gut CLint

3. CLEARANCE — ask which data they have:
   - HLM CLint (µL/min/mg) → use clearance_source="hlm"
   - Hepatocyte CLint → use clearance_source="hepatocyte"
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
