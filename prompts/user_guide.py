"""
PBPK MCP User Input Guide — Interactive parameter collection workflow.

When a user starts a PBPK simulation, Claude should guide them through
3 tiers of input collection:

  Tier 1 (REQUIRED): Without these, simulation cannot run
  Tier 2 (RECOMMENDED): Greatly improves accuracy
  Tier 3 (OPTIONAL): Fine-tuning for advanced users
"""


WELCOME_PROMPT = """# PBPK Simulation Setup

I'll help you run a whole-body PBPK simulation. Let me collect the necessary information.

## What do you have?

1. **Drug name only** → I'll look up properties from ChEMBL/database
2. **Drug name + in vitro data** → Most common scenario
3. **Full parameter set** → You have everything ready
4. **PK-Sim model file** → Import from .pkml/.xml

Which scenario fits you?

## Before I start: measurement audit

For the parameters below, **a measured value (when available) is always
preferred over a literature default or model prediction.** I will ask
you about each one explicitly. If you don't have a measurement, I will
fall back to literature consensus or in silico prediction and mark the
source clearly in the final parameter table (M / L / P).

**Priority-1 (most impactful — predictions diverge widely):**
- `fu_hep` (hepatocyte unbound fraction) — RED assay; affects CL by ≥2×
- `fu_inc` (HLM unbound fraction) — RED assay
- `R_bp` (blood:plasma ratio) — Bp/p assay; affects all Kp_blood values
- `Peff` or Caco-2 `Papp` — Caco-2 / PAMPA; determines Fg

**Priority-2:**
- Tissue Kp (rat tissue distribution) — supersedes any Kp method
- `ka` (absorption rate) — from oral concentration-time data
- EHC parameters (`CL_bile`, deconjugation rate, ...) — from bile-cannulation studies

Tell me which of these you have measured for your compound."""


TIER1_REQUIRED = {
    "header": "## Tier 1: Required Parameters (simulation cannot run without these)",
    "params": [
        {
            "name": "Drug name",
            "key": "name",
            "unit": "",
            "ask": "What is the drug name?",
            "example": "Midazolam",
            "auto": "Can look up MW, logP from ChEMBL if name is known",
        },
        {
            "name": "Molecular weight",
            "key": "mw",
            "unit": "g/mol",
            "ask": "What is the molecular weight?",
            "example": "325.8",
            "auto": "Auto from ChEMBL/PubChem by drug name",
        },
        {
            "name": "Dose",
            "key": "dose_mg",
            "unit": "mg",
            "ask": "What dose are you simulating?",
            "example": "5",
        },
        {
            "name": "Route",
            "key": "route",
            "unit": "",
            "ask": "Administration route? (oral / iv_bolus / iv_infusion)",
            "example": "oral",
        },
    ],
}

TIER2_RECOMMENDED = {
    "header": "## Tier 2: Recommended Parameters (greatly improves accuracy)",
    "sections": {
        "Physicochemical": [
            {"name": "logP", "key": "logP", "unit": "", "ask": "Octanol:water logP?",
             "why": "Determines tissue distribution (Kp)", "auto": "Predicted from SMILES via SwissADME"},
            {"name": "pKa", "key": "pKa", "unit": "", "ask": "Dissociation constant (pKa)?",
             "why": "Determines ionization → affects Kp and absorption", "auto": "Predicted via ChemAxon"},
            {"name": "Compound type", "key": "compound_type", "unit": "",
             "ask": "Is this an acid, base, or neutral? (acid/strong_base/moderate_base/weak_base/neutral/zwitterion)",
             "why": "Selects correct Kp equation (R&R Type 1 vs 2)", "auto": "Determined from pKa + structure"},
            {"name": "fu_p", "key": "fu_p", "unit": "fraction",
             "ask": "Fraction unbound in plasma? (0-1)", "why": "Critical for CL and Vd",
             "auto": "Predicted from logP if not measured (less accurate)"},
            {"name": "R_bp", "key": "R_bp", "unit": "ratio",
             "ask": "Blood:plasma ratio?", "why": "Converts plasma to blood concentration",
             "auto": "Predicted from RBC partitioning if not measured"},
        ],
        "Clearance (choose one source)": [
            {"name": "HLM CLint", "key": "CLint_vitro_hlm", "unit": "µL/min/mg",
             "ask": "Human liver microsomal intrinsic clearance?",
             "why": "Most common in vitro CL source → IVIVE scaled to in vivo",
             "group": "clearance"},
            {"name": "Hepatocyte CLint", "key": "CLint_vitro_hep", "unit": "µL/min/10⁶ cells",
             "ask": "Hepatocyte intrinsic clearance?",
             "why": "Better than HLM (includes Phase II + transporter)", "group": "clearance"},
            {"name": "rCYP CLint", "key": "CLint_per_cyp", "unit": "µL/min/pmol",
             "ask": "Recombinant CYP CLint per enzyme? (e.g., CYP3A4:0.15,CYP2D6:0.08)",
             "why": "Most mechanistic — gives fm per CYP automatically", "group": "clearance"},
            {"name": "Direct CL_int", "key": "CL_int", "unit": "L/h",
             "ask": "In vivo intrinsic clearance?",
             "why": "If already scaled or from clinical data", "group": "clearance"},
            {"name": "Renal clearance", "key": "CL_renal", "unit": "L/h",
             "ask": "Renal clearance? (0 if negligible)",
             "why": "For renally eliminated drugs. Default: GFR × fu_p", "group": "clearance"},
        ],
        "CYP Fraction Metabolized": [
            {"name": "fm per CYP", "key": "fm_per_cyp", "unit": "fraction",
             "ask": "Fraction metabolized per CYP? (e.g., CYP3A4:0.8,CYP2C9:0.15,UGT:0.05)",
             "why": "Used for: (1) DDI prediction, (2) auto gut CLint derivation, (3) rCYP IVIVE",
             "auto": "From reaction phenotyping or chemical inhibition studies"},
        ],
    },
}

TIER3_OPTIONAL = {
    "header": "## Tier 3: Optional Parameters (advanced fine-tuning)",
    "sections": {
        "Absorption (oral only)": [
            {"name": "ka", "key": "ka", "unit": "1/h", "default": 1.0,
             "desc": "First-order absorption rate constant"},
            {"name": "Fa", "key": "Fa", "unit": "fraction", "default": 1.0,
             "desc": "Fraction absorbed from GI tract"},
            {"name": "Fg", "key": "Fg", "unit": "fraction", "default": 1.0,
             "desc": "Fraction escaping gut wall metabolism"},
            {"name": "Peff", "key": "Peff", "unit": "×10⁻⁴ cm/s", "default": None,
             "desc": "Human jejunal permeability (for ACAT model)"},
            {"name": "Caco-2 Papp", "key": "papp_caco2", "unit": "cm/s", "default": None,
             "desc": "Caco-2 permeability → auto-converted to Peff"},
            {"name": "Solubility (S0)", "key": "S0", "unit": "mg/mL", "default": None,
             "desc": "Intrinsic solubility of neutral form (for ACAT dissolution)"},
            {"name": "Particle size", "key": "particle_radius_um", "unit": "µm", "default": 25,
             "desc": "Mean particle radius (affects dissolution rate)"},
            {"name": "Absorption model", "key": "absorption_model", "unit": "",
             "default": "first_order", "desc": "first_order or acat"},
        ],
        "Distribution": [
            {"name": "Kp method", "key": "kp_method", "unit": "", "default": "rodgers_rowland",
             "desc": "rodgers_rowland / lukacova / schmitt / poulin_theil / berezhkovskiy / pksim_standard / kp_membrane"},
            {"name": "Distribution model", "key": "distribution_model", "unit": "",
             "default": "perfusion_limited", "desc": "perfusion_limited or permeability_limited"},
            {"name": "Kp override", "key": "kp_override", "unit": "dict", "default": None,
             "desc": "Manual Kp per organ (e.g., for validated models)"},
        ],
        "Transporters (Km in µM, Vmax in pmol/min)": [
            {"name": "Liver OATP Km/Vmax", "key": "liver_oatp_km/vmax", "default": None,
             "desc": "OATP1B1/1B3 hepatic uptake (statins, sartans)"},
            {"name": "Liver MRP2 Km/Vmax", "key": "liver_mrp2_km/vmax", "default": None,
             "desc": "MRP2/BCRP biliary efflux"},
            {"name": "Kidney OCT2 Km/Vmax", "key": "kidney_oct2_km/vmax", "default": None,
             "desc": "OCT2 renal tubular uptake (metformin, cimetidine)"},
            {"name": "Kidney MATE1 Km/Vmax", "key": "kidney_mate1_km/vmax", "default": None,
             "desc": "MATE1 renal apical efflux"},
            {"name": "Gut P-gp Km/Vmax", "key": "gut_pgp_km/vmax", "default": None,
             "desc": "P-gp intestinal efflux (digoxin, loperamide)"},
        ],
        "Subject / Population": [
            {"name": "Body weight", "key": "body_weight", "unit": "kg", "default": 73,
             "desc": "Subject weight (auto-scales organs and CO)"},
            {"name": "Sex", "key": "sex", "unit": "", "default": "male",
             "desc": "male or female (affects organ volumes, HCT, GFR)"},
            {"name": "Age", "key": "age", "unit": "years", "default": 30,
             "desc": "Affects GFR (aging/maturation), organ sizes"},
            {"name": "Population sim", "key": "n_individuals", "unit": "", "default": None,
             "desc": "Run N virtual individuals (10-500) for variability"},
        ],
        "Dosing Regimen": [
            {"name": "Number of doses", "key": "n_doses", "unit": "", "default": 1,
             "desc": "For multiple dosing (QD, BID, TID)"},
            {"name": "Dosing interval", "key": "interval_h", "unit": "h", "default": 24,
             "desc": "Hours between doses (12=BID, 8=TID, 24=QD)"},
            {"name": "Infusion duration", "key": "infusion_duration_h", "unit": "h", "default": 0.5,
             "desc": "For IV infusion route"},
            {"name": "Simulation duration", "key": "duration_h", "unit": "h", "default": 24,
             "desc": "Total simulation time"},
        ],
        "DDI": [
            {"name": "Ki", "key": "Ki", "unit": "µM", "default": None,
             "desc": "Reversible CYP inhibition constant"},
            {"name": "KI", "key": "KI", "unit": "µM", "default": None,
             "desc": "Mechanism-based inhibition concentration"},
            {"name": "kinact", "key": "kinact", "unit": "1/h", "default": None,
             "desc": "MBI maximum inactivation rate"},
            {"name": "Emax", "key": "Emax", "unit": "fold", "default": None,
             "desc": "CYP induction maximum fold-change"},
            {"name": "EC50", "key": "EC50", "unit": "µM", "default": None,
             "desc": "Induction half-maximal concentration"},
            {"name": "Inhibitor concentration", "key": "I_h_u", "unit": "µM", "default": None,
             "desc": "Unbound inhibitor/inducer at liver"},
        ],
        "Special Features": [
            {"name": "EHC", "key": "enable_ehc", "unit": "bool", "default": False,
             "desc": "Enterohepatic recirculation (mycophenolate, sorafenib)"},
            {"name": "Circadian", "key": "enable_circadian", "unit": "bool", "default": False,
             "desc": "Circadian CYP expression variation"},
            {"name": "Disease state", "key": "disease", "unit": "", "default": None,
             "desc": "ckd_moderate / ckd_severe / hepatic_mild_A / hepatic_severe_C"},
            {"name": "Pregnancy GA", "key": "gestational_age", "unit": "weeks", "default": None,
             "desc": "Gestational age for pregnancy physiology"},
        ],
    },
}


def format_user_guide() -> str:
    """Format the complete user input guide as markdown."""
    lines = [WELCOME_PROMPT, ""]

    # Tier 1
    lines.append(TIER1_REQUIRED["header"])
    lines.append("")
    lines.append("| # | Parameter | Unit | Example | Auto-available? |")
    lines.append("|---|-----------|------|---------|-----------------|")
    for i, p in enumerate(TIER1_REQUIRED["params"], 1):
        auto = p.get("auto", "No")
        lines.append(f"| {i} | **{p['name']}** | {p['unit']} | {p['example']} | {auto} |")

    lines.append("")

    # Tier 2
    lines.append(TIER2_RECOMMENDED["header"])
    for section, params in TIER2_RECOMMENDED["sections"].items():
        lines.append(f"\n### {section}\n")
        lines.append("| Parameter | Unit | Why needed | Auto? |")
        lines.append("|-----------|------|-----------|-------|")
        for p in params:
            auto = p.get("auto", "No")
            lines.append(f"| {p['name']} | {p.get('unit', '')} | {p['why']} | {auto} |")

    lines.append("")

    # Tier 3
    lines.append(TIER3_OPTIONAL["header"])
    for section, params in TIER3_OPTIONAL["sections"].items():
        lines.append(f"\n### {section}\n")
        lines.append("| Parameter | Unit | Default | Description |")
        lines.append("|-----------|------|---------|-------------|")
        for p in params:
            default = p.get("default", "—")
            if default is None:
                default = "—"
            lines.append(f"| {p['name']} | {p.get('unit', '')} | {default} | {p['desc']} |")

    # Summary
    lines.extend([
        "",
        "---",
        "## Auto-Predicted from Your Input",
        "",
        "These are calculated automatically — you do NOT need to provide them:",
        "",
        "| Predicted | From | Method |",
        "|-----------|------|--------|",
        "| Kp (13 organs) | logP, pKa, fu_p | Rodgers-Rowland / 6 other methods |",
        "| fu_tissue, fu_int, fu_cell | Kp, albumin ratio | Schmitt / R&R |",
        "| R_bp | logP, pKa, fu_p | RBC partitioning |",
        "| fu_inc | logP | Austin 2002 / Hallifax 2006 |",
        "| CL_int (in vivo) | CLint_vitro + fu_inc | IVIVE scaling |",
        "| Gut CLint per CYP | Liver CLint + fm | Gut/liver CYP ratio |",
        "| GFR | Age, weight, sex | Rhodin 2009 + renal aging |",
        "| Organ volumes | Body weight, sex | ICRP 89 |",
        "| Blood flows | Cardiac output | Williams & Leggett 1989 |",
        "| Fa (if ACAT) | Peff, solubility | 9-segment ACAT model |",
        "| Fg (if ACAT) | Gut CLint, fu_gut | Qgut model |",
    ])

    return "\n".join(lines)


# Total parameter count
def count_all_parameters() -> dict:
    """Count parameters by tier."""
    t1 = len(TIER1_REQUIRED["params"])
    t2 = sum(len(v) for v in TIER2_RECOMMENDED["sections"].values())
    t3 = sum(len(v) for v in TIER3_OPTIONAL["sections"].values())
    return {"tier1_required": t1, "tier2_recommended": t2, "tier3_optional": t3,
            "total": t1 + t2 + t3}
