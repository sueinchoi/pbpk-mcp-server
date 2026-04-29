"""
In Vitro to In Vivo Extrapolation (IVIVE) pipeline.

Converts in vitro clearance measurements to in vivo intrinsic clearance:
  1. Microsomal CLint scaling (HLM → in vivo)
  2. Hepatocyte CLint scaling
  3. Recombinant CYP CLint scaling (ISEF/RAF)
  4. UGT CLint corrections

References:
  - Obach RS. Drug Metab Dispos 1999;27:1350-1359
  - Houston JB. Biochem Pharmacol 1994;47:1469-1479
  - Proctor NJ et al. Xenobiotica 2004;34:151-178
  - Barter ZE et al. Curr Drug Metab 2007;8:33-45
"""

import math
from dataclasses import dataclass
from typing import Optional

from .tissue_binding import predict_fu_inc


# ===================================================================
# Constants
# ===================================================================

# Microsomal protein per gram liver (mg/g)
MPPGL_ADULT = 40.0       # Barter et al. 2007 meta-analysis mean = 39.8 mg/g
MPPGL_ADULT_SD = 11.5    # ~29% CV (Barter 2007)

# Hepatocellularity (10^6 cells/g liver)
HPGL_ADULT = 99.0        # Wilson et al. 2003 (mean = 99 × 10^6 cells/g)
HPGL_SIMCYP = 120.0      # Older convention (Bayliss 1999)

# Liver weight defaults (g)
LIVER_WEIGHT_MALE_PER_KG = 25.7    # g/kg BW
LIVER_WEIGHT_FEMALE_PER_KG = 24.6

# CYP abundances in liver microsomes (pmol/mg protein)
# CYP abundances in liver microsomes (pmol/mg protein)
# Primary source: Rodrigues AD. Biochem Pharmacol 1999;57:465-480
# CYP3A4 range 60-200; using Rodrigues mean. For proteomic data see Achour 2014.
CYP_ABUNDANCE = {
    "CYP1A2": 52.0,    # Rodrigues 1999
    "CYP2B6": 17.0,    # Achour 2014
    "CYP2C8": 24.0,    # Rodrigues 1999 (not 64; Achour 2014 reports 15-64 range)
    "CYP2C9": 96.0,    # Rodrigues 1999
    "CYP2C19": 14.0,   # Rodrigues 1999
    "CYP2D6": 10.0,    # Rodrigues 1999
    "CYP2E1": 49.0,    # Rodrigues 1999
    "CYP3A4": 108.0,   # Rodrigues 1999 (not 137; proteomic range 93-137)
    "CYP3A5": 1.0,     # *3/*3 genotype (most Caucasians)
}

# CYP abundances in gut (enterocytes, nmol total)
CYP_ABUNDANCE_GUT = {
    "CYP3A4": 70.0,   # nmol total in small intestine (Paine 2006)
    "CYP3A5": 5.0,
    "CYP2C9": 3.0,
    "CYP2C19": 1.0,
    "CYP2D6": 0.5,
}

# UGT abundances in liver microsomes (pmol/mg protein)
UGT_ABUNDANCE = {
    "UGT1A1": 26.0,
    "UGT1A3": 8.0,
    "UGT1A4": 18.0,
    "UGT1A6": 17.0,
    "UGT1A9": 21.0,
    "UGT2B7": 86.0,
    "UGT2B15": 10.0,
}

# Default Inter-System Extrapolation Factors (ISEF)
DEFAULT_ISEF = {
    "CYP1A2": 1.0,
    "CYP2B6": 1.0,
    "CYP2C8": 1.0,
    "CYP2C9": 1.0,
    "CYP2C19": 1.0,
    "CYP2D6": 0.4,   # Typically lower for CYP2D6
    "CYP2E1": 1.0,
    "CYP3A4": 1.0,
    "CYP3A5": 1.0,
}


# ===================================================================
# Microsomal CLint scaling
# ===================================================================

def scale_microsomal_clint(
    clint_vitro: float,
    fu_inc: Optional[float] = None,
    logP: Optional[float] = None,
    protein_conc: float = 1.0,
    mppgl: float = MPPGL_ADULT,
    liver_weight_g: Optional[float] = None,
    body_weight: float = 73.0,
    sex: str = "male",
) -> dict:
    """
    Scale microsomal CLint (in vitro) to in vivo CLint.

    CLint_in_vivo = (CLint_vitro / fu_inc) * MPPGL * liver_weight

    Args:
        clint_vitro: Measured microsomal CLint (uL/min/mg protein).
        fu_inc: Fraction unbound in incubation. If None, predicted from logP.
        logP: logP for fu_inc prediction (required if fu_inc is None).
        protein_conc: Microsomal protein concentration (mg/mL).
        mppgl: Microsomal protein per gram liver (mg/g).
        liver_weight_g: Total liver weight (g). If None, from body_weight.
        body_weight: Body weight (kg).
        sex: "male" or "female".

    Returns:
        Dict with CLint_in_vivo (L/h), fu_inc, scaling_factor, etc.
    """
    fu_inc_source = "measured"
    if fu_inc is None:
        if logP is None:
            raise ValueError("Either fu_inc or logP must be provided")
        if logP == 0.0:
            # Sentinel logP from a tool default — Hallifax/Austin prediction
            # from logP=0 yields fu_inc≈1.0, masking high-binding compounds.
            raise ValueError(
                "scale_microsomal_clint: fu_inc is None AND logP=0.0 (sentinel "
                "default). Predicting fu_inc from logP=0 silently assumes ~100% "
                "unbound, which is wrong for most lipophilic drugs and shifts "
                "CLint_in_vivo by >2x. Provide a measured fu_inc OR a real logP."
            )
        from .compound import CompoundSpec, CompoundType
        dummy = CompoundSpec(name="tmp", mw=300, logP=logP, pKa=7.0,
                            fu_p=1.0, compound_type=CompoundType.NEUTRAL)
        fu_inc = predict_fu_inc(dummy, protein_conc)
        fu_inc_source = "predicted_from_logP"

    if liver_weight_g is None:
        factor = LIVER_WEIGHT_MALE_PER_KG if sex == "male" else LIVER_WEIGHT_FEMALE_PER_KG
        liver_weight_g = body_weight * factor

    # CLint_vitro is in uL/min/mg protein
    # Convert to L/h: * mppgl * LW_g / 1e6 * 60
    clint_unbound = clint_vitro / fu_inc if fu_inc > 0 else clint_vitro
    clint_in_vivo = clint_unbound * mppgl * liver_weight_g * 60.0 / 1e6

    return {
        "CLint_in_vivo_L_per_h": clint_in_vivo,
        "CLint_vitro_uL_min_mg": clint_vitro,
        "CLint_unbound_uL_min_mg": clint_unbound,
        "fu_inc": fu_inc,
        "fu_inc_source": fu_inc_source,
        "MPPGL": mppgl,
        "liver_weight_g": liver_weight_g,
        "scaling_factor": mppgl * liver_weight_g * 60.0 / 1e6,
    }


# ===================================================================
# Hepatocyte CLint scaling
# ===================================================================

def scale_hepatocyte_clint(
    clint_hep: float,
    fu_hep: Optional[float] = None,
    logP: Optional[float] = None,
    hpgl: float = HPGL_ADULT,
    liver_weight_g: Optional[float] = None,
    body_weight: float = 73.0,
    sex: str = "male",
) -> dict:
    """
    Scale hepatocyte CLint (in vitro) to in vivo CLint.

    CLint_in_vivo = (CLint_hep / fu_hep) * HPGL * liver_weight

    Args:
        clint_hep: Measured hepatocyte CLint (uL/min/10^6 cells).
        fu_hep: Fraction unbound in hepatocyte incubation. If None, ≈ fu_inc.
        logP: logP for fu_hep prediction.
        hpgl: Hepatocellularity (10^6 cells/g liver).
        liver_weight_g: Total liver weight (g).
        body_weight, sex: For liver weight estimation.

    Returns:
        Dict with CLint_in_vivo (L/h) and intermediate values.
    """
    fu_hep_source = "measured"
    if fu_hep is None:
        if logP is None:
            # Refuse-to-default: silently assuming fu_hep=1.0 yields the
            # uncorrected CLint and looks like a successful IVIVE run.
            raise ValueError(
                "scale_hepatocyte_clint: fu_hep is None AND logP is None. "
                "Falling back to fu_hep=1.0 silently produces an uncorrected "
                "in-vivo CLint (no binding correction), which is wrong for "
                "lipophilic drugs by >2x. Provide a measured fu_hep OR a "
                "real logP for prediction."
            )
        if logP == 0.0:
            raise ValueError(
                "scale_hepatocyte_clint: fu_hep is None AND logP=0.0 "
                "(sentinel default). Austin/Hallifax prediction from logP=0 "
                "yields fu_hep≈1.0, masking high-binding compounds. Provide "
                "a measured fu_hep OR a real logP."
            )
        from .compound import CompoundSpec, CompoundType
        dummy = CompoundSpec(name="tmp", mw=300, logP=logP, pKa=7.0,
                            fu_p=1.0, compound_type=CompoundType.NEUTRAL)
        fu_hep = predict_fu_inc(dummy, 1.0)  # approximation
        fu_hep_source = "predicted_from_logP"

    if liver_weight_g is None:
        factor = LIVER_WEIGHT_MALE_PER_KG if sex == "male" else LIVER_WEIGHT_FEMALE_PER_KG
        liver_weight_g = body_weight * factor

    clint_unbound = clint_hep / fu_hep if fu_hep > 0 else clint_hep
    clint_in_vivo = clint_unbound * hpgl * liver_weight_g * 60.0 / 1e6

    return {
        "CLint_in_vivo_L_per_h": clint_in_vivo,
        "CLint_hep_uL_min_1e6cells": clint_hep,
        "fu_hep": fu_hep,
        "fu_hep_source": fu_hep_source,
        "HPGL": hpgl,
        "liver_weight_g": liver_weight_g,
        "scaling_factor": hpgl * liver_weight_g * 60.0 / 1e6,
    }


# ===================================================================
# Recombinant CYP CLint scaling (ISEF approach)
# ===================================================================

def scale_recombinant_clint(
    clint_per_cyp: dict[str, float],
    isef: Optional[dict[str, float]] = None,
    mppgl: float = MPPGL_ADULT,
    liver_weight_g: Optional[float] = None,
    body_weight: float = 73.0,
    sex: str = "male",
) -> dict:
    """
    Scale recombinant CYP CLint to in vivo using ISEF approach.

    CLint_in_vivo = Σ(CLint_rCYPj * ISEFj * abundance_j) * MPPGL * LW

    Args:
        clint_per_cyp: {CYP_name: CLint in uL/min/pmol CYP}
        isef: {CYP_name: ISEF value}. Defaults used if None.
        mppgl, liver_weight_g, body_weight, sex: Scaling parameters.

    Returns:
        Dict with total CLint_in_vivo and per-CYP contributions.
    """
    if isef is None:
        isef = DEFAULT_ISEF

    if liver_weight_g is None:
        factor = LIVER_WEIGHT_MALE_PER_KG if sex == "male" else LIVER_WEIGHT_FEMALE_PER_KG
        liver_weight_g = body_weight * factor

    per_cyp = {}
    total_clint_per_mg = 0.0

    for cyp, clint_pmol in clint_per_cyp.items():
        abundance = CYP_ABUNDANCE.get(cyp, 0.0)
        isef_val = isef.get(cyp, 1.0)

        # CLint contribution per mg protein
        clint_mg = clint_pmol * isef_val * abundance  # uL/min/mg protein
        total_clint_per_mg += clint_mg

        per_cyp[cyp] = {
            "CLint_rCYP": clint_pmol,
            "ISEF": isef_val,
            "abundance_pmol_per_mg": abundance,
            "CLint_per_mg": clint_mg,
            "fraction": 0.0,  # filled below
        }

    # Scale to whole liver
    clint_in_vivo = total_clint_per_mg * mppgl * liver_weight_g * 60.0 / 1e6

    # Fill fractions
    for cyp in per_cyp:
        if total_clint_per_mg > 0:
            per_cyp[cyp]["fraction"] = per_cyp[cyp]["CLint_per_mg"] / total_clint_per_mg

    return {
        "CLint_in_vivo_L_per_h": clint_in_vivo,
        "CLint_per_mg_total": total_clint_per_mg,
        "per_CYP": per_cyp,
        "MPPGL": mppgl,
        "liver_weight_g": liver_weight_g,
    }


# ===================================================================
# MPPGL maturation (pediatric)
# ===================================================================

def mppgl_pediatric(age_years: float) -> float:
    """
    MPPGL for pediatric subjects.
    Barter et al. 2007: log10(MPPGL) = 0.8016 + 0.2743 * log10(age_years)
    """
    if age_years <= 0:
        return 10.0  # neonatal estimate (~10 mg/g)
    import math
    mppgl = 10.0 ** (0.8016 + 0.2743 * math.log10(age_years))
    return min(mppgl, MPPGL_ADULT)


def cyp_maturation(cyp_name: str, age_years: float) -> float:
    """
    CYP ontogeny (fraction of adult level) using Hill function.

    f_mat = Fmax * age^n / (TM50^n + age^n)

    Returns fraction of adult abundance (0 to 1).
    """
    params = {
        # CYP: (TM50_years, Hill_n, Fmax)
        "CYP3A4": (0.7, 2.1, 1.0),
        "CYP3A7": (0.1, 1.5, 0.0),   # decreases (inverted)
        "CYP1A2": (1.2, 2.0, 1.0),
        "CYP2D6": (0.2, 2.5, 1.0),
        "CYP2C9": (0.5, 2.0, 1.0),
        "CYP2C19": (0.5, 2.0, 1.0),
        "CYP2E1": (0.3, 2.0, 1.0),
        "UGT1A1": (2.5, 1.5, 1.0),
        "UGT2B7": (0.5, 2.0, 1.0),
    }

    if cyp_name not in params:
        return 1.0

    tm50, n, fmax = params[cyp_name]

    if cyp_name == "CYP3A7":
        # CYP3A7 decreases with age (predominant in neonates)
        return max(1.0 - age_years ** n / (tm50 ** n + age_years ** n), 0.0)

    if age_years <= 0:
        return 0.0

    f_mat = fmax * age_years ** n / (tm50 ** n + age_years ** n)
    return min(f_mat, 1.0)


# ===================================================================
# Formatting
# ===================================================================

def format_ivive_result(result: dict, method: str = "microsomal") -> str:
    """Format IVIVE result as markdown."""
    lines = [f"## IVIVE Result ({method.capitalize()} Scaling)\n"]
    lines.append(f"**CLint in vivo = {result['CLint_in_vivo_L_per_h']:.2f} L/h**\n")

    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")

    for key, val in result.items():
        if key == "per_CYP":
            continue
        if isinstance(val, float):
            lines.append(f"| {key} | {val:.4g} |")

    if "per_CYP" in result:
        lines.append("\n### Per-CYP Contributions\n")
        lines.append("| CYP | CLint/pmol | ISEF | Abundance | fm |")
        lines.append("|-----|-----------|------|-----------|-----|")
        for cyp, data in result["per_CYP"].items():
            lines.append(
                f"| {cyp} | {data['CLint_rCYP']:.3f} | {data['ISEF']:.2f} | "
                f"{data['abundance_pmol_per_mg']:.0f} | {data['fraction']:.3f} |"
            )

    return "\n".join(lines)
