"""
Disease state models for PBPK parameter adjustment.

Disease states modify physiological parameters as multipliers of healthy baseline:
  1. CKD (Chronic Kidney Disease) — eGFR-based staging
  2. Hepatic Impairment — Child-Pugh score-based
  3. Obesity — BMI-based adjustments

References:
  - Malik PRV et al. Clin Pharmacol Ther 2020;107:1209-1220 (CKD PBPK)
  - Johnson TN et al. Clin Pharmacokinet 2010;49:189-206 (hepatic impairment)
  - PK-Sim PKSimDB disease state parameterizations
  - FDA Guidance: Pharmacokinetics in Patients with Impaired Renal/Hepatic Function (2020)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CKDStage(str, Enum):
    NORMAL = "normal"           # eGFR >= 90
    MILD = "mild"               # eGFR 60-89 (Stage 2)
    MODERATE = "moderate"       # eGFR 30-59 (Stage 3)
    SEVERE = "severe"           # eGFR 15-29 (Stage 4)
    ESRD = "esrd"               # eGFR < 15  (Stage 5)


class ChildPugh(str, Enum):
    NORMAL = "normal"     # Score 5-6
    MILD = "mild_A"       # Score 5-6 (A)
    MODERATE = "moderate_B"  # Score 7-9 (B)
    SEVERE = "severe_C"   # Score 10-15 (C)


# ===================================================================
# CKD parameter multipliers (relative to healthy)
# Source: Malik 2020, Nolin 2008, PK-Sim CKD module
# ===================================================================

CKD_MULTIPLIERS = {
    CKDStage.NORMAL: {
        "GFR": 1.0, "fu_p": 1.0, "hematocrit": 1.0,
        "CYP3A4": 1.0, "CYP2C9": 1.0, "CYP2D6": 1.0,
        "CYP1A2": 1.0, "CYP2C19": 1.0,
        "gastric_pH": 1.0, "GI_transit": 1.0,
        "kidney_volume": 1.0, "cardiac_output": 1.0,
        "plasma_albumin": 1.0, "plasma_AAG": 1.0,
    },
    CKDStage.MILD: {
        "GFR": 0.75, "fu_p": 1.05, "hematocrit": 0.97,
        "CYP3A4": 1.0, "CYP2C9": 0.95, "CYP2D6": 1.0,
        "CYP1A2": 0.95, "CYP2C19": 1.0,
        "gastric_pH": 1.0, "GI_transit": 1.0,
        "kidney_volume": 0.95, "cardiac_output": 1.0,
        "plasma_albumin": 0.97, "plasma_AAG": 1.05,
    },
    CKDStage.MODERATE: {
        "GFR": 0.37, "fu_p": 1.15, "hematocrit": 0.88,
        "CYP3A4": 0.85, "CYP2C9": 0.80, "CYP2D6": 0.90,
        "CYP1A2": 0.80, "CYP2C19": 0.85,
        "gastric_pH": 1.10, "GI_transit": 1.10,
        "kidney_volume": 0.80, "cardiac_output": 0.95,
        "plasma_albumin": 0.90, "plasma_AAG": 1.20,
    },
    CKDStage.SEVERE: {
        "GFR": 0.18, "fu_p": 1.30, "hematocrit": 0.78,
        "CYP3A4": 0.70, "CYP2C9": 0.65, "CYP2D6": 0.80,
        "CYP1A2": 0.65, "CYP2C19": 0.70,
        "gastric_pH": 1.20, "GI_transit": 1.15,
        "kidney_volume": 0.65, "cardiac_output": 0.90,
        "plasma_albumin": 0.80, "plasma_AAG": 1.40,
    },
    CKDStage.ESRD: {
        "GFR": 0.08, "fu_p": 1.50, "hematocrit": 0.68,
        "CYP3A4": 0.60, "CYP2C9": 0.50, "CYP2D6": 0.70,
        "CYP1A2": 0.50, "CYP2C19": 0.60,
        "gastric_pH": 1.30, "GI_transit": 1.20,
        "kidney_volume": 0.50, "cardiac_output": 0.85,
        "plasma_albumin": 0.70, "plasma_AAG": 1.60,
    },
}


# ===================================================================
# Hepatic impairment multipliers (Child-Pugh)
# Source: Johnson 2010, Edginton 2006, PK-Sim HI module
# ===================================================================

HEPATIC_MULTIPLIERS = {
    ChildPugh.NORMAL: {
        "liver_volume": 1.0, "portal_blood_flow": 1.0,
        "hepatic_artery_flow": 1.0, "fu_p": 1.0,
        "hematocrit": 1.0, "cardiac_output": 1.0,
        "CYP3A4": 1.0, "CYP2C9": 1.0, "CYP2D6": 1.0,
        "CYP1A2": 1.0, "CYP2C19": 1.0, "CYP2E1": 1.0,
        "UGT1A1": 1.0, "UGT2B7": 1.0,
        "plasma_albumin": 1.0, "MPPGL": 1.0,
    },
    ChildPugh.MILD: {
        "liver_volume": 0.90, "portal_blood_flow": 0.85,
        "hepatic_artery_flow": 1.10, "fu_p": 1.10,
        "hematocrit": 0.95, "cardiac_output": 1.05,
        "CYP3A4": 0.80, "CYP2C9": 0.80, "CYP2D6": 0.90,
        "CYP1A2": 0.75, "CYP2C19": 0.75, "CYP2E1": 0.80,
        "UGT1A1": 0.85, "UGT2B7": 0.85,
        "plasma_albumin": 0.90, "MPPGL": 0.85,
    },
    ChildPugh.MODERATE: {
        "liver_volume": 0.75, "portal_blood_flow": 0.65,
        "hepatic_artery_flow": 1.20, "fu_p": 1.30,
        "hematocrit": 0.88, "cardiac_output": 1.15,
        "CYP3A4": 0.55, "CYP2C9": 0.55, "CYP2D6": 0.75,
        "CYP1A2": 0.50, "CYP2C19": 0.45, "CYP2E1": 0.55,
        "UGT1A1": 0.65, "UGT2B7": 0.65,
        "plasma_albumin": 0.75, "MPPGL": 0.65,
    },
    ChildPugh.SEVERE: {
        "liver_volume": 0.60, "portal_blood_flow": 0.45,
        "hepatic_artery_flow": 1.30, "fu_p": 1.60,
        "hematocrit": 0.80, "cardiac_output": 1.25,
        "CYP3A4": 0.30, "CYP2C9": 0.30, "CYP2D6": 0.55,
        "CYP1A2": 0.25, "CYP2C19": 0.20, "CYP2E1": 0.30,
        "UGT1A1": 0.40, "UGT2B7": 0.40,
        "plasma_albumin": 0.55, "MPPGL": 0.40,
    },
}


def get_ckd_multipliers(stage: CKDStage) -> dict:
    if isinstance(stage, str):
        stage = CKDStage(stage)
    return CKD_MULTIPLIERS[stage].copy()


def get_hepatic_multipliers(child_pugh: ChildPugh) -> dict:
    if isinstance(child_pugh, str):
        child_pugh = ChildPugh(child_pugh)
    return HEPATIC_MULTIPLIERS[child_pugh].copy()


def apply_disease_to_compound(compound_dict: dict, multipliers: dict) -> dict:
    """Apply disease multipliers to compound parameters."""
    result = compound_dict.copy()
    if "fu_p" in multipliers and "fu_p" in result:
        result["fu_p"] = min(result["fu_p"] * multipliers["fu_p"], 1.0)
    if "CYP3A4" in multipliers and "CL_int" in result:
        result["CL_int"] = result["CL_int"] * multipliers.get("CYP3A4", 1.0)
    return result


def format_disease_profile(disease_type: str, stage: str) -> str:
    """Format disease state multipliers as markdown."""
    # Aliases: CKD-style severity → Child-Pugh enum value
    hepatic_aliases = {
        "mild": "mild_A", "a": "mild_A",
        "moderate": "moderate_B", "b": "moderate_B",
        "severe": "severe_C", "c": "severe_C",
    }
    if disease_type == "ckd":
        mults = get_ckd_multipliers(stage)
        title = f"CKD Stage: {stage}"
    elif disease_type == "hepatic":
        stage_canon = hepatic_aliases.get(stage.lower(), stage)
        mults = get_hepatic_multipliers(stage_canon)
        title = f"Hepatic Impairment: {stage_canon}"
    else:
        return f"Unknown disease type: {disease_type}"

    lines = [
        f"## Disease State — {title}\n",
        "| Parameter | Multiplier (vs healthy) |",
        "|-----------|------------------------|",
    ]
    for param, mult in mults.items():
        direction = "↑" if mult > 1.05 else ("↓" if mult < 0.95 else "→")
        lines.append(f"| {param} | {mult:.2f} {direction} |")

    return "\n".join(lines)
