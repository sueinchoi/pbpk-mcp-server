"""
Multi-species physiological data and allometric scaling.

Species: Human, Rat, Mouse, Dog (Beagle), Monkey (Cynomolgus)

Data sources:
  - Brown RP et al. Toxicol Ind Health 1997;13:407-484
  - Davies B, Morris T. Pharm Res 1993;10:1093-1095
  - PK-Sim PKSimDB (Open Systems Pharmacology)

Allometric scaling:
  CL_human = CL_animal * (BW_human/BW_animal)^0.75
  Vss_human = Vss_animal * (BW_human/BW_animal)^1.0
  t½_human = t½_animal * (BW_human/BW_animal)^0.25
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Species(str, Enum):
    HUMAN = "human"
    RAT = "rat"
    MOUSE = "mouse"
    DOG = "dog"
    MONKEY = "monkey"


# ===================================================================
# Species-specific physiological data
# Reference BW, organ fractions (L/kg BW), blood flow fractions (of CO)
# Sources: Brown 1997, Davies 1993, PK-Sim database
# ===================================================================

SPECIES_DATA = {
    Species.HUMAN: {
        "BW_ref": 73.0,  # kg
        "CO_coeff": 15.0,  # CO = a * BW^0.74 (L/h)
        "CO_exp": 0.74,
        "MPPGL": 40.0,  # mg/g liver (Barter 2007)
        "HPGL": 99.0,   # x10^6 cells/g liver (Wilson 2003)
        "liver_frac": 0.025,  # L/kg BW
        "kidney_frac": 0.0044,
        "heart_frac": 0.0045,
        "brain_frac": 0.020,
        "lung_frac": 0.0076,
        "muscle_frac": 0.400,
        "adipose_frac": 0.214,
        "skin_frac": 0.037,
        "gut_frac": 0.017,
        "spleen_frac": 0.0021,
        "bone_frac": 0.085,
        "pancreas_frac": 0.0019,
        "Q_kidney_frac": 0.190,
        "Q_liver_HA_frac": 0.065,
        "Q_gut_frac": 0.146,
        "Q_spleen_frac": 0.030,
        "Q_brain_frac": 0.120,
        "Q_heart_frac": 0.040,
        "Q_muscle_frac": 0.170,
        "Q_skin_frac": 0.050,
        "Q_adipose_frac": 0.052,
        "Q_bone_frac": 0.050,
        "HCT": 0.45,
        "GFR_mL_min": 120.0,
        "plasma_albumin_gL": 42.0,
    },
    Species.RAT: {
        "BW_ref": 0.25,
        "CO_coeff": 15.0,
        "CO_exp": 0.74,
        "MPPGL": 45.0,
        "HPGL": 120.0,    # x10^6 cells/g
        "liver_frac": 0.034,  # Brown 1997 (was 0.037)
        "kidney_frac": 0.0073,
        "heart_frac": 0.0033,
        "brain_frac": 0.0057,
        "lung_frac": 0.0050,
        "muscle_frac": 0.404,
        "adipose_frac": 0.070,
        "skin_frac": 0.190,
        "gut_frac": 0.027,
        "spleen_frac": 0.0020,
        "bone_frac": 0.061,
        "pancreas_frac": 0.0018,
        "Q_kidney_frac": 0.141,
        "Q_liver_HA_frac": 0.024,
        "Q_gut_frac": 0.141,
        "Q_spleen_frac": 0.010,
        "Q_brain_frac": 0.020,
        "Q_heart_frac": 0.051,
        "Q_muscle_frac": 0.278,
        "Q_skin_frac": 0.058,
        "Q_adipose_frac": 0.070,
        "Q_bone_frac": 0.122,
        "HCT": 0.46,
        "GFR_mL_min": 1.31,
        "plasma_albumin_gL": 31.0,
    },
    Species.MOUSE: {
        "BW_ref": 0.025,
        "CO_coeff": 16.0,
        "CO_exp": 0.74,
        "MPPGL": 45.0,
        "HPGL": 120.0,
        "liver_frac": 0.055,
        "kidney_frac": 0.017,
        "heart_frac": 0.005,
        "brain_frac": 0.017,
        "lung_frac": 0.007,
        "muscle_frac": 0.384,
        "adipose_frac": 0.100,
        "skin_frac": 0.165,
        "gut_frac": 0.042,
        "spleen_frac": 0.004,
        "bone_frac": 0.060,
        "pancreas_frac": 0.006,
        "Q_kidney_frac": 0.091,
        "Q_liver_HA_frac": 0.016,
        "Q_gut_frac": 0.141,
        "Q_spleen_frac": 0.012,
        "Q_brain_frac": 0.033,
        "Q_heart_frac": 0.066,
        "Q_muscle_frac": 0.158,
        "Q_skin_frac": 0.058,
        "Q_adipose_frac": 0.070,
        "Q_bone_frac": 0.122,
        "HCT": 0.45,
        "GFR_mL_min": 0.28,
        "plasma_albumin_gL": 25.0,
    },
    Species.DOG: {
        "BW_ref": 10.5,
        "CO_coeff": 15.0,
        "CO_exp": 0.74,
        "MPPGL": 30.0,
        "HPGL": 120.0,
        "liver_frac": 0.032,
        "kidney_frac": 0.0054,
        "heart_frac": 0.0072,
        "brain_frac": 0.0074,
        "lung_frac": 0.0095,
        "muscle_frac": 0.457,
        "adipose_frac": 0.153,
        "skin_frac": 0.088,
        "gut_frac": 0.021,
        "spleen_frac": 0.0090,
        "bone_frac": 0.095,
        "pancreas_frac": 0.0024,
        "Q_kidney_frac": 0.173,
        "Q_liver_HA_frac": 0.050,
        "Q_gut_frac": 0.113,
        "Q_spleen_frac": 0.030,
        "Q_brain_frac": 0.040,
        "Q_heart_frac": 0.042,
        "Q_muscle_frac": 0.211,
        "Q_skin_frac": 0.055,
        "Q_adipose_frac": 0.052,
        "Q_bone_frac": 0.050,
        "HCT": 0.45,
        "GFR_mL_min": 3.96,
        "plasma_albumin_gL": 31.0,
    },
    Species.MONKEY: {
        "BW_ref": 5.0,
        "CO_coeff": 15.0,
        "CO_exp": 0.74,
        "MPPGL": 30.0,
        "HPGL": 120.0,
        "liver_frac": 0.027,
        "kidney_frac": 0.0044,
        "heart_frac": 0.0045,
        "brain_frac": 0.017,
        "lung_frac": 0.0065,
        "muscle_frac": 0.460,
        "adipose_frac": 0.080,
        "skin_frac": 0.070,
        "gut_frac": 0.020,
        "spleen_frac": 0.0020,
        "bone_frac": 0.095,
        "pancreas_frac": 0.0015,
        "Q_kidney_frac": 0.160,
        "Q_liver_HA_frac": 0.050,
        "Q_gut_frac": 0.130,
        "Q_spleen_frac": 0.020,
        "Q_brain_frac": 0.060,
        "Q_heart_frac": 0.040,
        "Q_muscle_frac": 0.200,
        "Q_skin_frac": 0.050,
        "Q_adipose_frac": 0.052,
        "Q_bone_frac": 0.050,
        "HCT": 0.41,
        "GFR_mL_min": 2.10,
        "plasma_albumin_gL": 40.0,
    },
}


# ===================================================================
# Allometric scaling
# ===================================================================

def allometric_scale(
    value_animal: float,
    BW_animal: float,
    BW_human: float = 73.0,
    exponent: float = 0.75,
) -> float:
    """
    Allometric scaling: value_human = value_animal * (BW_human/BW_animal)^exp

    Default exponents:
      CL: 0.75 (clearance)
      Vss: 1.0 (volume)
      t½: 0.25 (half-life)
      CO: 0.74 (cardiac output)
    """
    if BW_animal <= 0:
        return value_animal
    return value_animal * (BW_human / BW_animal) ** exponent


# ===================================================================
# Brain weight + Maximum Life-span Potential (MLP) for vertical
# allometry corrections (Mahmood 1996, Sacher 1959).
#
# Brain weight (g) — Boxenbaum 1980, ICRP 89, Davies 1993
# MLP (years)      — Sacher 1959 / Mahmood 1996 reference values
# ===================================================================

BRAIN_WEIGHT_G = {
    Species.MOUSE:  0.40,
    Species.RAT:    1.80,
    Species.DOG:    72.0,    # Beagle
    Species.MONKEY: 95.0,    # Cynomolgus
    Species.HUMAN:  1400.0,
}

MLP_YEARS = {
    # Maximum lifespan potential — Mahmood 1996, Sacher 1959
    Species.MOUSE:  2.7,
    Species.RAT:    4.7,
    Species.DOG:    20.0,
    Species.MONKEY: 22.0,
    Species.HUMAN:  93.0,
}


def scale_preclinical_to_human(
    CL_animal: float,
    Vss_animal: float,
    BW_animal: float,
    species: Species,
    BW_human: float = 73.0,
) -> dict:
    """
    Scale preclinical PK from animal to human.

    Args:
        CL_animal: Animal clearance (L/h).
        Vss_animal: Animal Vss (L).
        BW_animal: Animal body weight (kg).
        species: Animal species.
        BW_human: Target human BW (kg).

    Returns:
        Dict with predicted human CL, Vss, t½.
    """
    CL_human = allometric_scale(CL_animal, BW_animal, BW_human, 0.75)
    Vss_human = allometric_scale(Vss_animal, BW_animal, BW_human, 1.0)
    t_half_human = 0.693 * Vss_human / CL_human if CL_human > 0 else float("inf")

    return {
        "CL_human_L_per_h": CL_human,
        "Vss_human_L": Vss_human,
        "t_half_human_h": t_half_human,
        "species": species.value,
        "BW_animal_kg": BW_animal,
        "BW_human_kg": BW_human,
        "scaling_factor_CL": (BW_human / BW_animal) ** 0.75,
        "scaling_factor_Vss": BW_human / BW_animal,
    }


# ===================================================================
# Advanced allometric scaling methods
# ===================================================================

def fu_corrected_allometry(
    CL_animal: float,
    BW_animal: float,
    fu_animal: float,
    fu_human: float,
    BW_human: float = 73.0,
    exponent: float = 0.75,
) -> dict:
    """
    Fu-corrected single-species allometric scaling (Tang & Mayersohn 2005).

    The standard simple allometry assumes protein binding is conserved
    across species. For drugs where fu_p differs (warfarin: rat fu ~0.04
    vs human fu ~0.005, an 8-fold difference), simple BW^0.75 scaling
    can be off by 5-10×. Tang 2005 corrects with an explicit fu ratio:

        CL_human = CL_animal * (BW_human/BW_animal)^b * (fu_human/fu_animal)

    Args:
        CL_animal: Animal clearance (L/h).
        BW_animal: Animal body weight (kg).
        fu_animal: Animal fraction unbound in plasma.
        fu_human: Human fraction unbound in plasma.
        BW_human: Target human BW (kg).
        exponent: Allometric exponent (default 0.75).

    Returns:
        Dict with predicted human CL, naive (uncorrected) CL, and the
        fu-correction factor.

    Reference:
        Tang H, Mayersohn M. Drug Metab Dispos 2005;33:1294-1296.
    """
    if BW_animal <= 0:
        raise ValueError(f"BW_animal must be > 0 (got {BW_animal})")
    if not (0 < fu_animal <= 1.0):
        raise ValueError(f"fu_animal must be in (0, 1] (got {fu_animal})")
    if not (0 < fu_human <= 1.0):
        raise ValueError(f"fu_human must be in (0, 1] (got {fu_human})")

    naive = CL_animal * (BW_human / BW_animal) ** exponent
    fu_factor = fu_human / fu_animal
    corrected = naive * fu_factor
    return {
        "CL_human_L_per_h": corrected,
        "CL_human_naive_L_per_h": naive,
        "fu_correction_factor": fu_factor,
        "exponent": exponent,
        "method": "Tang-Mayersohn 2005 (fu-corrected single-species)",
        "BW_animal_kg": BW_animal,
        "BW_human_kg": BW_human,
        "fu_animal": fu_animal,
        "fu_human": fu_human,
    }


def vertical_allometry_brain_weight(
    CL_animal: float,
    BW_animal: float,
    species: Species,
    BW_human: float = 73.0,
    BrW_human_g: float = 1400.0,
) -> dict:
    """
    Vertical allometry with brain weight correction (Mahmood 1996).

    For drugs with high hepatic clearance or where simple BW^0.75 fails,
    Mahmood proposed correcting with brain weight:

        CL × BrW vs BW × BrW   (or)   CL_human = CL_animal × (BW_h/BW_a)^b × (BrW_h/BrW_a)

    The single-species variant used here:
        CL_human = CL_animal × (BW_human × BrW_human) / (BW_animal × BrW_animal)

    This applies to drugs where the brain-weight correction would have
    been chosen by the multi-species Rule of Exponents (b in 0.7-1.0).

    Args:
        CL_animal: Animal clearance (L/h).
        BW_animal: Animal body weight (kg).
        species: Animal species (uses BRAIN_WEIGHT_G lookup).
        BW_human: Target human BW (kg).
        BrW_human_g: Human brain weight (g).

    Returns:
        Dict with predicted human CL.

    Reference:
        Mahmood I, Balian JD. Xenobiotica 1996;26:887-895.
        Boxenbaum H. J Pharmacokinet Biopharm 1980;8:165-176.
    """
    if BW_animal <= 0:
        raise ValueError(f"BW_animal must be > 0 (got {BW_animal})")
    if species not in BRAIN_WEIGHT_G:
        raise ValueError(f"No brain weight data for {species.value}; "
                         f"available: {list(BRAIN_WEIGHT_G)}")
    BrW_animal = BRAIN_WEIGHT_G[species]
    factor = (BW_human * BrW_human_g) / (BW_animal * BrW_animal)
    CL_human = CL_animal * factor
    return {
        "CL_human_L_per_h": CL_human,
        "BrW_animal_g": BrW_animal,
        "BrW_human_g": BrW_human_g,
        "scaling_factor": factor,
        "method": "Vertical allometry × brain weight (Mahmood 1996)",
        "species": species.value,
        "BW_animal_kg": BW_animal,
        "BW_human_kg": BW_human,
    }


def mahmood_rule_of_exponents(
    species_data: list[tuple[Species, float, float]],
    BW_human: float = 73.0,
    BrW_human_g: float = 1400.0,
    MLP_human_y: float = 93.0,
) -> dict:
    """
    Multi-species allometric scaling with the Rule of Exponents (Mahmood 1996).

    Workflow:
      1. Fit log(CL) = log(a) + b·log(BW) across ≥3 species.
      2. Choose correction by exponent b:
         - b < 0.55:        simple allometry
         - 0.55 ≤ b ≤ 0.70: MLP correction (CL × MLP vs BW)
         - 0.70 < b ≤ 1.00: brain-weight correction (CL × BrW vs BW)
         - b > 1.00:        ROE inapplicable, return error
      3. Refit in the corrected coordinate, predict human CL,
         divide back by the human MLP or BrW.

    Args:
        species_data: list of (species, CL_obs_L_per_h, BW_kg) — at least 3.
        BW_human: target human body weight (kg).
        BrW_human_g: human brain weight (g).
        MLP_human_y: human maximum lifespan potential (years).

    Returns:
        Dict with predicted CL_human, the chosen method, the fitted
        exponent, fit R², and an applicability verdict.

    Reference:
        Mahmood I, Balian JD. Xenobiotica 1996;26:887-895.
        Mahmood I. Curr Drug Metab 2009;10:130-134.
    """
    import numpy as np

    if len(species_data) < 3:
        raise ValueError(
            f"Mahmood ROE requires ≥3 species, got {len(species_data)}. "
            f"Use fu_corrected_allometry for single-species scaling."
        )
    for sp, cl, bw in species_data:
        if cl <= 0 or bw <= 0:
            raise ValueError(
                f"Invalid data for {sp.value}: CL={cl}, BW={bw}; "
                f"both must be > 0."
            )
        if sp not in BRAIN_WEIGHT_G or sp not in MLP_YEARS:
            raise ValueError(
                f"No brain weight / MLP data for {sp.value}; "
                f"available: {list(BRAIN_WEIGHT_G)}"
            )

    BW = np.array([d[2] for d in species_data], dtype=float)
    CL = np.array([d[1] for d in species_data], dtype=float)
    BrW = np.array([BRAIN_WEIGHT_G[d[0]] for d in species_data], dtype=float)
    MLP = np.array([MLP_YEARS[d[0]] for d in species_data], dtype=float)

    # Step 1: fit simple allometry log(CL) = log(a) + b * log(BW)
    log_BW = np.log(BW)
    log_CL = np.log(CL)
    b_simple, log_a = np.polyfit(log_BW, log_CL, 1)
    a_simple = float(np.exp(log_a))
    pred_log = log_a + b_simple * log_BW
    ss_res = float(np.sum((log_CL - pred_log) ** 2))
    ss_tot = float(np.sum((log_CL - log_CL.mean()) ** 2))
    r2_simple = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Step 2: choose correction by Rule of Exponents
    method = ""
    CL_human = 0.0
    notes = ""
    applicability = "ok"

    if b_simple < 0.55:
        method = "simple allometry (b < 0.55)"
        CL_human = a_simple * BW_human ** b_simple
    elif b_simple <= 0.70:
        # MLP correction: regress CL × MLP vs BW
        y = np.log(CL * MLP)
        b_mlp, log_a_mlp = np.polyfit(log_BW, y, 1)
        a_mlp = float(np.exp(log_a_mlp))
        # Predicted (CL × MLP)_human, divide by MLP_human
        CL_MLP_human = a_mlp * BW_human ** b_mlp
        CL_human = CL_MLP_human / MLP_human_y
        method = f"MLP correction (0.55 ≤ b={b_simple:.3f} ≤ 0.70)"
        notes = (f"Refit exponent in (CL × MLP) coordinate: b'={b_mlp:.3f}. "
                 f"Final CL = (a' × BW^b') / MLP_human.")
    elif b_simple <= 1.00:
        # Brain-weight correction: regress CL × BrW vs BW
        y = np.log(CL * BrW)
        b_brw, log_a_brw = np.polyfit(log_BW, y, 1)
        a_brw = float(np.exp(log_a_brw))
        CL_BrW_human = a_brw * BW_human ** b_brw
        CL_human = CL_BrW_human / BrW_human_g
        method = f"Brain-weight correction (0.70 < b={b_simple:.3f} ≤ 1.00)"
        notes = (f"Refit exponent in (CL × BrW) coordinate: b'={b_brw:.3f}. "
                 f"Final CL = (a' × BW^b') / BrW_human.")
    else:
        method = f"INAPPLICABLE (b={b_simple:.3f} > 1.00)"
        applicability = "fail"
        notes = (
            "Exponent > 1.0 means CL grows faster than mass — the Rule "
            "of Exponents does not provide a reliable correction. "
            "Common causes: highly-protein-bound drug with cross-species "
            "fu differences (try fu_corrected_allometry), substrate of "
            "human-specific transporter, or species-specific metabolic "
            "pathway. Do NOT use the result for human prediction."
        )
        # Still report the simple-allometry extrapolation, but flagged
        CL_human = a_simple * BW_human ** b_simple

    return {
        "CL_human_L_per_h": float(CL_human),
        "exponent_simple": float(b_simple),
        "intercept_simple": a_simple,
        "r2_simple": float(r2_simple),
        "n_species": len(species_data),
        "method": method,
        "applicability": applicability,
        "notes": notes,
        "species_used": [d[0].value for d in species_data],
        "BW_human_kg": BW_human,
    }


def format_species_comparison(species_list: Optional[list[Species]] = None) -> str:
    """Format species physiological data comparison."""
    if species_list is None:
        species_list = list(Species)

    lines = [
        "## Species Physiological Comparison\n",
        "| Parameter | " + " | ".join(s.value.capitalize() for s in species_list) + " |",
        "|-----------|" + "|".join(["--------"] * len(species_list)) + "|",
    ]

    params = [
        ("BW ref (kg)", "BW_ref"),
        ("Liver (L/kg)", "liver_frac"),
        ("Kidney (L/kg)", "kidney_frac"),
        ("GFR (mL/min)", "GFR_mL_min"),
        ("MPPGL (mg/g)", "MPPGL"),
        ("Albumin (g/L)", "plasma_albumin_gL"),
        ("HCT", "HCT"),
    ]

    for label, key in params:
        vals = [f"{SPECIES_DATA[s][key]:.3g}" for s in species_list]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    return "\n".join(lines)
