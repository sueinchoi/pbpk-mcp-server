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
