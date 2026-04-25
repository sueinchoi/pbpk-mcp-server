"""
Human physiology database for whole-body PBPK modeling.

Data sources:
- Organ volumes & blood flows: ICRP Publication 89 (Valentin, 2002)
- Tissue composition: Rodgers & Rowland (2005, 2006), Schmitt (2008)
- Cardiac output allometry: Willmann et al. (2007)

Units:
- Volume: L
- Blood flow: fraction of cardiac output (Qco)
- Cardiac output: L/h
- Body weight: kg
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Organ(str, Enum):
    ADIPOSE = "adipose"
    BONE = "bone"
    BRAIN = "brain"
    GUT = "gut"
    HEART = "heart"
    KIDNEY = "kidney"
    LIVER = "liver"
    LUNG = "lung"
    MUSCLE = "muscle"
    PANCREAS = "pancreas"
    SKIN = "skin"
    SPLEEN = "spleen"
    REST = "rest"


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"


# ---------------------------------------------------------------
# Tissue composition data (Rodgers & Rowland 2005, 2006)
#
# f_EW  = fractional extracellular water volume
# f_IW  = fractional intracellular water volume
# f_NL  = fractional neutral lipid volume
# f_NP  = fractional neutral phospholipid volume
# f_AP  = fractional acidic phospholipid volume
# pH_IW = intracellular pH
# pH_EW = extracellular (interstitial) pH
# albumin_ratio = ratio of interstitial albumin to plasma albumin
# ---------------------------------------------------------------

# Tissue composition: R&R 2005 Table I, 2006 Table I
# Albumin ratios: R&R 2006 Table II (tissue interstitial albumin / plasma albumin)
TISSUE_COMPOSITION = {
    Organ.ADIPOSE: {
        "f_EW": 0.135, "f_IW": 0.017, "f_NL": 0.853,
        "f_NP": 0.0016, "f_AP": 0.00040,
        "pH_IW": 7.0, "pH_EW": 7.4, "albumin_ratio": 0.049,
    },
    Organ.BONE: {
        "f_EW": 0.100, "f_IW": 0.346, "f_NL": 0.017,
        "f_NP": 0.0017, "f_AP": 0.00067,
        "pH_IW": 7.0, "pH_EW": 7.4, "albumin_ratio": 0.100,
    },
    Organ.BRAIN: {
        "f_EW": 0.162, "f_IW": 0.620, "f_NL": 0.039,
        "f_NP": 0.0153, "f_AP": 0.00457,
        "pH_IW": 7.0, "pH_EW": 7.4, "albumin_ratio": 0.048,
    },
    Organ.GUT: {
        "f_EW": 0.282, "f_IW": 0.475, "f_NL": 0.038,
        "f_NP": 0.0128, "f_AP": 0.01630,
        "pH_IW": 7.0, "pH_EW": 7.4, "albumin_ratio": 0.158,
    },
    Organ.HEART: {
        "f_EW": 0.320, "f_IW": 0.456, "f_NL": 0.014,
        "f_NP": 0.0111, "f_AP": 0.01560,
        "pH_IW": 7.0, "pH_EW": 7.4, "albumin_ratio": 0.157,
    },
    Organ.KIDNEY: {
        "f_EW": 0.273, "f_IW": 0.483, "f_NL": 0.012,
        "f_NP": 0.0120, "f_AP": 0.01370,
        "pH_IW": 7.0, "pH_EW": 7.4, "albumin_ratio": 0.130,
    },
    Organ.LIVER: {
        "f_EW": 0.161, "f_IW": 0.573, "f_NL": 0.014,
        "f_NP": 0.0240, "f_AP": 0.02050,
        "pH_IW": 7.0, "pH_EW": 7.4, "albumin_ratio": 0.086,
    },
    Organ.LUNG: {
        "f_EW": 0.336, "f_IW": 0.446, "f_NL": 0.022,
        "f_NP": 0.0128, "f_AP": 0.00910,
        "pH_IW": 7.0, "pH_EW": 7.4, "albumin_ratio": 0.212,
    },
    Organ.MUSCLE: {
        "f_EW": 0.118, "f_IW": 0.630, "f_NL": 0.010,
        "f_NP": 0.0072, "f_AP": 0.00640,
        "pH_IW": 7.0, "pH_EW": 7.4, "albumin_ratio": 0.064,
    },
    Organ.PANCREAS: {
        "f_EW": 0.120, "f_IW": 0.664, "f_NL": 0.041,
        "f_NP": 0.0091, "f_AP": 0.00490,
        "pH_IW": 6.8, "pH_EW": 7.4, "albumin_ratio": 0.060,
    },
    Organ.SKIN: {
        "f_EW": 0.382, "f_IW": 0.291, "f_NL": 0.060,
        "f_NP": 0.0044, "f_AP": 0.00220,
        "pH_IW": 7.0, "pH_EW": 7.4, "albumin_ratio": 0.277,
    },
    Organ.SPLEEN: {
        "f_EW": 0.207, "f_IW": 0.579, "f_NL": 0.008,
        "f_NP": 0.0097, "f_AP": 0.01230,
        "pH_IW": 7.0, "pH_EW": 7.4, "albumin_ratio": 0.097,
    },
    Organ.REST: {
        "f_EW": 0.200, "f_IW": 0.400, "f_NL": 0.020,
        "f_NP": 0.0050, "f_AP": 0.00300,
        "pH_IW": 7.0, "pH_EW": 7.4, "albumin_ratio": 0.100,
    },
}

# Plasma composition (for Kp denominator calculation)
PLASMA_COMPOSITION = {
    "f_W": 0.945,      # Water fraction
    "f_NL": 0.0023,    # Neutral lipid fraction
    "f_NP": 0.0009,    # Neutral phospholipid fraction
    "f_AP": 0.00009,   # Acidic phospholipid fraction
    "f_protein": 0.0812,  # Plasma protein fraction (albumin ~4.4 g/dL)
    "pH": 7.4,
}

# ---------------------------------------------------------------
# Red blood cell (RBC) composition — Rodgers & Rowland (2005)
# Used for Ka_AP derivation from blood:plasma ratio
# ---------------------------------------------------------------

RBC_COMPOSITION = {
    "f_IW": 0.603,       # Intracellular water
    "f_NL": 0.0017,      # Neutral lipids
    "f_NP": 0.0029,      # Neutral phospholipids
    "f_AP": 0.0056,      # Acidic phospholipids
    "f_protein": 0.339,  # Hemoglobin + other proteins
    "pH": 7.22,          # RBC intracellular pH
}

HEMATOCRIT = 0.45  # Standard hematocrit

# ---------------------------------------------------------------
# Tissue protein fractions (Schmitt 2008, PK-Sim database)
# Used by Schmitt and PK-Sim Standard methods
# ---------------------------------------------------------------

TISSUE_PROTEIN_FRACTIONS = {
    Organ.ADIPOSE:  0.044,
    Organ.BONE:     0.204,
    Organ.BRAIN:    0.081,
    Organ.GUT:      0.070,
    Organ.HEART:    0.107,
    Organ.KIDNEY:   0.149,
    Organ.LIVER:    0.178,
    Organ.LUNG:     0.108,
    Organ.MUSCLE:   0.075,
    Organ.PANCREAS: 0.053,
    Organ.SKIN:     0.278,
    Organ.SPLEEN:   0.116,
    Organ.REST:     0.100,
}

# ---------------------------------------------------------------
# Tissue water + lipid fractions for Poulin-Theil / Berezhkovskiy
# (combined water, neutral lipid, phospholipid per tissue)
# ---------------------------------------------------------------

TISSUE_INTERSTITIAL_FRACTIONS = {
    # F_int = fractional interstitial volume (of total tissue)
    # F_cell = fractional cellular volume (of total tissue)
    # Source: Schmitt (2008) Table 1, Valentin (2002)
    Organ.ADIPOSE:  {"F_int": 0.135, "F_cell": 0.865},
    Organ.BONE:     {"F_int": 0.100, "F_cell": 0.900},
    Organ.BRAIN:    {"F_int": 0.162, "F_cell": 0.838},
    Organ.GUT:      {"F_int": 0.282, "F_cell": 0.718},
    Organ.HEART:    {"F_int": 0.320, "F_cell": 0.680},
    Organ.KIDNEY:   {"F_int": 0.273, "F_cell": 0.727},
    Organ.LIVER:    {"F_int": 0.161, "F_cell": 0.839},
    Organ.LUNG:     {"F_int": 0.336, "F_cell": 0.664},
    Organ.MUSCLE:   {"F_int": 0.118, "F_cell": 0.882},
    Organ.PANCREAS: {"F_int": 0.120, "F_cell": 0.880},
    Organ.SKIN:     {"F_int": 0.382, "F_cell": 0.618},
    Organ.SPLEEN:   {"F_int": 0.207, "F_cell": 0.793},
    Organ.REST:     {"F_int": 0.200, "F_cell": 0.800},
}


# ---------------------------------------------------------------
# Organ volumes (fraction of body weight, L/kg)
# Source: ICRP 89, Brown et al. (1997), PK-Sim defaults
# Density assumed ~1.0 kg/L for most organs except adipose (~0.92)
# ---------------------------------------------------------------

ORGAN_VOLUMES = {
    Sex.MALE: {
        # Reference: 73 kg adult male
        Organ.ADIPOSE:  0.214,   # ~15.6 L for 73 kg
        Organ.BONE:     0.085,   # ~6.2 L (cortical + trabecular)
        Organ.BRAIN:    0.020,   # ~1.45 L
        Organ.GUT:      0.017,   # ~1.2 L (wall only, SI+LI)
        Organ.HEART:    0.0045,  # ~0.33 L
        Organ.KIDNEY:   0.0044,  # ~0.32 L
        Organ.LIVER:    0.025,   # ~1.82 L
        Organ.LUNG:     0.0076,  # ~0.56 L
        Organ.MUSCLE:   0.400,   # ~29.2 L
        Organ.PANCREAS: 0.0019,  # ~0.14 L
        Organ.SKIN:     0.037,   # ~2.7 L
        Organ.SPLEEN:   0.0021,  # ~0.15 L
        # Blood volumes (fraction of BW in L)
        "V_arterial": 0.024,     # ~1.75 L
        "V_venous":   0.048,     # ~3.50 L
    },
    Sex.FEMALE: {
        # Reference: 60 kg adult female
        Organ.ADIPOSE:  0.325,   # ~19.5 L (higher fat fraction)
        Organ.BONE:     0.077,
        Organ.BRAIN:    0.022,
        Organ.GUT:      0.016,
        Organ.HEART:    0.0040,
        Organ.KIDNEY:   0.0042,
        Organ.LIVER:    0.024,
        Organ.LUNG:     0.0060,
        Organ.MUSCLE:   0.291,   # lower muscle fraction
        Organ.PANCREAS: 0.0017,
        Organ.SKIN:     0.037,
        Organ.SPLEEN:   0.0024,
        "V_arterial": 0.022,
        "V_venous":   0.044,
    },
}


# ---------------------------------------------------------------
# Blood flow fractions (fraction of cardiac output)
# Source: ICRP 89, Williams & Leggett (1989), PK-Sim defaults
#
# Portal circulation: gut + spleen → portal vein → liver
# Hepatic artery: direct arterial supply to liver
# ---------------------------------------------------------------

BLOOD_FLOWS = {
    Sex.MALE: {
        Organ.ADIPOSE:  0.052,
        Organ.BONE:     0.050,
        Organ.BRAIN:    0.120,
        Organ.GUT:      0.146,   # mesenteric (portal)
        Organ.HEART:    0.040,
        Organ.KIDNEY:   0.190,
        Organ.LUNG:     1.000,   # entire cardiac output
        Organ.MUSCLE:   0.170,
        Organ.PANCREAS: 0.010,
        Organ.SKIN:     0.050,
        Organ.SPLEEN:   0.030,   # portal
        "Q_hepatic_artery": 0.065,  # direct to liver
        # REST = 1.0 - sum(above except lung) - Q_HA
    },
    Sex.FEMALE: {
        Organ.ADIPOSE:  0.085,
        Organ.BONE:     0.050,
        Organ.BRAIN:    0.120,
        Organ.GUT:      0.143,
        Organ.HEART:    0.040,
        Organ.KIDNEY:   0.170,
        Organ.LUNG:     1.000,
        Organ.MUSCLE:   0.120,
        Organ.PANCREAS: 0.010,
        Organ.SKIN:     0.050,
        Organ.SPLEEN:   0.030,
        "Q_hepatic_artery": 0.065,
    },
}


@dataclass
class PhysiologyParams:
    """Resolved physiological parameters for a specific individual."""

    body_weight: float           # kg
    sex: Sex
    age_years: float = 30.0      # years
    height_cm: float = 176.0     # cm
    hematocrit: float = 0.45
    cardiac_output: float = 390.0  # L/h
    organ_volumes: dict = None   # {Organ: volume_L}
    blood_flows: dict = None     # {Organ: flow_L_per_h}
    V_arterial: float = 1.75     # L
    V_venous: float = 3.50       # L
    Q_hepatic_artery: float = 0.0  # L/h
    Q_portal: float = 0.0       # L/h (= Q_gut + Q_spleen)
    Q_liver_total: float = 0.0  # L/h (= Q_HA + Q_portal)
    GFR: float = 7.2            # L/h (= 120 mL/min)
    BSA: float = 1.73           # m^2


def calculate_bsa(weight_kg: float, height_cm: float) -> float:
    """Body surface area (m^2). Du Bois formula."""
    return 0.007184 * height_cm ** 0.725 * weight_kg ** 0.425


def calculate_gfr(
    age_years: float = 30.0,
    body_weight: float = 73.0,
    sex: Sex = Sex.MALE,
    bsa: float = 1.73,
) -> float:
    """
    Calculate GFR (L/h).

    For pediatric (<18y): Rhodin 2009 allometric model
      GFR = GFR_adult * (BW/70)^0.75 * PMA^3.4 / (47.7^3.4 + PMA^3.4)
    For adult: BSA-scaled with renal aging
      GFR = 120 * (BSA/1.73) * aging_factor

    References:
      - Rhodin MM et al. Pediatr Nephrol 2009;24:67-76
      - PK-Sim PKSimDB: renal aging formula
    """
    if age_years < 18.0:
        # Rhodin 2009: allometric + maturation
        GFR_ADULT = 121.2  # mL/min per 70 kg (Rhodin 2009)
        TM50 = 47.7        # weeks PMA
        HILL = 3.40

        F_size = (body_weight / 70.0) ** 0.75
        pma_weeks = age_years * 52.0 + 40.0  # postnatal weeks + 40 GA
        F_mat = pma_weeks ** HILL / (TM50 ** HILL + pma_weeks ** HILL)

        gfr = GFR_ADULT * F_size * F_mat
    else:
        # Adult: BSA-scaled
        gfr = 120.0 * (bsa / 1.73)

        # Sex correction
        if sex == Sex.FEMALE:
            gfr *= 0.95

        # Renal aging (>30y, PK-Sim Hill model)
        if age_years > 30.0:
            age_offset = age_years - 30.0
            aging_factor = 1.0 - 0.5 * age_offset ** 1.5 / (54.0 ** 1.5 + age_offset ** 1.5)
            gfr *= max(aging_factor, 0.3)

    # Convert mL/min to L/h
    return gfr * 60.0 / 1000.0


def estimate_height(weight_kg: float, sex: Sex, age_years: float = 30.0) -> float:
    """
    Estimate height (cm) from weight.

    Uses BMI-based estimation: height = sqrt(weight / BMI) * 100
    Default BMI: 24.5 male, 23.5 female (healthy adult).
    """
    if age_years < 18:
        # Very rough pediatric: CDC 50th percentile approximation
        if sex == Sex.MALE:
            return min(50.0 + 6.0 * age_years, 176.0)
        else:
            return min(49.0 + 5.5 * age_years, 163.0)
    bmi = 24.5 if sex == Sex.MALE else 23.5
    height_m = (weight_kg / bmi) ** 0.5
    return height_m * 100.0


def calculate_cardiac_output(body_weight: float, sex: Sex) -> float:
    """
    Cardiac output (L/h) scaled by body weight.

    Allometric scaling: Qco = 15 * BW^0.74 (L/h)
    Reference: Willmann et al. (2007), West et al.
    Standard: ~6.5 L/min = 390 L/h for 70 kg male
    """
    # Simple allometric: Qco = a * BW^b
    if sex == Sex.MALE:
        qco = 15.0 * (body_weight ** 0.74)
    else:
        qco = 13.5 * (body_weight ** 0.74)
    return qco  # L/h


def get_physiology(
    body_weight: float = 73.0,
    sex: Sex = Sex.MALE,
    cardiac_output: Optional[float] = None,
    age_years: float = 30.0,
    height_cm: Optional[float] = None,
    hematocrit: Optional[float] = None,
) -> PhysiologyParams:
    """
    Build resolved physiological parameters for a virtual individual.

    Organ volumes scale linearly with body weight.
    Blood flows scale with cardiac output.
    GFR is calculated from age, BSA, and sex.

    Args:
        body_weight: Body weight in kg
        sex: Sex (MALE or FEMALE)
        cardiac_output: Override cardiac output (L/h). If None, allometric.
        age_years: Age in years (for GFR aging).
        height_cm: Height in cm (for BSA). If None, estimated from weight.
        hematocrit: Override hematocrit. If None, sex-based default.

    Returns:
        PhysiologyParams with all values in absolute units (L, L/h)
    """
    if isinstance(sex, str):
        sex = Sex(sex)

    if cardiac_output is None:
        cardiac_output = calculate_cardiac_output(body_weight, sex)

    # Height, BSA, GFR, hematocrit
    if height_cm is None:
        height_cm = estimate_height(body_weight, sex, age_years)
    bsa = calculate_bsa(body_weight, height_cm)
    gfr = calculate_gfr(age_years, body_weight, sex, bsa)

    if hematocrit is None:
        hematocrit = 0.45 if sex == Sex.MALE else 0.40

    vol_fracs = ORGAN_VOLUMES[sex]
    flow_fracs = BLOOD_FLOWS[sex]

    # --- Organ volumes (L) ---
    organ_volumes = {}
    sum_named = 0.0
    for organ in Organ:
        if organ == Organ.REST:
            continue
        frac = vol_fracs.get(organ, 0.0)
        organ_volumes[organ] = frac * body_weight
        sum_named += frac

    # Rest of body: total body volume minus named organs minus blood
    v_art_frac = vol_fracs["V_arterial"]
    v_ven_frac = vol_fracs["V_venous"]
    rest_frac = max(1.0 - sum_named - v_art_frac - v_ven_frac, 0.01)
    organ_volumes[Organ.REST] = rest_frac * body_weight

    V_arterial = v_art_frac * body_weight
    V_venous = v_ven_frac * body_weight

    # --- Blood flows (L/h) ---
    blood_flows = {}
    sum_flow = 0.0
    for organ in Organ:
        if organ in (Organ.REST, Organ.LUNG, Organ.LIVER):
            continue
        frac = flow_fracs.get(organ, 0.0)
        blood_flows[organ] = frac * cardiac_output
        sum_flow += frac

    Q_ha = flow_fracs["Q_hepatic_artery"] * cardiac_output
    sum_flow += flow_fracs["Q_hepatic_artery"]

    # Rest of body flow
    rest_flow_frac = max(1.0 - sum_flow, 0.01)
    blood_flows[Organ.REST] = rest_flow_frac * cardiac_output

    # Lung receives entire cardiac output
    blood_flows[Organ.LUNG] = cardiac_output

    # Portal vein: gut + spleen + pancreas (Williams & Leggett 1989)
    Q_portal = blood_flows[Organ.GUT] + blood_flows[Organ.SPLEEN] + blood_flows.get(Organ.PANCREAS, 0)
    Q_liver_total = Q_ha + Q_portal
    blood_flows[Organ.LIVER] = Q_liver_total

    return PhysiologyParams(
        body_weight=body_weight,
        sex=sex,
        age_years=age_years,
        height_cm=height_cm,
        hematocrit=hematocrit,
        cardiac_output=cardiac_output,
        organ_volumes=organ_volumes,
        blood_flows=blood_flows,
        V_arterial=V_arterial,
        V_venous=V_venous,
        Q_hepatic_artery=Q_ha,
        Q_portal=Q_portal,
        Q_liver_total=Q_liver_total,
        GFR=gfr,
        BSA=bsa,
    )


def get_tissue_composition(organ: Organ) -> dict:
    """Get tissue composition for a specific organ."""
    if isinstance(organ, str):
        organ = Organ(organ)
    return TISSUE_COMPOSITION[organ].copy()


def get_plasma_composition() -> dict:
    """Get plasma composition data."""
    return PLASMA_COMPOSITION.copy()


# ---------------------------------------------------------------
# Permeability-limited model parameters (PK-Sim style)
# Vascular fraction, interstitial fraction per organ
# ---------------------------------------------------------------

VASCULAR_FRACTIONS = {
    # Fraction of organ volume occupied by blood (regional blood volume)
    Organ.ADIPOSE:  0.031,
    Organ.BONE:     0.041,
    Organ.BRAIN:    0.036,
    Organ.GUT:      0.032,
    Organ.HEART:    0.065,
    Organ.KIDNEY:   0.105,
    Organ.LIVER:    0.115,
    Organ.LUNG:     0.185,
    Organ.MUSCLE:   0.026,
    Organ.PANCREAS: 0.060,
    Organ.SKIN:     0.025,
    Organ.SPLEEN:   0.170,
    Organ.REST:     0.040,
}


def get_subcompartment_volumes(organ: Organ, V_organ: float) -> dict:
    """
    For permeability-limited model: split organ into 3 sub-compartments.

    Returns dict with V_vascular, V_interstitial, V_intracellular (all in L).
    PK-Sim approach: vascular = regional blood, interstitial = EW - vascular_plasma,
    intracellular = remainder.
    """
    tc = TISSUE_COMPOSITION[organ]
    f_vas = VASCULAR_FRACTIONS.get(organ, 0.04)

    V_vas = f_vas * V_organ
    V_int = tc["f_EW"] * V_organ  # interstitial ≈ extracellular water
    V_cell = V_organ - V_vas - V_int
    V_cell = max(V_cell, 0.001 * V_organ)  # ensure positive

    return {
        "V_vascular": V_vas,
        "V_interstitial": V_int,
        "V_intracellular": V_cell,
    }
