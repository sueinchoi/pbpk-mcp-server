"""
Time-varying physiology models.

1. Fed/Fasted state: GI parameter switching at meal times
2. Circadian CYP expression: cosine-based enzyme activity fluctuation
3. Pregnancy physiology: gestational age-dependent parameter scaling

References:
  - Jamei M et al. AAPS J 2009;11:225-237 (fed/fasted)
  - Riedmaier AE et al. AAPS J 2020;12:163 (food effect PBPK)
  - Zhang T et al. Clin Pharmacol Ther 2020;108:1191-1200 (circadian)
  - Dallmann A et al. CPT:PSP 2018;7:135-146 (pregnancy)
  - Abduljalil K et al. Br J Clin Pharmacol 2012;73:685-694 (pregnancy)
"""

import math
from dataclasses import dataclass
from typing import Optional


# ===================================================================
# 1. Fed/Fasted State Model
# ===================================================================

@dataclass
class GIStateParams:
    """GI parameters for a specific prandial state."""
    gastric_pH: float
    gastric_emptying_t50_h: float   # half-time for gastric emptying
    bile_salt_mM: float             # duodenal bile salt concentration
    stomach_fluid_mL: float
    SI_fluid_mL: float              # total small intestinal fluid
    splanchnic_flow_factor: float   # multiplier on mesenteric blood flow

FASTED_STATE = GIStateParams(
    gastric_pH=1.7,
    gastric_emptying_t50_h=0.25,
    bile_salt_mM=4.0,
    stomach_fluid_mL=50.0,
    SI_fluid_mL=100.0,
    splanchnic_flow_factor=1.0,
)

FED_STATE = GIStateParams(
    gastric_pH=4.5,
    gastric_emptying_t50_h=1.5,
    bile_salt_mM=12.0,
    stomach_fluid_mL=500.0,
    SI_fluid_mL=300.0,
    splanchnic_flow_factor=1.5,
)


def get_gi_state(
    t: float,
    meal_times_h: tuple = (0.0,),
    fed_duration_h: float = 3.0,
    transition_h: float = 0.5,
) -> tuple[GIStateParams, float]:
    """
    Get GI state at time t with smooth fed/fasted transition.

    Returns (GIStateParams, fed_fraction) where fed_fraction ∈ [0, 1].
    0 = fully fasted, 1 = fully fed.
    """
    fed_frac = 0.0
    for t_meal in meal_times_h:
        dt = t - t_meal
        if dt < 0:
            continue
        if dt < fed_duration_h:
            # Smooth onset (sigmoid)
            onset = 1.0 / (1.0 + math.exp(-10.0 * (dt - transition_h / 2) / transition_h))
            # Smooth offset
            offset = 1.0 / (1.0 + math.exp(10.0 * (dt - fed_duration_h + transition_h / 2) / transition_h))
            fed_frac = max(fed_frac, onset * offset)

    # Interpolate parameters
    params = GIStateParams(
        gastric_pH=_lerp(FASTED_STATE.gastric_pH, FED_STATE.gastric_pH, fed_frac),
        gastric_emptying_t50_h=_lerp(FASTED_STATE.gastric_emptying_t50_h,
                                      FED_STATE.gastric_emptying_t50_h, fed_frac),
        bile_salt_mM=_lerp(FASTED_STATE.bile_salt_mM, FED_STATE.bile_salt_mM, fed_frac),
        stomach_fluid_mL=_lerp(FASTED_STATE.stomach_fluid_mL,
                                FED_STATE.stomach_fluid_mL, fed_frac),
        SI_fluid_mL=_lerp(FASTED_STATE.SI_fluid_mL, FED_STATE.SI_fluid_mL, fed_frac),
        splanchnic_flow_factor=_lerp(FASTED_STATE.splanchnic_flow_factor,
                                      FED_STATE.splanchnic_flow_factor, fed_frac),
    )
    return params, fed_frac


def _lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * f


# ===================================================================
# 2. Circadian CYP Expression
# ===================================================================

# Circadian parameters: (amplitude_fraction, acrophase_h)
# amplitude = fraction of baseline (e.g., 0.2 = ±20%)
# acrophase = time of peak activity (24h clock)
# Source: Zhang 2020, Ozturk 2017, Guo 2024
CIRCADIAN_CYP = {
    "CYP3A4":  (0.20, 16.0),   # peak ~4 PM
    "CYP2E1":  (0.25, 12.0),   # peak ~noon
    "CYP1A2":  (0.15, 8.0),    # peak ~8 AM
    "CYP2D6":  (0.10, 14.0),   # modest variation
    "CYP2C9":  (0.10, 15.0),
    "CYP2C19": (0.12, 14.0),
}


def circadian_enzyme_factor(
    cyp_name: str,
    time_of_day_h: float,
) -> float:
    """
    Circadian modulation factor for CYP enzyme activity.

    factor(t) = 1 + A * cos(2π(t - φ) / 24)

    Returns multiplier on CL_int (0.8 to 1.2 for typical enzymes).
    """
    if cyp_name not in CIRCADIAN_CYP:
        return 1.0

    amplitude, acrophase = CIRCADIAN_CYP[cyp_name]
    hour = time_of_day_h % 24.0
    factor = 1.0 + amplitude * math.cos(2.0 * math.pi * (hour - acrophase) / 24.0)
    return max(factor, 0.5)


# ===================================================================
# 3. Pregnancy Physiology
# ===================================================================

# Multipliers by gestational age (weeks) relative to non-pregnant
# Interpolated from Dallmann 2018, Abduljalil 2012, Ke 2014
# Format: {param: [(GA_weeks, multiplier), ...]}

PREGNANCY_MULTIPLIERS = {
    "cardiac_output": [
        (0, 1.0), (13, 1.10), (26, 1.30), (40, 1.40),
    ],
    "GFR": [
        (0, 1.0), (13, 1.19), (26, 1.40), (40, 1.37),
    ],
    "plasma_volume": [
        (0, 1.0), (13, 1.10), (26, 1.30), (40, 1.50),
    ],
    "hematocrit": [
        (0, 1.0), (13, 0.97), (26, 0.91), (40, 0.88),
    ],
    "fu_p_aag": [  # AAG decrease → fu_p increase for AAG-bound drugs
        (0, 1.0), (13, 1.05), (26, 1.10), (40, 1.15),
    ],
    "CYP3A4": [
        (0, 1.0), (13, 1.35), (26, 1.50), (40, 1.75),
    ],
    "CYP2D6": [
        (0, 1.0), (13, 1.25), (26, 1.50), (40, 1.50),
    ],
    "CYP1A2": [
        (0, 1.0), (13, 0.75), (26, 0.65), (40, 0.65),
    ],
    "CYP2C19": [
        (0, 1.0), (13, 0.80), (26, 0.65), (40, 0.65),
    ],
    "CYP2C9": [
        (0, 1.0), (13, 1.0), (26, 1.10), (40, 1.20),
    ],
    "UGT1A4": [
        (0, 1.0), (13, 1.50), (26, 2.00), (40, 2.00),
    ],
    "UGT2B7": [
        (0, 1.0), (13, 1.0), (26, 1.10), (40, 1.20),
    ],
    "liver_blood_flow": [
        (0, 1.0), (13, 1.0), (26, 1.0), (40, 1.0),  # unchanged
    ],
    "body_weight": [
        (0, 1.0), (13, 1.03), (26, 1.10), (40, 1.18),
    ],
}


def pregnancy_factor(param_name: str, gestational_age_weeks: float) -> float:
    """
    Get pregnancy multiplier for a parameter at given gestational age.

    Linear interpolation between defined GA timepoints.
    """
    if param_name not in PREGNANCY_MULTIPLIERS:
        return 1.0

    points = PREGNANCY_MULTIPLIERS[param_name]
    if gestational_age_weeks <= points[0][0]:
        return points[0][1]
    if gestational_age_weeks >= points[-1][0]:
        return points[-1][1]

    for i in range(len(points) - 1):
        ga1, m1 = points[i]
        ga2, m2 = points[i + 1]
        if ga1 <= gestational_age_weeks <= ga2:
            frac = (gestational_age_weeks - ga1) / (ga2 - ga1)
            return m1 + (m2 - m1) * frac

    return 1.0


def get_pregnancy_profile(gestational_age_weeks: float) -> dict:
    """Get all pregnancy multipliers at a given GA."""
    return {
        param: pregnancy_factor(param, gestational_age_weeks)
        for param in PREGNANCY_MULTIPLIERS
    }


def format_pregnancy_profile(ga_weeks: float) -> str:
    """Format pregnancy profile as markdown."""
    profile = get_pregnancy_profile(ga_weeks)
    trimester = "T1" if ga_weeks <= 13 else ("T2" if ga_weeks <= 26 else "T3")

    lines = [
        f"## Pregnancy Physiology — GA {ga_weeks:.0f} weeks ({trimester})\n",
        "| Parameter | Multiplier (vs non-pregnant) |",
        "|-----------|----------------------------|",
    ]
    for param, mult in profile.items():
        direction = "↑" if mult > 1.05 else ("↓" if mult < 0.95 else "→")
        lines.append(f"| {param} | {mult:.2f} {direction} |")

    return "\n".join(lines)
