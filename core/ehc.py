"""
Enterohepatic Recirculation (EHC) model.

Adds biliary excretion from liver → gallbladder → intestinal deconjugation
→ reabsorption back to portal vein.

Compartments added to ODE system:
  [0] Bile duct (transit from liver to gallbladder)
  [1] Gallbladder
  [2] Intestinal deconjugation pool

Gallbladder emptying is meal-triggered (Gaussian pulse).

References:
  - Roberts MS et al. Clin Pharmacokinet 2002;41:751-790
  - Metry M et al. Pharmaceutics 2021;13:1-18 (cabozantinib PBPK)
  - Yang J et al. Clin Pharmacol Ther 2016;100:413-416
  - Lehr T et al. AAPS J 2010;12:163-172
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


N_EHC_STATES = 3  # bile_duct, gallbladder, deconjugation_pool


@dataclass
class EHCParams:
    """Enterohepatic recirculation parameters."""

    # Biliary excretion
    CL_bile: float = 0.0          # Biliary clearance (L/h), referenced to unbound liver conc
    f_bile_parent: float = 1.0    # Fraction excreted as parent drug (vs conjugate)

    # Gallbladder
    f_gallbladder: float = 0.6    # Fraction of bile stored in gallbladder (vs continuous)
    k_bile_transit: float = 2.0   # Bile duct transit rate (1/h)

    # Gallbladder emptying (meal-triggered)
    meal_times_h: tuple = (0.0, 5.0, 10.0)  # Default: breakfast, lunch, dinner
    gb_emptying_duration_h: float = 0.5       # Emptying duration per meal
    gb_emptying_sigma_h: float = 0.2          # Gaussian sigma

    # Intestinal deconjugation (for glucuronide/sulfate conjugates)
    k_deconjugation: float = 0.5  # Deconjugation rate (1/h), bacterial enzymes
    f_reabsorption: float = 0.8   # Fraction of deconjugated drug reabsorbed
    k_fecal_loss: float = 0.1     # Fecal elimination rate (1/h)

    # Reabsorption kinetics
    k_reabsorption: float = 1.0   # Rate of reabsorption from intestine (1/h)


def gallbladder_emptying_rate(t: float, params: EHCParams) -> float:
    """
    Gallbladder emptying rate as sum of Gaussian pulses at meal times.

    k_gb(t) = Σ k_max * exp(-(t - t_meal)^2 / (2σ^2))
    """
    k_max = 1.0 / params.gb_emptying_duration_h  # peak rate
    rate = 0.0
    for t_meal in params.meal_times_h:
        dt = t - t_meal
        if abs(dt) < 3.0 * params.gb_emptying_sigma_h + params.gb_emptying_duration_h:
            rate += k_max * math.exp(-dt * dt / (2.0 * params.gb_emptying_sigma_h ** 2))
    return rate


def ehc_rhs(
    y_ehc: np.ndarray,
    t: float,
    biliary_rate: float,
    params: EHCParams,
) -> tuple[np.ndarray, float]:
    """
    EHC ODE right-hand side.

    Args:
        y_ehc: [bile_duct, gallbladder, deconj_pool] amounts (mg)
        t: Current time (h)
        biliary_rate: Rate of drug entering bile from liver (mg/h)
        params: EHC parameters

    Returns:
        (dy_ehc, reabsorption_rate): derivatives and rate returning to portal vein
    """
    dy = np.zeros(N_EHC_STATES)

    A_bile = max(y_ehc[0], 0.0)
    A_gb = max(y_ehc[1], 0.0)
    A_deconj = max(y_ehc[2], 0.0)

    # Bile duct: receives from liver, transits to gallbladder/intestine
    continuous_fraction = 1.0 - params.f_gallbladder
    dy[0] = biliary_rate - params.k_bile_transit * A_bile

    # Gallbladder: receives from bile duct, empties at meals
    k_gb = gallbladder_emptying_rate(t, params)
    dy[1] = params.f_gallbladder * params.k_bile_transit * A_bile - k_gb * A_gb

    # Deconjugation pool: receives from gallbladder + continuous bile
    gb_to_intestine = k_gb * A_gb
    continuous_to_intestine = continuous_fraction * params.k_bile_transit * A_bile
    total_to_intestine = gb_to_intestine + continuous_to_intestine

    # Deconjugation and reabsorption
    deconj_rate = params.k_deconjugation * A_deconj  # mg/h
    fecal_rate = params.k_fecal_loss * A_deconj       # mg/h
    # Reabsorption: fraction of deconjugated drug that is reabsorbed (mg/h)
    reabs_rate = deconj_rate * params.f_reabsorption   # mg/h (not × k_reabs)

    dy[2] = total_to_intestine - deconj_rate - fecal_rate

    return dy, reabs_rate


def compute_biliary_rate(
    C_liver: float,
    V_liver: float,
    Kp_liver: float,
    fu_p: float,
    CL_bile: float,
) -> float:
    """
    Compute biliary excretion rate from liver concentration.

    Rate = CL_bile * fu_p * C_plasma_liver
    where C_plasma_liver = C_liver / Kp_liver
    """
    if V_liver <= 0 or Kp_liver <= 0:
        return 0.0
    C_plasma_liver = C_liver / (V_liver * Kp_liver)  # A_liver / (V * Kp) = C_plasma
    return CL_bile * fu_p * C_plasma_liver
