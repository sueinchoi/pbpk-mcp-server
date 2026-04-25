"""
Lymphatic absorption model for highly lipophilic drugs.

For drugs with logP > 5 and high triglyceride solubility, a significant
fraction of absorbed drug enters the systemic circulation via intestinal
lymphatics (thoracic duct), bypassing hepatic first-pass metabolism.

F_oral = Fa * [(1 - F_lymph) * Fg * Fh + F_lymph]

References:
  - Trevaskis NL et al. Pharm Res 2015;32:2241-2264
  - Trevaskis NL et al. Adv Drug Deliv Rev 2008;60:702-716
  - Dahan A, Hoffman A. Eur J Pharm Sci 2005;24:381-388
  - Charman WN, Porter CJH. Adv Drug Deliv Rev 1996;19:149-169
"""

import math
from dataclasses import dataclass
from typing import Optional


# Lymph flow rate (L/h) — thoracic duct, fasted human
LYMPH_FLOW_RATE = 0.06  # ~1.4 L/day = 0.06 L/h (fasted)
LYMPH_FLOW_RATE_FED = 0.12  # ~2-3 L/day postprandially

# Lymphatic transit time (h) — thoracic duct to venous blood
LYMPH_TRANSIT_TIME = 2.0  # ~1-4 h


@dataclass
class LymphaticParams:
    """Lymphatic absorption parameters."""
    F_lymph: float          # Fraction of absorbed drug via lymphatics
    k_lymph_drain: float    # Lymph drainage rate constant (1/h)
    fed_state: bool = False

    @property
    def lymph_flow(self) -> float:
        return LYMPH_FLOW_RATE_FED if self.fed_state else LYMPH_FLOW_RATE


def estimate_lymphatic_fraction(
    logP: float,
    TG_solubility_mg_per_g: Optional[float] = None,
) -> float:
    """
    Estimate lymphatic absorption fraction from lipophilicity.

    Rules of thumb (Trevaskis 2015, Charman 1996):
      - logP < 5: F_lymph ≈ 0 (negligible lymphatic transport)
      - logP 5-6: F_lymph ≈ 0.05-0.15
      - logP 6-7: F_lymph ≈ 0.15-0.40
      - logP > 7: F_lymph ≈ 0.30-0.80 (if TG-soluble)

    A more quantitative estimate uses TG solubility when available:
      F_lymph = min(1, 0.02 * TG_sol)  for TG_sol in mg/g

    Args:
        logP: Octanol:water partition coefficient.
        TG_solubility_mg_per_g: Solubility in long-chain triglycerides (mg/g).

    Returns:
        F_lymph (0-1)
    """
    if logP < 5.0:
        return 0.0

    if TG_solubility_mg_per_g is not None:
        # Charman/Trevaskis empirical: F_lymph scales with TG solubility
        # ~50 mg/g TG solubility → ~100% lymphatic at logP > 5
        f = min(1.0, 0.02 * TG_solubility_mg_per_g) * (1.0 - math.exp(-(logP - 5.0)))
        return max(0.0, min(f, 0.9))

    # Empirical sigmoid from logP alone (fitted to Trevaskis 2015 data)
    # F_lymph = Fmax / (1 + exp(-k*(logP - logP_mid)))
    Fmax = 0.6
    k = 1.5
    logP_mid = 6.5
    f = Fmax / (1.0 + math.exp(-k * (logP - logP_mid)))
    return max(0.0, min(f, 0.9))


def get_lymphatic_params(
    logP: float,
    TG_solubility: Optional[float] = None,
    fed_state: bool = False,
) -> LymphaticParams:
    """Build lymphatic absorption parameters."""
    F_lymph = estimate_lymphatic_fraction(logP, TG_solubility)
    k_drain = 1.0 / LYMPH_TRANSIT_TIME  # 1/h

    return LymphaticParams(
        F_lymph=F_lymph,
        k_lymph_drain=k_drain,
        fed_state=fed_state,
    )


def oral_bioavailability_with_lymph(
    Fa: float, Fg: float, Fh: float, F_lymph: float,
) -> dict:
    """
    Calculate oral bioavailability accounting for lymphatic bypass.

    F_oral = Fa * [(1 - F_lymph) * Fg * Fh + F_lymph]

    The portal pathway: Fa * (1-F_lymph) * Fg * Fh
    The lymphatic pathway: Fa * F_lymph  (bypasses liver entirely)

    Returns dict with F_oral, F_portal, F_lymph_contribution.
    """
    F_portal = Fa * (1.0 - F_lymph) * Fg * Fh
    F_lymph_contrib = Fa * F_lymph
    F_oral = F_portal + F_lymph_contrib

    return {
        "F_oral": F_oral,
        "F_portal_pathway": F_portal,
        "F_lymph_pathway": F_lymph_contrib,
        "F_lymph_fraction_of_total": F_lymph_contrib / F_oral if F_oral > 0 else 0,
        "Fa": Fa, "Fg": Fg, "Fh": Fh, "F_lymph": F_lymph,
    }
