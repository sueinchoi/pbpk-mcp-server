"""
Fg (fraction escaping gut wall metabolism) prediction.

Methods:
  1. Qgut model (Yang et al. 2007) — Simcyp approach
     Fg = Qgut / (Qgut + fu_gut * CLint_gut)

  2. Static well-stirred gut (simplified)
     Fg = Q_villi / (Q_villi + fu_gut * CLint_gut)

  3. Peff-based absorption/metabolism competition

References:
  - Yang J et al. Curr Drug Metab 2007;8:676-684
  - Paine MF et al. Drug Metab Dispos 2006;34:880-886
  - Gertz M et al. Drug Metab Dispos 2010;38:25-31
"""

import math
from dataclasses import dataclass
from typing import Optional


# ===================================================================
# Constants
# ===================================================================

# Villous blood flow (L/h) — ~50% of mesenteric blood flow to enterocytes
Q_VILLI_DEFAULT = 18.0  # L/h (Simcyp default)

# Small intestine geometry
SI_RADIUS_CM = 1.75     # intestinal lumen radius (cm)
SI_LENGTH_CM = 350.0    # small intestine length (cm)
SI_SURFACE_AREA_CM2 = 2 * math.pi * SI_RADIUS_CM * SI_LENGTH_CM  # ~3850 cm^2

# CYP3A4 total in gut enterocytes
CYP3A4_GUT_NMOL = 70.0  # nmol (Paine 2006)

# Small intestinal enterocyte protein (mg)
ENTEROCYTE_PROTEIN_TOTAL_MG = 3000.0  # approximate


@dataclass
class FgResult:
    """Result of Fg prediction."""
    Fg: float
    Qgut: float       # L/h
    Q_villi: float     # L/h
    CLperm: float      # L/h (permeability clearance)
    CLint_gut: float   # L/h
    fu_gut: float
    method: str

    def to_markdown(self) -> str:
        lines = [
            "## Fg (Gut Wall Bioavailability) Prediction\n",
            f"**Fg = {self.Fg:.4f}** ({self.method})\n",
            "| Parameter | Value | Unit |",
            "|-----------|-------|------|",
            f"| Fg | {self.Fg:.4f} | — |",
            f"| Qgut | {self.Qgut:.2f} | L/h |",
            f"| Q_villi | {self.Q_villi:.2f} | L/h |",
            f"| CLperm | {self.CLperm:.2f} | L/h |",
            f"| CLint_gut | {self.CLint_gut:.2f} | L/h |",
            f"| fu_gut | {self.fu_gut:.4f} | — |",
        ]
        return "\n".join(lines)


# ===================================================================
# Qgut model (Yang et al. 2007)
# ===================================================================

def predict_fg_qgut(
    Peff: float,
    CLint_gut: float,
    fu_gut: float,
    Q_villi: float = Q_VILLI_DEFAULT,
    SA_cm2: float = SI_SURFACE_AREA_CM2,
) -> FgResult:
    """
    Predict Fg using the Qgut model (Yang et al. 2007).

    Fg = Qgut / (Qgut + fu_gut * CLint_gut)

    Qgut = (Q_villi * CLperm) / (Q_villi + CLperm)

    Args:
        Peff: Effective human jejunal permeability (x10^-4 cm/s).
        CLint_gut: Intrinsic clearance in gut wall (L/h).
        fu_gut: Fraction unbound in enterocytes.
        Q_villi: Villous blood flow (L/h).
        SA_cm2: Effective surface area of small intestine (cm^2).

    Returns:
        FgResult with Fg and intermediate parameters.
    """
    # CLperm = Peff * SA (convert to L/h)
    # Peff in 10^-4 cm/s → cm/s: * 1e-4
    # SA in cm^2, result in cm^3/s → L/h: * 3600 / 1000
    CLperm = Peff * 1e-4 * SA_cm2 * 3600.0 / 1000.0  # L/h

    # Qgut: effective gut blood flow accounting for permeability
    Qgut = (Q_villi * CLperm) / (Q_villi + CLperm) if (Q_villi + CLperm) > 0 else 0

    # Fg
    denom = Qgut + fu_gut * CLint_gut
    Fg = Qgut / denom if denom > 0 else 1.0
    Fg = min(max(Fg, 0.0), 1.0)

    return FgResult(
        Fg=Fg, Qgut=Qgut, Q_villi=Q_villi,
        CLperm=CLperm, CLint_gut=CLint_gut,
        fu_gut=fu_gut, method="Qgut (Yang 2007)",
    )


def predict_fg_wellstirred(
    CLint_gut: float,
    fu_gut: float,
    Q_villi: float = Q_VILLI_DEFAULT,
) -> FgResult:
    """
    Simple well-stirred gut model (no permeability consideration).

    Fg = Q_villi / (Q_villi + fu_gut * CLint_gut)
    """
    denom = Q_villi + fu_gut * CLint_gut
    Fg = Q_villi / denom if denom > 0 else 1.0
    Fg = min(max(Fg, 0.0), 1.0)

    return FgResult(
        Fg=Fg, Qgut=Q_villi, Q_villi=Q_villi,
        CLperm=float("inf"), CLint_gut=CLint_gut,
        fu_gut=fu_gut, method="Well-stirred gut",
    )


def scale_gut_clint_from_liver(
    CLint_liver_per_mg: float,
    cyp3a4_fm: float = 1.0,
) -> float:
    """
    Estimate gut wall CLint from hepatic microsomal CLint (CYP3A4 only).

    CLint_gut = CLint_liver_whole * fm_CYP3A4 * (CYP3A4_gut / CYP3A4_liver)

    Returns CLint_gut in L/h.
    """
    result = scale_gut_clint_per_cyp(CLint_liver_per_mg, {"CYP3A4": cyp3a4_fm})
    return sum(result.values())


def scale_gut_clint_per_cyp(
    CLint_liver_per_mg: float,
    fm_per_cyp: dict[str, float],
    mppgl: float = 40.0,
    liver_weight_g: float = 1800.0,
) -> dict[str, float]:
    """
    Scale hepatic CLint to per-CYP gut wall CLint using relative enzyme content.

    For each CYP:
      CLint_gut_CYP = CLint_liver_whole * fm_CYP * (CYP_gut_total / CYP_liver_total)

    Where:
      CYP_liver_total = abundance_pmol_per_mg * MPPGL * liver_weight
      CYP_gut_total = from Paine 2006 (nmol total in small intestine)

    Args:
        CLint_liver_per_mg: HLM CLint (uL/min/mg protein).
        fm_per_cyp: {"CYP3A4": 0.8, "CYP2C9": 0.15, "UGT": 0.05}
        mppgl: Microsomal protein per gram liver (mg/g).
        liver_weight_g: Liver weight (g).

    Returns:
        {"CYP3A4": CLint_gut_3A4_L_per_h, "CYP2C9": ..., ...}

    References:
        - Paine MF et al. Drug Metab Dispos 2006;34:880-886
        - Rodrigues AD. Biochem Pharmacol 1999;57:465-480
    """
    # CYP abundances in liver (pmol/mg protein, Rodrigues 1999)
    CYP_LIVER_ABUNDANCE = {
        "CYP3A4": 108.0, "CYP2C9": 96.0, "CYP2C19": 14.0,
        "CYP2D6": 10.0, "CYP1A2": 52.0, "CYP2E1": 49.0,
    }

    # CYP total content in small intestine (nmol, Paine 2006)
    CYP_GUT_TOTAL_NMOL = {
        "CYP3A4": 70.0,   # dominant
        "CYP2C9": 3.0,    # ~15% of intestinal P450 pie
        "CYP2C19": 1.0,
        "CYP2D6": 0.5,
        "CYP1A2": 0.0,    # not expressed in gut
        "CYP2E1": 0.0,
        "UGT": 50.0,      # UGT1A1/1A8/1A10 (gut-specific UGTs, Strassburg 2000)
    }

    CLint_liver_whole = CLint_liver_per_mg * mppgl * liver_weight_g  # uL/min

    result = {}
    for cyp, fm in fm_per_cyp.items():
        gut_nmol = CYP_GUT_TOTAL_NMOL.get(cyp, 0.0)
        liver_abundance = CYP_LIVER_ABUNDANCE.get(cyp, 50.0)  # default if unknown
        liver_nmol = liver_abundance * mppgl * liver_weight_g / 1e3  # pmol→nmol

        if liver_nmol > 0:
            ratio = gut_nmol / liver_nmol
        else:
            ratio = 0.0

        clint_gut_cyp = CLint_liver_whole * fm * ratio  # uL/min
        result[cyp] = clint_gut_cyp * 60.0 / 1e6  # → L/h

    return result


# ===================================================================
# Peff prediction from Caco-2
# ===================================================================

def peff_from_caco2(
    Papp_cm_per_s: float,
    method: str = "sun",
) -> float:
    """
    Predict human jejunal Peff from Caco-2 Papp.

    Args:
        Papp_cm_per_s: Caco-2 apparent permeability (cm/s).
        method: "sun" (Sun 2002) or "simcyp" (Simcyp default).

    Returns:
        Peff in 10^-4 cm/s.
    """
    if method == "sun":
        # Sun et al. 2002: log(Peff) = 0.6836 * log(Papp) - 0.5579
        # Input: Papp in cm/s, Output: Peff in 10^-4 cm/s
        log_Papp_e6 = math.log10(Papp_cm_per_s * 1e6)  # Papp in 10^-6 cm/s
        log_Peff_e4 = 0.6836 * log_Papp_e6 - 0.5579
        return 10.0 ** log_Peff_e4  # in 10^-4 cm/s

    elif method == "simcyp":
        # Simcyp default
        log_Peff = 0.4926 * math.log10(Papp_cm_per_s * 1e6) - 0.1454
        return 10.0 ** log_Peff

    # Default: simple scaling factor
    return Papp_cm_per_s * 1e4 * 3.6  # rough 3.6x scaling


def predict_fa_cat(
    Peff_e4: float,
    SITT: float = 3.32,
    n_compartments: int = 7,
    R_cm: float = SI_RADIUS_CM,
) -> float:
    """
    Predict Fa using CAT model (Yu & Amidon 1999).

    Fa = 1 - (1 + ka/kt)^(-n)

    Args:
        Peff_e4: Effective permeability in 10^-4 cm/s.
        SITT: Small intestinal transit time (h).
        n_compartments: Number of transit compartments.
        R_cm: Intestinal radius (cm).

    Returns:
        Fa (fraction absorbed from lumen).
    """
    # ka = 2 * Peff / R (in 1/s, then convert to 1/h)
    ka = 2.0 * Peff_e4 * 1e-4 / R_cm * 3600.0  # 1/h

    # kt = n / SITT
    kt = n_compartments / SITT  # 1/h

    # Fa = 1 - (1 + ka/kt)^(-n)
    Fa = 1.0 - (1.0 + ka / kt) ** (-n_compartments) if kt > 0 else 1.0
    return min(max(Fa, 0.0), 1.0)
