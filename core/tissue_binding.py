"""
Tissue binding prediction — fraction unbound in tissue sub-compartments.

Implements PK-Sim/Simcyp-style fu predictions:
  - fu_tissue: overall fraction unbound in tissue
  - fu_interstitial: fraction unbound in interstitial (extracellular) space
  - fu_intracellular: fraction unbound in intracellular space
  - fu_gut: fraction unbound in enterocytes (gut wall)
  - fu_inc: fraction unbound in microsomal incubation (IVIVE)

Key relationships:
  Kp = fu_p / fu_tissue  (at equilibrium, perfusion-limited)
  Therefore: fu_tissue = fu_p / Kp

For permeability-limited model:
  fu_interstitial and fu_intracellular are needed separately.

References:
  - Rodgers T, Rowland M. Pharm Res 2007;24:918-933
  - Schmitt W. Toxicol In Vitro 2008;22:457-467
  - Austin RP et al. Drug Metab Dispos 2002;30:1497-1503 (fu_inc)
  - Poulin P, Theil F-P. J Pharm Sci 2002;91:1358-1370
"""

import math
from typing import Optional

from .compound import CompoundSpec, CompoundType
from .physiology import (
    Organ,
    TISSUE_COMPOSITION,
    PLASMA_COMPOSITION,
    TISSUE_PROTEIN_FRACTIONS,
    TISSUE_INTERSTITIAL_FRACTIONS,
)
from .partition_coeff import (
    KpMethod,
    predict_kp_single,
    _P_ow,
    _P_vo,
    _ionization_excess,
    _is_base,
    ALPHA_LIPID,
    AP_FACTOR_BASE,
    AP_FACTOR_ACID,
)


def predict_fu_tissue(
    compound: CompoundSpec,
    organ: Organ,
    method: KpMethod = KpMethod.RODGERS_ROWLAND,
    kp_override: Optional[float] = None,
) -> float:
    """
    Predict overall fraction unbound in tissue.

    fu_tissue = fu_p / Kp

    At equilibrium (perfusion-limited), unbound concentrations are equal
    in all compartments. The Kp reflects total tissue binding.

    Args:
        compound: Drug compound specification.
        organ: Target organ.
        method: Kp prediction method used.
        kp_override: Use this Kp instead of predicting.

    Returns:
        fu_tissue (dimensionless, 0 to 1)
    """
    if kp_override is not None:
        Kp = kp_override
    else:
        Kp = predict_kp_single(compound, organ, method)

    if Kp <= 0:
        return 1.0

    fu_t = compound.fu_p / Kp
    return min(max(fu_t, 1e-6), 1.0)


def predict_fu_interstitial(
    compound: CompoundSpec,
    organ: Organ,
) -> float:
    """
    Predict fraction unbound in interstitial (extracellular) space.

    In PK-Sim: interstitial space contains albumin at a fraction
    of the plasma concentration (albumin ratio, AR).

    fu_int = 1 / (1 + AR * (1/fu_p - 1))

    For tissues with no albumin access (e.g., brain behind BBB):
    fu_int ≈ 1.0 (no protein binding in interstitial space)

    Args:
        compound: Drug compound.
        organ: Target organ.

    Returns:
        fu_interstitial (dimensionless, 0 to 1)
    """
    tc = TISSUE_COMPOSITION[organ]
    fu_p = compound.fu_p
    AR = tc.get("albumin_ratio", 0.5)

    if fu_p >= 1.0 or AR <= 0:
        return 1.0

    fu_int = 1.0 / (1.0 + AR * (1.0 / fu_p - 1.0))
    return min(max(fu_int, 1e-6), 1.0)


def predict_fu_intracellular(
    compound: CompoundSpec,
    organ: Organ,
) -> float:
    """
    Predict fraction unbound in intracellular space.

    Uses the Schmitt (2008) approach: intracellular binding is determined
    by partitioning into intracellular lipids, phospholipids, acidic
    phospholipids, and proteins.

    fu_cell = 1 / K_cell

    Where K_cell = f_W + K_NL*f_NL + K_NP*f_NP + K_AP*f_AP + K_prot*f_prot
    (all fractions relative to cell volume, not tissue volume)

    Args:
        compound: Drug compound.
        organ: Target organ.

    Returns:
        fu_intracellular (dimensionless, 0 to 1)
    """
    tc = TISSUE_COMPOSITION[organ]
    frac_data = TISSUE_INTERSTITIAL_FRACTIONS.get(organ, {"F_int": 0.2, "F_cell": 0.8})
    F_cell = frac_data["F_cell"]

    if F_cell <= 0:
        return 1.0

    logP = compound.logP
    pKa = compound.pKa
    ctype = compound.compound_type
    pH_IW = tc["pH_IW"]

    # Partition coefficients (Schmitt approach)
    K_n_pl = _P_ow(logP)
    K_protein = 0.163 + 0.0221 * K_n_pl

    W = _ionization_excess(pKa, pH_IW, ctype)

    # Neutral lipid PC (ionization-corrected)
    if organ == Organ.ADIPOSE:
        K_n_l = _P_vo(logP)
        if W > 0:
            K_n_l *= ((1.0 - ALPHA_LIPID) / (1.0 + W) + ALPHA_LIPID)
    else:
        if W > 0:
            K_n_l = K_n_pl * ((1.0 - ALPHA_LIPID) / (1.0 + W) + ALPHA_LIPID)
        else:
            K_n_l = K_n_pl

    # Acidic phospholipid PC (charge-dependent)
    if _is_base(ctype) and W > 0:
        K_a_pl = K_n_pl * (1.0 / (1.0 + W) + AP_FACTOR_BASE * (1.0 - 1.0 / (1.0 + W)))
    elif ctype == CompoundType.ACID and W > 0:
        K_a_pl = K_n_pl * (1.0 / (1.0 + W) + AP_FACTOR_ACID * (1.0 - 1.0 / (1.0 + W)))
    else:
        K_a_pl = K_n_pl

    # Intracellular fractions (normalized to cell volume)
    f_IW_cell = tc["f_IW"] / F_cell
    f_NL_cell = tc["f_NL"] / F_cell
    f_NP_cell = tc["f_NP"] / F_cell
    f_AP_cell = tc["f_AP"] / F_cell
    f_prot_cell = TISSUE_PROTEIN_FRACTIONS.get(organ, 0.1) / F_cell

    K_cell = (
        f_IW_cell
        + K_n_l * f_NL_cell
        + K_n_pl * f_NP_cell
        + K_a_pl * f_AP_cell
        + K_protein * f_prot_cell
    )

    fu_cell = 1.0 / K_cell if K_cell > 0 else 1.0
    return min(max(fu_cell, 1e-6), 1.0)


def predict_fu_gut(compound: CompoundSpec) -> float:
    """
    Predict fraction unbound in enterocytes (gut wall).

    Uses the same approach as fu_intracellular for the gut organ.
    Important for first-pass gut wall metabolism calculations.

    fu_gut = fu_intracellular(gut)

    Returns:
        fu_gut (dimensionless)
    """
    return predict_fu_intracellular(compound, Organ.GUT)


def predict_fu_inc(
    compound: CompoundSpec,
    microsomal_protein_conc: float = 1.0,
    method: str = "austin",
) -> float:
    """
    Predict fraction unbound in microsomal incubation (fu_inc).

    Methods:
      "austin": Austin et al. (2002) Drug Metab Dispos 30:1497-1503
        log(1/fu-1) = 0.072*logP^2 + 0.067*logP - 1.126 + log(C_prot)
      "hallifax": Hallifax & Houston (2006) Drug Metab Dispos 34:724-726
        log(1/fu-1) = 0.0566*logP^2 + 0.0345*logP - 1.1279 + log(C_prot)
        Better for logP > 3.5.

    Args:
        compound: Drug compound.
        microsomal_protein_conc: Protein concentration (mg/mL).
        method: "austin" or "hallifax".

    Returns:
        fu_inc (dimensionless, 0 to 1)
    """
    logP = compound.logP

    if method == "hallifax":
        # Hallifax & Houston 2006
        log_binding = (
            0.0566 * logP ** 2
            + 0.0345 * logP
            - 1.1279
            + math.log10(max(microsomal_protein_conc, 0.01))
        )
    else:
        # Austin et al. 2002 (default)
        log_binding = (
            0.072 * logP ** 2
            + 0.067 * logP
            - 1.126
            + math.log10(max(microsomal_protein_conc, 0.01))
        )

    binding_ratio = 10.0 ** log_binding
    fu_inc = 1.0 / (1.0 + binding_ratio)

    return min(max(fu_inc, 0.001), 1.0)


def predict_fu_hepatocyte(
    compound: CompoundSpec,
    hepatocyte_conc: float = 1.0e6,
) -> float:
    """
    Predict fraction unbound in hepatocyte incubation.

    Kilford et al. (2008):
      fu_hep = 1 / (1 + 125 * VR * (1/fu_inc - 1))

    Where VR = volume ratio (cell to incubation medium),
    and fu_inc is predicted from logP.

    For practical IVIVE: fu_hep ≈ fu_inc for most compounds.

    Args:
        compound: Drug compound.
        hepatocyte_conc: Hepatocyte concentration (cells/mL).

    Returns:
        fu_hepatocyte (dimensionless)
    """
    fu_inc = predict_fu_inc(compound)
    VR = hepatocyte_conc * 3.9e-9  # approximate cell volume ratio
    if fu_inc >= 1.0:
        return 1.0
    fu_hep = 1.0 / (1.0 + 125.0 * VR * (1.0 / fu_inc - 1.0))
    return min(max(fu_hep, 0.001), 1.0)


def predict_all_fu(
    compound: CompoundSpec,
    method: KpMethod = KpMethod.RODGERS_ROWLAND,
) -> dict:
    """
    Predict all tissue binding parameters.

    Returns dict with:
      - fu_tissue: {Organ: fu} for all 13 organs
      - fu_interstitial: {Organ: fu_int} for all organs
      - fu_intracellular: {Organ: fu_cell} for all organs
      - fu_gut: scalar
      - fu_inc: scalar (at 1 mg/mL microsomal protein)
    """
    from .physiology import Organ

    fu_tissue = {}
    fu_int = {}
    fu_cell = {}

    for organ in Organ:
        fu_tissue[organ] = predict_fu_tissue(compound, organ, method)
        fu_int[organ] = predict_fu_interstitial(compound, organ)
        fu_cell[organ] = predict_fu_intracellular(compound, organ)

    return {
        "fu_tissue": fu_tissue,
        "fu_interstitial": fu_int,
        "fu_intracellular": fu_cell,
        "fu_gut": predict_fu_gut(compound),
        "fu_inc_1mgmL": predict_fu_inc(compound, 1.0),
        "fu_inc_0.5mgmL": predict_fu_inc(compound, 0.5),
    }


def format_fu_table(compound: CompoundSpec, method: KpMethod = KpMethod.RODGERS_ROWLAND) -> str:
    """Format tissue binding predictions as markdown."""
    all_fu = predict_all_fu(compound, method)

    lines = [
        f"## Tissue Binding — {compound.name}\n",
        f"fu_p = {compound.fu_p}, logP = {compound.logP}\n",
        "### Organ-level\n",
        "| Tissue | fu_tissue | fu_interstitial | fu_intracellular |",
        "|--------|-----------|-----------------|------------------|",
    ]

    for organ in Organ:
        fu_t = all_fu["fu_tissue"][organ]
        fu_i = all_fu["fu_interstitial"][organ]
        fu_c = all_fu["fu_intracellular"][organ]
        lines.append(
            f"| {organ.value.capitalize():10s} | {fu_t:.4f} | {fu_i:.4f} | {fu_c:.6f} |"
        )

    lines.extend([
        "",
        "### In Vitro Binding\n",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| fu_gut (enterocyte) | {all_fu['fu_gut']:.4f} |",
        f"| fu_inc (1.0 mg/mL) | {all_fu['fu_inc_1mgmL']:.4f} |",
        f"| fu_inc (0.5 mg/mL) | {all_fu['fu_inc_0.5mgmL']:.4f} |",
    ])

    return "\n".join(lines)
