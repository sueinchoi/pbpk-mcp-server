"""
Tissue:plasma partition coefficient (Kp) prediction — 5 methods.

Methods implemented:
  1. Rodgers & Rowland (2005, 2006) — R&R
     Ka_AP from RBC partitioning (Type 1) or albumin binding (Type 2)
  2. Schmitt (2008) — SCH
     3 lipid sub-fractions, ionization-dependent AP binding (20x for bases)
  3. Poulin & Theil (2002) — PT
     Original method, simplest
  4. Berezhkovskiy (2004) — PTB
     Corrected Poulin-Theil with fu in water term
  5. PK-Sim Standard (Willmann 2003) — PKSIM
     Single lipid fraction, empirical K_protein, no pKa

References:
  - Rodgers T, Rowland M. Drug Metab Dispos 2005;33:1325-1332 (Type 1)
  - Rodgers T, Rowland M. Pharm Res 2006;23:56-70 (Type 2)
  - Schmitt W. Toxicol In Vitro 2008;22:457-467
  - Poulin P, Theil F-P. J Pharm Sci 2002;91:1358-1370
  - Berezhkovskiy LM. J Pharm Sci 2004;93:1628-1640
  - Willmann S et al. J Med Chem 2004;47:4022-4031
  - R implementations: github.com/metrumresearchgroup/PBPK_PC
"""

import math
from enum import Enum
from typing import Optional

from .compound import CompoundSpec, CompoundType
from .physiology import (
    Organ,
    TISSUE_COMPOSITION,
    PLASMA_COMPOSITION,
    RBC_COMPOSITION,
    HEMATOCRIT,
    TISSUE_PROTEIN_FRACTIONS,
    TISSUE_INTERSTITIAL_FRACTIONS,
)


# ===================================================================
# Enums and constants
# ===================================================================

class KpMethod(str, Enum):
    RODGERS_ROWLAND = "rodgers_rowland"
    LUKACOVA = "lukacova"          # R&R with continuous AP/PR weighting
    SCHMITT = "schmitt"
    POULIN_THEIL = "poulin_theil"
    BEREZHKOVSKIY = "berezhkovskiy"
    PKSIM_STANDARD = "pksim_standard"
    KP_MEMBRANE = "kp_membrane"


# Schmitt (2008): ratio of ionized to neutral species' lipid distribution
ALPHA_LIPID = 0.001

# Schmitt (2008): AP binding enhancement factors
AP_FACTOR_BASE = 20.0    # Cationic species bind 20x to acidic phospholipids
AP_FACTOR_ACID = 0.05    # Anionic species bind 0.05x (repulsion)
AP_FACTOR_NEUTRAL = 1.0  # No charge effect


# ===================================================================
# Helper functions
# ===================================================================

def _P_ow(logP: float) -> float:
    """n-Octanol:water partition coefficient (neutral species)."""
    return 10.0 ** logP


def _P_vo(logP: float) -> float:
    """Vegetable oil:water partition coefficient.
    logP_vo = 1.115 * logP_ow - 1.35 (Graham & Hein 1949)."""
    return 10.0 ** (1.115 * logP - 1.35)


def _D_vo(logP: float, pKa: float, pH: float, ctype: CompoundType) -> float:
    """Ionization-corrected vegetable oil:water distribution coefficient.
    Used for adipose tissue in Poulin-Theil and Berezhkovskiy methods."""
    logD_vo = 1.115 * logP - 1.35
    W = _ionization_excess(pKa, pH, ctype)
    if W > 0:
        logD_vo -= math.log10(1.0 + W)
    return 10.0 ** logD_vo


def _ionization_excess(pKa: float, pH: float, ctype: CompoundType) -> float:
    """Ionization term W = ratio of ionized to unionized species.
    For bases: W = 10^(pKa - pH)
    For acids: W = 10^(pH - pKa)
    For neutrals: W = 0
    """
    if ctype in (CompoundType.STRONG_BASE, CompoundType.MODERATE_BASE, CompoundType.WEAK_BASE):
        return 10.0 ** (pKa - pH)
    elif ctype == CompoundType.ACID:
        return 10.0 ** (pH - pKa)
    return 0.0


def _ionization_ratio(pKa: float, pH: float, ctype: CompoundType) -> float:
    """X = 1 + W = total/unionized ratio."""
    return 1.0 + _ionization_excess(pKa, pH, ctype)


def _fraction_neutral(pKa: float, pH: float, ctype: CompoundType) -> float:
    """Fraction of drug in neutral (unionized) form at given pH."""
    X = _ionization_ratio(pKa, pH, ctype)
    return 1.0 / X


def _is_base(ctype: CompoundType) -> bool:
    return ctype in (CompoundType.STRONG_BASE, CompoundType.MODERATE_BASE, CompoundType.WEAK_BASE)


def _is_acid(ctype: CompoundType) -> bool:
    return ctype == CompoundType.ACID


# ===================================================================
# Method 1: Rodgers & Rowland (2005, 2006)
# ===================================================================

def _kp_rodgers_rowland(compound: CompoundSpec, organ: Organ) -> float:
    """
    Rodgers & Rowland Kp prediction.

    Type 1 (strong bases, pKa >= 7):
      Ka_AP derived from blood:plasma ratio via RBC partitioning.
      Kp = fu_p * [f_EW + (X_t/X_p)*f_IW + (X_t/X_p)*Ka_AP*f_AP + lip]

    Type 2 (acids, weak/moderate bases, neutrals, zwitterions):
      Ka_PR derived from fu_p (protein binding).
      Kp = fu_p * [f_EW + (X_t/X_p)*f_IW + lip + AR*(1/fu_p-1)*f_EW]/(1/fu_p) ... simplified form
    """
    tc = TISSUE_COMPOSITION[organ]
    pc = PLASMA_COMPOSITION
    rbc = RBC_COMPOSITION

    fu_p = compound.fu_p
    logP = compound.logP
    pKa = compound.pKa
    ctype = compound.compound_type
    R_bp = compound.R_bp
    HCT = HEMATOCRIT

    # Use P_vo for adipose neutral lipids (PK-Sim convention), P_ow for others
    P = _P_ow(logP)
    P_adipose = _P_vo(logP) if organ == Organ.ADIPOSE else P

    pH_p = pc["pH"]       # 7.4
    pH_IW = tc["pH_IW"]   # tissue intracellular pH
    pH_rbc = rbc["pH"]    # 7.22

    X_p = _ionization_ratio(pKa, pH_p, ctype)
    X_IW = _ionization_ratio(pKa, pH_IW, ctype)
    X_rbc = _ionization_ratio(pKa, pH_rbc, ctype)

    # Tissue fractions
    f_EW = tc["f_EW"]
    f_IW = tc["f_IW"]
    f_NL = tc["f_NL"]
    f_NP = tc["f_NP"]
    f_AP = tc["f_AP"]

    # Lipid partitioning in tissue
    lip_tissue = P_adipose * f_NL + (0.3 * P + 0.7) * f_NP

    if ctype == CompoundType.STRONG_BASE:
        # --- Type 1: Ka_AP from RBC partitioning ---
        # Kpu_bc = (HCT - 1 + R_bp) / (HCT * fu_p)
        Kpu_bc = (HCT - 1.0 + R_bp) / (HCT * fu_p)

        # RBC lipid terms
        lip_rbc = P * rbc["f_NL"] + (0.3 * P + 0.7) * rbc["f_NP"]

        # Ka_AP = [Kpu_bc - X_rbc/X_p * f_IW_rbc - lip_rbc/X_p] * X_p / (f_AP_rbc * (X_rbc - 1))
        Z = X_rbc - 1.0  # ionized fraction ratio at RBC pH
        if Z > 1e-10 and rbc["f_AP"] > 0:
            Ka_AP = (
                (Kpu_bc - (X_rbc / X_p) * rbc["f_IW"] - lip_rbc / X_p)
                * X_p / (rbc["f_AP"] * Z)
            )
            Ka_AP = max(Ka_AP, 0.0)
        else:
            # Fallback: derive from fu_p (less accurate)
            lip_p = P * pc["f_NL"] + (0.3 * P + 0.7) * pc["f_NP"]
            Ka_AP = max((1.0 / fu_p - 1.0 - lip_p) / (pc["f_AP"] * X_p), 0.0)

        # R&R 2005 Eq. 2:
        # Kpu = f_EW + (X_IW/X_p)*f_IW + Ka_AP*f_AP*(X_IW-1)/X_p + lip/X_p
        # The AP term uses (X_IW - 1): only IONIZED species bind to AP.
        Kp = fu_p * (
            f_EW
            + (X_IW / X_p) * f_IW
            + Ka_AP * f_AP * (X_IW - 1.0) / X_p
            + lip_tissue / X_p
        )

    else:
        # --- Type 2: Protein binding in interstitial space ---
        AR = tc.get("albumin_ratio", 0.5)

        # Ka_PR from fu_p: protein-binding constant
        lip_p = P * pc["f_NL"] + (0.3 * P + 0.7) * pc["f_NP"]
        Ka_PR = max((1.0 / fu_p) - 1.0 - lip_p / X_p, 0.0)

        # Kpu = f_EW + (X_IW/X_p)*f_IW + lip/X_p + Ka_PR*AR*f_EW*(X_IW/X_p)
        Kpu = (
            f_EW
            + (X_IW / X_p) * f_IW
            + lip_tissue / X_p
            + Ka_PR * AR * f_EW
        )

        Kp = Kpu * fu_p

    return max(Kp, 0.01)


# ===================================================================
# Method 1b: Lukacova Combined (R&R without pKa=7 cutoff)
# ===================================================================

def _kp_lukacova(compound: CompoundSpec, organ: Organ) -> float:
    """
    Lukacova combined equation — eliminates the R&R pKa=7 hard cutoff.

    Instead of switching between Type 1 (Ka_AP) and Type 2 (Ka_PR) at
    pKa=7, uses the IONIZED FRACTION at plasma pH as a continuous weight:

      f_cation = fraction cationic at pH 7.4
      Ka_effective = f_cation * Ka_AP + (1 - f_cation) * Ka_PR * AR

    This gives smooth transition for moderate bases (pKa 4-8) that caused
    Kp and Vss overestimation in the original R&R method.

    References:
      - Lukacova V et al. AAPS 2008 Annual Meeting (poster)
      - GastroPlus tissue distribution equations (Method 2)
      - Graham H et al. Drug Metab Dispos 2012;40:1131-1141
    """
    tc = TISSUE_COMPOSITION[organ]
    pc = PLASMA_COMPOSITION
    rbc = RBC_COMPOSITION

    fu_p = compound.fu_p
    logP = compound.logP
    pKa = compound.pKa
    ctype = compound.compound_type
    R_bp = compound.R_bp
    HCT = HEMATOCRIT

    P = _P_ow(logP)
    P_adipose = _P_vo(logP) if organ == Organ.ADIPOSE else P

    pH_p = pc["pH"]
    pH_IW = tc["pH_IW"]
    pH_rbc = rbc["pH"]

    X_p = _ionization_ratio(pKa, pH_p, ctype)
    X_IW = _ionization_ratio(pKa, pH_IW, ctype)
    X_rbc = _ionization_ratio(pKa, pH_rbc, ctype)

    f_EW = tc["f_EW"]
    f_IW = tc["f_IW"]
    f_NL = tc["f_NL"]
    f_NP = tc["f_NP"]
    f_AP = tc["f_AP"]
    AR = tc.get("albumin_ratio", 0.5)

    lip_tissue = P_adipose * f_NL + (0.3 * P + 0.7) * f_NP
    lip_p = P * pc["f_NL"] + (0.3 * P + 0.7) * pc["f_NP"]

    # Fraction cationic at plasma pH (continuous weight)
    if _is_base(ctype):
        f_cation = (X_p - 1.0) / X_p  # ionized fraction at pH 7.4
    else:
        f_cation = 0.0  # acids/neutrals: no cationic species

    # Ka_AP from RBC (same as R&R Type 1, but may be 0 for non-bases)
    Ka_AP = 0.0
    if f_cation > 0.01 and R_bp > 0:
        Kpu_bc = (HCT - 1.0 + R_bp) / (HCT * fu_p)
        lip_rbc = P * rbc["f_NL"] + (0.3 * P + 0.7) * rbc["f_NP"]
        Z = X_rbc - 1.0
        if Z > 1e-10 and rbc["f_AP"] > 0:
            Ka_AP = max(
                (Kpu_bc - (X_rbc / X_p) * rbc["f_IW"] - lip_rbc / X_p)
                * X_p / (rbc["f_AP"] * Z), 0.0
            )

    # Ka_PR from fu_p (same as R&R Type 2)
    Ka_PR = max((1.0 / fu_p) - 1.0 - lip_p / X_p, 0.0)

    # Combined Kpu: weighted average of AP and PR binding
    AP_term = Ka_AP * f_AP * (X_IW - 1.0) / X_p  # ionized cation → AP
    PR_term = Ka_PR * AR * f_EW                     # protein in interstitial

    Kpu = (
        f_EW
        + (X_IW / X_p) * f_IW
        + lip_tissue / X_p
        + f_cation * AP_term          # weighted AP contribution
        + (1.0 - f_cation) * PR_term  # weighted PR contribution
    )

    Kp = Kpu * fu_p
    return max(Kp, 0.01)


# ===================================================================
# Method 2: Schmitt (2008)
# ===================================================================

def _kp_schmitt(compound: CompoundSpec, organ: Organ) -> float:
    """
    Schmitt (2008) method.

    Separates lipids into 3 sub-fractions with ionization-dependent
    binding coefficients. Key feature: factor of 20 for cation binding
    to acidic phospholipids.

    K_cell = f_W + K_n_l*f_NL + K_n_pl*f_NP + K_a_pl*f_AP + K_prot*f_prot
    Kp = K_cell * fu_p  (simplified, without interstitial correction)
    """
    tc = TISSUE_COMPOSITION[organ]
    pc = PLASMA_COMPOSITION

    fu_p = compound.fu_p
    logP = compound.logP
    pKa = compound.pKa
    ctype = compound.compound_type

    pH_IW = tc["pH_IW"]

    # Membrane affinity (logMA ≈ logP when direct data unavailable)
    K_n_pl = _P_ow(logP)  # neutral phospholipid:water PC

    # Protein:water partition coefficient (Schmitt 2008, Eq. 19)
    K_protein = 0.163 + 0.0221 * K_n_pl

    # Ionization at tissue pH
    W = _ionization_excess(pKa, pH_IW, ctype)

    # --- Neutral lipid:water PC (ionization-dependent) ---
    if W > 0:
        K_n_l = K_n_pl * ((1.0 - ALPHA_LIPID) / (1.0 + W) + ALPHA_LIPID)
    else:
        K_n_l = K_n_pl  # neutral compound: no ionization correction

    # --- Acidic phospholipid:water PC (charge-dependent) ---
    if _is_base(ctype):
        # Cationic species: 20-fold enhanced binding to AP
        if W > 0:
            K_a_pl = K_n_pl * (1.0 / (1.0 + W) + AP_FACTOR_BASE * (1.0 - 1.0 / (1.0 + W)))
        else:
            K_a_pl = K_n_pl
    elif _is_acid(ctype):
        # Anionic species: reduced binding to AP (electrostatic repulsion)
        if W > 0:
            K_a_pl = K_n_pl * (1.0 / (1.0 + W) + AP_FACTOR_ACID * (1.0 - 1.0 / (1.0 + W)))
        else:
            K_a_pl = K_n_pl
    else:
        K_a_pl = K_n_pl  # neutral

    # Use P_vo for adipose neutral lipids (better for triglyceride-rich tissue)
    if organ == Organ.ADIPOSE:
        K_n_l_adj = _P_vo(logP)
        if W > 0:
            K_n_l_adj = K_n_l_adj * ((1.0 - ALPHA_LIPID) / (1.0 + W) + ALPHA_LIPID)
    else:
        K_n_l_adj = K_n_l

    # Tissue fractions
    f_NL = tc["f_NL"]
    f_NP = tc["f_NP"]
    f_AP = tc["f_AP"]
    f_W = tc["f_EW"] + tc["f_IW"]
    f_prot = TISSUE_PROTEIN_FRACTIONS.get(organ, 0.1)

    # Cellular partition coefficient
    K_cell = f_W + K_n_l_adj * f_NL + K_n_pl * f_NP + K_a_pl * f_AP + K_protein * f_prot

    # Interstitial correction (Schmitt Eq. 4)
    AR = tc.get("albumin_ratio", 0.5)
    frac_data = TISSUE_INTERSTITIAL_FRACTIONS.get(organ, {"F_int": 0.2, "F_cell": 0.8})
    F_int = frac_data["F_int"]
    F_cell = frac_data["F_cell"]

    # K_int: interstitial partition (accounts for interstitial albumin)
    f_prot_p = pc.get("f_protein", 0.08)
    K_int = tc["f_EW"] + (AR * f_prot / f_prot_p) * (1.0 / fu_p - pc["f_W"])

    # pH gradient correction (KAPPA)
    f_n_p = _fraction_neutral(pKa, pc["pH"], ctype)
    f_n_t = _fraction_neutral(pKa, pH_IW, ctype)
    KAPPA = f_n_p / f_n_t if f_n_t > 1e-10 else 1.0

    # Total Kp
    Kp = (F_int * K_int + KAPPA * F_cell * K_cell) * fu_p

    return max(Kp, 0.01)


# ===================================================================
# Method 3: Poulin & Theil (2002)
# ===================================================================

def _kp_poulin_theil(compound: CompoundSpec, organ: Organ) -> float:
    """
    Poulin & Theil (2002) method.

    Kp = [P*(f_NL + 0.3*f_NP) + (f_W + 0.7*f_NP)] /
         [P*(f_NL_p + 0.3*f_NP_p) + (f_W_p + 0.7*f_NP_p)] * (fu_p / fu_t)

    For adipose: uses ionization-corrected vegetable oil:water D* instead of P.
    fu_t approximated as: 1 / (1 + ((1-fu_p)/fu_p) * 0.5)
    """
    tc = TISSUE_COMPOSITION[organ]
    pc = PLASMA_COMPOSITION

    fu_p = compound.fu_p
    logP = compound.logP
    pKa = compound.pKa
    ctype = compound.compound_type

    f_NL = tc["f_NL"]
    f_NP = tc["f_NP"]
    f_W = tc["f_EW"] + tc["f_IW"]

    NL_p = pc["f_NL"]
    NP_p = pc["f_NP"]
    W_p = pc["f_W"]

    # Lipophilicity measure
    if organ == Organ.ADIPOSE:
        P = _D_vo(logP, pKa, 7.4, ctype)
    else:
        P = _P_ow(logP)

    # Tissue partition (unbound)
    num = P * (f_NL + 0.3 * f_NP) + (f_W + 0.7 * f_NP)
    den = P * (NL_p + 0.3 * NP_p) + (W_p + 0.7 * NP_p)

    Kp_u = num / den if den > 0 else 1.0

    # fu_t approximation (Poulin-Theil default)
    AR = tc.get("albumin_ratio", 0.5)
    fu_t = 1.0 / (1.0 + ((1.0 - fu_p) / fu_p) * AR) if fu_p > 0 else 1.0

    Kp = Kp_u * fu_p / fu_t

    return max(Kp, 0.01)


# ===================================================================
# Method 4: Berezhkovskiy (2004) — Corrected Poulin-Theil
# ===================================================================

def _kp_berezhkovskiy(compound: CompoundSpec, organ: Organ) -> float:
    """
    Berezhkovskiy (2004) correction to Poulin-Theil.

    Key change: fu_p and fu_t are applied INSIDE the water terms,
    not as an external ratio. This gives more physiological results.

    Kp = [P*(f_NL + 0.3*f_NP) + 0.7*f_NP + f_W/fu_t] /
         [P*(f_NL_p + 0.3*f_NP_p) + 0.7*f_NP_p + f_W_p/fu_p]
    """
    tc = TISSUE_COMPOSITION[organ]
    pc = PLASMA_COMPOSITION

    fu_p = compound.fu_p
    logP = compound.logP
    pKa = compound.pKa
    ctype = compound.compound_type

    f_NL = tc["f_NL"]
    f_NP = tc["f_NP"]
    f_W = tc["f_EW"] + tc["f_IW"]

    NL_p = pc["f_NL"]
    NP_p = pc["f_NP"]
    W_p = pc["f_W"]

    if organ == Organ.ADIPOSE:
        P = _D_vo(logP, pKa, 7.4, ctype)
    else:
        P = _P_ow(logP)

    # fu_t estimation
    AR = tc.get("albumin_ratio", 0.5)
    fu_t = 1.0 / (1.0 + ((1.0 - fu_p) / fu_p) * AR) if fu_p > 0 else 1.0

    # Berezhkovskiy formulation: fu inside water terms
    num = P * (f_NL + 0.3 * f_NP) + 0.7 * f_NP + f_W / fu_t
    den = P * (NL_p + 0.3 * NP_p) + 0.7 * NP_p + W_p / fu_p

    Kp = num / den if den > 0 else 1.0

    return max(Kp, 0.01)


# ===================================================================
# Method 5: PK-Sim Standard (Willmann 2003/2004)
# ===================================================================

def _kp_pksim_standard(compound: CompoundSpec, organ: Organ) -> float:
    """
    PK-Sim Standard method (Willmann et al. 2003).

    Simplest method: no pKa, no ionization, single lipid fraction.
    Uses empirical K_protein derived from membrane affinity.

    Kp = (f_water + K_n_pl * f_lipids + K_protein * f_proteins) * fu_p

    K_protein = 0.163 + 0.0221 * K_n_pl  (Schmitt/Walter 2008)
    """
    tc = TISSUE_COMPOSITION[organ]

    fu_p = compound.fu_p
    logP = compound.logP

    # Membrane affinity
    K_n_pl = _P_ow(logP)

    # Protein:water partition (empirical)
    K_protein = 0.163 + 0.0221 * K_n_pl

    # Tissue fractions
    f_water = tc["f_EW"] + tc["f_IW"]
    f_lipids = tc["f_NL"] + tc["f_NP"]  # combined lipids
    f_proteins = TISSUE_PROTEIN_FRACTIONS.get(organ, 0.1)

    # For adipose: use P_vo for lipid term
    if organ == Organ.ADIPOSE:
        K_lipid = _P_vo(logP)
    else:
        K_lipid = K_n_pl

    Kp = (f_water + K_lipid * f_lipids + K_protein * f_proteins) * fu_p

    return max(Kp, 0.01)


# ===================================================================
# Method 6: Kp,mem — Membrane Partitioning (Poulin & Bhatt 2019)
# ===================================================================

def _kp_membrane(compound: CompoundSpec, organ: Organ) -> float:
    """
    Kp,mem method (Poulin 2017; Poulin & Bhatt 2019).

    Derives membrane:water partition coefficient from microsomal fu.
    Correct formulation (Poulin 2017, Pharm Res 34:1085):

      K_mem = (1 - fu_mic) / (fu_mic * f_lipid_mic)
      Kpu = f_W + K_mem * f_lipid_tissue + K_protein * f_protein
      Kp = Kpu * fu_p

    References:
      - Poulin P. Pharm Res 2017;34:1085-1095
      - Poulin P, Bhatt DK. J Pharm Sci 2019;108:1801-1811
      - Margolis JM, Obach RS. Drug Metab Dispos 2003;31:983-985
    """
    tc = TISSUE_COMPOSITION[organ]
    fu_p = compound.fu_p

    from .tissue_binding import predict_fu_inc
    fu_mic = predict_fu_inc(compound, 1.0)

    # Microsomal lipid fraction at 1 mg/mL protein
    # Margolis & Obach 2003: ~0.45 mg lipid / mL incubation
    f_lipid_mic = 0.00045  # fraction (0.45 mg/mL = 0.00045 g/mL ≈ 0.00045 v/v)

    # Membrane:water partition coefficient (Poulin 2017 Eq. 3)
    if fu_mic < 1.0 and f_lipid_mic > 0:
        K_mem = (1.0 - fu_mic) / (fu_mic * f_lipid_mic)
    else:
        K_mem = 0.0

    # Tissue fractions — include ALL lipids (NL + NP + AP)
    f_water = tc["f_EW"] + tc["f_IW"]
    f_lipid = tc["f_NL"] + tc["f_NP"] + tc["f_AP"]
    f_protein = TISSUE_PROTEIN_FRACTIONS.get(organ, 0.1)
    K_protein = 0.163 + 0.0221 * _P_ow(compound.logP)  # Schmitt, uses P_ow not K_mem

    # Adipose: use full NL fraction (no artificial amplification)
    # K_mem already captures lipophilicity from fu_mic

    # Kpu (unbound tissue:plasma), then Kp
    Kpu = f_water + K_mem * f_lipid + K_protein * f_protein
    Kp = Kpu * fu_p
    return max(Kp, 0.01)


# ===================================================================
# Unified interface
# ===================================================================

def predict_kp_single(
    compound: CompoundSpec,
    organ: Organ,
    method: KpMethod = KpMethod.RODGERS_ROWLAND,
) -> float:
    """
    Predict Kp for one organ using the specified method.

    Args:
        compound: Drug compound specification.
        organ: Target organ.
        method: Kp prediction method.

    Returns:
        Kp (tissue:plasma partition coefficient, dimensionless)
    """
    if isinstance(organ, str):
        organ = Organ(organ)
    if isinstance(method, str):
        method = KpMethod(method)

    dispatch = {
        KpMethod.RODGERS_ROWLAND: _kp_rodgers_rowland,
        KpMethod.LUKACOVA: _kp_lukacova,
        KpMethod.SCHMITT: _kp_schmitt,
        KpMethod.POULIN_THEIL: _kp_poulin_theil,
        KpMethod.BEREZHKOVSKIY: _kp_berezhkovskiy,
        KpMethod.PKSIM_STANDARD: _kp_pksim_standard,
        KpMethod.KP_MEMBRANE: _kp_membrane,
    }
    fn = dispatch[method]
    kp = fn(compound, organ)

    # Apply optional per-organ empirical correction. Rodgers-Rowland and related
    # in silico methods systematically over-predict adipose Kp for lipophilic bases
    # (Jansson 2008, Graham 2012). Users can supply compound.kp_scale to inject
    # published in vivo Kp ratios.
    if compound.kp_scale:
        scale = compound.kp_scale.get(organ.value.lower(), 1.0)
        kp = kp * scale

    return max(kp, 0.01)


def predict_kp_all(
    compound: CompoundSpec,
    method: KpMethod = KpMethod.RODGERS_ROWLAND,
) -> dict[Organ, float]:
    """Predict Kp for all 13 organs."""
    return {organ: predict_kp_single(compound, organ, method) for organ in Organ}


def predict_kpb_all(
    compound: CompoundSpec,
    method: KpMethod = KpMethod.RODGERS_ROWLAND,
) -> dict[Organ, float]:
    """Predict Kp:blood = Kp:plasma / R_bp for all organs."""
    kp = predict_kp_all(compound, method)
    R_bp = compound.R_bp
    return {organ: val / R_bp for organ, val in kp.items()}


def predict_kp_all_methods(compound: CompoundSpec) -> dict[str, dict[Organ, float]]:
    """Predict Kp using ALL 5 methods for comparison."""
    results = {}
    for method in KpMethod:
        results[method.value] = predict_kp_all(compound, method)
    return results


def predict_vss(
    compound: CompoundSpec,
    organ_volumes: dict,
    V_plasma: float,
    method: KpMethod = KpMethod.RODGERS_ROWLAND,
) -> float:
    """Predict Vss from Kp: Vss = V_plasma + sum(Kp_i * V_i)."""
    kp = predict_kp_all(compound, method)
    vss = V_plasma
    for organ in Organ:
        if organ in organ_volumes:
            vss += kp[organ] * organ_volumes[organ]
    return vss


# ===================================================================
# Formatting
# ===================================================================

def format_kp_comparison_table(
    compound: CompoundSpec,
    methods: Optional[list[KpMethod]] = None,
) -> str:
    """Format Kp comparison across methods as markdown."""
    if methods is None:
        methods = list(KpMethod)

    all_kp = {m: predict_kp_all(compound, m) for m in methods}

    method_names = {
        KpMethod.RODGERS_ROWLAND: "R&R",
        KpMethod.LUKACOVA: "Lukacova",
        KpMethod.SCHMITT: "Schmitt",
        KpMethod.POULIN_THEIL: "PT",
        KpMethod.BEREZHKOVSKIY: "PTB",
        KpMethod.PKSIM_STANDARD: "PK-Sim",
        KpMethod.KP_MEMBRANE: "Kp_mem",
    }

    header = f"## Kp Comparison — {compound.name}\n\n"
    header += f"logP={compound.logP}, pKa={compound.pKa}, fu_p={compound.fu_p}, "
    header += f"type={compound.compound_type.value}, R_bp={compound.R_bp}\n\n"

    cols = [method_names[m] for m in methods]
    lines = [header]
    lines.append("| Tissue | " + " | ".join(cols) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(cols)) + "|")

    for organ in Organ:
        vals = [f"{all_kp[m][organ]:.3f}" for m in methods]
        lines.append(f"| {organ.value.capitalize():10s} | " + " | ".join(f"{v:>6s}" for v in vals) + " |")

    lines.append(f"\n*R&R=Rodgers-Rowland, PT=Poulin-Theil, PTB=Berezhkovskiy*")
    return "\n".join(lines)


def format_kpb_table(
    kp_values: dict[Organ, float],
    kpb_values: dict[Organ, float],
    compound_name: str = "",
    R_bp: float = 1.0,
    method_name: str = "Rodgers & Rowland",
) -> str:
    """Format Kp and Kp:blood values as markdown."""
    header = f"## Partition Coefficients"
    if compound_name:
        header += f" — {compound_name}"
    header += f"\n\nBlood:Plasma ratio (R_bp) = {R_bp:.3f}\n\n"

    lines = [
        header,
        "| Tissue | Kp (tissue:plasma) | Kp:blood (tissue:blood) |",
        "|--------|-------------------|------------------------|",
    ]

    for organ in Organ:
        kp = kp_values.get(organ, 0)
        kpb = kpb_values.get(organ, 0)
        lines.append(
            f"| {organ.value.capitalize():10s} | {kp:17.3f} | {kpb:22.3f} |"
        )

    lines.append(f"\n*Method: {method_name}*")
    return "\n".join(lines)
