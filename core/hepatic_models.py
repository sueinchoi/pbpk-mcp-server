"""
Hepatic clearance models + R_bp prediction + DDI static equations.

Hepatic models:
  1. Well-stirred (already in main code, repeated here for completeness)
  2. Parallel-tube (Pang & Rowland 1977)
  3. Dispersion model (Roberts & Rowland 1986)
  4. Extended clearance concept (transporter-mediated)

R_bp prediction:
  From Rodgers-Rowland RBC partitioning or simplified regression.

DDI static models:
  - Reversible (competitive) inhibition
  - Mechanism-based inhibition (MBI, time-dependent)
  - CYP induction
  - Net effect (combined)
  - FDA 2020 basic DDI equations

References:
  - Pang KS, Rowland M. J Pharmacokinet Biopharm 1977;5:625-653
  - Roberts MS, Rowland M. J Pharm Pharmacol 1986;38:177-181
  - Shitara Y et al. Pharmacol Ther 2006;112:57-76
  - FDA DDI Guidance 2020
"""

import math
from dataclasses import dataclass
from typing import Optional

from .compound import CompoundSpec, CompoundType
from .physiology import (
    Organ, RBC_COMPOSITION, HEMATOCRIT, PLASMA_COMPOSITION,
)
from .partition_coeff import _P_ow, _ionization_ratio


# ===================================================================
# Hepatic clearance models
# ===================================================================

def cl_hepatic_wellstirred(
    CLint: float, fu_p: float, R_bp: float, Q_h: float,
) -> dict:
    """Well-stirred model: CL_h = Q_h * fu_b * CLint / (Q_h + fu_b * CLint)."""
    fu_b = fu_p / R_bp
    denom = Q_h + fu_b * CLint
    CL_h = Q_h * fu_b * CLint / denom if denom > 0 else 0
    E_h = CL_h / Q_h if Q_h > 0 else 0
    return {"CL_h": CL_h, "E_h": E_h, "F_h": 1 - E_h, "model": "well-stirred"}


def cl_hepatic_parallel_tube(
    CLint: float, fu_p: float, R_bp: float, Q_h: float,
) -> dict:
    """
    Parallel-tube (undistributed sinusoidal) model.

    E_h = 1 - exp(-fu_b * CLint / Q_h)
    CL_h = Q_h * (1 - exp(-fu_b * CLint / Q_h))
    """
    fu_b = fu_p / R_bp
    ratio = fu_b * CLint / Q_h if Q_h > 0 else 0
    E_h = 1.0 - math.exp(-ratio)
    CL_h = Q_h * E_h
    return {"CL_h": CL_h, "E_h": E_h, "F_h": 1 - E_h, "model": "parallel-tube"}


def cl_hepatic_dispersion(
    CLint: float, fu_p: float, R_bp: float, Q_h: float,
    DN: float = 0.17,
) -> dict:
    """
    Dispersion model (Roberts & Rowland 1986).

    F_h = 4a / ((1+a)^2 * exp(a/(2*DN)) - (1-a)^2 * exp(-a/(2*DN)))
    a = sqrt(1 + 4 * R_N * DN)
    R_N = fu_b * CLint / Q_h

    DN = 0.17 is the standard human liver dispersion number.
    DN → ∞: approaches well-stirred
    DN → 0: approaches parallel-tube
    """
    fu_b = fu_p / R_bp
    R_N = fu_b * CLint / Q_h if Q_h > 0 else 0

    a = math.sqrt(1.0 + 4.0 * R_N * DN)
    half_DN = 1.0 / (2.0 * DN)
    exp_pos = math.exp(min(a * half_DN, 500))        # exp(a/(2DN))
    exp_neg = math.exp(max(-a * half_DN, -500))       # exp(-a/(2DN))
    exp_center = math.exp(min(half_DN, 500))           # exp(1/(2DN))

    # Roberts & Rowland (1986) Eq. 10:
    # F = 4a * exp(1/(2DN)) / [(1+a)^2 * exp(a/(2DN)) - (1-a)^2 * exp(-a/(2DN))]
    denom = (1.0 + a) ** 2 * exp_pos - (1.0 - a) ** 2 * exp_neg
    F_h = 4.0 * a * exp_center / denom if denom > 0 else 1.0
    F_h = min(max(F_h, 0.0), 1.0)
    E_h = 1.0 - F_h
    CL_h = Q_h * E_h

    return {"CL_h": CL_h, "E_h": E_h, "F_h": F_h, "DN": DN, "model": "dispersion"}


def cl_hepatic_extended(
    PS_inf: float, PS_eff: float, CLint_met: float,
    fu_p: float, R_bp: float, Q_h: float,
    CL_bile: float = 0.0,
    hepatic_model: str = "well-stirred",
) -> dict:
    """
    Extended clearance concept for transporter-mediated hepatic clearance.

    CLint_overall = PS_inf * (CLint_met + CL_bile) / (PS_inf + CLint_met + CL_bile + PS_eff)
    Kp_uu = PS_inf / (PS_eff + CLint_met + CL_bile)

    Then apply chosen hepatic model using CLint_overall.

    Args:
        PS_inf: Sinusoidal influx clearance (L/h).
        PS_eff: Sinusoidal efflux clearance (L/h).
        CLint_met: Intracellular metabolic intrinsic clearance (L/h).
        fu_p, R_bp, Q_h: Standard PK parameters.
        CL_bile: Biliary efflux intrinsic clearance (L/h).
        hepatic_model: "well-stirred", "parallel-tube", or "dispersion".
    """
    denom = PS_inf + CLint_met + CL_bile + PS_eff
    CLint_overall = PS_inf * (CLint_met + CL_bile) / denom if denom > 0 else 0

    Kp_uu = PS_inf / (PS_eff + CLint_met + CL_bile) if (PS_eff + CLint_met + CL_bile) > 0 else 1.0

    if hepatic_model == "parallel-tube":
        result = cl_hepatic_parallel_tube(CLint_overall, fu_p, R_bp, Q_h)
    elif hepatic_model == "dispersion":
        result = cl_hepatic_dispersion(CLint_overall, fu_p, R_bp, Q_h)
    else:
        result = cl_hepatic_wellstirred(CLint_overall, fu_p, R_bp, Q_h)

    result["CLint_overall"] = CLint_overall
    result["Kp_uu_liver"] = Kp_uu
    result["PS_inf"] = PS_inf
    result["PS_eff"] = PS_eff
    result["CL_bile"] = CL_bile
    result["model"] = f"extended-{hepatic_model}"
    return result


def compare_hepatic_models(
    CLint: float, fu_p: float, R_bp: float, Q_h: float,
) -> str:
    """Compare all 3 hepatic models as markdown."""
    ws = cl_hepatic_wellstirred(CLint, fu_p, R_bp, Q_h)
    pt = cl_hepatic_parallel_tube(CLint, fu_p, R_bp, Q_h)
    dm = cl_hepatic_dispersion(CLint, fu_p, R_bp, Q_h)

    lines = [
        "## Hepatic Clearance Model Comparison\n",
        f"CLint = {CLint:.2f} L/h, fu_p = {fu_p:.4f}, R_bp = {R_bp:.2f}, Q_h = {Q_h:.1f} L/h\n",
        "| Model | CL_h (L/h) | E_h | F_h |",
        "|-------|-----------|-----|-----|",
        f"| Well-stirred | {ws['CL_h']:.2f} | {ws['E_h']:.3f} | {ws['F_h']:.3f} |",
        f"| Parallel-tube | {pt['CL_h']:.2f} | {pt['E_h']:.3f} | {pt['F_h']:.3f} |",
        f"| Dispersion (DN=0.17) | {dm['CL_h']:.2f} | {dm['E_h']:.3f} | {dm['F_h']:.3f} |",
    ]
    return "\n".join(lines)


# ===================================================================
# R_bp prediction
# ===================================================================

def predict_rbp(compound: CompoundSpec) -> dict:
    """
    Predict blood:plasma ratio from Rodgers-Rowland RBC partitioning.

    R_bp = 1 - HCT + HCT * Kp_RBC

    For strong bases: uses Ka_AP (from fu_p).
    For others: simplified lipid + water partitioning.
    """
    rbc = RBC_COMPOSITION
    pc = PLASMA_COMPOSITION

    logP = compound.logP
    pKa = compound.pKa
    fu_p = compound.fu_p
    ctype = compound.compound_type
    HCT = HEMATOCRIT
    P = _P_ow(logP)

    pH_p = pc["pH"]
    pH_rbc = rbc["pH"]

    X_p = _ionization_ratio(pKa, pH_p, ctype)
    X_rbc = _ionization_ratio(pKa, pH_rbc, ctype)

    # RBC lipid terms
    lip_rbc = P * rbc["f_NL"] / X_p + (0.3 * P + 0.7) * rbc["f_NP"] / X_p

    if ctype == CompoundType.STRONG_BASE:
        # Type 1: Ka_AP from fu_p
        lip_p = P * pc["f_NL"] + (0.3 * P + 0.7) * pc["f_NP"]
        Ka_AP = max((1.0 / fu_p - 1.0 - lip_p) / (pc["f_AP"] * X_p), 0.0)

        Z = X_rbc - 1.0  # ionized fraction at RBC pH
        Kpu_rbc = (
            (X_rbc / X_p) * rbc["f_IW"]
            + Ka_AP * rbc["f_AP"] * Z / X_p
            + lip_rbc
        )
        Kp_RBC = Kpu_rbc * fu_p
    else:
        # Type 2: simplified
        Kpu_rbc = (X_rbc / X_p) * rbc["f_IW"] + lip_rbc
        Kp_RBC = Kpu_rbc * fu_p

    R_bp = 1.0 - HCT + HCT * Kp_RBC
    R_bp = max(R_bp, 0.5)  # physiological minimum

    return {
        "R_bp": R_bp,
        "Kp_RBC": Kp_RBC,
        "HCT": HCT,
    }


# ===================================================================
# DDI static models (FDA 2020 approach)
# ===================================================================

@dataclass
class DDIResult:
    """DDI prediction result."""
    AUC_ratio: float        # AUCinhibited / AUCcontrol
    mechanism: str
    details: dict

    def to_markdown(self) -> str:
        # FDA 2020 DDI classification (both inhibition and induction)
        r = self.AUC_ratio
        if r >= 5:
            cls = "Strong inhibition"
        elif r >= 2:
            cls = "Moderate inhibition"
        elif r >= 1.25:
            cls = "Weak inhibition"
        elif r <= 0.2:
            cls = "Strong induction"
        elif r <= 0.5:
            cls = "Moderate induction"
        elif r <= 0.8:
            cls = "Weak induction"
        else:
            cls = "No interaction"
        lines = [
            f"## DDI Prediction — {self.mechanism}\n",
            f"**AUC Ratio = {self.AUC_ratio:.2f}** ({cls})\n",
            "| Parameter | Value |",
            "|-----------|-------|",
        ]
        for k, v in self.details.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4g} |")
            else:
                lines.append(f"| {k} | {v} |")
        return "\n".join(lines)


def ddi_reversible_inhibition(
    Ki: float,
    I_h_u: float,
    fm: float,
    I_gut: Optional[float] = None,
    Fg_baseline: float = 1.0,
) -> DDIResult:
    """
    Reversible (competitive) CYP inhibition DDI prediction.

    AUC_ratio_h = 1 / (fm / (1 + I_h_u/Ki) + (1 - fm))

    For oral (with gut wall):
    AUC_ratio_oral = AUC_ratio_h / (Fg + (1-Fg)/(1+I_gut/Ki))

    Args:
        Ki: Inhibition constant (uM).
        I_h_u: Unbound inhibitor concentration at liver (uM).
        fm: Fraction of substrate metabolized by affected CYP.
        I_gut: Inhibitor concentration in enterocytes (uM).
        Fg_baseline: Baseline Fg of substrate.
    """
    R_h = 1.0 + I_h_u / Ki
    AUC_ratio_h = 1.0 / (fm / R_h + (1.0 - fm))

    AUC_ratio = AUC_ratio_h
    if I_gut is not None and Fg_baseline < 1.0:
        R_g = 1.0 + I_gut / Ki
        Fg_inh = 1.0 / (1.0 / Fg_baseline + (1.0 / Fg_baseline - 1.0) / R_g)
        AUC_ratio = AUC_ratio_h * Fg_inh / Fg_baseline

    return DDIResult(
        AUC_ratio=AUC_ratio,
        mechanism="Reversible CYP inhibition",
        details={"Ki_uM": Ki, "I_h_u_uM": I_h_u, "fm": fm, "R_h": R_h},
    )


def ddi_mechanism_based_inhibition(
    KI: float,
    kinact: float,
    I_h_u: float,
    fm: float,
    kdeg: float = 0.019,
    I_gut: Optional[float] = None,
    Fg_baseline: float = 1.0,
    kdeg_gut: float = 0.029,
) -> DDIResult:
    """
    Mechanism-based (time-dependent) CYP inhibition.

    kobs = kinact * I_u / (KI + I_u)
    AUC_ratio_h = 1 / (fm * kdeg/(kdeg + kobs) + (1-fm))

    Args:
        KI: Inhibitor concentration for half-maximal inactivation (uM).
        kinact: Maximum inactivation rate constant (1/h).
        I_h_u: Unbound inhibitor concentration at liver (uM).
        fm: Fraction metabolized by affected CYP.
        kdeg: CYP degradation rate constant in liver (1/h).
        I_gut: Inhibitor in enterocytes (uM).
        Fg_baseline: Baseline Fg.
        kdeg_gut: CYP degradation rate in gut (1/h).
    """
    kobs_h = kinact * I_h_u / (KI + I_h_u) if (KI + I_h_u) > 0 else 0
    AUC_ratio_h = 1.0 / (fm * kdeg / (kdeg + kobs_h) + (1.0 - fm))

    AUC_ratio = AUC_ratio_h
    if I_gut is not None and Fg_baseline < 1.0:
        kobs_g = kinact * I_gut / (KI + I_gut)
        Fg_new = Fg_baseline / (
            Fg_baseline + (1 - Fg_baseline) * kdeg_gut / (kdeg_gut + kobs_g)
        )
        AUC_ratio = AUC_ratio_h * Fg_new / Fg_baseline

    return DDIResult(
        AUC_ratio=AUC_ratio,
        mechanism="Mechanism-based CYP inhibition",
        details={
            "KI_uM": KI, "kinact_per_h": kinact, "I_h_u_uM": I_h_u,
            "fm": fm, "kdeg_per_h": kdeg, "kobs_per_h": kobs_h,
        },
    )


def ddi_induction(
    Emax: float,
    EC50: float,
    I_h_u: float,
    fm: float,
    d: float = 1.0,
) -> DDIResult:
    """
    CYP induction DDI prediction.

    AUC_ratio = 1 / (fm * (1 + d*Emax*I_u/(EC50+I_u)) + (1-fm))

    Args:
        Emax: Maximum fold-induction (dimensionless).
        EC50: Inducer concentration for half-maximal effect (uM).
        I_h_u: Unbound inducer at liver (uM).
        fm: Fraction metabolized by induced CYP.
        d: Scaling/calibration factor (default 1.0).
    """
    induction_fold = 1.0 + d * Emax * I_h_u / (EC50 + I_h_u) if (EC50 + I_h_u) > 0 else 1.0
    AUC_ratio = 1.0 / (fm * induction_fold + (1.0 - fm))

    return DDIResult(
        AUC_ratio=AUC_ratio,
        mechanism="CYP induction",
        details={
            "Emax": Emax, "EC50_uM": EC50, "I_h_u_uM": I_h_u,
            "fm": fm, "d": d, "induction_fold": induction_fold,
        },
    )


def ddi_net_effect(
    Ki: Optional[float] = None,
    KI: Optional[float] = None,
    kinact: Optional[float] = None,
    Emax: Optional[float] = None,
    EC50: Optional[float] = None,
    I_h_u: float = 0.0,
    fm: float = 1.0,
    kdeg: float = 0.019,
    d: float = 1.0,
) -> DDIResult:
    """
    Net DDI effect combining reversible inhibition, MBI, and induction.

    CLint_ratio = induction * MBI * reversible
    AUC_ratio = 1 / (fm * CLint_ratio + (1-fm))
    """
    rev_factor = 1.0 / (1.0 + I_h_u / Ki) if Ki and Ki > 0 else 1.0

    mbi_factor = 1.0
    if KI is not None and kinact is not None:
        kobs = kinact * I_h_u / (KI + I_h_u) if (KI + I_h_u) > 0 else 0
        mbi_factor = kdeg / (kdeg + kobs)

    ind_factor = 1.0
    if Emax is not None and EC50 is not None:
        ind_factor = 1.0 + d * Emax * I_h_u / (EC50 + I_h_u) if (EC50 + I_h_u) > 0 else 1.0

    CLint_ratio = ind_factor * mbi_factor * rev_factor
    AUC_ratio = 1.0 / (fm * CLint_ratio + (1.0 - fm))

    return DDIResult(
        AUC_ratio=AUC_ratio,
        mechanism="Net effect (inhibition + induction)",
        details={
            "reversible_factor": rev_factor,
            "MBI_factor": mbi_factor,
            "induction_factor": ind_factor,
            "CLint_ratio": CLint_ratio,
            "fm": fm,
        },
    )


# ===================================================================
# FDA Mechanistic Static Model (MSM) — ICH M12 (2024)
# ===================================================================

def ddi_msm(
    Ki: Optional[float] = None,
    KI: Optional[float] = None,
    kinact: Optional[float] = None,
    Emax: Optional[float] = None,
    EC50: Optional[float] = None,
    fm: float = 0.9,
    kdeg_liver: float = 0.019,
    kdeg_gut: float = 0.029,
    Fg_victim: float = 1.0,
    # Perpetrator PK
    Cmax_ss: float = 1.0,         # µM (unbound systemic Cmax at SS)
    Dose_perp: float = 200.0,     # mg
    fu_p_perp: float = 0.01,
    Fa_perp: float = 1.0,
    Fg_perp: float = 1.0,
    ka_perp: float = 6.0,         # 1/h (default 0.1/min per FDA)
    R_bp_perp: float = 1.0,
    MW_perp: float = 500.0,
    Q_h: float = 97.0,            # L/h (hepatic blood flow)
    Q_ent: float = 18.0,          # L/h (enterocytic villous flow)
) -> DDIResult:
    """
    FDA/ICH M12 Mechanistic Static Model for DDI prediction.

    Uses inlet hepatic [I] and enterocytic [I] per FDA 2020 guidance.

    [I]h,u = fu_p * (Cmax_ss + Fa*Fg*ka*Dose / (Qh*Rbp))
    [I]g = Fa * ka * Dose / Qent

    References:
      - FDA 2020 DDI Guidance
      - ICH M12 (2024)
    """
    # Hepatic inlet unbound concentration (µM)
    # Portal vein contribution: Fa*Fg*ka*Dose / (Qh * Rbp)
    portal_term = Fa_perp * Fg_perp * ka_perp * Dose_perp / (Q_h * R_bp_perp)
    # Convert mg/h / (L/h) = mg/L → µM: * 1000 / MW
    portal_uM = portal_term * 1000.0 / MW_perp
    I_h_u = fu_p_perp * (Cmax_ss + portal_uM)

    # Gut (enterocyte) concentration (µM)
    I_gut = Fa_perp * ka_perp * Dose_perp / Q_ent * 1000.0 / MW_perp

    # --- Hepatic term ---
    A_h = 1.0
    if Ki and Ki > 0:
        A_h *= 1.0 / (1.0 + I_h_u / Ki)
    if KI and kinact:
        kobs = kinact * I_h_u / (KI + I_h_u)
        A_h *= kdeg_liver / (kdeg_liver + kobs)
    if Emax and EC50:
        A_h *= (1.0 + Emax * I_h_u / (EC50 + I_h_u))

    hepatic_term = fm * A_h + (1.0 - fm)

    # --- Gut term ---
    A_g = 1.0
    if Ki and Ki > 0:
        A_g = 1.0 / (1.0 + I_gut / Ki)
    if KI and kinact:
        kobs_g = kinact * I_gut / (KI + I_gut)
        A_g *= kdeg_gut / (kdeg_gut + kobs_g)
    Fg_new = 1.0 - (1.0 - Fg_victim) * A_g
    gut_term = Fg_new / Fg_victim if Fg_victim > 0 else 1.0

    AUC_ratio = 1.0 / (hepatic_term * (1.0 / gut_term))
    # Simplified: AUC_ratio = gut_term / hepatic_term

    return DDIResult(
        AUC_ratio=AUC_ratio,
        mechanism=f"MSM (FDA/ICH M12): [I]h,u={I_h_u:.2f}µM, [I]gut={I_gut:.1f}µM",
        details={
            "I_h_u_uM": I_h_u,
            "I_gut_uM": I_gut,
            "portal_term_uM": portal_uM,
            "hepatic_term": hepatic_term,
            "gut_term": gut_term,
            "Fg_new": Fg_new,
            "fm": fm,
        },
    )


# ===================================================================
# Guest Criteria (Guest et al. 2011)
# ===================================================================

def guest_criteria(predicted_ratio: float, observed_ratio: float, delta: float = 2.0) -> dict:
    """
    Evaluate DDI prediction using Guest et al. 2011 criteria.

    The criterion is applied to (R-1), not R itself.
    Upper = 1 + delta * (R_obs - 1)
    Lower = 1 + (R_obs - 1) / delta

    Reference: Guest EJ et al. Drug Metab Dispos 2011;39:170-173.
    """
    if observed_ratio >= 1.0:
        upper = 1.0 + delta * (observed_ratio - 1.0)
        lower = 1.0 + (observed_ratio - 1.0) / delta
    else:
        upper = 1.0 - (1.0 - observed_ratio) / delta
        lower = 1.0 - delta * (1.0 - observed_ratio)
        lower = max(lower, 0.01)

    within = lower <= predicted_ratio <= upper

    return {
        "predicted": predicted_ratio,
        "observed": observed_ratio,
        "lower_limit": lower,
        "upper_limit": upper,
        "within_guest": within,
        "classification": "PASS" if within else "FAIL",
    }
