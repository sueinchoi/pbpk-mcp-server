"""
Physiological and physicochemical invariants enforced server-side.

These checks run at simulation entry — they do not depend on LLM-side
prompting. A parameter that fails an invariant raises ValueError with
the parameter name, value, and acceptable range.

Reference ranges are conservative — they exclude obviously non-physical
values rather than narrow "typical drug" windows. The intent is to
catch fabrication / unit confusion / typos, not to reject unusual but
real drugs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------
# Physicochemical / PK ranges
# ---------------------------------------------------------------------

PHYSCHEM_RANGES = {
    # name              (min,        max,         unit,                 reason for bound)
    "mw":               (50.0,       3000.0,      "g/mol",              "small molecule to large peptide"),
    "logP":             (-5.0,       10.0,        "log10",              "extreme hydrophilic to extreme lipophilic"),
    "pKa":              (-2.0,       16.0,        "dimensionless",      "physical pKa range"),
    "fu_p":             (1e-5,       1.0,         "fraction",           "0.001%-100% unbound; warfarin ~0.005, caffeine ~0.7"),
    "R_bp":             (0.05,       20.0,        "ratio",              "acids ~0.55, lipophilic bases up to ~10"),
    "ka":               (0.01,       50.0,        "1/h",                "very slow ER to fast IR"),
    "Fa":               (0.0,        1.0,         "fraction",           "definitionally bounded"),
    "Fg":               (0.0,        1.0,         "fraction",           "definitionally bounded"),
    "Peff":             (0.001,      100.0,       "1e-4 cm/s",          "human jejunal Peff range"),
}

CLEARANCE_RANGES = {
    "CL_int":           (0.0,        1.0e6,       "L/h",                "intrinsic CL after IVIVE; metformin ~50, midaz ~1000"),
    "CL_renal":         (0.0,        1000.0,      "L/h",                "renal CL; GFR ~7.5, max ~10x for active secretion"),
    "CLint_vitro_hlm":  (0.001,      10000.0,     "uL/min/mg",          "HLM CLint dynamic range"),
    "CLint_vitro_hep":  (0.001,      1000.0,      "uL/min/1e6 cells",   "hepatocyte CLint dynamic range"),
    "Vmax":             (0.0,        1.0e6,       "mg/h",               "max metabolism rate"),
    "Km":               (1e-6,       1.0e4,       "mg/L",               "Michaelis constant"),
}

DDI_RANGES = {
    "Ki":               (1e-6,       1.0e3,       "uM",                 "competitive inhibition constant"),
    "KI":               (1e-6,       1.0e3,       "uM",                 "MBI inactivation constant"),
    "kinact":           (0.0,        100.0,       "1/h",                "MBI inactivation rate"),
    "Emax":             (0.0,        100.0,       "fold",               "induction Emax (1 = no induction)"),
    "EC50":             (1e-6,       1.0e3,       "uM",                 "induction EC50"),
    "fm":               (0.0,        1.0,         "fraction",           "definitionally bounded"),
}

DOSE_SUBJECT_RANGES = {
    "dose_mg":          (0.001,      1.0e5,       "mg",                 "1 µg to 100 g"),
    "duration_h":       (0.1,        2400.0,      "h",                  "6 min to 100 days"),
    "n_doses":          (1,          1000,        "count",              ""),
    "interval_h":       (0.5,        168.0,       "h",                  "30 min to 1 week"),
    "infusion_duration_h": (0.001,   72.0,        "h",                  ""),
    "body_weight":      (1.0,        300.0,       "kg",                 "neonate to bariatric"),
    "age":              (0.0,        120.0,       "years",              ""),
    "n_individuals":    (10,         500,         "count",              "Monte Carlo precision"),
    "n_liver_segments": (1,          20,          "count",              "dispersion model resolution"),
}


@dataclass
class InvariantViolation:
    """One failed invariant check."""
    parameter: str
    value: object
    expected: str
    why: str


def _check_range(name: str, value: float, table: dict) -> Optional[InvariantViolation]:
    if name not in table:
        return None
    if value is None:
        return None
    lo, hi, unit, reason = table[name]
    if not (lo <= value <= hi):
        return InvariantViolation(
            parameter=name, value=value,
            expected=f"[{lo}, {hi}] {unit}",
            why=reason or "out of physiological range",
        )
    return None


def check_compound_ranges(
    *,
    mw: Optional[float] = None,
    logP: Optional[float] = None,
    pKa: Optional[float] = None,
    fu_p: Optional[float] = None,
    R_bp: Optional[float] = None,
    ka: Optional[float] = None,
    Fa: Optional[float] = None,
    Fg: Optional[float] = None,
    Peff: Optional[float] = None,
    CL_int: Optional[float] = None,
    CL_renal: Optional[float] = None,
    CLint_vitro_hlm: Optional[float] = None,
    CLint_vitro_hep: Optional[float] = None,
    Vmax: Optional[float] = None,
    Km: Optional[float] = None,
) -> list[InvariantViolation]:
    """Check every supplied physicochemical / clearance value."""
    out: list[InvariantViolation] = []
    for name, val, table in [
        ("mw", mw, PHYSCHEM_RANGES),
        ("logP", logP, PHYSCHEM_RANGES),
        ("pKa", pKa, PHYSCHEM_RANGES),
        ("fu_p", fu_p, PHYSCHEM_RANGES),
        ("R_bp", R_bp, PHYSCHEM_RANGES),
        ("ka", ka, PHYSCHEM_RANGES),
        ("Fa", Fa, PHYSCHEM_RANGES),
        ("Fg", Fg, PHYSCHEM_RANGES),
        ("Peff", Peff, PHYSCHEM_RANGES),
        ("CL_int", CL_int, CLEARANCE_RANGES),
        ("CL_renal", CL_renal, CLEARANCE_RANGES),
        ("CLint_vitro_hlm", CLint_vitro_hlm, CLEARANCE_RANGES),
        ("CLint_vitro_hep", CLint_vitro_hep, CLEARANCE_RANGES),
        ("Vmax", Vmax, CLEARANCE_RANGES),
        ("Km", Km, CLEARANCE_RANGES),
    ]:
        v = _check_range(name, val, table)
        if v:
            out.append(v)
    return out


def check_dose_subject_ranges(
    *,
    dose_mg: Optional[float] = None,
    duration_h: Optional[float] = None,
    n_doses: Optional[int] = None,
    interval_h: Optional[float] = None,
    infusion_duration_h: Optional[float] = None,
    body_weight: Optional[float] = None,
    age: Optional[float] = None,
    n_individuals: Optional[int] = None,
    n_liver_segments: Optional[int] = None,
) -> list[InvariantViolation]:
    out: list[InvariantViolation] = []
    for name, val in [
        ("dose_mg", dose_mg), ("duration_h", duration_h),
        ("n_doses", n_doses), ("interval_h", interval_h),
        ("infusion_duration_h", infusion_duration_h),
        ("body_weight", body_weight), ("age", age),
        ("n_individuals", n_individuals),
        ("n_liver_segments", n_liver_segments),
    ]:
        v = _check_range(name, val, DOSE_SUBJECT_RANGES)
        if v:
            out.append(v)
    return out


def check_ddi_ranges(
    *,
    Ki: Optional[float] = None,
    KI: Optional[float] = None,
    kinact: Optional[float] = None,
    Emax: Optional[float] = None,
    EC50: Optional[float] = None,
    fm: Optional[float] = None,
) -> list[InvariantViolation]:
    out: list[InvariantViolation] = []
    for name, val in [
        ("Ki", Ki), ("KI", KI), ("kinact", kinact),
        ("Emax", Emax), ("EC50", EC50), ("fm", fm),
    ]:
        v = _check_range(name, val, DDI_RANGES)
        if v:
            out.append(v)
    return out


# ---------------------------------------------------------------------
# Mass balance / physiology integrity
# ---------------------------------------------------------------------

def check_blood_flow_balance(
    organ_blood_flow_fractions: dict,
    tolerance: float = 0.02,
) -> Optional[InvariantViolation]:
    """All organ blood flows (as fraction of cardiac output) must sum to 1.0
    within tolerance. The 'rest' compartment exists to absorb residual."""
    total = sum(organ_blood_flow_fractions.values())
    if abs(total - 1.0) > tolerance:
        return InvariantViolation(
            parameter="blood_flow_fractions",
            value=f"sum={total:.4f}",
            expected=f"1.0 ± {tolerance}",
            why="organ blood flows do not sum to cardiac output — physiology table corrupted",
        )
    return None


def check_organ_volume_balance(
    organ_volumes: dict,
    body_weight_kg: float,
    tolerance: float = 0.10,
) -> Optional[InvariantViolation]:
    """Organ volumes should sum within ±10% of body weight (1 L/kg approx).
    Slightly looser tolerance than blood flow because the model adipose
    fraction varies and 'rest' absorbs error."""
    total = sum(organ_volumes.values())
    if total < body_weight_kg * (1 - tolerance) or total > body_weight_kg * (1 + tolerance):
        return InvariantViolation(
            parameter="organ_volumes",
            value=f"sum={total:.2f} L for BW={body_weight_kg} kg",
            expected=f"{body_weight_kg*(1-tolerance):.2f} - {body_weight_kg*(1+tolerance):.2f} L",
            why="organ volumes do not approximate body mass",
        )
    return None


def check_dose_self_consistency(
    *,
    dose_mg: float,
    n_doses: int,
    interval_h: float,
    duration_h: float,
    route: str,
) -> Optional[InvariantViolation]:
    """A multi-dose regimen with n*interval > duration silently truncates.
    Better to flag it explicitly."""
    if n_doses > 1 and interval_h * (n_doses - 1) > duration_h:
        return InvariantViolation(
            parameter="(n_doses, interval_h, duration_h)",
            value=f"n_doses={n_doses}, interval_h={interval_h}, duration_h={duration_h}",
            expected=f"interval_h * (n_doses-1) <= duration_h "
                     f"(currently {interval_h*(n_doses-1)} > {duration_h})",
            why="last dose falls outside simulation window — increase duration_h or reduce n_doses",
        )
    return None


# ---------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------

def raise_on_violations(violations: list[InvariantViolation]) -> None:
    if not violations:
        return
    msg_lines = [
        f"Input invariant violation(s) — {len(violations)} parameter(s) out of physiological range:",
    ]
    for v in violations:
        msg_lines.append(
            f"  • {v.parameter}={v.value!r} expected {v.expected} — {v.why}"
        )
    raise ValueError("\n".join(msg_lines))


# ---------------------------------------------------------------------
# Post-simulation dose recovery (mass balance assertion)
# ---------------------------------------------------------------------

def check_dose_recovery(
    *,
    result,                 # SimulationResult — has time, concentrations, venous_plasma
    model,                  # PBPKModel — has compound, phys, kp
    dose_mg: float,
    n_doses: int,
    route: str,
    tolerance: float = 0.01,
    oral_tolerance: float = 0.05,
) -> Optional[InvariantViolation]:
    """
    Assert mass balance in the simulation. Failure means the ODE lost
    or created mass somewhere — a serious physical bug (wrong volume
    scaling, missing compartment, integration tolerance too loose).

    Computation:
        body_burden(T) = Σ_organs V_organ × C_organ(T)
                       + V_arterial × C_art(T) + V_venous × C_ven(T)
        eliminated(T)  ≈ ∫₀ᵀ ( CL_int × fu_p × C_liver / Kp_liver
                              + CL_renal × C_kidney / Kp_kidney ) dt
        input(T)       = dose_mg × n_doses_completed_by_T
                          (for oral, multiplied by Fa)

    Assert |input - (body_burden + eliminated)| / input < tolerance.

    For multi-dose simulations, the check is run at the simulation end.
    A 1% tolerance corresponds to typical BDF integration tolerance
    (atol=1e-10, rtol=1e-8) on dosed mg amounts.
    """
    import numpy as np

    if dose_mg <= 0:
        return None  # nothing to check

    # --- Total input mass (mg) ---
    is_iv = route in ("iv_bolus", "iv_infusion")
    if is_iv:
        total_input = dose_mg * n_doses
        effective_tolerance = tolerance
    else:
        # Oral: drug enters systemic circulation = dose × Fa × Fg.
        #   Fa is the fraction absorbed from lumen into enterocyte.
        #   Fg is the fraction escaping gut-wall metabolism.
        # The (1-Fg) fraction is metabolized in the gut wall before
        # reaching portal vein and is lost from the systemic mass
        # balance perspective.
        Fa = getattr(model.compound, "Fa", 1.0)
        Fg = getattr(model.compound, "Fg", 1.0)
        total_input = dose_mg * n_doses * Fa * Fg
        # Trapezoidal post-hoc integration of CL × C_liver does not
        # exactly match BDF ODE integration — typical agreement
        # 1-3% on oral. IV bolus integrates more cleanly.
        effective_tolerance = oral_tolerance

    # --- Body burden at simulation end ---
    t = result.time
    if t is None or len(t) < 2:
        return None
    final_idx = len(t) - 1
    p = model.phys

    body_burden = 0.0
    for organ_name, C_t in result.concentrations.items():
        if organ_name in ("arterial", "venous"):
            continue
        # organ_name is the .value of an Organ enum
        from .physiology import Organ as _O
        try:
            organ_enum = _O(organ_name)
        except ValueError:
            continue
        V = p.organ_volumes.get(organ_enum, 0.0)
        body_burden += V * C_t[final_idx]

    # Blood pools — concentrations are in mg/L too
    if hasattr(result, "arterial_plasma") and result.arterial_plasma is not None:
        body_burden += p.V_arterial * result.arterial_plasma[final_idx]
    if result.venous_plasma is not None:
        body_burden += p.V_venous * result.venous_plasma[final_idx]

    # Drug remaining in oral lumen (oral only) — not "lost", just
    # not yet absorbed
    lumen_remaining = 0.0
    if not is_iv and hasattr(result, "lumen_amount"):
        lumen_amount = getattr(result, "lumen_amount", None)
        if lumen_amount is not None:
            lumen_remaining = lumen_amount[final_idx]

    # --- Eliminated mass — integrate CL × C_u_liver and CL_r × C_u_kidney ---
    fu_p = model.compound.fu_p
    kp_liver = model.kp.get(_O.LIVER, 1.0) if hasattr(model, "kp") else 1.0
    kp_kidney = model.kp.get(_O.KIDNEY, 1.0) if hasattr(model, "kp") else 1.0
    R_bp = model.compound.R_bp

    C_liver = result.concentrations.get("liver")
    C_kidney = result.concentrations.get("kidney")

    eliminated = 0.0
    if C_liver is not None and model.compound.CL_int > 0:
        # CL term in ODE is CL_int × fu_p × (C_liver / Kp_liver)
        # Integrate by trapezoidal rule
        rate_liver = model.compound.CL_int * fu_p * (C_liver / max(kp_liver, 1e-3))
        eliminated += float(np.trapezoid(rate_liver, t))
    if C_kidney is not None and model.compound.CL_renal > 0:
        rate_kidney = model.compound.CL_renal * (C_kidney / R_bp / max(kp_kidney, 1e-3))
        eliminated += float(np.trapezoid(rate_kidney, t))

    accounted = body_burden + lumen_remaining + eliminated
    if total_input <= 0:
        return None
    rel_err = abs(total_input - accounted) / total_input

    if rel_err > effective_tolerance:
        return InvariantViolation(
            parameter="dose_recovery",
            value=(
                f"input={total_input:.4f} mg, "
                f"body_burden={body_burden:.4f} mg, "
                f"lumen_remaining={lumen_remaining:.4f} mg, "
                f"eliminated={eliminated:.4f} mg, "
                f"rel_err={rel_err:.4%}"
            ),
            expected=f"|input - accounted| / input < {effective_tolerance:.1%}",
            why=(
                "post-simulation mass balance failed — drug created or destroyed "
                "in the ODE. Common causes: corrupt physiology table (run "
                "get_physiology mass-balance check), distribution_model mismatch "
                "between organ volumes and ODE state, integration tolerance too "
                "loose (lower atol/rtol)"
            ),
        )
    return None
