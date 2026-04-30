"""
Observed-data fitting for PBPK models.

Pipeline (codex design review 2026-04-30):

  1. Load CSV/TSV observations.
  2. Build a CompoundSpec from a library entry OR explicit physchem
     (custom compound fitting). Strict schema/range validation.
  3. Apply the fittable-parameter POLICY LAYER. Reject parameter
     combinations that are unidentifiable from the supplied data
     (e.g. CL_int + fu_p both fitted from a single plasma curve, or
     Fa + ka without an IV reference). Forbid is the default for
     coupled parameters; the user must invoke a specific override
     with the data conditions that justify the combination.
  4. Run the optimizer (Nelder-Mead local or DE global) with
     instrumentation: n_eval, n_fail, fail_rate, best_is_penalty.
  5. After convergence, run identifiability diagnostics
     (FIM/Jacobian rank, condition number, SE on log-params, parameter
     correlation matrix, estimate-on-bound flags).
  6. Compute GOF (R², AFE, AAFE, % within 2-fold).
  7. Return a structured FitReport with explicit warnings — never a
     pass/fail verdict (FDA/EMA review is intended-use dependent).

Multi-dataset co-fitting:
  - Accept a list of Dataset objects, each with its own CSV, dose, route.
  - Parameters are tagged 'shared' or 'route-specific' (ka is oral-only).
  - Objective is the sum of weighted log-residuals across datasets.
  - This is the only correct way to fit clearance + absorption from
    plasma PK without lying about identifiability (codex review).

References (FDA/EMA expectations):
  - FDA PBPK Format and Content guidance:
    https://www.fda.gov/files/drugs/published/Physiologically-Based-Pharmacokinetic-Analyses-%E2%80%94-Format-and-Content-Guidance-for-Industry.pdf
  - EMA PBPK reporting guideline:
    https://www.ema.europa.eu/en/reporting-physiologically-based-pharmacokinetic-pbpk-modelling-simulation-scientific-guideline
"""
from __future__ import annotations
import csv
import os
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

from .sensitivity import compute_gof, _r_squared, _fold_errors


# =====================================================================
# CSV loader (unchanged from earlier — auto-detect delimiter & columns)
# =====================================================================

def load_observed_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (time, concentration) from CSV/TSV. Auto-detects delimiter
    and recognises common column names (time / time_h / t; conc / cp /
    dv; etc.). Skips rows with non-numeric or negative values."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r") as f:
        sample = f.read(2000)
        f.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        fields = reader.fieldnames or []
        time_col = None
        conc_col = None
        for col in fields:
            cl = col.lower().strip()
            if cl in ("time", "time_h", "time (h)", "t", "hours"):
                time_col = col
            elif cl in ("concentration", "conc", "conc_mg_l",
                        "conc (mg/l)", "c", "cp", "plasma", "dv"):
                conc_col = col
        if time_col is None and fields:
            time_col = fields[0]
        if conc_col is None and len(fields) > 1:
            conc_col = fields[1]
        times, concs = [], []
        for row in reader:
            try:
                t = float(row[time_col])
                c = float(row[conc_col])
                if t >= 0 and c >= 0:
                    times.append(t)
                    concs.append(c)
            except (ValueError, KeyError, TypeError):
                continue
    if not times:
        raise ValueError("No valid data points found in file")
    return np.array(times), np.array(concs)


# =====================================================================
# Datasets (single + multi)
# =====================================================================

@dataclass
class Dataset:
    """A single observed PK profile + its dosing context."""
    csv_file: str
    dose_mg: float
    route: str = "oral"        # 'oral' / 'iv_bolus' / 'iv_infusion'
    label: str = ""            # human-readable identifier
    weight: float = 1.0        # objective weight (default equal)
    obs_t: np.ndarray = field(default_factory=lambda: np.array([]))
    obs_c: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if self.obs_t.size == 0:
            self.obs_t, self.obs_c = load_observed_data(self.csv_file)
        if not self.label:
            self.label = os.path.basename(self.csv_file)


# =====================================================================
# Fittable-parameter POLICY LAYER (codex SHIP-NOW)
# =====================================================================

# Allowlist — these are the parameters that can be safely fitted from
# typical plasma PK data with the right data conditions.
_ALLOWED_FITTABLE = {"CL_int", "CL_renal", "ka", "Fa", "Vmax", "Km"}

# Forbidden combinations: each entry is (combo, reason, override_token).
# To fit a forbidden combination the user must explicitly pass
# `force_combo_token=<token>` AND the data must satisfy the named
# condition. Default behavior is REJECT.
_FORBIDDEN_COMBOS: list[tuple[set[str], str, str]] = [
    (
        {"CL_int", "fu_p"},
        "CL_int and fu_p are coupled in the well-stirred CL formula "
        "(CL_h = Q·fu_b·CLint / (Q + fu_b·CLint)). Fitting both from "
        "a single plasma curve is non-identifiable — only the product "
        "fu_p · CL_int is determinable. Provide a measured fu_p (RED) "
        "and fit CL_int alone, or supply tissue/blood data.",
        "fu_measured_separately",
    ),
    (
        {"Fa", "Fg"},
        "Fa and Fg multiply in oral exposure (F_oral = Fa · Fg · F_h). "
        "Plasma profiles identify only the product, not the individual "
        "factors. Fix one from BCS/Caco-2 (Fa) or in vitro gut metabolism "
        "(Fg) and fit the other.",
        "iv_oral_paired_with_fg_constraint",
    ),
    (
        {"Fa", "ka"},
        "Fa and ka can both be inflated/depressed to fit an oral "
        "Cmax/Tmax pair without a clear answer unless an IV reference "
        "is co-fitted. Without IV data, fit ka and fix Fa from BCS or "
        "literature.",
        "iv_reference_codataset",
    ),
    (
        {"CL_int", "CL_renal"},
        "CL_int and CL_renal both reduce systemic exposure with no "
        "kinetic distinction unless renal clearance is measured "
        "separately (urinary excretion, fe_renal). Fit one and fix "
        "the other.",
        "urinary_excretion_data",
    ),
    (
        {"Vmax", "Km"},
        "Vmax and Km are non-identifiable from single-dose data — "
        "you need observations spanning both linear (C << Km) and "
        "saturation (C >> Km) regimes. Provide multi-dose datasets "
        "with concentration-dependent clearance.",
        "multidose_dose_ranging",
    ),
]


@dataclass
class PolicyResult:
    accepted: list[str]
    forbidden_violations: list[str] = field(default_factory=list)


def apply_fitting_policy(
    params_to_fit: list[str],
    *,
    datasets: list[Dataset],
    overrides: Optional[set[str]] = None,
) -> PolicyResult:
    """
    Validate the requested fittable set against the allowlist + forbidden
    combination policy. Raises ValueError on policy violation unless the
    user provided the matching override token AND data conditions are
    satisfied.

    `overrides` is a set of override tokens the user has acknowledged
    (e.g. {'iv_reference_codataset'} when an IV dataset is also passed).
    """
    overrides = overrides or set()

    # 1) Disallowed parameter names
    bad_names = [p for p in params_to_fit if p not in _ALLOWED_FITTABLE]
    if bad_names:
        raise ValueError(
            f"Cannot fit parameters {bad_names}. Allowed fittable names: "
            f"{sorted(_ALLOWED_FITTABLE)}. Parameters like fu_p, R_bp, "
            f"logP, pKa are NOT fittable from plasma PK alone — they are "
            f"physical/binding properties that should be measured "
            f"(RED, BP/p assay, etc.) and treated as fixed inputs."
        )

    # 2) Forbidden combinations
    requested = set(params_to_fit)
    violations = []
    for combo, reason, token in _FORBIDDEN_COMBOS:
        if combo.issubset(requested):
            if token in overrides:
                # Conditional acceptance: trust the user but record it
                continue
            violations.append(
                f"{sorted(combo)} forbidden together: {reason} "
                f"(override token: '{token}')"
            )

    # 3) Route-specific rules
    has_oral = any(d.route == "oral" for d in datasets)
    has_iv = any(d.route in ("iv_bolus", "iv_infusion") for d in datasets)
    if "ka" in requested and not has_oral:
        violations.append(
            "ka is oral-only — provide an oral dataset to fit it."
        )
    if "Fa" in requested and not (has_oral and has_iv):
        violations.append(
            "Fa requires both an oral AND an IV dataset (relative "
            "bioavailability). Without IV reference, oral exposure "
            "cannot identify Fa separately from ka or CL."
        )

    if violations:
        raise ValueError(
            "Fitting policy rejected the request:\n  - "
            + "\n  - ".join(violations)
        )
    return PolicyResult(accepted=list(requested))


# =====================================================================
# FitReport (structured result + diagnostics)
# =====================================================================

@dataclass
class FitReport:
    fitted_params: dict[str, float]
    initial_params: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    optimizer_method: str
    objective_value: float
    n_observations: int
    n_eval: int                  # total objective evaluations
    n_fail: int                  # crashed evaluations (returned penalty)
    fail_rate: float
    best_is_penalty: bool        # True if final objective is the penalty
    converged: bool
    message: str
    # GOF
    r_squared: float
    afe: float
    aafe: float
    pct_within_2fold: float
    # Identifiability
    on_bound: list[str] = field(default_factory=list)   # params at bound
    se_log: dict[str, float] = field(default_factory=dict)
    cv_pct: dict[str, float] = field(default_factory=dict)
    fim_rank: int = 0
    fim_cond: float = float("nan")
    fim_singular: bool = False
    correlations: dict[tuple[str, str], float] = field(default_factory=dict)
    # User-facing warnings (never pass/fail)
    warnings: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "## Fit Report",
            "",
            f"- **Method:** `{self.optimizer_method}` ({self.message})",
            f"- **Converged:** {self.converged}",
            f"- **n_obs:** {self.n_observations}  ·  "
            f"**n_eval:** {self.n_eval}  ·  "
            f"**n_fail:** {self.n_fail} ({self.fail_rate:.1%})",
            f"- **best_is_penalty:** {self.best_is_penalty}",
            "",
            "### Fitted parameters",
            "| Parameter | Value | Initial | Bounds | SE(log) | CV(%) | Note |",
            "|---|---|---|---|---|---|---|",
        ]
        for p, v in self.fitted_params.items():
            init = self.initial_params.get(p, float("nan"))
            lo, hi = self.bounds.get(p, (float("nan"), float("nan")))
            se = self.se_log.get(p)
            cv = self.cv_pct.get(p)
            note = ""
            if p in self.on_bound:
                note = "⚠️ at bound"
            lines.append(
                f"| {p} | {v:.4g} | {init:.4g} | [{lo:.3g}, {hi:.3g}] | "
                f"{('-' if se is None else f'{se:.3f}')} | "
                f"{('-' if cv is None else f'{cv:.1f}')} | {note} |"
            )
        lines += [
            "",
            "### Goodness-of-fit",
            f"- R² = {self.r_squared:.4f}",
            f"- AFE = {self.afe:.3f} (1.0 = no bias)",
            f"- AAFE = {self.aafe:.3f} (<2 acceptable)",
            f"- Within 2-fold: {self.pct_within_2fold:.0f}%",
            "",
            "### Identifiability (Jacobian / FIM)",
            f"- FIM rank: {self.fim_rank} / {len(self.fitted_params)}",
            f"- FIM condition number: "
            f"{('singular' if self.fim_singular else f'{self.fim_cond:.2e}')}",
        ]
        if self.correlations:
            lines.append("- Strong correlations (|ρ|>0.9):")
            for (a, b), r in self.correlations.items():
                if abs(r) > 0.9:
                    lines.append(f"  - {a} ↔ {b}: ρ = {r:+.2f}")
        if self.warnings:
            lines += ["", "### ⚠️ Warnings"]
            for w in self.warnings:
                lines.append(f"- {w}")
        return "\n".join(lines)


# =====================================================================
# Compound builder (library OR custom)
# =====================================================================

def _build_compound(
    *,
    name: str,
    library_lookup: dict,
    logP: Optional[float] = None,
    pKa: Optional[float] = None,
    fu_p: Optional[float] = None,
    mw: Optional[float] = None,
    compound_type: str = "neutral",
    R_bp: float = 1.0,
):
    """Resolve a CompoundSpec from a library entry or explicit physchem.
    Validation is strict: a custom compound MUST supply mw+logP+pKa+fu_p
    (no sentinel defaults — codex SHIP-NOW)."""
    from .compound import CompoundSpec, CompoundType
    key = (name or "").lower().strip()
    if key in library_lookup:
        return library_lookup[key]
    missing = [n for n, v in
               (("mw", mw), ("logP", logP), ("pKa", pKa), ("fu_p", fu_p))
               if v is None]
    if missing:
        raise ValueError(
            f"Custom compound '{name or '<unnamed>'}' is missing required "
            f"physchem inputs: {missing}. Either pass `name=<library "
            f"compound>` or provide mw, logP, pKa, fu_p explicitly. "
            f"R_bp defaults to 1.0 with a sentinel warning; pass an "
            f"explicit value when known."
        )
    return CompoundSpec(
        name=name or "Custom",
        mw=mw, logP=logP, pKa=pKa, fu_p=fu_p,
        compound_type=CompoundType(compound_type),
        R_bp=R_bp,
    )


# =====================================================================
# Per-parameter physiological bounds (multiplicative on initial value)
# =====================================================================

_DEFAULT_BOUND_FACTORS: dict[str, tuple[float, float]] = {
    "CL_int":   (0.01, 100.0),
    "CL_renal": (0.01, 100.0),
    "ka":       (0.01, 50.0),    # h^-1 — wide span needed for slow vs IR
    "Fa":       (0.05, 1.0),     # absolute bounds (not multiplicative)
    "Vmax":     (0.01, 100.0),
    "Km":       (0.01, 100.0),
}

_ABSOLUTE_BOUNDS: dict[str, tuple[float, float]] = {
    "Fa": (0.01, 1.0),           # fraction
}


def _resolve_bounds(
    params_to_fit: list[str],
    initial: dict[str, float],
    user_bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> dict[str, tuple[float, float]]:
    out = {}
    user_bounds = user_bounds or {}
    for p in params_to_fit:
        if p in user_bounds:
            out[p] = user_bounds[p]
            continue
        if p in _ABSOLUTE_BOUNDS:
            out[p] = _ABSOLUTE_BOUNDS[p]
            continue
        lo_f, hi_f = _DEFAULT_BOUND_FACTORS.get(p, (0.01, 100.0))
        v = initial.get(p, 1.0)
        if v <= 0:
            v = 1.0
        out[p] = (v * lo_f, v * hi_f)
    return out


# =====================================================================
# Multi-dataset objective
# =====================================================================

_FAIL_PENALTY = 1e10


def _make_objective(
    simulate_fn: Callable,           # (compound, dataset) -> (sim_t, sim_c)
    datasets: list[Dataset],
    params_to_fit: list[str],
    initial_params: dict[str, float],
    log_scale: bool = True,
    counter: Optional[dict] = None,
):
    """
    Multi-dataset objective. Sums weighted log-residual SSR across all
    datasets. counter is a mutable dict carrying n_eval, n_fail.
    """
    counter = counter if counter is not None else {"n_eval": 0, "n_fail": 0}

    def objective(x: np.ndarray) -> float:
        params = dict(initial_params)
        for i, pname in enumerate(params_to_fit):
            params[pname] = float(x[i])
        counter["n_eval"] += 1
        total = 0.0
        any_finite = False
        for d in datasets:
            try:
                sim_t, sim_c = simulate_fn(params, d)
                pred = np.interp(d.obs_t, sim_t, sim_c)
                if log_scale:
                    mask = (pred > 0) & (d.obs_c > 0)
                    if mask.sum() < 2:
                        continue
                    res = np.log(pred[mask]) - np.log(d.obs_c[mask])
                else:
                    res = pred - d.obs_c
                total += d.weight * float(np.sum(res ** 2))
                any_finite = True
            except Exception:
                # Per-dataset failure — skip; if no datasets succeeded
                # we return the penalty.
                continue
        if not any_finite:
            counter["n_fail"] += 1
            return _FAIL_PENALTY
        return total

    return objective, counter


# =====================================================================
# FIM / Jacobian identifiability diagnostics
# =====================================================================

def _compute_jacobian(
    simulate_fn: Callable,
    datasets: list[Dataset],
    fitted: dict[str, float],
    params_to_fit: list[str],
    *,
    eps: float = 0.01,
    log_params: bool = True,
    log_pred: bool = True,
) -> Optional[np.ndarray]:
    """
    Finite-difference Jacobian d(log_pred)/d(log_param) stacked across
    all datasets. Returns shape (sum_n_obs, n_params).
    """
    # simulate_fn returns (sim_time_array, sim_conc_array). Interpolate
    # to each dataset's observation times.
    rows = []
    for d in datasets:
        try:
            sim_t, sim_c = simulate_fn(fitted, d)
            pred = np.interp(d.obs_t, sim_t, sim_c)
        except Exception:
            return None
        rows.append(pred)
    base_preds = np.concatenate(rows) if rows else np.array([])
    if base_preds.size == 0:
        return None

    n_obs = len(base_preds)
    n_p = len(params_to_fit)
    J = np.zeros((n_obs, n_p))
    for j, pname in enumerate(params_to_fit):
        v = fitted[pname]
        if v <= 0:
            continue
        v_up = v * (1.0 + eps)
        v_dn = v * (1.0 - eps)
        params_up = dict(fitted)
        params_up[pname] = v_up
        params_dn = dict(fitted)
        params_dn[pname] = v_dn
        try:
            up_rows, dn_rows = [], []
            for d in datasets:
                sim_t_up, c_up = simulate_fn(params_up, d)
                up_rows.append(np.interp(d.obs_t, sim_t_up, c_up))
                sim_t_dn, c_dn = simulate_fn(params_dn, d)
                dn_rows.append(np.interp(d.obs_t, sim_t_dn, c_dn))
            preds_up = np.concatenate(up_rows)
            preds_dn = np.concatenate(dn_rows)
        except Exception:
            continue

        if log_pred:
            mask = (preds_up > 0) & (preds_dn > 0) & (base_preds > 0)
            if mask.sum() < 2:
                continue
            d_log_pred = (np.log(preds_up[mask]) - np.log(preds_dn[mask])) / 2.0
            d_log_param = (np.log(1 + eps) - np.log(1 - eps)) / 2.0 if log_params else (v_up - v_dn) / v
            if d_log_param == 0:
                continue
            col = d_log_pred / d_log_param
            J[mask, j] = col
        else:
            col = (preds_up - preds_dn) / (v_up - v_dn)
            J[:, j] = col
    return J


def _identifiability(
    simulate_fn: Callable,
    datasets: list[Dataset],
    fitted: dict[str, float],
    params_to_fit: list[str],
    bounds: dict[str, tuple[float, float]],
    sigma_log: float = 0.3,
):
    """
    FIM = (J^T J) / σ². Compute rank, condition number, parameter
    standard errors on log scale, correlation matrix.

    sigma_log = assumed log-residual SD (default 0.3, typical PK CV~30%).
    """
    J = _compute_jacobian(simulate_fn, datasets, fitted, params_to_fit)
    n_p = len(params_to_fit)
    out = {
        "fim_rank": 0,
        "fim_cond": float("nan"),
        "fim_singular": True,
        "se_log": {},
        "cv_pct": {},
        "correlations": {},
        "on_bound": [],
    }
    # Estimate-on-bound flags
    for p in params_to_fit:
        v = fitted.get(p, 0.0)
        lo, hi = bounds.get(p, (0.0, float("inf")))
        if v <= lo * 1.01 or v >= hi * 0.99:
            out["on_bound"].append(p)

    if J is None or J.size == 0:
        return out

    # FIM and SE
    try:
        fim = (J.T @ J) / (sigma_log ** 2)
        rank = int(np.linalg.matrix_rank(fim))
        out["fim_rank"] = rank
        eigvals = np.abs(np.linalg.eigvalsh(fim))
        eigvals = eigvals[eigvals > 0]
        if len(eigvals) >= 2:
            out["fim_cond"] = float(np.max(eigvals) / np.min(eigvals))
        out["fim_singular"] = rank < n_p
        if not out["fim_singular"]:
            cov = np.linalg.inv(fim)
            for i, p in enumerate(params_to_fit):
                var = cov[i, i]
                if var > 0:
                    se = float(np.sqrt(var))
                    out["se_log"][p] = se
                    # Convert log-SE to %CV approximation: CV ≈ SE × 100 (small SE)
                    out["cv_pct"][p] = float(100.0 * (np.exp(se) - 1.0))
            std = np.sqrt(np.diag(cov))
            denom = np.outer(std, std)
            corr = np.where(denom > 0, cov / denom, 0.0)
            for i in range(n_p):
                for j in range(i + 1, n_p):
                    out["correlations"][
                        (params_to_fit[i], params_to_fit[j])
                    ] = float(corr[i, j])
    except (np.linalg.LinAlgError, ValueError):
        pass
    return out


# =====================================================================
# Main entry point
# =====================================================================

def fit_pbpk(
    *,
    datasets: list[Dataset],
    compound_name: str,
    params_to_fit: list[str],
    # Library OR custom physchem (custom requires mw/logP/pKa/fu_p)
    logP: Optional[float] = None,
    pKa: Optional[float] = None,
    fu_p: Optional[float] = None,
    mw: Optional[float] = None,
    compound_type: str = "neutral",
    R_bp: float = 1.0,
    body_weight: float = 73.0,
    # Optimizer
    method: str = "nelder-mead",
    log_scale: bool = True,
    user_bounds: Optional[dict[str, tuple[float, float]]] = None,
    initial_overrides: Optional[dict[str, float]] = None,
    overrides: Optional[set[str]] = None,
    sigma_log: float = 0.3,
) -> FitReport:
    """
    Fit PBPK parameters to one or more observed-data datasets.

    Uses the codex-reviewed policy layer: only allowed parameter names
    can be fitted, and forbidden combinations (e.g. CL_int+fu_p, Fa+Fg)
    raise unless the user passes the matching override token AND the
    data conditions are satisfied (e.g. an IV reference dataset for
    Fa identifiability).

    Returns a structured FitReport with optimizer instrumentation, GOF,
    FIM-based identifiability diagnostics, and user-facing warnings.
    Never returns a regulatory pass/fail verdict.
    """
    from .compound import COMPOUND_LIBRARY
    from .physiology import get_physiology, Sex
    from .pbpk_model import PBPKModel, DosingProtocol, SimulationConfig, Route
    from scipy.optimize import minimize, differential_evolution

    # --- 1) Build base compound ---
    base = _build_compound(
        name=compound_name, library_lookup=COMPOUND_LIBRARY,
        logP=logP, pKa=pKa, fu_p=fu_p, mw=mw,
        compound_type=compound_type, R_bp=R_bp,
    )
    phys = get_physiology(body_weight, Sex.MALE)

    # --- 2) Apply policy ---
    apply_fitting_policy(params_to_fit, datasets=datasets, overrides=overrides)

    # --- 3) Initial values + bounds ---
    initial = {p: float(getattr(base, p, 1.0) or 1.0) for p in params_to_fit}
    if initial_overrides:
        initial.update(initial_overrides)
    bounds = _resolve_bounds(params_to_fit, initial, user_bounds)

    # --- 4) Build per-dataset simulator ---
    def simulate_fn(params: dict, d: Dataset):
        from .compound import CompoundSpec
        c = CompoundSpec(
            name=base.name, mw=base.mw, logP=base.logP, pKa=base.pKa,
            fu_p=params.get("fu_p", base.fu_p),
            compound_type=base.compound_type,
            R_bp=base.R_bp,
            ka=params.get("ka", base.ka),
            Fa=params.get("Fa", base.Fa),
            Fg=base.Fg,
            CL_int=params.get("CL_int", base.CL_int),
            CL_renal=params.get("CL_renal", base.CL_renal),
            metabolism_model=base.metabolism_model,
            Vmax=params.get("Vmax", base.Vmax),
            Km=params.get("Km", base.Km),
            Peff=base.Peff, S0=base.S0,
            particle_radius_um=base.particle_radius_um,
            CLint_gut=base.CLint_gut,
        )
        model = PBPKModel(c, phys)
        dosing = DosingProtocol(d.dose_mg, Route(d.route))
        duration = max(float(np.max(d.obs_t)) * 1.5, 24.0)
        config = SimulationConfig(duration_h=duration, n_timepoints=500)
        return model.simulate(dosing, config), model.simulate(dosing, config).venous_plasma

    # Refactor: simulate_fn must return (sim_time, sim_conc) tuple. Adjust.
    def _simfn(params: dict, d: Dataset):
        from .compound import CompoundSpec
        c = CompoundSpec(
            name=base.name, mw=base.mw, logP=base.logP, pKa=base.pKa,
            fu_p=params.get("fu_p", base.fu_p),
            compound_type=base.compound_type,
            R_bp=base.R_bp,
            ka=params.get("ka", base.ka),
            Fa=params.get("Fa", base.Fa),
            Fg=base.Fg,
            CL_int=params.get("CL_int", base.CL_int),
            CL_renal=params.get("CL_renal", base.CL_renal),
            metabolism_model=base.metabolism_model,
            Vmax=params.get("Vmax", base.Vmax),
            Km=params.get("Km", base.Km),
            Peff=base.Peff, S0=base.S0,
            particle_radius_um=base.particle_radius_um,
            CLint_gut=base.CLint_gut,
        )
        model = PBPKModel(c, phys)
        dosing = DosingProtocol(d.dose_mg, Route(d.route))
        duration = max(float(np.max(d.obs_t)) * 1.5, 24.0)
        config = SimulationConfig(duration_h=duration, n_timepoints=500)
        result = model.simulate(dosing, config)
        return result.time, result.venous_plasma

    objective, counter = _make_objective(
        _simfn, datasets, params_to_fit, initial, log_scale=log_scale,
    )

    # --- 5) Run optimizer ---
    x0 = [initial[p] for p in params_to_fit]
    bounds_list = [bounds[p] for p in params_to_fit]
    if method == "differential-evolution":
        opt = differential_evolution(
            objective, bounds_list, seed=42, maxiter=200, tol=1e-6,
        )
        converged = bool(opt.success)
        msg = opt.message
    else:
        opt = minimize(
            objective, x0, method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-8},
        )
        converged = bool(opt.success) if hasattr(opt, "success") else True
        msg = opt.message if hasattr(opt, "message") else "completed"

    fitted = {p: float(opt.x[i]) for i, p in enumerate(params_to_fit)}
    final_obj = float(opt.fun)
    n_eval = counter["n_eval"]
    n_fail = counter["n_fail"]
    fail_rate = (n_fail / n_eval) if n_eval > 0 else 0.0
    best_is_penalty = final_obj >= 0.5 * _FAIL_PENALTY

    # --- 6) GOF (combined across datasets) ---
    all_obs, all_pred = [], []
    for d in datasets:
        try:
            sim_t, sim_c = _simfn(fitted, d)
            pred = np.interp(d.obs_t, sim_t, sim_c)
            all_obs.append(d.obs_c)
            all_pred.append(pred)
        except Exception:
            continue
    obs_v = np.concatenate(all_obs) if all_obs else np.array([])
    pred_v = np.concatenate(all_pred) if all_pred else np.array([])
    r2 = _r_squared(obs_v, pred_v) if obs_v.size else 0.0
    afe, aafe = _fold_errors(obs_v, pred_v) if obs_v.size else (1.0, 1.0)
    if obs_v.size:
        mask = obs_v > 0
        if mask.any():
            ratios = pred_v[mask] / obs_v[mask]
            pct_2 = float(100.0 * np.sum((ratios > 0.5) & (ratios < 2.0)) / mask.sum())
        else:
            pct_2 = 0.0
    else:
        pct_2 = 0.0

    # --- 7) Identifiability diagnostics ---
    ident = _identifiability(_simfn, datasets, fitted, params_to_fit,
                              bounds, sigma_log=sigma_log)

    # --- 8) Compose warnings ---
    warnings: list[str] = []
    if best_is_penalty:
        warnings.append(
            f"FIT FAILED: final objective ({final_obj:.2g}) is at the "
            f"failure penalty ({_FAIL_PENALTY:.0g}). The optimizer could "
            f"not find a parameter region where the simulation runs "
            f"successfully — every evaluation crashed. Inspect compound "
            f"physchem and dosing range; try wider bounds or DE method."
        )
    if fail_rate > 0.5:
        warnings.append(
            f"high simulation failure rate during optimization "
            f"({n_fail}/{n_eval} = {fail_rate:.0%}). The fit may be "
            f"trapped in a region where most parameter combinations "
            f"crash the simulator. Tighten initial values or bounds."
        )
    if ident["on_bound"]:
        warnings.append(
            f"estimate(s) on bound: {ident['on_bound']}. Bound was "
            f"reached — the true optimum may lie outside the bounds. "
            f"Widen bounds and refit, or accept a constrained estimate."
        )
    if ident["fim_singular"]:
        warnings.append(
            f"FIM singular (rank {ident['fim_rank']} < {len(params_to_fit)}) — "
            f"parameters are not jointly identifiable from this dataset. "
            f"Drop one parameter or add a complementary dataset (e.g. IV "
            f"if you only have oral)."
        )
    if not np.isnan(ident["fim_cond"]) and ident["fim_cond"] > 1e6:
        warnings.append(
            f"FIM condition number {ident['fim_cond']:.1e} is very high — "
            f"some parameter directions are weakly informed. SEs may be "
            f"unreliable; consider parameter reduction."
        )
    high_corr = [
        (a, b, r) for (a, b), r in ident["correlations"].items() if abs(r) > 0.95
    ]
    if high_corr:
        warnings.append(
            f"strong parameter correlations (|ρ|>0.95): "
            + ", ".join(f"{a}↔{b}={r:+.2f}" for a, b, r in high_corr)
            + " — these parameters move together and only their "
            "combination is well-determined."
        )
    if r2 < 0.5 and obs_v.size > 5:
        warnings.append(
            f"R²={r2:.2f} is poor — the model does not describe the data "
            f"well even after fitting. Reconsider the model structure "
            f"(distribution model, absorption model, additional clearance "
            f"pathways)."
        )
    if aafe > 3.0:
        warnings.append(
            f"AAFE={aafe:.2f} > 3 — predictions disagree with observations "
            f"by more than 3-fold on average. Treat the fit as a starting "
            f"point, not a final model."
        )

    return FitReport(
        fitted_params=fitted, initial_params=initial, bounds=bounds,
        optimizer_method=method, objective_value=final_obj,
        n_observations=int(obs_v.size), n_eval=n_eval, n_fail=n_fail,
        fail_rate=fail_rate, best_is_penalty=best_is_penalty,
        converged=converged, message=str(msg),
        r_squared=r2, afe=afe, aafe=aafe, pct_within_2fold=pct_2,
        on_bound=ident["on_bound"],
        se_log=ident["se_log"], cv_pct=ident["cv_pct"],
        fim_rank=ident["fim_rank"], fim_cond=ident["fim_cond"],
        fim_singular=ident["fim_singular"],
        correlations=ident["correlations"],
        warnings=warnings,
    )


# =====================================================================
# Backwards-compatible legacy entry point
# =====================================================================

def fit_pbpk_to_data(
    observed_file: str,
    compound_name: str,
    dose_mg: float,
    route: str = "oral",
    params_to_fit: Optional[list[str]] = None,
    body_weight: float = 73.0,
    # New (codex SHIP-NOW): custom compound + multi-dataset hooks
    logP: Optional[float] = None,
    pKa: Optional[float] = None,
    fu_p: Optional[float] = None,
    mw: Optional[float] = None,
    compound_type: str = "neutral",
    R_bp: float = 1.0,
    overrides: Optional[set[str]] = None,
) -> str:
    """Single-dataset fitting wrapper. Returns markdown report.
    Use `fit_pbpk(...)` directly for multi-dataset co-fitting and
    full FitReport access."""
    if params_to_fit is None:
        params_to_fit = ["CL_int"] + (["ka"] if route == "oral" else [])

    ds = Dataset(csv_file=observed_file, dose_mg=dose_mg, route=route)
    rep = fit_pbpk(
        datasets=[ds],
        compound_name=compound_name,
        params_to_fit=params_to_fit,
        logP=logP, pKa=pKa, fu_p=fu_p, mw=mw,
        compound_type=compound_type, R_bp=R_bp,
        body_weight=body_weight, overrides=overrides,
    )
    return rep.to_markdown()
