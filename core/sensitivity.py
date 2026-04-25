"""
Sensitivity analysis and parameter fitting for PBPK models.

1. Local sensitivity analysis (one-at-a-time, OAT)
2. Parameter fitting to observed data (scipy.optimize)
3. Goodness-of-fit metrics

References:
  - McNally K et al. Front Pharmacol 2020;11:1-15
  - OSPSuite.ParameterIdentification (HJKB, BOBYQA patterns)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional
from scipy.optimize import minimize, differential_evolution


# ===================================================================
# Local Sensitivity Analysis (OAT)
# ===================================================================

@dataclass
class SensitivityResult:
    """Result of local sensitivity analysis."""
    param_names: list[str]
    sensitivities: dict[str, float]   # param -> normalized sensitivity
    pk_metric: str                     # which PK metric was analyzed

    def to_markdown(self) -> str:
        lines = [
            f"## Sensitivity Analysis — {self.pk_metric}\n",
            "| Parameter | Normalized Sensitivity | Rank |",
            "|-----------|----------------------|------|",
        ]
        sorted_params = sorted(
            self.sensitivities.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for rank, (param, sens) in enumerate(sorted_params, 1):
            direction = "+" if sens > 0 else "-"
            lines.append(f"| {param} | {direction}{abs(sens):.3f} | {rank} |")
        lines.append(
            "\n*Normalized: fractional change in PK / fractional change in parameter*"
        )
        return "\n".join(lines)


def local_sensitivity(
    simulate_fn: Callable,
    base_params: dict,
    param_names: list[str],
    pk_metric: str = "AUC_0_inf",
    perturbation: float = 0.05,
) -> SensitivityResult:
    """
    One-at-a-time local sensitivity analysis.

    For each parameter, perturb by ±perturbation fraction and measure
    the change in the specified PK metric.

    Normalized sensitivity = (ΔPK/PK_base) / (Δparam/param_base)

    Args:
        simulate_fn: Function(params_dict) -> PKParameters
        base_params: Dict of parameter name -> base value
        param_names: List of parameters to analyze
        pk_metric: Attribute of PKParameters to measure
        perturbation: Fractional perturbation (default 5%)

    Returns:
        SensitivityResult with normalized sensitivities
    """
    # Baseline simulation
    pk_base = simulate_fn(base_params)
    pk_base_val = getattr(pk_base, pk_metric, 0)
    if pk_base_val == 0:
        pk_base_val = 1e-10

    sensitivities = {}
    for pname in param_names:
        if pname not in base_params or base_params[pname] == 0:
            sensitivities[pname] = 0.0
            continue

        base_val = base_params[pname]

        # Perturb up
        params_up = base_params.copy()
        params_up[pname] = base_val * (1.0 + perturbation)
        pk_up = simulate_fn(params_up)
        pk_up_val = getattr(pk_up, pk_metric, 0)

        # Perturb down
        params_down = base_params.copy()
        params_down[pname] = base_val * (1.0 - perturbation)
        pk_down = simulate_fn(params_down)
        pk_down_val = getattr(pk_down, pk_metric, 0)

        # Central difference normalized sensitivity
        dpk = (pk_up_val - pk_down_val) / (2.0 * perturbation * pk_base_val)
        sensitivities[pname] = dpk

    return SensitivityResult(
        param_names=param_names,
        sensitivities=sensitivities,
        pk_metric=pk_metric,
    )


# ===================================================================
# Parameter Fitting
# ===================================================================

@dataclass
class FitResult:
    """Result of parameter fitting."""
    fitted_params: dict[str, float]
    objective_value: float
    n_observations: int
    r_squared: float
    afe: float           # Average fold error
    aafe: float          # Absolute average fold error
    success: bool
    message: str

    def to_markdown(self) -> str:
        lines = [
            "## Parameter Fitting Result\n",
            f"**Success:** {self.success}",
            f"**Objective (WSSR):** {self.objective_value:.4g}",
            f"**R²:** {self.r_squared:.4f}",
            f"**AFE:** {self.afe:.3f} (1.0 = no bias)",
            f"**AAFE:** {self.aafe:.3f} (<2.0 = acceptable)\n",
            "### Fitted Parameters\n",
            "| Parameter | Value |",
            "|-----------|-------|",
        ]
        for k, v in self.fitted_params.items():
            lines.append(f"| {k} | {v:.4g} |")
        return "\n".join(lines)


def fit_parameters(
    simulate_fn: Callable,
    observed_time: np.ndarray,
    observed_conc: np.ndarray,
    param_names: list[str],
    param_bounds: dict[str, tuple[float, float]],
    initial_params: dict[str, float],
    log_scale: bool = True,
    method: str = "nelder-mead",
) -> FitResult:
    """
    Fit PBPK model parameters to observed concentration-time data.

    Args:
        simulate_fn: Function(params_dict) -> (time_array, conc_array)
        observed_time: Observed time points (h)
        observed_conc: Observed concentrations (mg/L)
        param_names: Names of parameters to fit
        param_bounds: {param: (lower, upper)} bounds
        initial_params: Starting values
        log_scale: Use log-transformed residuals (better for PK)
        method: "nelder-mead" (local) or "differential-evolution" (global)

    Returns:
        FitResult with optimized parameters and GOF metrics
    """
    n_obs = len(observed_time)

    def objective(x):
        params = initial_params.copy()
        for i, pname in enumerate(param_names):
            params[pname] = x[i]

        try:
            sim_time, sim_conc = simulate_fn(params)
            # Interpolate simulated to observed time points
            pred = np.interp(observed_time, sim_time, sim_conc)

            if log_scale:
                mask = (pred > 0) & (observed_conc > 0)
                if np.sum(mask) < 2:
                    return 1e10
                residuals = np.log(pred[mask]) - np.log(observed_conc[mask])
            else:
                residuals = pred - observed_conc

            return np.sum(residuals ** 2)
        except Exception:
            return 1e10

    x0 = [initial_params[p] for p in param_names]
    bounds_list = [param_bounds.get(p, (x0[i] * 0.01, x0[i] * 100)) for i, p in enumerate(param_names)]

    if method == "differential-evolution":
        result = differential_evolution(objective, bounds_list, seed=42, maxiter=200, tol=1e-6)
    else:
        result = minimize(objective, x0, method="Nelder-Mead",
                         options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-8})

    # Extract fitted params
    fitted = {}
    for i, pname in enumerate(param_names):
        fitted[pname] = result.x[i]

    # Compute GOF
    params_final = initial_params.copy()
    params_final.update(fitted)
    try:
        sim_time, sim_conc = simulate_fn(params_final)
        pred = np.interp(observed_time, sim_time, sim_conc)
    except Exception:
        pred = np.zeros_like(observed_conc)

    r2 = _r_squared(observed_conc, pred)
    afe, aafe = _fold_errors(observed_conc, pred)

    return FitResult(
        fitted_params=fitted,
        objective_value=result.fun,
        n_observations=n_obs,
        r_squared=r2,
        afe=afe,
        aafe=aafe,
        success=result.success if hasattr(result, 'success') else True,
        message=result.message if hasattr(result, 'message') else "completed",
    )


# ===================================================================
# Goodness-of-Fit Metrics
# ===================================================================

def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    mask = (observed > 0) & (predicted > 0)
    if np.sum(mask) < 2:
        return 0.0
    obs = observed[mask]
    pred = predicted[mask]
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _fold_errors(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """
    Average Fold Error (AFE) and Absolute Average Fold Error (AAFE).

    AFE = 10^(mean(log10(pred/obs)))  — bias measure, 1.0 = no bias
    AAFE = 10^(mean(|log10(pred/obs)|)) — precision, <2 = acceptable
    """
    mask = (observed > 0) & (predicted > 0)
    if np.sum(mask) < 1:
        return 1.0, 1.0
    log_ratios = np.log10(predicted[mask] / observed[mask])
    afe = 10.0 ** np.mean(log_ratios)
    aafe = 10.0 ** np.mean(np.abs(log_ratios))
    return float(afe), float(aafe)


def compute_gof(
    observed_time: np.ndarray,
    observed_conc: np.ndarray,
    simulated_time: np.ndarray,
    simulated_conc: np.ndarray,
) -> str:
    """Compute and format GOF metrics."""
    pred = np.interp(observed_time, simulated_time, simulated_conc)
    r2 = _r_squared(observed_conc, pred)
    afe, aafe = _fold_errors(observed_conc, pred)

    within_2fold = np.sum(
        (pred[observed_conc > 0] / observed_conc[observed_conc > 0] > 0.5) &
        (pred[observed_conc > 0] / observed_conc[observed_conc > 0] < 2.0)
    )
    total_nonzero = np.sum(observed_conc > 0)
    pct_2fold = within_2fold / total_nonzero * 100 if total_nonzero > 0 else 0

    lines = [
        "## Goodness-of-Fit\n",
        "| Metric | Value | Interpretation |",
        "|--------|-------|----------------|",
        f"| R² | {r2:.4f} | {'Good' if r2 > 0.8 else 'Fair' if r2 > 0.5 else 'Poor'} |",
        f"| AFE | {afe:.3f} | {'No bias' if 0.8 < afe < 1.25 else 'Biased'} |",
        f"| AAFE | {aafe:.3f} | {'Good' if aafe < 2 else 'Acceptable' if aafe < 3 else 'Poor'} |",
        f"| Within 2-fold | {pct_2fold:.0f}% | {'Good' if pct_2fold > 80 else 'Fair'} |",
    ]
    return "\n".join(lines)
