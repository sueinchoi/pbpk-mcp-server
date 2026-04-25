"""
Observed data loading and parameter fitting pipeline.

Reads CSV/TSV files with observed PK data, fits PBPK model parameters,
and generates goodness-of-fit reports.

CSV format expected:
  time,concentration
  0.5,1.23
  1.0,2.45
  ...

Or with header variations: Time,Conc / time_h,conc_mg_L / etc.
"""

import csv
import os
import numpy as np
from typing import Optional
from dataclasses import dataclass

from .sensitivity import fit_parameters, compute_gof, FitResult


def load_observed_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load observed concentration-time data from CSV/TSV file.

    Auto-detects delimiter and header names.
    Returns (time_array, concentration_array).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        # Detect delimiter
        sample = f.read(2000)
        f.seek(0)
        if "\t" in sample:
            delimiter = "\t"
        else:
            delimiter = ","

        reader = csv.DictReader(f, delimiter=delimiter)
        fields = reader.fieldnames

        # Find time and concentration columns
        time_col = None
        conc_col = None
        for col in fields:
            cl = col.lower().strip()
            if cl in ("time", "time_h", "time (h)", "t", "hours"):
                time_col = col
            elif cl in ("concentration", "conc", "conc_mg_l", "conc (mg/l)",
                        "c", "cp", "plasma", "dv"):
                conc_col = col

        if time_col is None:
            time_col = fields[0]
        if conc_col is None:
            conc_col = fields[1] if len(fields) > 1 else fields[0]

        times = []
        concs = []
        for row in reader:
            try:
                t = float(row[time_col])
                c = float(row[conc_col])
                if t >= 0 and c >= 0:
                    times.append(t)
                    concs.append(c)
            except (ValueError, KeyError):
                continue

    if not times:
        raise ValueError("No valid data points found in file")

    return np.array(times), np.array(concs)


def fit_pbpk_to_data(
    observed_file: str,
    compound_name: str,
    dose_mg: float,
    route: str = "oral",
    params_to_fit: Optional[list[str]] = None,
    body_weight: float = 73.0,
) -> str:
    """
    Fit PBPK model to observed data from CSV file.

    Default fitted parameters: CL_int, ka (for oral), Vss-proxy via Kp scaling.

    Args:
        observed_file: Path to CSV with time,concentration columns.
        compound_name: Drug name (from library) or for labeling.
        dose_mg: Dose amount (mg).
        route: "oral" or "iv_bolus".
        params_to_fit: List of parameter names to fit. Default: ["CL_int", "ka"].
        body_weight: Subject body weight (kg).

    Returns:
        Markdown report with fitted parameters and GOF.
    """
    from .compound import COMPOUND_LIBRARY, CompoundSpec, CompoundType
    from .physiology import get_physiology, Sex
    from .pbpk_model import PBPKModel, DosingProtocol, SimulationConfig, Route

    # Load data
    obs_t, obs_c = load_observed_data(observed_file)

    # Get base compound
    if compound_name.lower() in COMPOUND_LIBRARY:
        base_compound = COMPOUND_LIBRARY[compound_name.lower()]
    else:
        raise ValueError(f"Compound '{compound_name}' not in library. Provide properties manually.")

    phys = get_physiology(body_weight, Sex.MALE)
    route_enum = Route(route)

    if params_to_fit is None:
        params_to_fit = ["CL_int"]
        if route == "oral":
            params_to_fit.append("ka")

    # Build simulation function
    def simulate_fn(params):
        c = CompoundSpec(
            name=base_compound.name,
            mw=base_compound.mw,
            logP=base_compound.logP,
            pKa=base_compound.pKa,
            fu_p=base_compound.fu_p,
            compound_type=base_compound.compound_type,
            R_bp=base_compound.R_bp,
            ka=params.get("ka", base_compound.ka),
            Fa=base_compound.Fa,
            Fg=base_compound.Fg,
            CL_int=params.get("CL_int", base_compound.CL_int),
            CL_renal=base_compound.CL_renal,
        )
        model = PBPKModel(c, phys)
        dosing = DosingProtocol(dose_mg, route_enum)
        duration = max(obs_t) * 1.5
        config = SimulationConfig(duration_h=duration, n_timepoints=500)
        result = model.simulate(dosing, config)
        return result.time, result.venous_plasma

    # Initial params and bounds
    initial = {}
    bounds = {}
    for p in params_to_fit:
        val = getattr(base_compound, p, 1.0)
        initial[p] = val if val > 0 else 1.0
        bounds[p] = (val * 0.01, val * 100)

    # Fit
    fit_result = fit_parameters(
        simulate_fn, obs_t, obs_c,
        params_to_fit, bounds, initial,
        log_scale=True, method="nelder-mead",
    )

    # Generate GOF with fitted params
    sim_t, sim_c = simulate_fn(fit_result.fitted_params)
    gof_str = compute_gof(obs_t, obs_c, sim_t, sim_c)

    # Report
    lines = [
        f"## Parameter Fitting — {compound_name}\n",
        f"Observed data: {os.path.basename(observed_file)} ({len(obs_t)} points)",
        f"Dose: {dose_mg} mg {route}\n",
        fit_result.to_markdown(),
        "",
        gof_str,
    ]
    return "\n".join(lines)
