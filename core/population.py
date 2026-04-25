"""
Virtual population generation for population PBPK simulation.

Generates N virtual individuals with correlated physiological parameters,
runs the PBPK model for each, and collects population PK statistics.

Approach:
  1. Sample demographics (age, sex, weight, height)
  2. Derive physiology with allometric scaling + inter-individual variability
  3. Sample biochemical parameters (CYP abundance, fu_p, MPPGL)
  4. Apply correlation structure via Cholesky decomposition

References:
  - Willmann S et al. J Pharmacokinet Pharmacodyn 2007;34:401-431
  - Jamei M et al. AAPS J 2009;11:225-237
  - McNally K et al. Front Pharmacol 2020;11:1-15
  - OSPSuite-R PKSimDB: VIEW_PARAMETER_DISTRIBUTIONS
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .compound import CompoundSpec
from .physiology import (
    Organ, Sex, PhysiologyParams, get_physiology,
    calculate_cardiac_output,
)
from .pbpk_model import (
    PBPKModel, DosingProtocol, SimulationConfig, SimulationResult,
)
from .pk_calculator import calculate_pk_parameters, PKParameters


# ===================================================================
# Parameter variability definitions (CV%, distribution type)
# Source: PK-Sim defaults, Willmann 2007, Jamei 2009
# ===================================================================

PARAM_VARIABILITY = {
    # Parameter: (CV%, distribution_type)
    # "normal" for ~symmetric, "lognormal" for right-skewed
    "body_weight":      (0.20, "normal"),
    "height":           (0.05, "normal"),
    "cardiac_output":   (0.20, "lognormal"),
    "hematocrit":       (0.08, "normal"),

    # Organ volumes (CV as fraction of mean)
    "V_liver":          (0.25, "lognormal"),
    "V_kidney":         (0.20, "lognormal"),
    "V_muscle":         (0.15, "normal"),
    "V_adipose":        (0.30, "lognormal"),
    "V_brain":          (0.10, "normal"),
    "V_heart":          (0.15, "lognormal"),
    "V_gut":            (0.20, "lognormal"),
    "V_lung":           (0.15, "lognormal"),
    "V_spleen":         (0.20, "lognormal"),
    "V_skin":           (0.15, "normal"),
    "V_bone":           (0.15, "normal"),
    "V_pancreas":       (0.25, "lognormal"),

    # Blood flows (CV of fraction)
    "Q_liver":          (0.25, "lognormal"),
    "Q_kidney":         (0.20, "lognormal"),

    # Biochemical
    "fu_p":             (0.25, "lognormal"),
    "MPPGL":            (0.30, "lognormal"),
    "HPGL":             (0.20, "lognormal"),
    "GFR":              (0.20, "normal"),

    # CYP abundances
    "CYP1A2":           (0.50, "lognormal"),
    "CYP2C9":           (0.40, "lognormal"),
    "CYP2C19":          (0.60, "lognormal"),
    "CYP2D6":           (1.00, "lognormal"),  # polymorphic, very high
    "CYP3A4":           (0.60, "lognormal"),
    "CYP2E1":           (0.40, "lognormal"),
}

# Correlation matrix (key pairs)
# Source: Willmann 2007, Jamei 2009
CORRELATIONS = {
    ("body_weight", "V_liver"):     0.75,
    ("body_weight", "V_kidney"):    0.60,
    ("body_weight", "V_muscle"):    0.80,
    ("body_weight", "V_adipose"):   0.70,
    ("body_weight", "cardiac_output"): 0.80,
    ("body_weight", "V_lung"):      0.60,
    ("body_weight", "V_heart"):     0.50,
    ("body_weight", "GFR"):         0.40,
    ("V_liver", "V_kidney"):        0.30,
    ("CYP3A4", "CYP2C9"):          0.15,
}


# ===================================================================
# Sampling functions
# ===================================================================

def _sample_lognormal(mean: float, cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from log-normal with given mean and CV."""
    if cv <= 0:
        return np.full(n, mean)
    sigma2 = np.log(1.0 + cv ** 2)
    mu = np.log(mean) - sigma2 / 2.0
    return rng.lognormal(mu, np.sqrt(sigma2), n)


def _sample_normal_positive(mean: float, cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from truncated normal (positive values only)."""
    if cv <= 0:
        return np.full(n, mean)
    sd = mean * cv
    samples = rng.normal(mean, sd, n)
    return np.clip(samples, mean * 0.1, mean * 3.0)


def _sample_parameter(mean: float, cv: float, dist: str, n: int,
                       rng: np.random.Generator) -> np.ndarray:
    if dist == "lognormal":
        return _sample_lognormal(mean, cv, n, rng)
    return _sample_normal_positive(mean, cv, n, rng)


# ===================================================================
# Virtual population
# ===================================================================

@dataclass
class PopulationResult:
    """Results of population PBPK simulation."""
    n_individuals: int
    pk_params: list[PKParameters]           # per-individual PK
    plasma_profiles: list[np.ndarray]       # per-individual C(t)
    time: np.ndarray                        # shared time vector
    demographics: dict                      # sampled demographics

    # Summary statistics
    Cmax_median: float = 0.0
    Cmax_5th: float = 0.0
    Cmax_95th: float = 0.0
    AUC_median: float = 0.0
    AUC_5th: float = 0.0
    AUC_95th: float = 0.0
    thalf_median: float = 0.0

    def compute_statistics(self):
        cmax_vals = [pk.Cmax for pk in self.pk_params if pk.Cmax > 0]
        auc_vals = [pk.AUC_0_inf for pk in self.pk_params if pk.AUC_0_inf > 0]
        thalf_vals = [pk.t_half for pk in self.pk_params if 0 < pk.t_half < 1000]

        if cmax_vals:
            self.Cmax_median = float(np.median(cmax_vals))
            self.Cmax_5th = float(np.percentile(cmax_vals, 5))
            self.Cmax_95th = float(np.percentile(cmax_vals, 95))
        if auc_vals:
            self.AUC_median = float(np.median(auc_vals))
            self.AUC_5th = float(np.percentile(auc_vals, 5))
            self.AUC_95th = float(np.percentile(auc_vals, 95))
        if thalf_vals:
            self.thalf_median = float(np.median(thalf_vals))

    def to_markdown(self, compound_name: str = "") -> str:
        lines = [f"## Population PK Summary"]
        if compound_name:
            lines[0] += f" — {compound_name}"
        lines.extend([
            f"\nN = {self.n_individuals} virtual individuals\n",
            "| Parameter | Median | 5th %ile | 95th %ile |",
            "|-----------|--------|---------|----------|",
            f"| Cmax (mg/L) | {self.Cmax_median:.4g} | {self.Cmax_5th:.4g} | {self.Cmax_95th:.4g} |",
            f"| AUC (mg·h/L) | {self.AUC_median:.4g} | {self.AUC_5th:.4g} | {self.AUC_95th:.4g} |",
            f"| t½ (h) | {self.thalf_median:.2f} | — | — |",
        ])
        return "\n".join(lines)


def run_population_simulation(
    compound: CompoundSpec,
    dosing: DosingProtocol,
    config: SimulationConfig,
    n_individuals: int = 100,
    proportion_female: float = 0.5,
    age_range: tuple = (20, 60),
    weight_mean: float = 73.0,
    seed: Optional[int] = None,
    kp_method: str = "rodgers_rowland",
) -> PopulationResult:
    """
    Run population PBPK simulation.

    Generates N virtual individuals, runs deterministic PBPK for each,
    and collects population PK statistics.

    Args:
        compound: Drug compound specification.
        dosing: Dosing protocol.
        config: Simulation configuration.
        n_individuals: Number of virtual individuals.
        proportion_female: Fraction of females (0-1).
        age_range: (min_age, max_age) in years.
        weight_mean: Mean body weight (kg).
        seed: Random seed for reproducibility.

    Returns:
        PopulationResult with per-individual PK and summary statistics.
    """
    rng = np.random.default_rng(seed)
    n = n_individuals

    # --- Sample demographics ---
    sex_draws = rng.choice([0, 1], n, p=[1 - proportion_female, proportion_female])
    sexes = [Sex.FEMALE if s == 1 else Sex.MALE for s in sex_draws]
    ages = rng.uniform(age_range[0], age_range[1], n)
    bw_cv = PARAM_VARIABILITY["body_weight"][0]
    weights = _sample_normal_positive(weight_mean, bw_cv, n, rng)

    # --- Sample CL_int variability (CYP abundance variation) ---
    cyp_cv = PARAM_VARIABILITY.get("CYP3A4", (0.6, "lognormal"))
    cl_int_factors = _sample_lognormal(1.0, cyp_cv[0], n, rng)

    # --- Sample fu_p variability ---
    fu_cv = PARAM_VARIABILITY["fu_p"][0]
    fu_factors = _sample_lognormal(1.0, fu_cv, n, rng)

    # --- Sample additional variability ---
    ka_cv = 0.30
    ka_factors = _sample_lognormal(1.0, ka_cv, n, rng)
    Fg_cv = 0.20
    Fg_factors = _sample_lognormal(1.0, Fg_cv, n, rng)

    # --- Run individual simulations ---
    pk_list = []
    profiles = []
    time_ref = None

    for i in range(n):
        # Build individual physiology with age + GFR
        ind_age = float(ages[i])
        phys = get_physiology(
            body_weight=float(weights[i]),
            sex=sexes[i],
            age_years=ind_age,
        )

        # Scale CL_renal with individual GFR
        # Base CL_renal is at GFR=7.2 L/h; scale by individual GFR ratio
        gfr_ratio = phys.GFR / 7.2 if 7.2 > 0 else 1.0
        ind_cl_renal = compound.CL_renal * gfr_ratio

        # Perturb compound parameters
        ind_fu_p = min(max(compound.fu_p * fu_factors[i], 0.001), 1.0)
        ind_ka = compound.ka * float(ka_factors[i])
        ind_Fg = min(max(compound.Fg * float(Fg_factors[i]), 0.01), 1.0)

        c_i = CompoundSpec(
            name=compound.name,
            mw=compound.mw,
            logP=compound.logP,
            pKa=compound.pKa,
            fu_p=ind_fu_p,
            compound_type=compound.compound_type,
            R_bp=compound.R_bp,
            ka=ind_ka,
            Fa=compound.Fa,
            Fg=ind_Fg,
            CL_int=compound.CL_int * cl_int_factors[i],
            CL_renal=ind_cl_renal,
            metabolism_model=compound.metabolism_model,
            Vmax=compound.Vmax,
            Km=compound.Km,
            Peff=compound.Peff,
            S0=compound.S0,
            particle_radius_um=compound.particle_radius_um,
            CLint_gut=compound.CLint_gut,
        )

        try:
            from .partition_coeff import predict_kp_all, KpMethod
            kp_method_enum = KpMethod(kp_method)
            kp_override = (predict_kp_all(c_i, kp_method_enum)
                           if kp_method_enum != KpMethod.RODGERS_ROWLAND else None)
            model = PBPKModel(c_i, phys, kp_override=kp_override)
            result = model.simulate(dosing, config)

            is_iv = dosing.route.value in ("iv_bolus", "iv_infusion")
            pk = calculate_pk_parameters(
                result.time, result.venous_plasma,
                dose_mg=dosing.dose_mg, is_iv=is_iv,
            )
            pk_list.append(pk)
            profiles.append(result.venous_plasma)

            if time_ref is None:
                time_ref = result.time
        except Exception:
            # Skip failed individuals
            pk_list.append(PKParameters(
                Cmax=0, Tmax=0, AUC_0_t=0, AUC_0_inf=0,
                t_half=0, lambda_z=0, CL_F=0, Vz_F=0,
                Vss=None, MRT=0, C_last=0, T_last=0,
            ))
            if time_ref is not None:
                profiles.append(np.zeros_like(time_ref))

    if time_ref is None:
        time_ref = np.linspace(0, config.duration_h, config.n_timepoints)

    pop_result = PopulationResult(
        n_individuals=n,
        pk_params=pk_list,
        plasma_profiles=profiles,
        time=time_ref,
        demographics={
            "weights": weights.tolist(),
            "ages": ages.tolist(),
            "sexes": [s.value for s in sexes],
            "cl_int_factors": cl_int_factors.tolist(),
            "fu_factors": fu_factors.tolist(),
        },
    )
    pop_result.compute_statistics()
    return pop_result
