"""
Non-compartmental analysis (NCA) PK parameter calculator.

Computes standard PK parameters from concentration-time profiles:
- Cmax, Tmax
- AUC (linear-log trapezoidal)
- Terminal half-life (t½)
- Apparent clearance (CL/F)
- Apparent volume of distribution (Vz/F, Vss)
- Mean Residence Time (MRT)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PKParameters:
    """Standard NCA PK parameters."""

    Cmax: float           # Maximum concentration (mg/L)
    Tmax: float           # Time of Cmax (h)
    AUC_0_t: float        # AUC from 0 to last measurable time (mg*h/L)
    AUC_0_inf: float      # AUC from 0 to infinity (mg*h/L)
    t_half: float         # Terminal elimination half-life (h)
    lambda_z: float       # Terminal elimination rate constant (1/h)
    CL_F: float           # Apparent clearance (L/h)
    Vz_F: float           # Apparent volume of distribution (L)
    Vss: Optional[float]  # Steady-state Vd (L), only for IV
    MRT: float            # Mean Residence Time (h)
    C_last: float         # Last measurable concentration (mg/L)
    T_last: float         # Time of last measurable concentration (h)
    # --- NCA reliability fields (FDA/EMA criteria) ---
    extrapolation_fraction: float = 0.0   # (AUC_inf - AUC_0_t) / AUC_inf
    n_terminal_points: int = 0             # points used for lambda_z fit
    terminal_r2: float = 0.0               # R² of terminal log-linear fit
    duration_per_t_half: float = 0.0       # duration / t_half (≥3 is good)
    reliability_flags: list = None         # human-readable warnings

    def __post_init__(self):
        if self.reliability_flags is None:
            self.reliability_flags = []

    def is_reliable(self) -> bool:
        """True if all FDA/EMA reliability criteria pass."""
        return not self.reliability_flags

    def to_markdown(self, compound_name: str = "", dose_mg: float = 0) -> str:
        """Format as markdown table."""
        lines = ["## PK Parameters (NCA)"]
        if compound_name:
            lines[0] += f" — {compound_name}"
        if dose_mg > 0:
            lines.append(f"\nDose: {dose_mg:.1f} mg\n")

        lines.extend([
            "",
            "| Parameter | Value | Unit |",
            "|-----------|-------|------|",
            f"| Cmax | {self.Cmax:.4g} | mg/L |",
            f"| Tmax | {self.Tmax:.2f} | h |",
            f"| AUC(0-t) | {self.AUC_0_t:.4g} | mg·h/L |",
            f"| AUC(0-inf) | {self.AUC_0_inf:.4g} | mg·h/L |",
            f"| t½ | {self.t_half:.2f} | h |",
            f"| λz | {self.lambda_z:.4g} | 1/h |",
            f"| CL/F | {self.CL_F:.4g} | L/h |",
            f"| Vz/F | {self.Vz_F:.4g} | L |",
        ])
        if self.Vss is not None:
            lines.append(f"| Vss | {self.Vss:.4g} | L |")
        lines.extend([
            f"| MRT | {self.MRT:.2f} | h |",
            f"| C_last | {self.C_last:.4g} | mg/L |",
            f"| T_last | {self.T_last:.2f} | h |",
        ])
        # NCA reliability footer (always emit so silent-fallback is impossible)
        lines.append("")
        lines.append("### NCA reliability")
        lines.append(f"- Extrapolation fraction: {self.extrapolation_fraction:.1%} "
                     f"(FDA/EMA: <20% recommended)")
        lines.append(f"- Terminal-phase points: {self.n_terminal_points} "
                     f"(>= 3 required for valid λz)")
        lines.append(f"- Terminal R²: {self.terminal_r2:.3f}")
        lines.append(f"- Duration / t½: {self.duration_per_t_half:.1f} "
                     f"(>= 3 recommended)")
        if self.reliability_flags:
            lines.append("\n⚠️ **NCA reliability warnings:**")
            for f in self.reliability_flags:
                lines.append(f"- {f}")
        else:
            lines.append("\n_NCA reliability criteria all met._")
        return "\n".join(lines)


def _auc_linear_log_trapezoidal(time: np.ndarray, conc: np.ndarray) -> float:
    """
    Linear-log trapezoidal AUC.

    Uses linear trapezoidal rule for ascending concentrations and
    log-linear trapezoidal rule for descending concentrations.
    """
    auc = 0.0
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        c1 = conc[i - 1]
        c2 = conc[i]

        if dt <= 0 or c1 <= 0 or c2 <= 0:
            # Linear if any concentration is zero
            auc += 0.5 * (c1 + c2) * dt
        elif c2 >= c1:
            # Ascending: linear trapezoidal
            auc += 0.5 * (c1 + c2) * dt
        else:
            # Descending: log-linear trapezoidal
            log_ratio = np.log(c1 / c2)
            if log_ratio > 0:
                auc += (c1 - c2) * dt / log_ratio
            else:
                auc += 0.5 * (c1 + c2) * dt

    return auc


def _estimate_terminal_slope(
    time: np.ndarray,
    conc: np.ndarray,
    min_points: int = 3,
) -> tuple[float, float, int]:
    """
    Estimate terminal elimination rate constant (lambda_z) by log-linear
    regression on the terminal phase.

    Selects the best-fit terminal portion (highest R²) with >= min_points.

    Returns:
        (lambda_z, R², n_terminal_points)
    """
    # Filter out zero/negative concentrations
    mask = conc > 0
    t_pos = time[mask]
    c_pos = conc[mask]

    if len(t_pos) < min_points:
        return 0.0, 0.0, 0

    ln_c = np.log(c_pos)

    best_lz = 0.0
    best_r2 = 0.0
    best_n = 0

    # Try different starting points for terminal phase
    # Start from Cmax onwards (descending phase)
    cmax_idx = np.argmax(c_pos)
    start_range = range(cmax_idx, len(t_pos) - min_points + 1)

    for start in start_range:
        t_seg = t_pos[start:]
        ln_seg = ln_c[start:]

        if len(t_seg) < min_points:
            continue

        # Linear regression: ln(C) = a - lambda_z * t
        n = len(t_seg)
        sum_t = np.sum(t_seg)
        sum_lnc = np.sum(ln_seg)
        sum_t2 = np.sum(t_seg ** 2)
        sum_t_lnc = np.sum(t_seg * ln_seg)

        denom = n * sum_t2 - sum_t ** 2
        if abs(denom) < 1e-30:
            continue

        slope = (n * sum_t_lnc - sum_t * sum_lnc) / denom

        # R² calculation
        mean_lnc = sum_lnc / n
        ss_tot = np.sum((ln_seg - mean_lnc) ** 2)
        intercept = (sum_lnc - slope * sum_t) / n
        predicted = intercept + slope * t_seg
        ss_res = np.sum((ln_seg - predicted) ** 2)

        if ss_tot > 0:
            r2 = 1.0 - ss_res / ss_tot
        else:
            r2 = 0.0

        lz = -slope  # lambda_z is positive

        # Select best R² with positive lambda_z
        if lz > 0 and r2 > best_r2:
            best_lz = lz
            best_r2 = r2
            best_n = n

    return best_lz, best_r2, best_n


def calculate_pk_parameters(
    time: np.ndarray,
    concentration: np.ndarray,
    dose_mg: float = 0.0,
    is_iv: bool = False,
) -> PKParameters:
    """
    Calculate NCA PK parameters from a concentration-time profile.

    Args:
        time: Time points (h)
        concentration: Plasma concentration (mg/L)
        dose_mg: Dose amount (mg) for CL/F and Vz/F calculation
        is_iv: True for IV dosing (affects Vss calculation)

    Returns:
        PKParameters dataclass
    """
    if len(time) == 0 or len(concentration) == 0:
        return PKParameters(
            Cmax=0, Tmax=0, AUC_0_t=0, AUC_0_inf=0,
            t_half=0, lambda_z=0, CL_F=0, Vz_F=0,
            Vss=None, MRT=0, C_last=0, T_last=0,
        )

    # Cmax, Tmax
    cmax_idx = np.argmax(concentration)
    Cmax = float(concentration[cmax_idx])
    Tmax = float(time[cmax_idx])

    # Last measurable concentration
    nonzero_mask = concentration > Cmax * 1e-6  # threshold: 0.0001% of Cmax
    if np.any(nonzero_mask):
        last_idx = np.where(nonzero_mask)[0][-1]
        C_last = float(concentration[last_idx])
        T_last = float(time[last_idx])
    else:
        C_last = 0.0
        T_last = float(time[-1])

    # AUC(0-t) using linear-log trapezoidal
    AUC_0_t = _auc_linear_log_trapezoidal(time, concentration)

    # Terminal slope estimation
    lambda_z, r2, n_terminal = _estimate_terminal_slope(time, concentration)

    # AUC(0-inf) = AUC(0-t) + C_last / lambda_z
    if lambda_z > 0 and C_last > 0:
        AUC_0_inf = AUC_0_t + C_last / lambda_z
    else:
        AUC_0_inf = AUC_0_t

    # Half-life
    t_half = np.log(2) / lambda_z if lambda_z > 0 else float("inf")

    # Clearance and Volume of distribution
    if dose_mg > 0 and AUC_0_inf > 0:
        CL_F = dose_mg / AUC_0_inf
    else:
        CL_F = 0.0

    if CL_F > 0 and lambda_z > 0:
        Vz_F = CL_F / lambda_z
    else:
        Vz_F = 0.0

    # MRT (Mean Residence Time)
    # AUMC = ∫ t * C(t) dt
    aumc = _auc_linear_log_trapezoidal(time, time * concentration)
    if lambda_z > 0 and C_last > 0:
        aumc_inf = aumc + T_last * C_last / lambda_z + C_last / (lambda_z ** 2)
    else:
        aumc_inf = aumc

    MRT = aumc_inf / AUC_0_inf if AUC_0_inf > 0 else 0.0

    # Vss (for IV only): Vss = CL * MRT
    Vss = CL_F * MRT if is_iv and CL_F > 0 else None

    # --- NCA reliability metrics (FDA/EMA criteria) ---
    flags: list[str] = []
    extrapolation_fraction = 0.0
    duration_per_t_half = 0.0
    if AUC_0_inf > 0:
        extrapolation_fraction = max(0.0, (AUC_0_inf - AUC_0_t) / AUC_0_inf)
    if t_half > 0 and t_half != float("inf"):
        sim_duration = float(time[-1] - time[0])
        duration_per_t_half = sim_duration / t_half
    if extrapolation_fraction > 0.20:
        flags.append(
            f"AUC extrapolation fraction = {extrapolation_fraction:.1%} "
            f"(> 20% FDA/EMA threshold) — terminal phase undersampled, "
            f"AUC_inf and CL/F are unreliable. Increase duration_h."
        )
    if n_terminal < 3:
        flags.append(
            f"Terminal-phase has only {n_terminal} points (< 3 required) — "
            f"λz fit and t½ are unreliable."
        )
    if r2 > 0 and r2 < 0.85:
        flags.append(
            f"Terminal log-linear R² = {r2:.3f} (< 0.85) — terminal phase "
            f"may not be truly mono-exponential or contains noise."
        )
    if 0 < duration_per_t_half < 3.0 and t_half != float("inf"):
        flags.append(
            f"Simulation duration covers only {duration_per_t_half:.1f}× t½ "
            f"(< 3× recommended). t½ extrapolated from short tail; use "
            f"`duration_h >= {3 * t_half:.0f}` for reliable estimate."
        )

    return PKParameters(
        Cmax=Cmax,
        Tmax=Tmax,
        AUC_0_t=AUC_0_t,
        AUC_0_inf=AUC_0_inf,
        t_half=t_half,
        lambda_z=lambda_z,
        CL_F=CL_F,
        Vz_F=Vz_F,
        Vss=Vss,
        MRT=MRT,
        extrapolation_fraction=extrapolation_fraction,
        n_terminal_points=n_terminal,
        terminal_r2=r2,
        duration_per_t_half=duration_per_t_half,
        reliability_flags=flags,
        C_last=C_last,
        T_last=T_last,
    )
