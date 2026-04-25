"""
Concentration-dependent protein binding models.

For drugs where C_total approaches or exceeds binding site concentration,
fu_p changes with concentration. This affects Kp, clearance, and PK.

Models:
  1. Langmuir single-site binding
  2. Two-site binding (albumin site I/II, or albumin + AAG)
  3. Competitive displacement (DDI at binding level)

Key drugs: valproic acid, phenytoin, warfarin (high dose), diazepam

References:
  - Bohnert T, Gan LS. J Pharm Sci 2013;102:2953-2994
  - Berezhkovskiy LM. J Pharm Sci 2018;107:1079-1085
  - Benet LZ, Bowman CM. J Pharm Sci 2024;113:1-5
  - Oie S, Tozer TN. J Pharm Sci 1979;68:1203-1205
"""

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np


# Typical plasma protein concentrations
ALBUMIN_CONC_UM = 630.0     # 42 g/L / 66,472 Da = 632 uM
AAG_CONC_UM = 20.0          # ~0.8 g/L, MW 42 kDa → ~19 uM


@dataclass
class BindingSite:
    """Single protein binding site."""
    Bmax_uM: float    # Maximum binding capacity (uM)
    Kd_uM: float      # Dissociation constant (uM)
    protein: str = "albumin"  # albumin, AAG, or custom


@dataclass
class BindingModel:
    """Protein binding model specification."""
    sites: list[BindingSite]

    def fu_at_concentration(self, C_total_uM: float) -> float:
        """
        Calculate fu at given total drug concentration.

        Solves: C_total = C_u + Σ(Bmax_i * C_u / (Kd_i + C_u))

        Uses Newton-Raphson iteration.
        """
        if C_total_uM <= 0:
            return 1.0

        # Initial guess: C_u = C_total * fu_at_low_conc
        fu_low = self.fu_at_low_concentration()
        Cu = C_total_uM * fu_low

        for _ in range(50):
            # f(Cu) = Cu + Σ(Bmax * Cu / (Kd + Cu)) - Ctotal = 0
            f_val = Cu - C_total_uM
            df_val = 1.0
            for site in self.sites:
                denom = site.Kd_uM + Cu
                f_val += site.Bmax_uM * Cu / denom
                df_val += site.Bmax_uM * site.Kd_uM / (denom * denom)

            if abs(f_val) < 1e-12:
                break
            Cu -= f_val / df_val
            Cu = max(Cu, 1e-15)

        fu = Cu / C_total_uM if C_total_uM > 0 else 1.0
        return max(min(fu, 1.0), 1e-6)

    def fu_at_low_concentration(self) -> float:
        """fu at infinitely low concentration (linear binding)."""
        # At C→0: fu = 1 / (1 + Σ(Bmax_i / Kd_i))
        sum_ratio = sum(s.Bmax_uM / s.Kd_uM for s in self.sites)
        return 1.0 / (1.0 + sum_ratio)

    def fu_profile(self, C_range_uM: np.ndarray) -> np.ndarray:
        """Compute fu across a concentration range."""
        return np.array([self.fu_at_concentration(c) for c in C_range_uM])


def single_site_albumin(fu_p: float) -> BindingModel:
    """
    Create single-site albumin binding model from measured fu_p.

    Derives Kd from fu_p assuming binding at low concentration:
      fu_p = 1 / (1 + Bmax/Kd)  →  Kd = Bmax * fu_p / (1 - fu_p)
    """
    Bmax = ALBUMIN_CONC_UM
    if fu_p >= 1.0:
        return BindingModel(sites=[BindingSite(Bmax, 1e6)])
    Kd = Bmax * fu_p / (1.0 - fu_p)
    return BindingModel(sites=[BindingSite(Bmax, Kd, "albumin")])


def two_site_albumin_aag(
    fu_p: float,
    fraction_albumin: float = 0.7,
) -> BindingModel:
    """
    Two-site model: albumin + AAG binding.

    Splits total binding between albumin (fraction) and AAG (1 - fraction).
    Derives Kd for each from fu_p.
    """
    if fu_p >= 1.0:
        return BindingModel(sites=[])

    total_binding = 1.0 / fu_p - 1.0

    # Albumin contribution
    alb_binding = total_binding * fraction_albumin
    Kd_alb = ALBUMIN_CONC_UM / alb_binding if alb_binding > 0 else 1e6

    # AAG contribution
    aag_binding = total_binding * (1.0 - fraction_albumin)
    Kd_aag = AAG_CONC_UM / aag_binding if aag_binding > 0 else 1e6

    return BindingModel(sites=[
        BindingSite(ALBUMIN_CONC_UM, Kd_alb, "albumin"),
        BindingSite(AAG_CONC_UM, Kd_aag, "AAG"),
    ])


def displacement_fu(
    binding_model: BindingModel,
    C_substrate_uM: float,
    C_displacer_uM: float,
    Kd_displacer_uM: float,
    displaced_site_index: int = 0,
) -> float:
    """
    Calculate fu of substrate in presence of a competing displacer.

    Modifies the effective Bmax of the displaced site:
      Bmax_eff = Bmax / (1 + C_displacer_u / Kd_displacer)

    Then solves for Cu of substrate with reduced binding capacity.
    """
    modified_sites = []
    for i, site in enumerate(binding_model.sites):
        if i == displaced_site_index:
            reduction = 1.0 + C_displacer_uM / Kd_displacer_uM
            modified_sites.append(BindingSite(
                site.Bmax_uM / reduction, site.Kd_uM, site.protein
            ))
        else:
            modified_sites.append(site)

    modified_model = BindingModel(sites=modified_sites)
    return modified_model.fu_at_concentration(C_substrate_uM)


def format_binding_profile(
    binding_model: BindingModel,
    compound_name: str = "",
    mw: float = 300.0,
) -> str:
    """Format binding model as markdown."""
    lines = [f"## Concentration-Dependent Binding"]
    if compound_name:
        lines[0] += f" — {compound_name}"

    fu_low = binding_model.fu_at_low_concentration()
    lines.extend([
        f"\nfu at low concentration: {fu_low:.4f}\n",
        "### Binding Sites\n",
        "| Site | Protein | Bmax (uM) | Kd (uM) |",
        "|------|---------|-----------|---------|",
    ])
    for i, s in enumerate(binding_model.sites):
        lines.append(f"| {i+1} | {s.protein} | {s.Bmax_uM:.1f} | {s.Kd_uM:.1f} |")

    # Profile at selected concentrations
    concs_uM = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    concs_mg_L = [c * mw / 1e6 * 1e3 for c in concs_uM]  # rough
    lines.extend([
        "\n### fu vs Concentration\n",
        "| C_total (uM) | fu |",
        "|-------------|-----|",
    ])
    for c in concs_uM:
        fu = binding_model.fu_at_concentration(c)
        lines.append(f"| {c:.1f} | {fu:.4f} |")

    return "\n".join(lines)
