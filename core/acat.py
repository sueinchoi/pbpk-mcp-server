"""
ACAT (Advanced Compartmental Absorption and Transit) model.

9-segment GI tract model with:
  - Segment-specific pH, transit time, surface area, blood flow
  - Noyes-Whitney dissolution with pH-dependent solubility
  - Permeability-limited absorption (Peff × ASF)
  - Segment-specific gut wall metabolism (CYP3A4)
  - Precipitation/supersaturation
  - Carrier-mediated efflux (optional P-gp)

Segments:
  [0] Stomach
  [1] Duodenum
  [2] Jejunum 1
  [3] Jejunum 2
  [4] Ileum 1
  [5] Ileum 2
  [6] Ileum 3
  [7] Caecum
  [8] Ascending Colon

Per segment, 3 states:
  - Undissolved drug in lumen (solid)
  - Dissolved drug in lumen (solution)
  - Drug in enterocyte (absorbed, pre-portal vein)

Total ACAT states: 9 × 3 = 27

References:
  - Agoram B et al. J Controlled Release 2001;71:109-126
  - Yu LX, Amidon GL. Int J Pharm 1999;186:119-125
  - Noyes A, Whitney W. J Am Chem Soc 1897;19:930-934
  - Sugano K. Biopharmaceutics Modeling and Simulations. Wiley, 2012
  - Jamei M et al. AAPS J 2009;11:225-237 (Simcyp ADAM)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ===================================================================
# GI Segment definitions
# ===================================================================

class GISegment(int, Enum):
    STOMACH = 0
    DUODENUM = 1
    JEJUNUM_1 = 2
    JEJUNUM_2 = 3
    ILEUM_1 = 4
    ILEUM_2 = 5
    ILEUM_3 = 6
    CAECUM = 7
    ASC_COLON = 8


N_SEGMENTS = 9
N_STATES_PER_SEG = 3  # undissolved, dissolved, enterocyte
N_ACAT_STATES = N_SEGMENTS * N_STATES_PER_SEG  # 27

# Sub-state offsets within each segment
UNDISSOLVED = 0
DISSOLVED = 1
ENTEROCYTE = 2


# ===================================================================
# Segment physiological data (Simcyp ADAM / GastroPlus defaults)
# ===================================================================

SEGMENT_DATA = {
    # pH, length(cm), radius(cm), transit(h), SA_factor, blood_flow(L/h)
    # Transit times scaled so SI total = 3.32 h (Yu & Amidon 1999, Davis 1986)
    # Duodenum ~11%, Jejunum ~40%, Ileum ~49% of SITT
    GISegment.STOMACH:    (1.7,   20.0,      5.0,      0.25,      0.0,       0.5),
    GISegment.DUODENUM:   (6.0,   25.0,      1.75,     0.37,      1.0,       1.8),
    GISegment.JEJUNUM_1:  (6.2,   90.0,      1.75,     0.66,      1.0,       4.5),
    GISegment.JEJUNUM_2:  (6.4,   90.0,      1.50,     0.66,      0.67,      3.8),
    GISegment.ILEUM_1:    (6.6,   90.0,      1.50,     0.66,      0.45,      3.0),
    GISegment.ILEUM_2:    (6.9,   60.0,      1.25,     0.49,      0.30,      2.2),
    GISegment.ILEUM_3:    (7.4,   60.0,      1.25,     0.49,      0.20,      1.7),
    GISegment.CAECUM:     (6.4,   13.0,      3.25,     4.0,       0.01,      0.3),
    GISegment.ASC_COLON:  (6.8,   20.0,      2.50,     9.0,       0.005,     0.2),
}
# SI transit sum: 0.37+0.66+0.66+0.66+0.49+0.49 = 3.33 h ≈ SITT 3.32 h

# Absorption Scale Factors (GastroPlus convention)
# Relative Peff scaling per segment
ASF_FACTORS = {
    GISegment.STOMACH:    0.0,    # no absorption from stomach
    GISegment.DUODENUM:   1.0,
    GISegment.JEJUNUM_1:  1.0,
    GISegment.JEJUNUM_2:  0.67,
    GISegment.ILEUM_1:    0.45,
    GISegment.ILEUM_2:    0.30,
    GISegment.ILEUM_3:    0.20,
    GISegment.CAECUM:     0.01,
    GISegment.ASC_COLON:  0.005,
}

# Paracellular pore radii per segment (Angstrom)
# Adson 1994/1995; Sugano 2002; Avdeef 2010
PORE_RADIUS_A = {
    GISegment.STOMACH:    0.0,    # no paracellular
    GISegment.DUODENUM:   9.0,
    GISegment.JEJUNUM_1:  8.0,
    GISegment.JEJUNUM_2:  7.0,
    GISegment.ILEUM_1:    5.5,
    GISegment.ILEUM_2:    4.5,
    GISegment.ILEUM_3:    4.0,
    GISegment.CAECUM:     8.0,    # leak pathway
    GISegment.ASC_COLON:  8.0,
}

# Porosity/path length ratio (cm^-1) per segment
# Sugano 2002; Avdeef & Tam 2010
POROSITY_PATH = {
    GISegment.STOMACH:    0.0,
    GISegment.DUODENUM:   8.0,
    GISegment.JEJUNUM_1:  7.0,
    GISegment.JEJUNUM_2:  5.5,
    GISegment.ILEUM_1:    4.0,
    GISegment.ILEUM_2:    3.0,
    GISegment.ILEUM_3:    2.0,
    GISegment.CAECUM:     1.0,
    GISegment.ASC_COLON:  0.5,
}

# CYP3A4 relative expression per segment (fraction of total gut CYP3A4)
# Paine et al. 2006; highest in jejunum, decreasing distally
# CYP regional expression fractions per segment (Paine 1997, 2006)
# Each dict sums to ~1.0. Different CYPs have different gradients.
GUT_CYP_EXPRESSION = {
    "CYP3A4": {
        GISegment.STOMACH: 0.0, GISegment.DUODENUM: 0.15,
        GISegment.JEJUNUM_1: 0.30, GISegment.JEJUNUM_2: 0.25,
        GISegment.ILEUM_1: 0.15, GISegment.ILEUM_2: 0.08,
        GISegment.ILEUM_3: 0.05, GISegment.CAECUM: 0.01, GISegment.ASC_COLON: 0.01,
    },
    "CYP2C9": {
        GISegment.STOMACH: 0.0, GISegment.DUODENUM: 0.20,
        GISegment.JEJUNUM_1: 0.25, GISegment.JEJUNUM_2: 0.25,
        GISegment.ILEUM_1: 0.15, GISegment.ILEUM_2: 0.08,
        GISegment.ILEUM_3: 0.05, GISegment.CAECUM: 0.01, GISegment.ASC_COLON: 0.01,
    },
    "CYP2C19": {
        GISegment.STOMACH: 0.0, GISegment.DUODENUM: 0.25,
        GISegment.JEJUNUM_1: 0.25, GISegment.JEJUNUM_2: 0.20,
        GISegment.ILEUM_1: 0.15, GISegment.ILEUM_2: 0.08,
        GISegment.ILEUM_3: 0.05, GISegment.CAECUM: 0.01, GISegment.ASC_COLON: 0.01,
    },
    "CYP2D6": {
        GISegment.STOMACH: 0.0, GISegment.DUODENUM: 0.20,
        GISegment.JEJUNUM_1: 0.20, GISegment.JEJUNUM_2: 0.20,
        GISegment.ILEUM_1: 0.15, GISegment.ILEUM_2: 0.10,
        GISegment.ILEUM_3: 0.10, GISegment.CAECUM: 0.03, GISegment.ASC_COLON: 0.02,
    },
    "UGT": {  # UGT1A1/1A8/1A10 combined (Strassburg 2000)
        GISegment.STOMACH: 0.0, GISegment.DUODENUM: 0.10,
        GISegment.JEJUNUM_1: 0.15, GISegment.JEJUNUM_2: 0.15,
        GISegment.ILEUM_1: 0.15, GISegment.ILEUM_2: 0.15,
        GISegment.ILEUM_3: 0.15, GISegment.CAECUM: 0.10, GISegment.ASC_COLON: 0.05,
    },
}
# Backward compatibility alias
CYP3A4_EXPRESSION = GUT_CYP_EXPRESSION["CYP3A4"]

# Fluid volumes per segment (mL, fasted state)
# Fluid volumes per segment (mL, fasted state)
# Total SI ~105 mL (Schiller 2005, Mudie 2010)
FLUID_VOLUMES = {
    GISegment.STOMACH:    50.0,
    GISegment.DUODENUM:   15.0,
    GISegment.JEJUNUM_1:  25.0,
    GISegment.JEJUNUM_2:  20.0,
    GISegment.ILEUM_1:    18.0,
    GISegment.ILEUM_2:    15.0,
    GISegment.ILEUM_3:    12.0,
    GISegment.CAECUM:     13.0,
    GISegment.ASC_COLON:  13.0,
}

# Bile salt concentration per segment (mM, fasted)
BILE_SALTS = {
    GISegment.STOMACH:    0.0,
    GISegment.DUODENUM:   4.0,
    GISegment.JEJUNUM_1:  4.0,
    GISegment.JEJUNUM_2:  3.0,
    GISegment.ILEUM_1:    2.0,
    GISegment.ILEUM_2:    1.5,
    GISegment.ILEUM_3:    1.0,
    GISegment.CAECUM:     0.5,
    GISegment.ASC_COLON:  0.3,
}


# ===================================================================
# Drug formulation / dissolution parameters
# ===================================================================

@dataclass
class FormulationSpec:
    """Drug formulation and dissolution properties."""
    # Intrinsic solubility (neutral form, mg/mL)
    S0: float = 1.0

    # Particle properties
    particle_radius_um: float = 25.0   # mean particle radius (um)
    particle_density: float = 1.2      # g/cm^3

    # Dissolution layer thickness (um)
    diffusion_layer_um: float = 30.0

    # Precipitation
    precipitation_time_h: float = 1e6  # time to nucleation (h); very large = no precipitation
    supersaturation_ratio: float = 1.0 # max fold supersaturation before precipitation

    # P-gp efflux (optional)
    pgp_efflux_ratio: float = 1.0      # ER from Caco-2 B-A/A-B; 1.0 = no efflux

    @property
    def particle_radius_cm(self) -> float:
        return self.particle_radius_um * 1e-4

    @property
    def diffusion_layer_cm(self) -> float:
        return self.diffusion_layer_um * 1e-4


def _hydrodynamic_radius_A(mw: float) -> float:
    """Hydrodynamic radius (Angstrom) from MW. Sugano 2002."""
    return 0.485 * mw ** (1.0 / 3.0)


def _renkin_sieving(solute_radius_A: float, pore_radius_A: float) -> float:
    """
    Renkin molecular sieving function F(λ) for size-restricted diffusion.
    λ = a/R (solute/pore radius ratio).
    F(λ) = (1-λ)^2 * [1 - 2.104λ + 2.089λ^3 - 0.948λ^5]
    Renkin EM. J Gen Physiol 1954;38:225-243.
    """
    if pore_radius_A <= 0 or solute_radius_A >= pore_radius_A:
        return 0.0
    lam = solute_radius_A / pore_radius_A
    F = (1.0 - lam) ** 2 * (1.0 - 2.104 * lam + 2.089 * lam ** 3 - 0.948 * lam ** 5)
    return max(F, 0.0)


def _paracellular_peff(
    mw: float,
    D_aq: float,
    seg: 'GISegment',
) -> float:
    """
    Paracellular permeability (cm/s) for a given GI segment.
    P_para = (epsilon/delta) * D_aq * F(a/R)
    Adson 1994/1995; Sugano 2002.
    """
    pore_R = PORE_RADIUS_A.get(seg, 0.0)
    eps_delta = POROSITY_PATH.get(seg, 0.0)
    if pore_R <= 0 or eps_delta <= 0:
        return 0.0
    a = _hydrodynamic_radius_A(mw)
    F = _renkin_sieving(a, pore_R)
    return eps_delta * D_aq * F  # cm/s


# ===================================================================
# Pre-computed segment parameters
# ===================================================================

@dataclass
class SegmentParams:
    """Pre-computed parameters for one GI segment."""
    segment: GISegment
    pH: float
    length_cm: float
    radius_cm: float
    transit_time_h: float
    SA_cm2: float         # cylindrical surface area
    k_transit: float      # transit rate constant (1/h)
    k_abs: float          # absorption rate constant (1/h)
    V_fluid_mL: float     # luminal fluid volume (mL)
    solubility_mg_mL: float  # pH-corrected solubility
    k_dissolution: float  # dissolution rate coefficient (1/h)
    CL_gut_segment: float # gut wall CLint for this segment (L/h)
    Q_blood: float        # villous blood flow for this segment (L/h)


def _solubility_at_pH(S0: float, pKa: float, pH: float, compound_type: str) -> float:
    """
    Henderson-Hasselbalch pH-dependent solubility.
    S(pH) = S0 * (1 + ionization_term)
    """
    if compound_type in ("strong_base", "moderate_base", "weak_base"):
        return S0 * (1.0 + 10.0 ** (pKa - pH))
    elif compound_type == "acid":
        return S0 * (1.0 + 10.0 ** (pH - pKa))
    return S0  # neutral


def _diffusion_coefficient(mw: float) -> float:
    """Aqueous diffusion coefficient (cm^2/s). Hayduk-Laudie approximation."""
    return 9.9e-6 * mw ** (-0.453)


def _P_ow_from_mw(mw: float) -> float:
    """Very rough logP estimate from MW when logP unavailable. NOT for production use."""
    return max(10.0 ** (0.005 * mw - 1.0), 0.1)


def build_segment_params(
    Peff_e4: float,
    mw: float,
    pKa: float,
    compound_type: str,
    S0: float,
    formulation: FormulationSpec,
    CLint_gut_total: float = 0.0,
    fu_gut: float = 1.0,
    logP: float = 2.0,
    gut_cyp_clint: Optional[dict[str, float]] = None,
) -> list[SegmentParams]:
    """
    Build pre-computed parameters for all 9 ACAT segments.

    Args:
        Peff_e4: Reference human jejunal Peff (10^-4 cm/s).
        mw: Molecular weight (g/mol).
        pKa: Dissociation constant.
        compound_type: Acid/base/neutral classification string.
        S0: Intrinsic solubility (mg/mL).
        formulation: FormulationSpec with particle and dissolution data.
        CLint_gut_total: Total gut wall intrinsic clearance (L/h).
            Used when gut_cyp_clint is None (all assigned to CYP3A4).
        fu_gut: Fraction unbound in enterocytes.
        gut_cyp_clint: Per-CYP gut CLint dict, e.g.:
            {"CYP3A4": 30.0, "CYP2C9": 5.0, "UGT": 10.0}
            Each value is total gut CLint (L/h) for that enzyme.
            If provided, CLint_gut_total is ignored.

    Returns:
        List of 9 SegmentParams.
    """
    D_aq = _diffusion_coefficient(mw)
    r0 = formulation.particle_radius_cm
    h = formulation.diffusion_layer_cm
    rho = formulation.particle_density

    segments = []
    for seg in GISegment:
        pH, length, radius, t_transit, asf, Q_blood = SEGMENT_DATA[seg]
        V_fluid = FLUID_VOLUMES[seg]

        # Surface area (cylindrical, no villus amplification — Peff accounts for it)
        SA = 2.0 * math.pi * radius * length

        # Transit rate
        k_transit = 1.0 / t_transit if t_transit > 0 else 0.0

        # Absorption rate: ka = 2 * (Peff_trans + Peff_para) / radius
        # Transcellular: Peff * ASF (10^-4 cm/s → cm/s)
        Peff_trans = Peff_e4 * 1e-4 * asf

        # Paracellular: Renkin pore model (Adson 1994, Sugano 2002)
        Peff_para = _paracellular_peff(mw, D_aq, seg)

        Peff_total = Peff_trans + Peff_para
        k_abs = 2.0 * Peff_total / radius * 3600.0 if radius > 0 else 0.0

        # P-gp efflux correction (only transcellular component)
        if formulation.pgp_efflux_ratio > 1.0:
            k_abs_trans = 2.0 * Peff_trans / radius * 3600.0 if radius > 0 else 0.0
            k_abs_para = 2.0 * Peff_para / radius * 3600.0 if radius > 0 else 0.0
            k_abs = k_abs_trans / formulation.pgp_efflux_ratio + k_abs_para

        # pH-dependent solubility + bile salt micellar enhancement
        sol = _solubility_at_pH(S0, pKa, pH, compound_type)
        # Micellar solubilization: lipophilic drugs (logP > 2) benefit from bile salts
        # S_total = S_aq + BS * SR, where SR = solubilization ratio (mg/mL per mM BS)
        # Fagerberg JH et al. Mol Pharmaceutics 2015;12:2523: log(SR) = 0.72*logP - 2.76
        bs_conc = BILE_SALTS.get(seg, 0.0)  # mM
        if bs_conc > 0 and logP > 2.0:
            SR = 10.0 ** (0.72 * logP - 2.76)  # mg/mL per mM bile salt
            sol += bs_conc * SR

        # Dissolution rate coefficient (Noyes-Whitney, spherical particles)
        # dM/dt = -3*D*Cs/(rho*h*r) * M * (1-C/Cs)
        # Units: D(cm²/s), Cs(mg/mL=mg/cm³), rho(g/cm³=1000 mg/cm³), h(cm), r(cm)
        # k_diss = 3*D*Cs / (rho_mg * h * r) * 3600 [1/h]
        # Divide by 1000 to convert rho from g/cm³ to mg/cm³
        if r0 > 0 and h > 0:
            k_diss = 3.0 * D_aq * sol / (rho * 1000.0 * h * r0) * 3600.0
        else:
            k_diss = 1e6  # instant dissolution

        # Gut wall metabolism for this segment
        # Gut wall metabolism: sum contributions from each CYP enzyme
        # Each CYP has its own regional expression gradient
        if gut_cyp_clint:
            CL_gut_seg = 0.0
            for cyp_name, cyp_clint in gut_cyp_clint.items():
                expr = GUT_CYP_EXPRESSION.get(cyp_name, GUT_CYP_EXPRESSION["CYP3A4"])
                CL_gut_seg += cyp_clint * expr.get(seg, 0.0) * fu_gut
        else:
            # Default: all CLint assigned to CYP3A4 expression pattern
            CL_gut_seg = CLint_gut_total * CYP3A4_EXPRESSION.get(seg, 0.0) * fu_gut

        segments.append(SegmentParams(
            segment=seg,
            pH=pH,
            length_cm=length,
            radius_cm=radius,
            transit_time_h=t_transit,
            SA_cm2=SA,
            k_transit=k_transit,
            k_abs=k_abs,
            V_fluid_mL=V_fluid,
            solubility_mg_mL=sol,
            k_dissolution=k_diss,
            CL_gut_segment=CL_gut_seg,
            Q_blood=Q_blood,
        ))

    return segments


# ===================================================================
# ACAT ODE right-hand side
# ===================================================================

def acat_rhs(
    y_acat: np.ndarray,
    seg_params: list[SegmentParams],
    formulation: FormulationSpec,
    dose_mg: float,
) -> tuple[np.ndarray, float]:
    """
    Compute ACAT ODE derivatives for the 27 GI states.

    State layout per segment i (3 states each):
      y[i*3 + 0] = undissolved drug in lumen (mg)
      y[i*3 + 1] = dissolved drug in lumen (mg)
      y[i*3 + 2] = drug in enterocyte (mg)

    Args:
        y_acat: State vector (27 elements).
        seg_params: Pre-computed segment parameters.
        formulation: Formulation specification.
        dose_mg: Total dose (for mass balance checks).

    Returns:
        (dy_acat, total_absorbed_rate): derivatives and the rate of drug
        entering the portal vein (mg/h) from all segments.
    """
    dy = np.zeros_like(y_acat)
    total_to_portal = 0.0

    for i, sp in enumerate(seg_params):
        idx_u = i * 3 + UNDISSOLVED
        idx_d = i * 3 + DISSOLVED
        idx_e = i * 3 + ENTEROCYTE

        M_undissolved = max(y_acat[idx_u], 0.0)
        M_dissolved = max(y_acat[idx_d], 0.0)
        M_enterocyte = max(y_acat[idx_e], 0.0)

        V_mL = sp.V_fluid_mL
        C_dissolved = M_dissolved / V_mL if V_mL > 0 else 0.0  # mg/mL

        # --- Dissolution (Noyes-Whitney) ---
        # Rate = k_diss * M_undissolved * (1 - C/Cs)
        # Only dissolve when below saturation
        Cs = sp.solubility_mg_mL
        if Cs > 0 and C_dissolved < Cs:
            dissolution_rate = sp.k_dissolution * M_undissolved * (1.0 - C_dissolved / Cs)
        else:
            dissolution_rate = 0.0
        dissolution_rate = max(dissolution_rate, 0.0)

        # --- Precipitation ---
        # If supersaturated beyond threshold, drug precipitates
        precip_rate = 0.0
        if Cs > 0 and C_dissolved > Cs * formulation.supersaturation_ratio:
            # First-order precipitation
            k_precip = 1.0 / formulation.precipitation_time_h
            excess = M_dissolved - Cs * formulation.supersaturation_ratio * V_mL
            if excess > 0:
                precip_rate = k_precip * excess

        # --- Transit (input from previous segment, output to next) ---
        # Transit in (from previous segment)
        transit_in_u = 0.0
        transit_in_d = 0.0
        if i > 0:
            prev_sp = seg_params[i - 1]
            prev_u = max(y_acat[(i - 1) * 3 + UNDISSOLVED], 0.0)
            prev_d = max(y_acat[(i - 1) * 3 + DISSOLVED], 0.0)
            transit_in_u = prev_sp.k_transit * prev_u
            transit_in_d = prev_sp.k_transit * prev_d

        # Transit out
        transit_out_u = sp.k_transit * M_undissolved
        transit_out_d = sp.k_transit * M_dissolved

        # --- Absorption (dissolved → enterocyte) ---
        absorption_rate = sp.k_abs * M_dissolved

        # --- Gut wall metabolism (enterocyte → eliminated) ---
        metabolism_rate = sp.CL_gut_segment * M_enterocyte / (V_mL / 1000.0) if V_mL > 0 else 0.0
        # CL_gut_segment is in L/h; M_enterocyte/V is concentration; rate = CL * C

        # --- Transfer to portal vein (enterocyte → systemic) ---
        # Rate proportional to blood flow: enterocyte drains to portal vein
        # Using well-stirred assumption within each segment's enterocyte
        Q_seg = sp.Q_blood
        C_enterocyte = M_enterocyte / (V_mL / 1000.0) if V_mL > 0 else 0.0
        to_portal = Q_seg * C_enterocyte / 1.0  # simplified: Kp_enterocyte ~ 1
        # Cap at available amount
        to_portal = min(to_portal, M_enterocyte / 0.001) if M_enterocyte > 0 else 0.0

        total_to_portal += to_portal

        # --- Derivatives ---
        # Undissolved: +transit_in -transit_out -dissolution +precipitation
        dy[idx_u] = transit_in_u - transit_out_u - dissolution_rate + precip_rate

        # Dissolved: +transit_in -transit_out +dissolution -precipitation -absorption
        dy[idx_d] = transit_in_d - transit_out_d + dissolution_rate - precip_rate - absorption_rate

        # Enterocyte: +absorption -metabolism -to_portal
        dy[idx_e] = absorption_rate - metabolism_rate - to_portal

    return dy, total_to_portal


# ===================================================================
# Standalone ACAT simulation (for diagnostics / Fa prediction)
# ===================================================================

@dataclass
class ACATResult:
    """Result of standalone ACAT simulation."""
    time: np.ndarray
    Fa_cumulative: np.ndarray       # cumulative fraction absorbed vs time
    Fg_cumulative: np.ndarray       # cumulative fraction escaping gut wall
    Fa_final: float                 # total Fa at end
    Fg_final: float                 # total Fg at end
    Fa_by_segment: dict             # {segment: fraction absorbed}
    dissolution_profile: np.ndarray # fraction dissolved vs time
    portal_rate: np.ndarray         # rate entering portal vein (mg/h)


def simulate_acat_standalone(
    dose_mg: float,
    Peff_e4: float,
    mw: float,
    pKa: float,
    compound_type: str,
    S0: float,
    formulation: Optional[FormulationSpec] = None,
    CLint_gut_total: float = 0.0,
    fu_gut: float = 1.0,
    logP: float = 2.0,
    duration_h: float = 24.0,
    n_points: int = 500,
) -> ACATResult:
    """
    Run a standalone ACAT simulation (GI tract only, no systemic).

    Useful for predicting Fa, Fg, dissolution profiles, and
    regional absorption before running full PBPK.

    Args:
        dose_mg: Oral dose (mg).
        Peff_e4: Human jejunal Peff (10^-4 cm/s).
        mw: Molecular weight (g/mol).
        pKa: pKa.
        compound_type: "acid", "base", "neutral", etc.
        S0: Intrinsic solubility (mg/mL).
        formulation: FormulationSpec (default: standard IR tablet).
        CLint_gut_total: Total gut wall CLint (L/h).
        fu_gut: Enterocyte unbound fraction.
        duration_h: Simulation duration.
        n_points: Output time points.

    Returns:
        ACATResult with Fa, Fg, dissolution, and absorption profiles.
    """
    from scipy.integrate import solve_ivp

    if formulation is None:
        formulation = FormulationSpec(S0=S0)

    seg_params = build_segment_params(
        Peff_e4, mw, pKa, compound_type, S0,
        formulation, CLint_gut_total, fu_gut, logP,
    )

    # Initial conditions: all drug as undissolved solid in stomach
    y0 = np.zeros(N_ACAT_STATES)
    y0[GISegment.STOMACH * 3 + UNDISSOLVED] = dose_mg

    # Track cumulative absorbed and metabolized
    absorbed_total = [0.0]
    metabolized_total = [0.0]

    def rhs(t, y):
        dy, portal_rate = acat_rhs(y, seg_params, formulation, dose_mg)
        return dy

    t_eval = np.linspace(0, duration_h, n_points)
    sol = solve_ivp(
        rhs, [0, duration_h], y0,
        method="BDF", t_eval=t_eval,
        rtol=1e-8, atol=1e-10, max_step=0.05,
    )

    if not sol.success:
        raise RuntimeError(f"ACAT solver failed: {sol.message}")

    time = sol.t
    y_full = sol.y  # (27, n_points)

    # Compute profiles
    n_t = len(time)
    dissolved_profile = np.zeros(n_t)
    absorbed_profile = np.zeros(n_t)
    enterocyte_total = np.zeros(n_t)
    portal_rates = np.zeros(n_t)
    fa_by_seg = {seg: 0.0 for seg in GISegment}

    for k in range(n_t):
        y_k = y_full[:, k]
        total_undissolved = sum(max(y_k[i * 3 + UNDISSOLVED], 0) for i in range(N_SEGMENTS))
        total_dissolved = sum(max(y_k[i * 3 + DISSOLVED], 0) for i in range(N_SEGMENTS))
        total_enterocyte = sum(max(y_k[i * 3 + ENTEROCYTE], 0) for i in range(N_SEGMENTS))
        dissolved_profile[k] = (dose_mg - total_undissolved) / dose_mg if dose_mg > 0 else 0
        enterocyte_total[k] = total_enterocyte
        # Cumulative absorbed = dose - remaining_in_lumen
        remaining = total_undissolved + total_dissolved
        absorbed_profile[k] = (dose_mg - remaining - total_enterocyte) / dose_mg

        _, pr = acat_rhs(y_k, seg_params, formulation, dose_mg)
        portal_rates[k] = pr

    # Fa and Fg at final time
    y_end = y_full[:, -1]
    total_remaining_lumen = sum(
        max(y_end[i * 3 + UNDISSOLVED], 0) + max(y_end[i * 3 + DISSOLVED], 0)
        for i in range(N_SEGMENTS)
    )
    Fa_final = 1.0 - total_remaining_lumen / dose_mg if dose_mg > 0 else 1.0
    Fa_final = max(min(Fa_final, 1.0), 0.0)

    # Fg: fraction of absorbed drug that escaped gut wall metabolism
    total_in_enterocyte = sum(max(y_end[i * 3 + ENTEROCYTE], 0) for i in range(N_SEGMENTS))
    drug_past_gut = dose_mg - total_remaining_lumen - total_in_enterocyte
    Fg_final = drug_past_gut / (Fa_final * dose_mg) if Fa_final * dose_mg > 0 else 1.0
    Fg_final = max(min(Fg_final, 1.0), 0.0)

    # Per-segment Fa
    for seg in GISegment:
        i = seg.value
        seg_remaining = max(y_end[i * 3 + UNDISSOLVED], 0) + max(y_end[i * 3 + DISSOLVED], 0)
        seg_enterocyte = max(y_end[i * 3 + ENTEROCYTE], 0)
        # Absorbed from this segment ≈ what left it
        if i == 0:
            seg_input = dose_mg
        else:
            seg_input = 0  # complex to track; use enterocyte as proxy
        fa_by_seg[seg] = max(y_end[i * 3 + ENTEROCYTE], 0) / dose_mg if dose_mg > 0 else 0

    return ACATResult(
        time=time,
        Fa_cumulative=np.clip(1.0 - np.array([
            sum(max(y_full[i * 3 + UNDISSOLVED, k], 0) + max(y_full[i * 3 + DISSOLVED, k], 0)
                for i in range(N_SEGMENTS)) / dose_mg
            for k in range(n_t)
        ]), 0, 1),
        Fg_cumulative=np.ones(n_t) * Fg_final,  # simplified
        Fa_final=Fa_final,
        Fg_final=Fg_final,
        Fa_by_segment=fa_by_seg,
        dissolution_profile=dissolved_profile,
        portal_rate=portal_rates,
    )


def format_acat_result(result: ACATResult, compound_name: str = "") -> str:
    """Format ACAT result as markdown."""
    lines = [f"## ACAT Absorption Prediction"]
    if compound_name:
        lines[0] += f" — {compound_name}"

    lines.extend([
        "",
        f"**Fa = {result.Fa_final:.4f}** (fraction absorbed)",
        f"**Fg = {result.Fg_final:.4f}** (fraction escaping gut wall)",
        f"**Fa × Fg = {result.Fa_final * result.Fg_final:.4f}**",
        "",
        "### Regional Absorption",
        "",
        "| Segment | Fraction in Enterocyte |",
        "|---------|----------------------|",
    ])
    for seg in GISegment:
        lines.append(f"| {seg.name.replace('_', ' ').title()} | {result.Fa_by_segment[seg]:.4f} |")

    lines.extend([
        "",
        f"Time to 50% dissolved: {_time_to_frac(result.time, result.dissolution_profile, 0.5):.2f} h",
        f"Time to 90% dissolved: {_time_to_frac(result.time, result.dissolution_profile, 0.9):.2f} h",
        f"Time to 50% absorbed: {_time_to_frac(result.time, result.Fa_cumulative, 0.5):.2f} h",
        f"Time to 90% absorbed: {_time_to_frac(result.time, result.Fa_cumulative, 0.9):.2f} h",
    ])
    return "\n".join(lines)


def _time_to_frac(time: np.ndarray, profile: np.ndarray, target: float) -> float:
    """Find time at which profile reaches target fraction."""
    idx = np.where(profile >= target)[0]
    if len(idx) > 0:
        return time[idx[0]]
    return float("inf")
