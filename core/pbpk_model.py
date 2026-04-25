"""
Whole-body PBPK ODE model — PK-Sim / Simcyp architecture.

Model structure:
  - 13 tissue compartments + arterial/venous blood pools + gut lumen
  - Perfusion-limited (default): 1 well-stirred compartment per organ
  - Permeability-limited (optional): 3 sub-compartments per organ
    (vascular, interstitial, intracellular) — PK-Sim "Standard" model
  - Portal circulation: gut + spleen → portal vein → liver
  - Dual liver input: hepatic artery + portal vein
  - Lung between venous and arterial blood pools

Circulation diagram:
                    ┌─────────┐
           ┌───────┤  Lung   ├───────┐
           │       └─────────┘       │
     ┌─────┴─────┐             ┌─────┴─────┐
     │  Venous   │             │ Arterial  │
     │  Blood    │             │  Blood    │
     └─────┬─────┘             └─────┬─────┘
           │                         │
    ┌──────┤  ┌──────────────────────┤
    │      │  │    ┌──────┐          │
    │  ┌───┴──┴──┐ │Brain │  ┌──────┴──────┐
    │  │  Liver  │ ├──────┤  │Heart,Kidney, │
    │  │(HA+PV)  │ │Muscle│  │Bone,Skin,   │
    │  └────┬────┘ ├──────┤  │Pancreas,Rest│
    │       │      │ etc. │  └─────────────┘
    │  ┌────┴────┐ └──────┘
    │  │Portal V.│
    │  └────┬────┘
    │  ┌────┴────┐  ┌────────┐
    │  │  Gut    ├──┤ Lumen  │ ← oral dose
    │  ├─────────┤  └────────┘
    │  │ Spleen  │
    │  └─────────┘

State vector (perfusion-limited, 16 states):
  [0] A_lumen     — gut lumen (oral absorption depot)
  [1] A_venous    — venous blood pool
  [2] A_arterial  — arterial blood pool
  [3] A_lung
  [4] A_adipose
  [5] A_bone
  [6] A_brain
  [7] A_gut       — gut wall tissue
  [8] A_heart
  [9] A_kidney
  [10] A_liver
  [11] A_muscle
  [12] A_pancreas
  [13] A_skin
  [14] A_spleen
  [15] A_rest

Units: amount (mg), time (h), volume (L), flow (L/h), concentration (mg/L)
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

from .compound import CompoundSpec, CompoundType, MetabolismModel, AbsorptionModel
from .physiology import (
    Organ,
    PhysiologyParams,
    get_physiology,
    get_subcompartment_volumes,
    VASCULAR_FRACTIONS,
    TISSUE_COMPOSITION,
)
from .partition_coeff import predict_kp_all, predict_kpb_all, KpMethod
from .acat import (
    N_ACAT_STATES, N_SEGMENTS, UNDISSOLVED, DISSOLVED, ENTEROCYTE,
    GISegment, FormulationSpec, build_segment_params, acat_rhs,
)
from .tissue_binding import predict_fu_gut
from .ehc import EHCParams, ehc_rhs, compute_biliary_rate, N_EHC_STATES
from .lymphatic import estimate_lymphatic_fraction, LYMPH_TRANSIT_TIME
from .transporters import OrganTransporters, compute_transport_rate
from .time_varying import circadian_enzyme_factor


class Route(str, Enum):
    IV_BOLUS = "iv_bolus"
    IV_INFUSION = "iv_infusion"
    ORAL = "oral"


class DistributionModel(str, Enum):
    PERFUSION_LIMITED = "perfusion_limited"
    PERMEABILITY_LIMITED = "permeability_limited"


# Index mapping for state vector
IDX = {
    "lumen": 0,
    "venous": 1,
    "arterial": 2,
    "lung": 3,
    "adipose": 4,
    "bone": 5,
    "brain": 6,
    "gut": 7,
    "heart": 8,
    "kidney": 9,
    "liver": 10,
    "muscle": 11,
    "pancreas": 12,
    "skin": 13,
    "spleen": 14,
    "rest": 15,
}

ORGAN_TO_IDX = {
    Organ.ADIPOSE: 4,
    Organ.BONE: 5,
    Organ.BRAIN: 6,
    Organ.GUT: 7,
    Organ.HEART: 8,
    Organ.KIDNEY: 9,
    Organ.LIVER: 10,
    Organ.LUNG: 3,
    Organ.MUSCLE: 11,
    Organ.PANCREAS: 12,
    Organ.SKIN: 13,
    Organ.SPLEEN: 14,
    Organ.REST: 15,
}

# Non-portal organs: return venous blood directly to venous pool
NON_PORTAL_ORGANS = [
    Organ.ADIPOSE, Organ.BONE, Organ.BRAIN, Organ.HEART,
    Organ.KIDNEY, Organ.MUSCLE, Organ.SKIN, Organ.REST,
]

# Portal organs: drain to portal vein → liver (Williams & Leggett 1989)
PORTAL_ORGANS = [Organ.GUT, Organ.SPLEEN, Organ.PANCREAS]

N_STATES_PERFUSION = 16

# Extended states for EHC + lymphatic (appended after index 15)
IDX_EHC_BILE = 16      # bile duct
IDX_EHC_GB = 17        # gallbladder
IDX_EHC_DECONJ = 18    # intestinal deconjugation pool
IDX_LYMPH = 19         # lymphatic compartment
N_STATES_EXTENDED = 20


@dataclass
class DosingProtocol:
    """Dosing regimen specification."""

    dose_mg: float              # Single dose amount (mg)
    route: Route = Route.ORAL
    n_doses: int = 1            # Number of doses
    interval_h: float = 24.0    # Dosing interval (h)
    infusion_duration_h: float = 0.5  # For IV infusion (h)

    def get_dose_times(self) -> list[float]:
        return [i * self.interval_h for i in range(self.n_doses)]


@dataclass
class SimulationConfig:
    """Simulation configuration."""

    duration_h: float = 24.0            # Total simulation time (h)
    n_timepoints: int = 1000            # Number of output time points
    distribution_model: DistributionModel = DistributionModel.PERFUSION_LIMITED
    absorption_model: str = "first_order"  # "first_order" or "acat"
    enable_ehc: bool = False            # Enterohepatic recirculation
    enable_lymphatic: bool = False      # Lymphatic absorption (logP > 5)
    enable_circadian: bool = False      # Circadian CYP expression variation
    circadian_cyp: str = "CYP3A4"      # Which CYP for circadian modulation
    dose_time_of_day_h: float = 8.0    # Clock time of first dose (for circadian)
    ehc_params: Optional[dict] = None  # EHC parameters override
    gut_cyp_clint: Optional[dict] = None  # Per-CYP gut CLint: {"CYP3A4": 30, "CYP2C9": 5}
    rtol: float = 1e-8
    atol: float = 1e-10
    max_step: float = 0.1


@dataclass
class SimulationResult:
    """Container for PBPK simulation output."""

    time: np.ndarray                    # Time points (h)
    amounts: dict[str, np.ndarray]      # {compartment: amount_mg array}
    concentrations: dict[str, np.ndarray]  # {compartment: conc_mg_per_L array}
    venous_plasma: np.ndarray           # Venous plasma concentration (mg/L)
    arterial_plasma: np.ndarray         # Arterial plasma concentration (mg/L)
    compound_name: str = ""
    dose_info: str = ""

    def get_organ_concentration(self, organ_name: str) -> np.ndarray:
        """Get tissue concentration for a named organ."""
        return self.concentrations.get(organ_name, np.zeros_like(self.time))

    def get_plasma_concentration(self) -> np.ndarray:
        """Venous plasma concentration (standard PK sampling site)."""
        return self.venous_plasma


class PBPKModel:
    """
    Whole-body PBPK model with PK-Sim/Simcyp-style ODE system.

    Usage:
        model = PBPKModel(compound, physiology)
        result = model.simulate(dosing, config)
    """

    def __init__(
        self,
        compound: CompoundSpec,
        physiology: Optional[PhysiologyParams] = None,
        kp_override: Optional[dict[Organ, float]] = None,
        transporters: Optional[dict[str, OrganTransporters]] = None,
    ):
        self.compound = compound
        self.phys = physiology or get_physiology()

        # Organ-specific transporters: {"liver": OrganTransporters, "kidney": ..., etc.}
        self.transporters = transporters or {}

        # Predict partition coefficients
        if kp_override:
            self.kp = kp_override
        else:
            self.kp = predict_kp_all(compound)

        # Kp:blood = Kp / R_bp (for blood-flow-based ODEs)
        self.kpb = {organ: kp / compound.R_bp for organ, kp in self.kp.items()}

        # Pre-compute ODE parameters for performance
        self._precompute_params()

    def _precompute_params(self):
        """Pre-compute all constant parameters used in ODE RHS."""
        p = self.phys
        c = self.compound

        # Volumes
        self._V = np.zeros(N_STATES_PERFUSION)
        self._V[IDX["venous"]] = p.V_venous
        self._V[IDX["arterial"]] = p.V_arterial
        for organ, idx in ORGAN_TO_IDX.items():
            self._V[idx] = p.organ_volumes[organ]

        # Blood flows
        self._Q = {}
        for organ in Organ:
            if organ != Organ.LUNG:
                self._Q[organ] = p.blood_flows.get(organ, 0.0)

        self._Qco = p.cardiac_output
        self._Q_ha = p.Q_hepatic_artery
        self._Q_portal = p.Q_portal
        self._Q_liver_total = p.Q_liver_total

        # Kp:blood ratio for each organ (C_tissue / C_blood at equilibrium)
        self._kpb = np.zeros(N_STATES_PERFUSION)
        for organ, idx in ORGAN_TO_IDX.items():
            self._kpb[idx] = self.kpb[organ]

        # Absorption
        self._ka = c.ka
        self._Fa = c.Fa
        self._Fg = c.Fg

        # Hepatic clearance parameters
        self._CL_int = c.CL_int
        self._fu_p = c.fu_p
        self._R_bp = c.R_bp
        self._metabolism = c.metabolism_model
        self._Vmax = c.Vmax
        self._Km = c.Km

        # Renal clearance (referenced to plasma)
        self._CL_renal = c.CL_renal

        # Lymphatic fraction (for logP > 5 drugs)
        self._F_lymph = estimate_lymphatic_fraction(c.logP)
        self._k_lymph_drain = 1.0 / LYMPH_TRANSIT_TIME

        # Kp for liver and kidney (tissue:plasma, for unbound calc)
        self._Kp_liver = self.kp[Organ.LIVER]
        self._Kp_kidney = self.kp[Organ.KIDNEY]

        # Precompute inverse of (V * kpb) for vectorized C_ven_org calculation.
        # Zero entries signal "no tissue partitioning" (used for blood pools).
        self._inv_VKpb = np.zeros(N_STATES_PERFUSION)
        for idx in range(N_STATES_PERFUSION):
            v = self._V[idx]
            k = self._kpb[idx]
            if v > 0 and k > 0:
                self._inv_VKpb[idx] = 1.0 / (v * k)

    def _rhs_perfusion_limited(self, t: float, y: np.ndarray, infusion_rate: float = 0.0) -> np.ndarray:
        """
        Right-hand side of the perfusion-limited PBPK ODE system.

        Each organ is a single well-stirred compartment:
          dA_t/dt = Q_t * (C_art_blood - C_t / Kpb_t)

        Special organs:
          - Lung: between venous and arterial pools
          - Gut: receives oral absorption, drains to portal vein
          - Spleen: drains to portal vein
          - Liver: receives hepatic artery + portal vein, hepatic clearance
          - Kidney: renal clearance
        """
        dy = np.zeros_like(y)

        V = self._V
        Qco = self._Qco

        # Vectorized C_ven_organ = y / (V * Kpb). Entries where inv_VKpb==0
        # (blood pools) correctly produce 0 — they're never read for those indices.
        C_ven_arr = y * self._inv_VKpb

        # Blood pool concentrations (direct mass/volume)
        C_ven = y[1] / V[1] if V[1] > 0 else 0.0
        C_art = y[2] / V[2] if V[2] > 0 else 0.0

        # --- [0] Lumen: oral absorption ---
        dy[0] = -self._ka * y[0]

        # --- [3] Lung: Qco * (C_ven - C_ven_lung) ---
        dy[3] = Qco * (C_ven - C_ven_arr[3])

        # --- [2] Arterial blood: Qco * (C_ven_lung - C_art) ---
        dy[2] = Qco * (C_ven_arr[3] - C_art)

        # --- [1] Venous blood: receives from non-portal organs + liver, loses to lung ---
        ven_inflow = 0.0

        # Non-portal organs → venous pool
        for organ in NON_PORTAL_ORGANS:
            idx = ORGAN_TO_IDX[organ]
            ven_inflow += self._Q[organ] * C_ven_arr[idx]

        # Liver → venous pool
        ven_inflow += self._Q_liver_total * C_ven_arr[IDX["liver"]]

        dy[1] = ven_inflow - Qco * C_ven + infusion_rate

        # --- Non-portal, non-eliminating organs ---
        for organ in [Organ.ADIPOSE, Organ.BONE, Organ.BRAIN,
                      Organ.HEART, Organ.MUSCLE, Organ.PANCREAS,
                      Organ.SKIN, Organ.REST]:
            idx = ORGAN_TO_IDX[organ]
            dy[idx] = self._Q[organ] * (C_art - C_ven_arr[idx])

        # --- [7] Gut wall: absorption + perfusion ---
        Q_gut = self._Q[Organ.GUT]
        absorption = self._ka * y[0] * self._Fg
        dy[7] = Q_gut * (C_art - C_ven_arr[7]) + absorption

        # --- [9] Kidney: perfusion + renal clearance ---
        Q_kid = self._Q[Organ.KIDNEY]
        C_plasma_kid = y[9] / (V[9] * self._Kp_kidney) if V[9] > 0 and self._Kp_kidney > 0 else 0.0
        renal_elim = self._CL_renal * C_plasma_kid
        dy[9] = Q_kid * (C_art - C_ven_arr[9]) - renal_elim

        # --- [14] Spleen: perfusion (drains to portal vein) ---
        Q_spl = self._Q[Organ.SPLEEN]
        dy[14] = Q_spl * (C_art - C_ven_arr[14])

        # --- [10] Liver: dual input + hepatic clearance ---
        C_portal = 0.0
        if self._Q_portal > 0:
            C_portal = (
                Q_gut * C_ven_arr[7]
                + Q_spl * C_ven_arr[14]
                + self._Q.get(Organ.PANCREAS, 0) * C_ven_arr[12]
            ) / self._Q_portal

        liver_inflow = self._Q_ha * C_art + self._Q_portal * C_portal
        liver_outflow = self._Q_liver_total * C_ven_arr[IDX["liver"]]

        # Hepatic metabolism
        # Unbound plasma concentration in liver (well-stirred)
        C_plasma_liver = y[10] / (V[10] * self._Kp_liver) if V[10] > 0 and self._Kp_liver > 0 else 0.0
        C_u_liver = self._fu_p * C_plasma_liver

        if self._metabolism == MetabolismModel.FIRST_ORDER:
            hepatic_elim = self._CL_int * C_u_liver
        elif self._metabolism == MetabolismModel.MICHAELIS_MENTEN:
            if self._Vmax is not None and self._Km is not None:
                hepatic_elim = self._Vmax * C_u_liver / (self._Km + C_u_liver) if C_u_liver > 0 else 0.0
            else:
                hepatic_elim = 0.0
        else:
            hepatic_elim = 0.0

        dy[10] = liver_inflow - liver_outflow - hepatic_elim

        return dy

    def _get_initial_conditions(self, dosing: DosingProtocol) -> np.ndarray:
        """Set initial conditions based on dosing route."""
        y0 = np.zeros(N_STATES_PERFUSION)

        if dosing.route == Route.ORAL:
            # Drug enters gut lumen; Fa applied at absorption from GI tract
            y0[IDX["lumen"]] = dosing.dose_mg * self._Fa
        elif dosing.route == Route.IV_BOLUS:
            # Bolus into venous blood
            y0[IDX["venous"]] = dosing.dose_mg
        elif dosing.route == Route.IV_INFUSION:
            # Infusion handled via forcing function in RHS
            pass

        return y0

    def simulate(
        self,
        dosing: DosingProtocol,
        config: Optional[SimulationConfig] = None,
    ) -> SimulationResult:
        """
        Run the PBPK simulation.

        For multiple doses, the simulation is run continuously with dose events
        applied at each dosing time.

        Args:
            dosing: Dosing protocol specification.
            config: Simulation configuration. Defaults to 24h, 1000 points.

        Returns:
            SimulationResult with time-concentration profiles for all compartments.
        """
        if config is None:
            config = SimulationConfig()

        if config.distribution_model == DistributionModel.PERMEABILITY_LIMITED:
            return self._simulate_permeability_limited(dosing, config)

        if config.absorption_model == "acat" and dosing.route == Route.ORAL:
            return self._simulate_with_acat(dosing, config)

        # Use extended model if EHC or lymphatic enabled
        if config.enable_ehc or config.enable_lymphatic or config.enable_circadian:
            return self._simulate_extended(dosing, config)

        return self._simulate_perfusion_limited(dosing, config)

    def _simulate_extended(
        self,
        dosing: DosingProtocol,
        config: SimulationConfig,
    ) -> SimulationResult:
        """
        Perfusion-limited PBPK with EHC, lymphatic, and circadian modules.

        Extended state vector [0..19]:
          [0-15]  Standard perfusion-limited states
          [16]    EHC: bile duct
          [17]    EHC: gallbladder
          [18]    EHC: deconjugation pool
          [19]    Lymphatic compartment
        """
        c = self.compound
        use_ehc = config.enable_ehc
        use_lymph = config.enable_lymphatic and self._F_lymph > 0.01
        use_circadian = config.enable_circadian

        # EHC parameters
        ehc_p = EHCParams()
        if config.ehc_params:
            for k, v in config.ehc_params.items():
                if hasattr(ehc_p, k):
                    setattr(ehc_p, k, v)

        V = self._V
        kpb = self._kpb
        Qco = self._Qco

        def rhs_extended(t, y):
            dy = np.zeros(N_STATES_EXTENDED)

            # --- Standard perfusion-limited part ---
            C_ven = y[1] / V[1] if V[1] > 0 else 0.0
            C_art = y[2] / V[2] if V[2] > 0 else 0.0

            def C_ven_org(idx):
                if V[idx] > 0 and kpb[idx] > 0:
                    return y[idx] / (V[idx] * kpb[idx])
                return 0.0

            dy[0] = -self._ka * y[0]
            dy[3] = Qco * (C_ven - C_ven_org(3))
            dy[2] = Qco * (C_ven_org(3) - C_art)

            ven_inflow = 0.0
            for organ in NON_PORTAL_ORGANS:
                idx = ORGAN_TO_IDX[organ]
                Q = self._Q[organ]
                dy[idx] = Q * (C_art - C_ven_org(idx))
                ven_inflow += Q * C_ven_org(idx)

            # Gut wall: receives oral absorption
            Q_gut = self._Q[Organ.GUT]
            absorption = self._ka * y[0] * self._Fg
            if use_lymph:
                # Split absorption: portal vs lymphatic
                portal_fraction = 1.0 - self._F_lymph
                dy[7] = Q_gut * (C_art - C_ven_org(7)) + absorption * portal_fraction
                dy[IDX_LYMPH] = absorption * self._F_lymph - self._k_lymph_drain * y[IDX_LYMPH]
                # Lymphatic drains to venous blood (bypasses liver!)
                # Added to ven_inflow as mass rate (mg/h) — handled in dy[1] below
                lymph_drain_rate = self._k_lymph_drain * max(y[IDX_LYMPH], 0.0)
            else:
                dy[7] = Q_gut * (C_art - C_ven_org(7)) + absorption

            # Kidney
            Q_kid = self._Q[Organ.KIDNEY]
            C_plasma_kid = y[9] / (V[9] * self._Kp_kidney) if V[9] > 0 and self._Kp_kidney > 0 else 0.0
            renal_elim = self._CL_renal * C_plasma_kid
            dy[9] = Q_kid * (C_art - C_ven_org(9)) - renal_elim

            # Spleen
            Q_spl = self._Q[Organ.SPLEEN]
            dy[14] = Q_spl * (C_art - C_ven_org(14))

            # Liver
            C_portal = 0.0
            if self._Q_portal > 0:
                C_portal = (Q_gut * C_ven_org(7) + Q_spl * C_ven_org(14) + self._Q.get(Organ.PANCREAS, 0) * C_ven_org(12)) / self._Q_portal
            liver_inflow = self._Q_ha * C_art + self._Q_portal * C_portal
            liver_outflow = self._Q_liver_total * C_ven_org(IDX["liver"])

            C_plasma_liver = y[10] / (V[10] * self._Kp_liver) if V[10] > 0 and self._Kp_liver > 0 else 0.0
            C_u_liver = self._fu_p * C_plasma_liver

            # Circadian modulation of CL_int
            cl_int_eff = self._CL_int
            if use_circadian:
                clock_time = (config.dose_time_of_day_h + t) % 24.0
                circ_factor = circadian_enzyme_factor(config.circadian_cyp, clock_time)
                cl_int_eff *= circ_factor

            if self._metabolism == MetabolismModel.FIRST_ORDER:
                hepatic_elim = cl_int_eff * C_u_liver
            elif self._metabolism == MetabolismModel.MICHAELIS_MENTEN:
                if self._Vmax is not None and self._Km is not None and C_u_liver > 0:
                    hepatic_elim = self._Vmax * C_u_liver / (self._Km + C_u_liver)
                else:
                    hepatic_elim = 0.0
            else:
                hepatic_elim = 0.0

            # EHC: biliary excretion from liver
            biliary_rate = 0.0
            ehc_reabs = 0.0
            if use_ehc:
                biliary_rate = compute_biliary_rate(
                    y[10], V[10], self._Kp_liver, self._fu_p, ehc_p.CL_bile
                )
                dy_ehc, ehc_reabs = ehc_rhs(
                    y[IDX_EHC_BILE:IDX_EHC_BILE + N_EHC_STATES],
                    t, biliary_rate, ehc_p,
                )
                dy[IDX_EHC_BILE:IDX_EHC_BILE + N_EHC_STATES] = dy_ehc
                # Reabsorbed drug re-enters gut wall (portal pathway)
                dy[7] += ehc_reabs

            dy[10] = liver_inflow - liver_outflow - hepatic_elim - biliary_rate

            ven_inflow += self._Q_liver_total * C_ven_org(IDX["liver"])
            dy[1] = ven_inflow - Qco * C_ven
            # Add lymphatic drain to venous pool (bypasses liver, single addition)
            if use_lymph:
                dy[1] += lymph_drain_rate

            return dy

        # Initial conditions
        y0 = np.zeros(N_STATES_EXTENDED)
        if dosing.route == Route.ORAL:
            y0[IDX["lumen"]] = dosing.dose_mg * self._Fa
        elif dosing.route == Route.IV_BOLUS:
            y0[IDX["venous"]] = dosing.dose_mg

        t_eval = np.linspace(0, config.duration_h, config.n_timepoints)
        sol = solve_ivp(
            rhs_extended, [0, config.duration_h], y0,
            method="BDF", t_eval=t_eval,
            rtol=config.rtol, atol=config.atol, max_step=config.max_step,
        )

        if not sol.success:
            raise RuntimeError(f"Extended ODE solver failed: {sol.message}")

        return self._build_result(sol.t, sol.y[:N_STATES_PERFUSION, :], dosing)

    def _simulate_perfusion_limited(
        self,
        dosing: DosingProtocol,
        config: SimulationConfig,
    ) -> SimulationResult:
        """Run perfusion-limited PBPK simulation."""

        dose_times = dosing.get_dose_times()
        t_end = config.duration_h
        t_eval = np.linspace(0, t_end, config.n_timepoints)

        # For infusion: compute rate
        infusion_rate = 0.0
        infusion_end = 0.0
        if dosing.route == Route.IV_INFUSION:
            infusion_rate = dosing.dose_mg / dosing.infusion_duration_h
            infusion_end = dosing.infusion_duration_h

        # Initial conditions
        y0 = self._get_initial_conditions(dosing)

        # Collect all results
        all_t = []
        all_y = []

        current_y = y0.copy()
        current_t = 0.0

        for dose_idx, dose_time in enumerate(dose_times):
            # Time segments: [dose_time, next_dose_time or t_end]
            if dose_idx < len(dose_times) - 1:
                segment_end = dose_times[dose_idx + 1]
            else:
                segment_end = t_end

            if dose_time > current_t:
                # Solve up to the next dose time
                seg_t_eval = t_eval[(t_eval >= current_t) & (t_eval < dose_time)]
                if len(seg_t_eval) > 0:
                    def rhs_seg(t, y):
                        inf = infusion_rate if t < infusion_end else 0.0
                        return self._rhs_perfusion_limited(t, y, inf)
                    sol = solve_ivp(
                        rhs_seg,
                        [current_t, dose_time],
                        current_y,
                        method="BDF",
                        t_eval=seg_t_eval,
                        rtol=config.rtol,
                        atol=config.atol,
                        max_step=config.max_step,
                    )
                    if sol.success:
                        all_t.append(sol.t)
                        all_y.append(sol.y)
                        current_y = sol.y[:, -1].copy()
                    current_t = dose_time

            # Apply dose
            if dose_idx > 0:  # First dose already in y0
                if dosing.route == Route.ORAL:
                    current_y[IDX["lumen"]] += dosing.dose_mg * self._Fa
                elif dosing.route == Route.IV_BOLUS:
                    current_y[IDX["venous"]] += dosing.dose_mg
                elif dosing.route == Route.IV_INFUSION:
                    # Reset infusion timing for this dose
                    infusion_end = dose_time + dosing.infusion_duration_h

            # Solve from dose time to segment end
            seg_t_eval = t_eval[(t_eval >= current_t) & (t_eval <= segment_end)]
            if len(seg_t_eval) == 0:
                seg_t_eval = np.array([current_t, segment_end])

            def rhs_final(t, y, _ie=infusion_end, _ir=infusion_rate):
                inf = _ir if t < _ie else 0.0
                return self._rhs_perfusion_limited(t, y, inf)

            sol = solve_ivp(
                rhs_final,
                [current_t, segment_end],
                current_y,
                method="BDF",
                t_eval=seg_t_eval,
                rtol=config.rtol,
                atol=config.atol,
                max_step=config.max_step,
            )

            if sol.success:
                all_t.append(sol.t)
                all_y.append(sol.y)
                current_y = sol.y[:, -1].copy()
            else:
                raise RuntimeError(
                    f"ODE solver failed at t={current_t:.2f}: {sol.message}"
                )

            current_t = segment_end

        # Concatenate results
        time = np.concatenate(all_t)
        y_full = np.concatenate(all_y, axis=1)

        # Remove duplicate time points
        _, unique_idx = np.unique(time, return_index=True)
        time = time[unique_idx]
        y_full = y_full[:, unique_idx]

        return self._build_result(time, y_full, dosing)

    def _simulate_with_acat(
        self,
        dosing: DosingProtocol,
        config: SimulationConfig,
    ) -> SimulationResult:
        """
        Perfusion-limited PBPK with ACAT absorption model.

        The state vector is expanded:
          [0..15]  = systemic PBPK states (lumen slot [0] unused, set to 0)
          [16..42] = 27 ACAT states (9 segments × 3: undissolved, dissolved, enterocyte)

        The ACAT enterocyte-to-portal-vein flux replaces the simple
        first-order absorption. Drug enters the gut wall compartment [7]
        from the ACAT portal output.
        """
        c = self.compound
        N_TOTAL = N_STATES_PERFUSION + N_ACAT_STATES  # 16 + 27 = 43

        # Build ACAT segment parameters
        Peff = c.Peff if c.Peff is not None else 5.0
        S0 = c.S0 if c.S0 is not None else 1.0
        fu_gut = predict_fu_gut(c) if c.fu_p < 1.0 else 1.0

        formulation = FormulationSpec(
            S0=S0,
            particle_radius_um=c.particle_radius_um,
        )
        seg_params = build_segment_params(
            Peff_e4=Peff,
            mw=c.mw,
            pKa=c.pKa,
            compound_type=c.compound_type.value,
            S0=S0,
            formulation=formulation,
            CLint_gut_total=c.CLint_gut,
            fu_gut=fu_gut,
            logP=c.logP,
            gut_cyp_clint=config.gut_cyp_clint,
        )

        V = self._V
        kpb = self._kpb
        Qco = self._Qco

        def rhs_acat_pbpk(t, y):
            dy = np.zeros(N_TOTAL)

            # Split state vector
            y_sys = y[:N_STATES_PERFUSION]
            y_gi = y[N_STATES_PERFUSION:]

            # --- ACAT GI tract ---
            dy_gi, total_portal_rate = acat_rhs(y_gi, seg_params, formulation, dosing.dose_mg)
            dy[N_STATES_PERFUSION:] = dy_gi

            # --- Systemic PBPK (same as perfusion-limited, but no lumen) ---
            C_ven = y_sys[1] / V[1] if V[1] > 0 else 0.0
            C_art = y_sys[2] / V[2] if V[2] > 0 else 0.0

            def C_ven_org(idx):
                if V[idx] > 0 and kpb[idx] > 0:
                    return y_sys[idx] / (V[idx] * kpb[idx])
                return 0.0

            # Lumen is unused in ACAT mode
            dy[0] = 0.0

            # Lung
            dy[3] = Qco * (C_ven - C_ven_org(3))

            # Arterial
            dy[2] = Qco * (C_ven_org(3) - C_art)

            # Venous pool
            ven_inflow = 0.0
            for organ in NON_PORTAL_ORGANS:
                idx = ORGAN_TO_IDX[organ]
                Q = self._Q[organ]
                ven_inflow += Q * C_ven_org(idx)
            ven_inflow += self._Q_liver_total * C_ven_org(IDX["liver"])
            dy[1] = ven_inflow - Qco * C_ven

            # Non-eliminating organs
            for organ in [Organ.ADIPOSE, Organ.BONE, Organ.BRAIN,
                          Organ.HEART, Organ.MUSCLE, Organ.PANCREAS,
                          Organ.SKIN, Organ.REST]:
                idx = ORGAN_TO_IDX[organ]
                Q = self._Q[organ]
                dy[idx] = Q * (C_art - C_ven_org(idx))

            # Gut wall: receives ACAT portal output instead of first-order absorption
            Q_gut = self._Q[Organ.GUT]
            dy[7] = Q_gut * (C_art - C_ven_org(7)) + total_portal_rate

            # Kidney
            Q_kid = self._Q[Organ.KIDNEY]
            C_plasma_kid = y_sys[9] / (V[9] * self._Kp_kidney) if V[9] > 0 and self._Kp_kidney > 0 else 0.0
            renal_elim = self._CL_renal * C_plasma_kid
            dy[9] = Q_kid * (C_art - C_ven_org(9)) - renal_elim

            # Spleen
            Q_spl = self._Q[Organ.SPLEEN]
            dy[14] = Q_spl * (C_art - C_ven_org(14))

            # Liver
            C_portal = 0.0
            if self._Q_portal > 0:
                C_portal = (Q_gut * C_ven_org(7) + Q_spl * C_ven_org(14) + self._Q.get(Organ.PANCREAS, 0) * C_ven_org(12)) / self._Q_portal
            liver_inflow = self._Q_ha * C_art + self._Q_portal * C_portal
            liver_outflow = self._Q_liver_total * C_ven_org(IDX["liver"])
            C_plasma_liver = y_sys[10] / (V[10] * self._Kp_liver) if V[10] > 0 and self._Kp_liver > 0 else 0.0
            C_u_liver = self._fu_p * C_plasma_liver
            if self._metabolism == MetabolismModel.FIRST_ORDER:
                hepatic_elim = self._CL_int * C_u_liver
            elif self._metabolism == MetabolismModel.MICHAELIS_MENTEN:
                if self._Vmax is not None and self._Km is not None and C_u_liver > 0:
                    hepatic_elim = self._Vmax * C_u_liver / (self._Km + C_u_liver)
                else:
                    hepatic_elim = 0.0
            else:
                hepatic_elim = 0.0
            dy[10] = liver_inflow - liver_outflow - hepatic_elim

            return dy

        # Initial conditions
        y0 = np.zeros(N_TOTAL)
        # ACAT: all dose as undissolved in stomach
        y0[N_STATES_PERFUSION + GISegment.STOMACH * 3 + UNDISSOLVED] = dosing.dose_mg

        t_eval = np.linspace(0, config.duration_h, config.n_timepoints)

        sol = solve_ivp(
            rhs_acat_pbpk,
            [0, config.duration_h],
            y0,
            method="BDF",
            t_eval=t_eval,
            rtol=config.rtol,
            atol=config.atol,
            max_step=config.max_step,
        )

        if not sol.success:
            raise RuntimeError(f"ACAT-PBPK solver failed: {sol.message}")

        # Extract systemic states only for result building
        time = sol.t
        y_sys = sol.y[:N_STATES_PERFUSION, :]
        return self._build_result(time, y_sys, dosing)

    def _build_result(
        self,
        time: np.ndarray,
        y: np.ndarray,
        dosing: DosingProtocol,
    ) -> SimulationResult:
        """Convert raw ODE output to SimulationResult."""
        V = self._V

        # Amounts for each compartment
        amounts = {}
        concentrations = {}

        compartment_names = list(IDX.keys())
        for name, idx in IDX.items():
            amounts[name] = y[idx, :]
            if V[idx] > 0:
                concentrations[name] = y[idx, :] / V[idx]
            else:
                concentrations[name] = np.zeros_like(time)

        # Venous and arterial PLASMA concentrations
        # C_plasma = C_blood / R_bp
        R_bp = self.compound.R_bp
        venous_plasma = concentrations["venous"] / R_bp
        arterial_plasma = concentrations["arterial"] / R_bp

        route_str = dosing.route.value.replace("_", " ").upper()
        dose_info = f"{dosing.dose_mg:.1f} mg {route_str}"
        if dosing.n_doses > 1:
            dose_info += f" x{dosing.n_doses} q{dosing.interval_h:.0f}h"

        return SimulationResult(
            time=time,
            amounts=amounts,
            concentrations=concentrations,
            venous_plasma=venous_plasma,
            arterial_plasma=arterial_plasma,
            compound_name=self.compound.name,
            dose_info=dose_info,
        )

    # -------------------------------------------------------------------
    # Permeability-limited model (PK-Sim "Standard" organ model)
    # -------------------------------------------------------------------

    def _simulate_permeability_limited(
        self,
        dosing: DosingProtocol,
        config: SimulationConfig,
    ) -> SimulationResult:
        """
        Permeability-limited PBPK simulation.

        Each organ has 3 sub-compartments:
          - Vascular (V_vas): blood in organ capillaries
          - Interstitial (V_int): extracellular, extravascular
          - Intracellular (V_cell): intracellular space

        Transport:
          Vascular ↔ Interstitial: endothelial permeability (PA_endo)
          Interstitial ↔ Intracellular: cell membrane permeability (PA_cell)

        For small molecules, PA is typically large → equivalent to perfusion-limited.
        For large molecules (biologics), PA becomes rate-limiting.
        """
        # Number of states: 3 per organ (13 organs) + 2 blood pools + 1 lumen = 42
        N_ORGANS = 13
        N_SUB = 3  # vascular, interstitial, intracellular
        N_STATES = 3 + N_ORGANS * N_SUB  # lumen(1) + ven(1) + art(1) + 13*3

        # Sub-compartment indices
        # 0: lumen, 1: venous, 2: arterial
        # Then for each organ: [vas, int, cell] starting at index 3
        organ_list = list(Organ)  # 13 organs in enum order

        def organ_sub_idx(organ_i: int, sub: int) -> int:
            return 3 + organ_i * N_SUB + sub

        # Pre-compute sub-compartment volumes
        sub_volumes = {}
        for i, organ in enumerate(organ_list):
            V_organ = self.phys.organ_volumes[organ]
            sub_v = get_subcompartment_volumes(organ, V_organ)
            sub_volumes[i] = sub_v

        # Permeability-surface area products (default estimates)
        # For small molecules: large PA → effectively perfusion-limited
        # PA scales with molecular weight: PA ∝ MW^(-0.5) approximately
        mw = self.compound.mw
        # Default: PA_endo = 10 L/h for MW=300 (small molecule), scaled
        # PA estimation: for small molecules (MW < 700), PA should be
        # much larger than organ blood flow (effectively perfusion-limited).
        # PK-Sim: P = (MW_eff/336)^(-6) * 10^logMA * scaling
        # Simplified: PA_base ~ 1000 * (300/MW)^0.5 for small molecules
        # For large molecules (MW > 700): PA decreases dramatically
        # PK-Sim approach: P = (MW_eff/336)^(-6) * 10^logMA * scaling
        # For small molecules PA >> Q → effectively perfusion-limited
        # For large molecules (biologics) PA << Q → permeability rate-limiting
        if mw < 700:
            pa_base_endo = 10000.0 * (300.0 / max(mw, 1.0)) ** 0.5
            pa_base_cell = 5000.0 * (300.0 / max(mw, 1.0)) ** 0.5
        else:
            pa_base_endo = 0.1 * (700.0 / max(mw, 1.0)) ** 3.0
            pa_base_cell = 0.05 * (700.0 / max(mw, 1.0)) ** 3.0

        # fu in interstitial space ≈ fu_p (same protein binding as plasma)
        fu_int = self.compound.fu_p
        # fu in intracellular space: approximate
        fu_cell = min(1.0, self.compound.fu_p * 2.0)

        phys = self.phys

        def rhs_perm(t, y, infusion_rate=0.0):
            dy = np.zeros(N_STATES)

            C_ven = y[1] / phys.V_venous if phys.V_venous > 0 else 0.0
            C_art = y[2] / phys.V_arterial if phys.V_arterial > 0 else 0.0

            # Lumen absorption
            dy[0] = -self._ka * y[0]

            ven_inflow = 0.0
            lung_idx = organ_list.index(Organ.LUNG)

            for i, organ in enumerate(organ_list):
                sv = sub_volumes[i]
                V_vas = sv["V_vascular"]
                V_int_c = sv["V_interstitial"]
                V_cell = sv["V_intracellular"]

                idx_vas = organ_sub_idx(i, 0)
                idx_int = organ_sub_idx(i, 1)
                idx_cell = organ_sub_idx(i, 2)

                C_vas = y[idx_vas] / V_vas if V_vas > 0 else 0.0
                C_int = y[idx_int] / V_int_c if V_int_c > 0 else 0.0
                C_cell = y[idx_cell] / V_cell if V_cell > 0 else 0.0

                PA_endo = pa_base_endo * V_vas  # scale with organ size
                PA_cell_val = pa_base_cell * V_int_c

                # Lung blood flow = cardiac output (total systemic circulation).
                # self._Q excludes lung by design (perfusion-limited handles it
                # separately via Qco*(C_ven - C_ven_org(lung))); perm-limited
                # needs to inject Qco here or lung receives no drug.
                if organ == Organ.LUNG:
                    Q = self._Qco
                else:
                    Q = self._Q.get(organ, 0.0)

                # --- Vascular sub-compartment ---
                # dA_vas/dt = Q*(C_in - C_vas) - PA_endo*(fu_p*C_vas/R_bp - fu_int*C_int)
                if organ == Organ.LUNG:
                    C_in = C_ven  # lung receives venous blood
                elif organ == Organ.LIVER:
                    # Liver receives HA + portal
                    # Handled separately below
                    C_in = C_art
                else:
                    C_in = C_art

                flux_endo = PA_endo * (
                    self._fu_p * C_vas / self._R_bp - fu_int * C_int
                )

                # Vascular
                if organ == Organ.LIVER:
                    # Portal vein contribution — ALL portal organs drain to liver
                    gut_i = organ_list.index(Organ.GUT)
                    spl_i = organ_list.index(Organ.SPLEEN)
                    pan_i = organ_list.index(Organ.PANCREAS)
                    C_gut_vas = y[organ_sub_idx(gut_i, 0)] / sub_volumes[gut_i]["V_vascular"] if sub_volumes[gut_i]["V_vascular"] > 0 else 0.0
                    C_spl_vas = y[organ_sub_idx(spl_i, 0)] / sub_volumes[spl_i]["V_vascular"] if sub_volumes[spl_i]["V_vascular"] > 0 else 0.0
                    C_pan_vas = y[organ_sub_idx(pan_i, 0)] / sub_volumes[pan_i]["V_vascular"] if sub_volumes[pan_i]["V_vascular"] > 0 else 0.0

                    Q_gut = self._Q[Organ.GUT]
                    Q_spl = self._Q[Organ.SPLEEN]
                    Q_pan = self._Q.get(Organ.PANCREAS, 0.0)
                    liver_in = (self._Q_ha * C_art
                                + Q_gut * C_gut_vas
                                + Q_spl * C_spl_vas
                                + Q_pan * C_pan_vas)
                    dy[idx_vas] = liver_in - self._Q_liver_total * C_vas - flux_endo
                else:
                    dy[idx_vas] = Q * (C_in - C_vas) - flux_endo

                # --- Active transport (transporter-mediated) ---
                # Michaelis-Menten flux: Vmax * C_u / (Km + C_u)
                # Vmax units should be in amount/time matching ODE (mg/h)
                # If Vmax is in pmol/min, user must pre-scale to mg/h:
                #   Vmax_mg_h = Vmax_pmol_min * MW * 60 / 1e9
                # Or provide Vmax as CLint_transport (L/h) with is_scaled=True
                organ_key = organ.value
                organ_trans = self.transporters.get(organ_key)
                active_uptake = 0.0   # vascular/interstitial → cell (mg/h)
                active_efflux = 0.0   # cell → bile/urine/lumen (mg/h)

                if organ_trans:
                    # Unbound concentrations at transport sites (mg/L)
                    C_u_vas = self._fu_p * C_vas / self._R_bp  # unbound plasma in vascular
                    C_u_cell_t = fu_cell * C_cell               # unbound intracellular

                    for t in organ_trans.influx:
                        # Uptake: Vmax * Cu_vas / (Km + Cu_vas)
                        # Scale: if Km in uM, convert Cu to uM: Cu_uM = Cu_mg_L * 1000 / MW
                        C_u_uM = C_u_vas * 1000.0 / max(mw, 1.0)
                        rate_uM = compute_transport_rate(t, C_u_uM)
                        # Convert rate back to mg/h: rate * MW / 1000 * V_cell (L)
                        active_uptake += rate_uM * mw / 1000.0 * V_cell

                    for t in organ_trans.efflux:
                        C_u_uM = C_u_cell_t * 1000.0 / max(mw, 1.0)
                        rate_uM = compute_transport_rate(t, C_u_uM)
                        active_efflux += rate_uM * mw / 1000.0 * V_cell

                # --- Interstitial sub-compartment ---
                flux_cell_passive = PA_cell_val * (fu_int * C_int - fu_cell * C_cell)
                dy[idx_int] = flux_endo - flux_cell_passive

                # --- Vascular: active uptake removes from vascular directly ---
                dy[idx_vas] -= active_uptake

                # --- Intracellular sub-compartment ---
                dy[idx_cell] = flux_cell_passive + active_uptake - active_efflux

                # Gut absorption into enterocyte (intracellular)
                if organ == Organ.GUT:
                    dy[idx_cell] += self._ka * y[0] * self._Fg

                # Metabolism in liver intracellular
                if organ == Organ.LIVER:
                    C_u = fu_cell * C_cell
                    if self._metabolism == MetabolismModel.FIRST_ORDER:
                        dy[idx_cell] -= self._CL_int * C_u
                    elif self._metabolism == MetabolismModel.MICHAELIS_MENTEN:
                        if self._Vmax and self._Km and C_u > 0:
                            dy[idx_cell] -= self._Vmax * C_u / (self._Km + C_u)

                # Renal clearance from kidney vascular.
                # To match the perfusion-limited model (where CL_renal acts on
                # tissue-equilibrated plasma = C_tissue/Kp_kidney), we divide by
                # Kp_kidney here. Without this, the same CL_renal input gives
                # Kp_kidney-fold faster elimination in perm-limited vs perfusion.
                # Users can still define active tubular secretion separately via
                # OCT2/MATE1 transporter parameters.
                if organ == Organ.KIDNEY:
                    Kp_kid_eff = max(self._Kp_kidney, 1e-3)
                    C_plasma_kid = (C_vas / self._R_bp) / Kp_kid_eff
                    dy[idx_vas] -= self._CL_renal * C_plasma_kid

                # Venous blood return
                if organ == Organ.LUNG:
                    # Lung output goes to arterial pool
                    pass
                elif organ in PORTAL_ORGANS:
                    # Portal organs drain to liver (handled in liver vascular)
                    pass
                elif organ == Organ.LIVER:
                    ven_inflow += self._Q_liver_total * C_vas
                else:
                    ven_inflow += Q * C_vas

            # Lung output → arterial
            lung_vas_idx = organ_sub_idx(lung_idx, 0)
            C_lung_vas = y[lung_vas_idx] / sub_volumes[lung_idx]["V_vascular"] if sub_volumes[lung_idx]["V_vascular"] > 0 else 0.0

            dy[2] = self._Qco * (C_lung_vas - C_art)  # arterial
            dy[1] = ven_inflow - self._Qco * C_ven + infusion_rate  # venous

            return dy

        # --- Multi-dose + IV infusion support (mirrors _simulate_perfusion_limited) ---
        dose_times = dosing.get_dose_times()
        t_end = config.duration_h
        t_eval = np.linspace(0, t_end, config.n_timepoints)

        infusion_rate = 0.0
        infusion_end = 0.0
        if dosing.route == Route.IV_INFUSION:
            infusion_rate = dosing.dose_mg / dosing.infusion_duration_h
            infusion_end = dosing.infusion_duration_h

        # Initial conditions (first dose)
        y0 = np.zeros(N_STATES)
        if dosing.route == Route.ORAL:
            y0[0] = dosing.dose_mg * self._Fa
        elif dosing.route == Route.IV_BOLUS:
            y0[1] = dosing.dose_mg
        # IV_INFUSION: dose enters continuously via infusion_rate in rhs

        all_t = []
        all_y = []
        current_y = y0.copy()
        current_t = 0.0

        for dose_idx, dose_time in enumerate(dose_times):
            segment_end = dose_times[dose_idx + 1] if dose_idx < len(dose_times) - 1 else t_end

            # Solve up to this dose time if we haven't reached it yet
            if dose_time > current_t:
                seg_t_eval = t_eval[(t_eval >= current_t) & (t_eval < dose_time)]
                if len(seg_t_eval) > 0:
                    def rhs_seg(t, y, _ie=infusion_end, _ir=infusion_rate):
                        inf = _ir if t < _ie else 0.0
                        return rhs_perm(t, y, inf)
                    sol = solve_ivp(
                        rhs_seg,
                        [current_t, dose_time],
                        current_y,
                        method="BDF",
                        t_eval=seg_t_eval,
                        rtol=config.rtol,
                        atol=config.atol,
                        max_step=config.max_step,
                    )
                    if sol.success:
                        all_t.append(sol.t)
                        all_y.append(sol.y)
                        current_y = sol.y[:, -1].copy()
                    current_t = dose_time

            # Apply dose at dose_time (first dose already in y0)
            if dose_idx > 0:
                if dosing.route == Route.ORAL:
                    current_y[0] += dosing.dose_mg * self._Fa
                elif dosing.route == Route.IV_BOLUS:
                    current_y[1] += dosing.dose_mg
                elif dosing.route == Route.IV_INFUSION:
                    infusion_end = dose_time + dosing.infusion_duration_h

            # Solve from current dose time to end of segment
            seg_t_eval = t_eval[(t_eval >= current_t) & (t_eval <= segment_end)]
            if len(seg_t_eval) == 0:
                seg_t_eval = np.array([current_t, segment_end])

            def rhs_final(t, y, _ie=infusion_end, _ir=infusion_rate):
                inf = _ir if t < _ie else 0.0
                return rhs_perm(t, y, inf)

            sol = solve_ivp(
                rhs_final,
                [current_t, segment_end],
                current_y,
                method="BDF",
                t_eval=seg_t_eval,
                rtol=config.rtol,
                atol=config.atol,
                max_step=config.max_step,
            )

            if sol.success:
                all_t.append(sol.t)
                all_y.append(sol.y)
                current_y = sol.y[:, -1].copy()
            else:
                raise RuntimeError(
                    f"ODE solver failed at t={current_t:.2f}: {sol.message}"
                )

            current_t = segment_end

        # Concatenate and deduplicate
        time = np.concatenate(all_t)
        y_full = np.concatenate(all_y, axis=1)
        _, unique_idx = np.unique(time, return_index=True)
        time = time[unique_idx]
        y_full = y_full[:, unique_idx]

        # Build result: aggregate sub-compartments for each organ
        amounts = {"lumen": y_full[0], "venous": y_full[1], "arterial": y_full[2]}
        concentrations = {
            "lumen": np.zeros_like(time),
            "venous": y_full[1] / self.phys.V_venous,
            "arterial": y_full[2] / self.phys.V_arterial,
        }

        for i, organ in enumerate(organ_list):
            name = organ.value
            V_organ = self.phys.organ_volumes[organ]
            total_amount = np.zeros_like(time)
            for sub in range(N_SUB):
                idx = organ_sub_idx(i, sub)
                total_amount += y_full[idx]
            amounts[name] = total_amount
            concentrations[name] = total_amount / V_organ if V_organ > 0 else np.zeros_like(time)

        R_bp = self.compound.R_bp
        venous_plasma = concentrations["venous"] / R_bp
        arterial_plasma = concentrations["arterial"] / R_bp

        route_str = dosing.route.value.replace("_", " ").upper()
        dose_info = f"{dosing.dose_mg:.1f} mg {route_str}"

        return SimulationResult(
            time=time,
            amounts=amounts,
            concentrations=concentrations,
            venous_plasma=venous_plasma,
            arterial_plasma=arterial_plasma,
            compound_name=self.compound.name,
            dose_info=dose_info,
        )
