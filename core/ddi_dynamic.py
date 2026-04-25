"""
Dynamic DDI simulation — two drugs simulated simultaneously in one ODE system.

The perpetrator (inhibitor/inducer) and victim (substrate) are simulated
together, with the perpetrator's time-varying concentration modulating
the victim's clearance in real-time.

Mechanisms modeled:
  1. Reversible inhibition: CL_int_eff = CL_int / (1 + C_inh_u/Ki)
  2. MBI (time-dependent): enzyme pool ODE with inactivation
     dE/dt = kdeg*(1 - E) - kinact*C_inh_u/(KI + C_inh_u) * E
  3. Induction: enzyme synthesis upregulation
     dE/dt = kdeg*((1 + Emax*C_inh_u/(EC50+C_inh_u)) - E)
  4. Combined: all three simultaneously

State vector layout:
  [0..15]  Victim drug — 16 perfusion-limited states
  [16..31] Perpetrator drug — 16 perfusion-limited states
  [32]     Liver CYP enzyme fraction (E, 0-1 relative to baseline)
  [33]     Gut CYP enzyme fraction (E_gut)

Total: 34 states

References:
  - Fahmi OA et al. Drug Metab Dispos 2009;37:1658-1666
  - Einolf HJ et al. J Pharmacol Exp Ther 2004;308:303-309
  - Rowland Yeo K et al. Clin Pharmacokinet 2010;49:651-667
  - Galetin A et al. J Pharmacol Exp Ther 2006;316:461-468
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Optional

from .compound import CompoundSpec, CompoundType, MetabolismModel
from .physiology import PhysiologyParams, Organ, get_physiology, Sex
from .partition_coeff import predict_kp_all
from .pk_calculator import calculate_pk_parameters, PKParameters


N_STATES_PER_DRUG = 16
# Total DDI states = 2 * N_STATES_PER_DRUG + n_segments + 1
# Layout: victim(16) | perpetrator(16) | liver_enzyme[0..N-1] | gut_enzyme

# Organ indices (same as pbpk_model.py)
_IDX = {"lumen": 0, "venous": 1, "arterial": 2, "lung": 3,
        "adipose": 4, "bone": 5, "brain": 6, "gut": 7,
        "heart": 8, "kidney": 9, "liver": 10, "muscle": 11,
        "pancreas": 12, "skin": 13, "spleen": 14, "rest": 15}

_ORGAN_IDX = {
    Organ.ADIPOSE: 4, Organ.BONE: 5, Organ.BRAIN: 6, Organ.GUT: 7,
    Organ.HEART: 8, Organ.KIDNEY: 9, Organ.LIVER: 10, Organ.LUNG: 3,
    Organ.MUSCLE: 11, Organ.PANCREAS: 12, Organ.SKIN: 13,
    Organ.SPLEEN: 14, Organ.REST: 15,
}

_NON_PORTAL = [Organ.ADIPOSE, Organ.BONE, Organ.BRAIN, Organ.HEART,
               Organ.KIDNEY, Organ.MUSCLE, Organ.SKIN, Organ.REST]


@dataclass
class DDIDrugSpec:
    """Drug specification for DDI simulation."""
    compound: CompoundSpec
    dose_mg: float
    route: str = "oral"              # "oral" or "iv_bolus"
    n_doses: int = 1
    interval_h: float = 24.0
    first_dose_time_h: float = 0.0   # when to start dosing this drug


@dataclass
class DDIMechanism:
    """DDI mechanism parameters."""
    # Reversible
    Ki: Optional[float] = None       # µM

    # MBI
    KI: Optional[float] = None       # µM
    kinact: Optional[float] = None   # 1/h

    # Induction
    Emax: Optional[float] = None     # fold
    EC50: Optional[float] = None     # µM

    # CYP degradation rates
    kdeg_liver: float = 0.019        # 1/h (CYP3A4, t½~36h)
    kdeg_gut: float = 0.029          # 1/h (CYP3A4 gut, t½~24h)

    # Which fraction of victim CL is affected
    fm: float = 0.9                  # fraction metabolized by target CYP


@dataclass
class DDIResult:
    """Dynamic DDI simulation result."""
    time: np.ndarray
    victim_plasma: np.ndarray        # victim venous plasma (mg/L)
    perpetrator_plasma: np.ndarray   # perpetrator venous plasma (mg/L)
    enzyme_liver: np.ndarray         # liver CYP fraction (0-1)
    enzyme_gut: np.ndarray           # gut CYP fraction (0-1)
    victim_alone_plasma: np.ndarray  # victim without DDI (control)
    pk_victim_ddi: PKParameters
    pk_victim_alone: PKParameters
    pk_perpetrator: PKParameters

    @property
    def auc_ratio(self) -> float:
        if self.pk_victim_alone.AUC_0_inf > 0:
            return self.pk_victim_ddi.AUC_0_inf / self.pk_victim_alone.AUC_0_inf
        return 1.0

    @property
    def cmax_ratio(self) -> float:
        if self.pk_victim_alone.Cmax > 0:
            return self.pk_victim_ddi.Cmax / self.pk_victim_alone.Cmax
        return 1.0

    def to_markdown(self) -> str:
        lines = [
            f"## Dynamic DDI Simulation\n",
            f"**AUC ratio: {self.auc_ratio:.2f}x** | "
            f"**Cmax ratio: {self.cmax_ratio:.2f}x**\n",
            "| Parameter | Victim Alone | Victim + Perpetrator | Ratio |",
            "|-----------|-------------|---------------------|-------|",
            f"| Cmax (mg/L) | {self.pk_victim_alone.Cmax:.4g} | {self.pk_victim_ddi.Cmax:.4g} | {self.cmax_ratio:.2f} |",
            f"| AUC (mg·h/L) | {self.pk_victim_alone.AUC_0_inf:.4g} | {self.pk_victim_ddi.AUC_0_inf:.4g} | {self.auc_ratio:.2f} |",
            f"| t½ (h) | {self.pk_victim_alone.t_half:.2f} | {self.pk_victim_ddi.t_half:.2f} | {self.pk_victim_ddi.t_half/max(self.pk_victim_alone.t_half,0.01):.2f} |",
            f"| CL/F (L/h) | {self.pk_victim_alone.CL_F:.4g} | {self.pk_victim_ddi.CL_F:.4g} | {self.pk_victim_ddi.CL_F/max(self.pk_victim_alone.CL_F,0.01):.2f} |",
            f"\nLiver CYP at victim Tmax: {self.enzyme_liver[np.argmax(self.victim_plasma)]:.2%} of baseline",
        ]

        classification = "No interaction"
        r = self.auc_ratio
        if r >= 5: classification = "Strong inhibition"
        elif r >= 2: classification = "Moderate inhibition"
        elif r >= 1.25: classification = "Weak inhibition"
        elif r <= 0.2: classification = "Strong induction"
        elif r <= 0.5: classification = "Moderate induction"
        elif r <= 0.8: classification = "Weak induction"
        lines.append(f"\n**Classification: {classification}** (FDA 2020)")

        return "\n".join(lines)


def _build_drug_params(compound, phys):
    """Pre-compute ODE parameters for one drug."""
    kp = predict_kp_all(compound)
    kpb = {o: v / compound.R_bp for o, v in kp.items()}

    V = np.zeros(N_STATES_PER_DRUG)
    V[1] = phys.V_venous
    V[2] = phys.V_arterial
    for organ, idx in _ORGAN_IDX.items():
        V[idx] = phys.organ_volumes[organ]

    Q = {}
    for organ in Organ:
        if organ != Organ.LUNG:
            Q[organ] = phys.blood_flows.get(organ, 0)

    kpb_arr = np.zeros(N_STATES_PER_DRUG)
    for organ, idx in _ORGAN_IDX.items():
        kpb_arr[idx] = kpb[organ]

    return {
        "kp": kp, "kpb": kpb, "V": V, "Q": Q, "kpb_arr": kpb_arr,
        "ka": compound.ka, "Fa": compound.Fa, "Fg": compound.Fg,
        "CL_int": compound.CL_int, "CL_renal": compound.CL_renal,
        "fu_p": compound.fu_p, "R_bp": compound.R_bp,
        "Kp_liver": kp[Organ.LIVER], "Kp_kidney": kp[Organ.KIDNEY],
        "Qco": phys.cardiac_output,
        "Q_ha": phys.Q_hepatic_artery,
        "Q_portal": phys.Q_portal,
        "Q_liver_total": phys.Q_liver_total,
        "mw": compound.mw,
    }


def _rhs_single_drug(y, p, offset, enzyme_factor=1.0):
    """RHS for one drug in the two-drug system."""
    dy = np.zeros(N_STATES_PER_DRUG)
    V = p["V"]
    kpb = p["kpb_arr"]
    Qco = p["Qco"]

    C_ven = y[offset + 1] / V[1] if V[1] > 0 else 0
    C_art = y[offset + 2] / V[2] if V[2] > 0 else 0

    def C_out(idx):
        if V[idx] > 0 and kpb[idx] > 0:
            return y[offset + idx] / (V[idx] * kpb[idx])
        return 0

    # Lumen
    dy[0] = -p["ka"] * y[offset + 0]

    # Lung
    dy[3] = Qco * (C_ven - C_out(3))

    # Arterial
    dy[2] = Qco * (C_out(3) - C_art)

    # Non-portal organs
    ven_inflow = 0
    for organ in _NON_PORTAL:
        idx = _ORGAN_IDX[organ]
        Q = p["Q"].get(organ, 0)
        dy[idx] = Q * (C_art - C_out(idx))
        ven_inflow += Q * C_out(idx)

    # Gut — enzyme_factor modulates gut wall CL_int, which shifts Fg.
    # Well-stirred gut: Fg_eff = Fg_base / (Fg_base + (1 - Fg_base) * factor)
    # Derivation: Fg = Q/(Q + fu*CL_int_gut); scaling CL_int_gut by `factor`
    # preserves Fg_eff ∈ (0,1] for any factor ≥ 0 (inhibition AND induction).
    Q_gut = p["Q"][Organ.GUT]
    Fg_base = p["Fg"]
    ef = max(enzyme_factor, 1e-6)
    Fg_eff = Fg_base / (Fg_base + (1.0 - Fg_base) * ef)
    absorption = p["ka"] * y[offset + 0] * Fg_eff
    dy[7] = Q_gut * (C_art - C_out(7)) + absorption

    # Kidney
    Q_kid = p["Q"][Organ.KIDNEY]
    C_plas_kid = y[offset + 9] / (V[9] * p["Kp_kidney"]) if V[9] > 0 and p["Kp_kidney"] > 0 else 0
    dy[9] = Q_kid * (C_art - C_out(9)) - p["CL_renal"] * C_plas_kid

    # Spleen
    Q_spl = p["Q"][Organ.SPLEEN]
    dy[14] = Q_spl * (C_art - C_out(14))

    # Pancreas (portal)
    Q_pan = p["Q"].get(Organ.PANCREAS, 0)
    dy[12] = Q_pan * (C_art - C_out(12))

    # Liver
    C_portal = (Q_gut * C_out(7) + Q_spl * C_out(14) + Q_pan * C_out(12)) / p["Q_portal"] if p["Q_portal"] > 0 else 0
    liver_in = p["Q_ha"] * C_art + p["Q_portal"] * C_portal
    liver_out = p["Q_liver_total"] * C_out(10)

    C_plas_liver = y[offset + 10] / (V[10] * p["Kp_liver"]) if V[10] > 0 and p["Kp_liver"] > 0 else 0
    C_u_liver = p["fu_p"] * C_plas_liver

    # Hepatic metabolism modulated by enzyme_factor
    hepatic_elim = p["CL_int"] * C_u_liver * enzyme_factor
    dy[10] = liver_in - liver_out - hepatic_elim

    # Venous pool
    ven_inflow += p["Q_liver_total"] * C_out(10)
    dy[1] = ven_inflow - Qco * C_ven

    return dy


def simulate_ddi(
    victim: DDIDrugSpec,
    perpetrator: DDIDrugSpec,
    mechanism: DDIMechanism,
    phys: Optional[PhysiologyParams] = None,
    duration_h: float = 72.0,
    n_timepoints: int = 1000,
    n_liver_segments: int = 5,
) -> DDIResult:
    """
    Run dynamic DDI simulation with two drugs.

    Drug mass balance uses a single well-stirred liver tank, but the liver
    enzyme pool is discretized into n_liver_segments CSTRs in series to
    capture the spatial concentration gradient of the perpetrator along
    liver sinusoids. Each segment sees a local C_perp (plug-flow decay from
    inlet), and the victim's effective CL_int accounts for the resulting
    enzyme heterogeneity via the cascaded extraction formula.

    n_liver_segments=1 recovers the classical well-stirred DDI model.
    n_liver_segments=5 (default) matches Simcyp's dispersion model and
    produces stronger (more realistic) DDI magnitudes for high-extraction
    perpetrators (e.g. ketoconazole).

    Args:
        victim: Victim drug specification.
        perpetrator: Perpetrator drug specification.
        mechanism: DDI mechanism parameters.
        phys: Physiology (default: 73kg male).
        duration_h: Total simulation duration.
        n_timepoints: Output time points.
        n_liver_segments: Number of sequential liver enzyme compartments
            (1 = well-stirred, 5 = dispersion).

    References:
        - Ito K et al. Pharm Res 1998;15:1546-1554 (dispersion model)
        - Rowland Yeo K et al. Clin Pharmacokinet 2010;49:651-667
    """
    if phys is None:
        phys = get_physiology()

    v_params = _build_drug_params(victim.compound, phys)
    p_params = _build_drug_params(perpetrator.compound, phys)

    fm = mechanism.fm
    kdeg_liver = mechanism.kdeg_liver
    kdeg_gut = mechanism.kdeg_gut
    N = max(1, int(n_liver_segments))

    # State vector layout (depends on N):
    #   0..15                victim drug (16 states)
    #   16..31               perpetrator drug (16 states)
    #   32..32+N-1           liver enzyme segments (N states)
    #   32+N                 gut enzyme (1 state)
    idx_enz_liver_0 = 2 * N_STATES_PER_DRUG
    idx_enz_gut = idx_enz_liver_0 + N
    n_ddi_states = idx_enz_gut + 1

    # Midpoint z/L for each segment (plug-flow sampling positions)
    z_mid = np.array([(i - 0.5) / N for i in range(1, N + 1)])

    # Dosing schedules
    v_doses = [(victim.first_dose_time_h + i * victim.interval_h, victim.dose_mg)
               for i in range(victim.n_doses)]
    p_doses = [(perpetrator.first_dose_time_h + i * perpetrator.interval_h, perpetrator.dose_mg)
               for i in range(perpetrator.n_doses)]

    def rhs(t, y):
        dy = np.zeros(n_ddi_states)

        E_liver_arr = np.clip(y[idx_enz_liver_0:idx_enz_liver_0 + N], 1e-3, None)
        E_gut = max(y[idx_enz_gut], 0.001)

        # --- Perpetrator liver inlet concentration ---
        V_liver = p_params["V"][10]
        Kp_liver_p = p_params["Kp_liver"]
        kpb_p = p_params["kpb_arr"]

        C_outlet_plasma = 0.0
        if V_liver > 0 and Kp_liver_p > 0:
            C_outlet_plasma = y[16 + 10] / (V_liver * Kp_liver_p)

        C_art_p = y[16 + 2] / p_params["V"][2] if p_params["V"][2] > 0 else 0
        V_gut_p = p_params["V"][7]
        V_spl_p = p_params["V"][14]
        C_gut_out_p = y[16 + 7] / (V_gut_p * kpb_p[7]) if V_gut_p > 0 and kpb_p[7] > 0 else 0
        C_spl_out_p = y[16 + 14] / (V_spl_p * kpb_p[14]) if V_spl_p > 0 and kpb_p[14] > 0 else 0
        Q_gut_p = p_params["Q"].get(Organ.GUT, 0)
        Q_spl_p = p_params["Q"].get(Organ.SPLEEN, 0)
        Q_pv_p = Q_gut_p + Q_spl_p
        C_portal_blood_p = (Q_gut_p * C_gut_out_p + Q_spl_p * C_spl_out_p) / Q_pv_p if Q_pv_p > 0 else 0
        Q_liv_p = p_params["Q_liver_total"]
        C_inlet_blood = ((p_params["Q_ha"] * C_art_p + Q_pv_p * C_portal_blood_p) /
                         Q_liv_p) if Q_liv_p > 0 else 0
        C_inlet_plasma = C_inlet_blood / max(p_params["R_bp"], 0.1)
        C_inlet_u = p_params["fu_p"] * C_inlet_plasma
        C_inlet_u_uM = C_inlet_u * 1000.0 / max(p_params["mw"], 1)

        # --- Plug-flow decay of perpetrator along liver axis ---
        # k = CL_int_perp * fu_perp / Q_liver (perpetrator assumed not self-inhibited)
        CL_int_perp = p_params["CL_int"]
        k_decay = CL_int_perp * p_params["fu_p"] / max(Q_liv_p, 0.01)
        C_perp_local_uM = C_inlet_u_uM * np.exp(-k_decay * z_mid)

        # --- Per-segment reversible inhibition, MBI, induction ---
        if mechanism.Ki and mechanism.Ki > 0:
            rev_arr = 1.0 / (1.0 + C_perp_local_uM / mechanism.Ki)
        else:
            rev_arr = np.ones(N)

        if mechanism.KI and mechanism.kinact:
            inact_arr = mechanism.kinact * C_perp_local_uM / (mechanism.KI + C_perp_local_uM)
        else:
            inact_arr = np.zeros(N)

        if mechanism.Emax and mechanism.EC50:
            ind_arr = 1.0 + mechanism.Emax * C_perp_local_uM / (mechanism.EC50 + C_perp_local_uM)
        else:
            ind_arr = np.ones(N)

        # Liver enzyme ODEs (per segment)
        dy[idx_enz_liver_0:idx_enz_liver_0 + N] = (
            kdeg_liver * (ind_arr - E_liver_arr) - inact_arr * E_liver_arr
        )

        # Gut enzyme driven by inlet (enterocyte concentration ~ portal vein tissue)
        C_gut_u_uM = p_params["fu_p"] * C_gut_out_p * 1000.0 / max(p_params["mw"], 1)
        rev_gut = 1.0 / (1.0 + C_gut_u_uM / mechanism.Ki) if mechanism.Ki else 1.0
        inact_gut = (mechanism.kinact * C_gut_u_uM / (mechanism.KI + C_gut_u_uM)
                     if mechanism.KI and mechanism.kinact else 0.0)
        ind_gut = (1.0 + mechanism.Emax * C_gut_u_uM / (mechanism.EC50 + C_gut_u_uM)
                   if mechanism.Emax and mechanism.EC50 else 1.0)
        dy[idx_enz_gut] = kdeg_gut * (ind_gut - E_gut) - inact_gut * E_gut

        # --- Effective victim CL_int from segmented extraction ---
        # Per-segment extraction: local_CL_seg = (fm*CL_int/N) * E_i * rev_i, fu is applied in ODE.
        # Survival of drug through segment i (intrinsic, no fu yet):
        # S_i = Q / (Q + (fm*CL_int/N)*E_i*rev_i*fu)
        # We fold fu into the effective CL_int below by computing unbound-basis extraction.
        CL_int_v_fm = fm * v_params["CL_int"]
        fu_v = v_params["fu_p"]
        CL_seg_local = (CL_int_v_fm / N) * E_liver_arr * rev_arr * fu_v
        # S_i in terms of unbound extraction; the well-stirred liver ODE below
        # multiplies effective CL_int by the outlet *unbound* concentration,
        # so we need CL_int_eff such that Q*E_extr = CL_int_eff * fu_v * <Cu_out>.
        # For well-stirred single tank: E_extr_WS = (CL_int_eff*fu_v) / (Q + CL_int_eff*fu_v).
        # Equate E_extr_WS = 1 - ∏(Q/(Q+CL_seg_local)):
        S_prod = np.prod(Q_liv_p / (Q_liv_p + CL_seg_local))
        E_extr = 1.0 - S_prod
        if E_extr >= 0.9999:
            CL_int_eff_v = 1e9
        elif E_extr <= 0.0:
            CL_int_eff_v = 0.0
        else:
            # CL_int_eff * fu_v = Q * E_extr / (1 - E_extr)
            CL_int_eff_v = Q_liv_p * E_extr / ((1.0 - E_extr) * max(fu_v, 1e-9))

        # Fraction of victim CL NOT going through the modulated enzyme keeps its baseline rate
        # Overall effective CL_int = CL_eff_modulated + (1-fm) * CL_int_v
        CL_int_v_eff = CL_int_eff_v + (1.0 - fm) * v_params["CL_int"]

        # --- Victim drug RHS with modified liver CL ---
        # We can't just pass a multiplier anymore — build victim RHS inline,
        # reusing _rhs_single_drug but with enzyme_factor mapping handled here.
        # Solution: temporarily swap v_params["CL_int"] via a helper factor.
        # Simplest: pass enzyme_factor = CL_int_v_eff / CL_int_v (preserves API).
        CL_int_v_base = max(v_params["CL_int"], 1e-9)
        liver_enzyme_factor = CL_int_v_eff / CL_int_v_base

        # Gut enzyme acts on Fg via well-stirred gut formula
        gut_enzyme_factor = fm * E_gut * rev_gut + (1.0 - fm)

        # Victim drug RHS (modulated by liver + gut)
        dy_v = _rhs_single_drug(y, v_params, 0, liver_enzyme_factor)
        # Overwrite gut ODE with the dedicated gut enzyme factor
        Q_gut_v = v_params["Q"][Organ.GUT]
        Fg_base_v = v_params["Fg"]
        gef_v = max(gut_enzyme_factor, 1e-6)
        Fg_eff_v = Fg_base_v / (Fg_base_v + (1.0 - Fg_base_v) * gef_v)
        abs_v = v_params["ka"] * y[0] * Fg_eff_v
        # Recompute gut ODE using gut enzyme factor
        V_v = v_params["V"]
        kpb_v = v_params["kpb_arr"]
        C_art_v = y[2] / V_v[2] if V_v[2] > 0 else 0
        C_gut_out_v = y[7] / (V_v[7] * kpb_v[7]) if V_v[7] > 0 and kpb_v[7] > 0 else 0
        dy_v[7] = Q_gut_v * (C_art_v - C_gut_out_v) + abs_v
        dy[:N_STATES_PER_DRUG] = dy_v

        # Perpetrator drug RHS (unmodulated)
        dy_p = _rhs_single_drug(y, p_params, N_STATES_PER_DRUG, 1.0)
        dy[N_STATES_PER_DRUG:2*N_STATES_PER_DRUG] = dy_p

        return dy

    # Initial conditions
    y0 = np.zeros(n_ddi_states)
    y0[idx_enz_liver_0:idx_enz_liver_0 + N] = 1.0  # baseline enzyme = 100%
    y0[idx_enz_gut] = 1.0

    # All dose times
    all_dose_times = sorted(set(
        [t for t, _ in v_doses] + [t for t, _ in p_doses] + [0, duration_h]
    ))

    t_eval = np.linspace(0, duration_h, n_timepoints)
    current_y = y0.copy()
    all_t = []
    all_y = []

    for seg_i in range(len(all_dose_times) - 1):
        t_start = all_dose_times[seg_i]
        t_end = all_dose_times[seg_i + 1]

        # Apply doses at t_start
        for dose_t, dose_mg in v_doses:
            if abs(dose_t - t_start) < 0.001:
                if victim.route == "oral":
                    current_y[0] += dose_mg * v_params["Fa"]
                else:
                    current_y[1] += dose_mg

        for dose_t, dose_mg in p_doses:
            if abs(dose_t - t_start) < 0.001:
                if perpetrator.route == "oral":
                    current_y[16 + 0] += dose_mg * p_params["Fa"]
                else:
                    current_y[16 + 1] += dose_mg

        seg_eval = t_eval[(t_eval >= t_start) & (t_eval < t_end)]
        if len(seg_eval) == 0:
            seg_eval = np.array([t_start, t_end])

        # Tighten max_step under strong induction to preserve mass balance
        # (effective CL can become very large, shrinking liver time constant)
        mech_max_step = 0.1
        if mechanism.Emax and mechanism.Emax > 3.0:
            mech_max_step = 0.02

        sol = solve_ivp(rhs, [t_start, t_end], current_y,
                       method="BDF", t_eval=seg_eval,
                       rtol=1e-8, atol=1e-11, max_step=mech_max_step)

        if sol.success:
            all_t.append(sol.t)
            all_y.append(sol.y)
            current_y = sol.y[:, -1].copy()

    time = np.concatenate(all_t)
    y_full = np.concatenate(all_y, axis=1)
    _, uniq = np.unique(time, return_index=True)
    time = time[uniq]
    y_full = y_full[:, uniq]

    # Extract results
    R_bp_v = victim.compound.R_bp
    R_bp_p = perpetrator.compound.R_bp
    V_ven = v_params["V"][1]

    victim_plasma = y_full[1, :] / V_ven / R_bp_v if V_ven > 0 else np.zeros_like(time)
    perpetrator_plasma = y_full[17, :] / V_ven / R_bp_p if V_ven > 0 else np.zeros_like(time)
    # Average enzyme over N segments (for reporting)
    enzyme_liver_segs = y_full[idx_enz_liver_0:idx_enz_liver_0 + N, :]
    enzyme_liver = enzyme_liver_segs.mean(axis=0) if N > 0 else y_full[idx_enz_liver_0, :]
    enzyme_gut = y_full[idx_enz_gut, :]

    # Victim PK with DDI
    # Find victim dose time to calculate PK from that point
    v_start = victim.first_dose_time_h
    mask = time >= v_start
    pk_ddi = calculate_pk_parameters(
        time[mask] - v_start, victim_plasma[mask], victim.dose_mg)

    # Control: victim alone — re-run the SAME segmented DDI ODE with perpetrator
    # dose=0 so that baseline CL matches the victim+DDI simulation exactly.
    # (Using a plain well-stirred PBPKModel here would introduce a spurious
    # bias because segmented extraction ≠ well-stirred at baseline.)
    perp_zero = DDIDrugSpec(
        compound=perpetrator.compound, dose_mg=0.0, route=perpetrator.route,
        n_doses=perpetrator.n_doses, interval_h=perpetrator.interval_h,
        first_dose_time_h=perpetrator.first_dose_time_h,
    )
    # Recursion guard: call simulate_ddi with no mechanism effect implicitly via dose=0
    # But to avoid infinite recursion or overhead, inline a minimal control here:
    y0_ctrl = np.zeros(n_ddi_states)
    y0_ctrl[idx_enz_liver_0:idx_enz_liver_0 + N] = 1.0
    y0_ctrl[idx_enz_gut] = 1.0
    current_y_c = y0_ctrl.copy()
    all_t_c = []; all_y_c = []
    for seg_i in range(len(all_dose_times) - 1):
        t_start = all_dose_times[seg_i]; t_end = all_dose_times[seg_i + 1]
        for dose_t, dose_mg in v_doses:
            if abs(dose_t - t_start) < 0.001:
                if victim.route == "oral":
                    current_y_c[0] += dose_mg * v_params["Fa"]
                else:
                    current_y_c[1] += dose_mg
        seg_eval = t_eval[(t_eval >= t_start) & (t_eval < t_end)]
        if len(seg_eval) == 0:
            seg_eval = np.array([t_start, t_end])
        sol = solve_ivp(rhs, [t_start, t_end], current_y_c,
                       method="BDF", t_eval=seg_eval,
                       rtol=1e-8, atol=1e-11, max_step=mech_max_step)
        if sol.success:
            all_t_c.append(sol.t); all_y_c.append(sol.y)
            current_y_c = sol.y[:, -1].copy()
    time_ctrl = np.concatenate(all_t_c)
    y_full_ctrl = np.concatenate(all_y_c, axis=1)
    _, uniq_c = np.unique(time_ctrl, return_index=True)
    time_ctrl = time_ctrl[uniq_c]; y_full_ctrl = y_full_ctrl[:, uniq_c]
    victim_alone_plasma_raw = y_full_ctrl[1, :] / V_ven / R_bp_v if V_ven > 0 else np.zeros_like(time_ctrl)
    mask_c = time_ctrl >= v_start
    pk_alone = calculate_pk_parameters(
        time_ctrl[mask_c] - v_start, victim_alone_plasma_raw[mask_c], victim.dose_mg)

    pk_perp = calculate_pk_parameters(
        time, perpetrator_plasma, perpetrator.dose_mg)

    return DDIResult(
        time=time,
        victim_plasma=victim_plasma,
        perpetrator_plasma=perpetrator_plasma,
        enzyme_liver=enzyme_liver,
        enzyme_gut=enzyme_gut,
        victim_alone_plasma=np.interp(time, time_ctrl, victim_alone_plasma_raw),
        pk_victim_ddi=pk_ddi,
        pk_victim_alone=pk_alone,
        pk_perpetrator=pk_perp,
    )
