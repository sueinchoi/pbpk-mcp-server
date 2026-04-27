"""
MCP tool definitions for PBPK modeling server.

Tools:
  - predict_kp: Predict tissue:plasma partition coefficients
  - run_pbpk_simulation: Run whole-body PBPK simulation
  - calculate_pk_parameters: Calculate NCA PK parameters from simulation
  - predict_hepatic_clearance: Well-stirred hepatic clearance model
  - list_physiology: Display physiological parameters
  - list_compounds: Show available reference compounds
  - plot_concentration: Generate concentration-time plot
"""

import json
import os
import tempfile
from typing import Optional

from mcp.server.fastmcp import FastMCP

from core.compound import (
    CompoundSpec,
    CompoundType,
    MetabolismModel,
    COMPOUND_LIBRARY,
)
from core.physiology import (
    Organ,
    Sex,
    PhysiologyParams,
    get_physiology,
    TISSUE_COMPOSITION,
    PLASMA_COMPOSITION,
)
from core.partition_coeff import (
    KpMethod,
    predict_kp_all,
    predict_kpb_all,
    format_kpb_table,
    format_kp_comparison_table,
)
from core.tissue_binding import (
    predict_all_fu,
    predict_fu_inc,
    format_fu_table,
)
from core.pbpk_model import (
    PBPKModel,
    DosingProtocol,
    SimulationConfig,
    SimulationResult,
    Route,
    DistributionModel,
)
from core.pk_calculator import calculate_pk_parameters
from core.ivive import (
    scale_microsomal_clint,
    scale_hepatocyte_clint,
    scale_recombinant_clint,
    format_ivive_result,
)
from core.fg_prediction import (
    predict_fg_qgut,
    predict_fg_wellstirred,
    scale_gut_clint_from_liver,
    peff_from_caco2,
    predict_fa_cat,
)
from core.acat import (
    simulate_acat_standalone,
    format_acat_result,
    FormulationSpec,
)
from core.lymphatic import (
    estimate_lymphatic_fraction,
    oral_bioavailability_with_lymph,
)
from core.protein_binding import (
    single_site_albumin,
    two_site_albumin_aag,
    format_binding_profile,
)
from core.population import run_population_simulation, PopulationResult
from core.species import (
    Species, SPECIES_DATA, scale_preclinical_to_human, format_species_comparison,
    fu_corrected_allometry, vertical_allometry_brain_weight,
    mahmood_rule_of_exponents,
    BRAIN_WEIGHT_G, MLP_YEARS,
)
from core.disease_states import (
    CKDStage, ChildPugh, format_disease_profile,
)
from core.sensitivity import compute_gof, local_sensitivity
from core.time_varying import (
    format_pregnancy_profile,
    circadian_enzyme_factor,
    CIRCADIAN_CYP,
)
from core.ddi_dynamic import (
    DDIDrugSpec, DDIMechanism, simulate_ddi,
)
from core.drug_lookup import lookup_drug
# Lazy imports (heavy deps: sqlite, csv, xml) — loaded only when tool invoked:
#   core.pksim_db, core.data_fitting, core.pksim_import
from core.transporters import (
    OrganTransporters, TransporterSpec, TransportDirection,
    statins_liver_transporters, renal_transporters, gut_efflux_transporters,
    compute_organ_transport_fluxes, format_transporter_profile,
)
from core.hepatic_models import (
    compare_hepatic_models,
    predict_rbp,
    ddi_reversible_inhibition,
    ddi_mechanism_based_inhibition,
    ddi_induction,
    ddi_net_effect,
)
from core.validation import (
    validate_run_pbpk_inputs,
    format_warnings_block,
)


def register_pbpk_tools(mcp: FastMCP):
    """Register all PBPK tools with the MCP server."""

    @mcp.tool()
    def predict_kp(
        name: str = "",
        logP: float = 0.0,
        pKa: float = 7.0,
        fu_p: float = 1.0,
        compound_type: str = "neutral",
        R_bp: float = 1.0,
        mw: float = 300.0,
        method: str = "rodgers_rowland",
    ) -> str:
        """
        Predict tissue:plasma partition coefficients (Kp).

        ANTI-FABRICATION: do not pass placeholder or "typical" physicochemical
        values. The server enforces physiological ranges; out-of-range
        rejects. If the user has not supplied logP/pKa/fu_p/R_bp, ask
        them or use `name` of a library compound — do not guess.

        Five methods available:
          - rodgers_rowland: R&R (2005/2006), mechanistic, Ka_AP from RBC
          - schmitt: Schmitt (2008), 3 lipid sub-fractions, AP binding 20x for bases
          - poulin_theil: PT (2002), simple, original method
          - berezhkovskiy: PTB (2004), corrected Poulin-Theil
          - pksim_standard: PK-Sim (Willmann 2003), simplest, no pKa

        Provide a known compound name or physicochemical properties.

        Args:
            name: Compound name (from library) or custom name.
            logP: Octanol:water partition coefficient (log, unionized).
            pKa: Dissociation constant.
            fu_p: Fraction unbound in plasma (0-1).
            compound_type: strong_base, moderate_base, weak_base, acid, neutral, zwitterion.
            R_bp: Blood:plasma ratio.
            mw: Molecular weight (g/mol).
            method: Kp prediction method (see above).

        Returns:
            Markdown table of Kp values for all 13 tissues.
        """
        from core.validation import require_compound_input
        require_compound_input(name=name, library=COMPOUND_LIBRARY,
                               logP=logP, pKa=pKa, fu_p=fu_p, mw=mw,
                               tool_name="predict_kp")
        name = name or ""   # defensive: None → ""
        if name.lower() in COMPOUND_LIBRARY:
            compound = COMPOUND_LIBRARY[name.lower()]
        else:
            ctype = CompoundType(compound_type)
            compound = CompoundSpec(
                name=name or "Custom Compound",
                mw=mw, logP=logP, pKa=pKa,
                fu_p=fu_p, compound_type=ctype, R_bp=R_bp,
            )

        kp_method = KpMethod(method)
        kp = predict_kp_all(compound, kp_method)
        kpb = predict_kpb_all(compound, kp_method)

        method_names = {
            "rodgers_rowland": "Rodgers & Rowland (2005/2006)",
            "schmitt": "Schmitt (2008)",
            "poulin_theil": "Poulin & Theil (2002)",
            "berezhkovskiy": "Berezhkovskiy (2004)",
            "pksim_standard": "PK-Sim Standard (Willmann 2003)",
        }
        result = format_kpb_table(kp, kpb, compound.name, compound.R_bp, method_names.get(method, method))

        result += f"\n\n### Input Parameters\n"
        result += f"- Name: {compound.name}\n"
        result += f"- MW: {compound.mw} g/mol\n"
        result += f"- logP: {compound.logP}\n"
        result += f"- pKa: {compound.pKa}\n"
        result += f"- fu,p: {compound.fu_p}\n"
        result += f"- Type: {compound.compound_type.value}\n"
        result += f"- R_bp: {compound.R_bp}\n"
        result += f"- **Method: {method_names.get(method, method)}**\n"

        return result

    @mcp.tool()
    def compare_kp_methods(
        name: str = "",
        logP: float = 0.0,
        pKa: float = 7.0,
        fu_p: float = 1.0,
        compound_type: str = "neutral",
        R_bp: float = 1.0,
        mw: float = 300.0,
    ) -> str:
        """
        Compare Kp predictions across ALL 5 methods side by side.

        Useful for understanding method differences and choosing
        the most appropriate method for your compound.

        Methods compared:
          R&R (Rodgers-Rowland), Schmitt, PT (Poulin-Theil),
          PTB (Berezhkovskiy), PK-Sim Standard

        Args:
            name: Compound name (from library) or custom name.
            logP, pKa, fu_p, compound_type, R_bp, mw: Compound properties.

        Returns:
            Side-by-side comparison table.
        """
        from core.validation import require_compound_input
        require_compound_input(name=name, library=COMPOUND_LIBRARY,
                               logP=logP, pKa=pKa, fu_p=fu_p, mw=mw,
                               tool_name="compare_kp_methods")
        if name.lower() in COMPOUND_LIBRARY:
            compound = COMPOUND_LIBRARY[name.lower()]
        else:
            compound = CompoundSpec(
                name=name or "Custom Compound",
                mw=mw, logP=logP, pKa=pKa,
                fu_p=fu_p, compound_type=CompoundType(compound_type), R_bp=R_bp,
            )

        return format_kp_comparison_table(compound)

    @mcp.tool()
    def predict_tissue_binding(
        name: str = "",
        logP: float = 0.0,
        pKa: float = 7.0,
        fu_p: float = 1.0,
        compound_type: str = "neutral",
        R_bp: float = 1.0,
        mw: float = 300.0,
        method: str = "rodgers_rowland",
    ) -> str:
        """
        Predict tissue binding parameters (fu_tissue, fu_interstitial,
        fu_intracellular, fu_gut, fu_inc).

        Used for:
          - Permeability-limited PBPK models (fu_int, fu_cell per organ)
          - IVIVE: fu_inc for in vitro microsomal CLint correction
          - First-pass gut metabolism: fu_gut

        Args:
            name: Compound name or custom.
            logP, pKa, fu_p, compound_type, R_bp, mw: Compound properties.
            method: Kp method for fu_tissue derivation.

        Returns:
            Markdown table of all tissue binding parameters.
        """
        from core.validation import require_compound_input
        require_compound_input(name=name, library=COMPOUND_LIBRARY,
                               logP=logP, pKa=pKa, fu_p=fu_p, mw=mw,
                               tool_name="predict_tissue_binding")
        if name.lower() in COMPOUND_LIBRARY:
            compound = COMPOUND_LIBRARY[name.lower()]
        else:
            compound = CompoundSpec(
                name=name or "Custom Compound",
                mw=mw, logP=logP, pKa=pKa,
                fu_p=fu_p, compound_type=CompoundType(compound_type), R_bp=R_bp,
            )

        return format_fu_table(compound, KpMethod(method))

    @mcp.tool()
    def run_pbpk_simulation(
        name: str = "",
        dose_mg: float = 100.0,
        route: str = "oral",
        duration_h: float = 24.0,
        n_doses: int = 1,
        interval_h: float = 24.0,
        infusion_duration_h: float = 0.5,
        body_weight: float = 73.0,
        sex: str = "male",
        age: float = 30.0,
        distribution_model: str = "perfusion_limited",
        logP: float = 0.0,
        pKa: float = 7.0,
        fu_p: float = 1.0,
        compound_type: str = "neutral",
        R_bp: float = 1.0,
        mw: float = 300.0,
        ka: float = 1.0,
        Fa: float = 1.0,
        Fg: float = 1.0,
        CL_int: float = 0.0,
        CL_renal: float = 0.0,
        Vmax: Optional[float] = None,
        Km: Optional[float] = None,
        absorption_model: str = "first_order",
        Peff: Optional[float] = None,
        S0: Optional[float] = None,
        particle_radius_um: float = 25.0,
        CLint_gut: float = 0.0,
        gut_cyp_clint: Optional[str] = None,
        fm_per_cyp: Optional[str] = None,
        # --- Clearance source (alternative to direct CL_int) ---
        clearance_source: str = "direct",
        CLint_vitro_hlm: Optional[float] = None,
        CLint_vitro_hep: Optional[float] = None,
        CLint_per_cyp: Optional[str] = None,
        protein_conc: float = 1.0,
        # --- Transporter parameters (JSON-like string) ---
        liver_oatp_km: Optional[float] = None,
        liver_oatp_vmax: Optional[float] = None,
        liver_mrp2_km: Optional[float] = None,
        liver_mrp2_vmax: Optional[float] = None,
        kidney_oct2_km: Optional[float] = None,
        kidney_oct2_vmax: Optional[float] = None,
        kidney_mate1_km: Optional[float] = None,
        kidney_mate1_vmax: Optional[float] = None,
        gut_pgp_km: Optional[float] = None,
        gut_pgp_vmax: Optional[float] = None,
        # --- Partition coefficient method ---
        kp_method: str = "rodgers_rowland",
    ) -> str:
        """
        Run a whole-body PBPK simulation (PK-Sim/Simcyp-style).

        The model uses 13 tissue compartments + arterial/venous blood pools.
        Provide a known compound name from the library or custom properties.

        Clearance sources (choose one):
          - direct: provide CL_int (L/h) directly (default)
          - hlm: provide CLint_vitro_hlm (µL/min/mg) → IVIVE scaled
          - hepatocyte: provide CLint_vitro_hep (µL/min/10^6 cells)
          - rcyp: provide CLint_per_cyp as "CYP3A4:0.1,CYP2D6:0.05"

        Transporter parameters (all optional, µM and pmol/min):
          Liver: liver_oatp_km/vmax (OATP1B1 uptake), liver_mrp2_km/vmax (biliary)
          Kidney: kidney_oct2_km/vmax (uptake), kidney_mate1_km/vmax (efflux)
          Gut: gut_pgp_km/vmax (P-gp apical efflux)

        Absorption models:
          - first_order: simple ka-based (default)
          - acat: 9-segment GI with dissolution, pH-dependent solubility

        Args:
            name: Compound name (from library: midazolam, metformin,
                  theophylline, diazepam, warfarin, caffeine) or custom name.
            dose_mg: Dose amount in mg.
            route: Dosing route: "oral", "iv_bolus", or "iv_infusion".
            duration_h: Simulation duration in hours.
            n_doses: Number of doses for multiple dosing.
            interval_h: Dosing interval in hours (for multiple doses).
            infusion_duration_h: IV infusion duration in hours.
            body_weight: Subject body weight in kg.
            sex: "male" or "female".
            distribution_model: "perfusion_limited" (default) or "permeability_limited".
            logP: Octanol:water logP (if custom compound).
            pKa: pKa (if custom compound).
            fu_p: Fraction unbound in plasma (if custom compound).
            compound_type: Compound type (if custom compound).
            R_bp: Blood:plasma ratio (if custom compound).
            mw: Molecular weight (if custom compound).
            ka: Absorption rate constant in 1/h (oral).
            Fa: Fraction absorbed (oral).
            Fg: Fraction escaping gut wall metabolism (oral).
            CL_int: Intrinsic hepatic clearance in L/h.
            CL_renal: Renal clearance in L/h.
            Vmax: Michaelis-Menten Vmax in mg/h (optional).
            Km: Michaelis-Menten Km in mg/L (optional).
            gut_cyp_clint: Per-CYP gut wall CLint (direct), e.g.:
                "CYP3A4:30,CYP2C9:5,UGT:10" (each in L/h).
            fm_per_cyp: Hepatic CYP fraction metabolized, e.g.:
                "CYP3A4:0.8,CYP2C9:0.15,UGT:0.05"
                → gut CLint per CYP auto-calculated from liver CLint
                using relative gut/liver enzyme content (Paine 2006).
                This is the RECOMMENDED input when gut CLint is unknown.
            kp_method: Partition coefficient method. One of:
                "rodgers_rowland" (default), "lukacova", "schmitt",
                "poulin_theil", "berezhkovskiy", "pksim_standard", "kp_membrane".
                For lipophilic bases (logP>3), "poulin_theil" typically gives
                better Vss prediction; "rodgers_rowland" is the standard default.

        ## Anti-fabrication

        Do NOT call this tool with placeholder, "typical", or estimated
        physicochemical or clearance values. If a measurement (RED for
        fu_inc/fu_hep, Bp/p assay for R_bp, Caco-2 for Peff, hepatocyte
        depletion for CLint) is unavailable for any required parameter,
        ask the user explicitly first. The server enforces physiological
        ranges and rejects out-of-range values. Sentinel defaults
        (fu_p=1.0, R_bp=1.0, CL_int=0) trigger soft warnings. Library
        compound lookup ignores any custom physicochemical values you
        pass alongside `name` — it warns, but the data still travels.

        Returns:
            Markdown summary with PK parameters and key concentrations.
        """
        # --- Resolve CL_int from clearance source ---
        cl_int_resolved = CL_int
        ivive_info = ""

        # Capture inputs for audit log (before validation transforms)
        _audit_inputs = {
            "name": name, "dose_mg": dose_mg, "route": route,
            "duration_h": duration_h, "n_doses": n_doses,
            "distribution_model": distribution_model,
            "kp_method": kp_method,
            "clearance_source": clearance_source,
            "CL_int": CL_int, "CLint_vitro_hlm": CLint_vitro_hlm,
            "CLint_vitro_hep": CLint_vitro_hep, "CLint_per_cyp": CLint_per_cyp,
            "logP": logP, "pKa": pKa, "fu_p": fu_p, "R_bp": R_bp, "mw": mw,
            "compound_type": compound_type, "body_weight": body_weight,
            "sex": sex, "age": age,
        }

        # Pre-IVIVE: hard validation of clearance_source vs provided inputs
        # AND the new clearance discriminated union (catches incomplete pairs).
        from core.validation import (
            validate_kp_method, validate_clearance_source_mismatch,
        )
        from core.clearance_spec import parse_clearance_from_legacy_args
        from core.transporter_spec import TransporterKwargs

        validate_kp_method(kp_method)
        validate_clearance_source_mismatch(
            clearance_source, CL_int, CLint_vitro_hlm,
            CLint_vitro_hep, CLint_per_cyp,
        )
        # Hard validation: absorption_model enum (typos like
        # 'first-order' previously silently fell back to first_order)
        from core.validation import validate_absorption_model, parse_cyp_dict
        validate_absorption_model(absorption_model)
        # Eager parse of CYP-dict strings — even if the downstream branch
        # would skip them, malformed input must be flagged loudly so the
        # user does not see "succeeded" when their CYP data was dropped.
        if fm_per_cyp:
            parse_cyp_dict(fm_per_cyp, parameter_name="fm_per_cyp")
        if gut_cyp_clint:
            parse_cyp_dict(gut_cyp_clint, parameter_name="gut_cyp_clint")
        # Schema-level: clearance discriminated union (re-validates ranges,
        # ensures the chosen source has its required input)
        try:
            _clearance_spec = parse_clearance_from_legacy_args(
                clearance_source=clearance_source,
                CL_int=CL_int, CLint_vitro_hlm=CLint_vitro_hlm,
                CLint_vitro_hep=CLint_vitro_hep,
                CLint_per_cyp=CLint_per_cyp,
                protein_conc=protein_conc,
            )
        except Exception as e:
            raise ValueError(f"Clearance schema validation failed: {e}") from e

        # Schema-level: transporter pairing (Km xor Vmax → reject)
        _transporter_spec = TransporterKwargs.from_legacy_kwargs(
            liver_oatp_km=liver_oatp_km, liver_oatp_vmax=liver_oatp_vmax,
            liver_mrp2_km=liver_mrp2_km, liver_mrp2_vmax=liver_mrp2_vmax,
            kidney_oct2_km=kidney_oct2_km, kidney_oct2_vmax=kidney_oct2_vmax,
            kidney_mate1_km=kidney_mate1_km, kidney_mate1_vmax=kidney_mate1_vmax,
            gut_pgp_km=gut_pgp_km, gut_pgp_vmax=gut_pgp_vmax,
        )

        if clearance_source == "hlm" and CLint_vitro_hlm is not None:
            from core.ivive import scale_microsomal_clint
            ivive_r = scale_microsomal_clint(CLint_vitro_hlm, logP=logP,
                protein_conc=protein_conc, body_weight=body_weight, sex=sex)
            cl_int_resolved = ivive_r["CLint_in_vivo_L_per_h"]
            ivive_info = f"IVIVE (HLM): {CLint_vitro_hlm} µL/min/mg → {cl_int_resolved:.1f} L/h"

        elif clearance_source == "hepatocyte" and CLint_vitro_hep is not None:
            from core.ivive import scale_hepatocyte_clint
            ivive_r = scale_hepatocyte_clint(CLint_vitro_hep, logP=logP,
                body_weight=body_weight, sex=sex)
            cl_int_resolved = ivive_r["CLint_in_vivo_L_per_h"]
            ivive_info = f"IVIVE (Hepatocyte): {CLint_vitro_hep} µL/min/10^6 → {cl_int_resolved:.1f} L/h"

        elif clearance_source == "rcyp" and CLint_per_cyp is not None:
            from core.ivive import scale_recombinant_clint
            from core.validation import parse_cyp_dict
            # Strict parsing — malformed entries raise (was: silently skipped)
            cyp_dict = parse_cyp_dict(CLint_per_cyp, parameter_name="CLint_per_cyp")
            ivive_r = scale_recombinant_clint(cyp_dict, body_weight=body_weight, sex=sex)
            cl_int_resolved = ivive_r["CLint_in_vivo_L_per_h"]
            ivive_info = f"IVIVE (rCYP): {cyp_dict} → {cl_int_resolved:.1f} L/h"

        # --- Build transporter dict (schema-validated above) ---
        transporter_dict = _transporter_spec.to_organ_transporters()

        # --- Build compound ---
        if name.lower() in COMPOUND_LIBRARY:
            compound = COMPOUND_LIBRARY[name.lower()]
        else:
            metabolism = MetabolismModel.FIRST_ORDER
            if Vmax is not None and Km is not None:
                metabolism = MetabolismModel.MICHAELIS_MENTEN

            compound = CompoundSpec(
                name=name or "Custom Compound",
                mw=mw, logP=logP, pKa=pKa,
                fu_p=fu_p, compound_type=CompoundType(compound_type),
                R_bp=R_bp, ka=ka, Fa=Fa, Fg=Fg,
                CL_int=cl_int_resolved, CL_renal=CL_renal,
                metabolism_model=metabolism, Vmax=Vmax, Km=Km,
                Peff=Peff, S0=S0, particle_radius_um=particle_radius_um,
                CLint_gut=CLint_gut,
            )

        # --- Build physiology ---
        phys = get_physiology(body_weight=body_weight, sex=Sex(sex), age_years=age)

        # --- Partition coefficients ---
        kp_override = None
        kp_warning = ""
        # If the user kept the default kp_method AND a library compound
        # carries a recommended_kp_method, USE the recommended method
        # (do not just emit a tip — that was the silent fallback in
        # earlier versions). The user is explicitly told this happened.
        user_kp_explicit = (kp_method != "rodgers_rowland")
        if (not user_kp_explicit
            and getattr(compound, "recommended_kp_method", None)
            and compound.recommended_kp_method != "rodgers_rowland"):
            kp_method = compound.recommended_kp_method  # ← effective override
            kp_warning = (
                f"\n> ℹ️ **Auto-selected Kp method:** `{kp_method}` "
                f"(library default for `{compound.name}`). "
                f"R&R was NOT used; the library specifies a more "
                f"accurate method for this compound class. To force "
                f"R&R, pass `kp_method=\"rodgers_rowland\"` explicitly.\n"
            )
        # kp_method already validated above — direct enum construction is safe
        kp_method_enum = KpMethod(kp_method)
        if kp_method_enum != KpMethod.RODGERS_ROWLAND:
            kp_override = predict_kp_all(compound, kp_method_enum)

        # --- Soft warnings: collect AFTER cl_int_resolved is finalized ---
        in_library_match = bool(name and name.lower() in COMPOUND_LIBRARY)
        user_overrides = {}
        if in_library_match:
            # Detect which custom params the user supplied that the
            # library lookup will ignore (uses tool-default sentinels)
            override_candidates = {
                "logP": (logP, 0.0), "pKa": (pKa, 7.0), "fu_p": (fu_p, 1.0),
                "R_bp": (R_bp, 1.0), "mw": (mw, 300.0),
                "ka": (ka, 1.0), "Fa": (Fa, 1.0), "Fg": (Fg, 1.0),
                "CL_int": (CL_int, 0.0), "CL_renal": (CL_renal, 0.0),
            }
            user_overrides = {
                k: v for k, v in override_candidates.items() if v[0] != v[1]
            }
        soft_warnings = validate_run_pbpk_inputs(
            name=name, library=COMPOUND_LIBRARY,
            distribution_model=distribution_model, kp_method=kp_method,
            clearance_source=clearance_source,
            CL_int=CL_int, CLint_vitro_hlm=CLint_vitro_hlm,
            CLint_vitro_hep=CLint_vitro_hep, CLint_per_cyp=CLint_per_cyp,
            CL_renal=CL_renal,
            has_transporters=bool(transporter_dict),
            fu_p=compound.fu_p, R_bp=compound.R_bp,
            cl_int_resolved=cl_int_resolved,
            compound_type=compound.compound_type.value,
            user_overrides=user_overrides,
            # Range-checked numeric params (None if user didn't supply)
            mw=(mw if mw and mw != 300.0 else None),
            logP=(logP if logP != 0.0 else None),
            pKa=(pKa if pKa != 7.0 else None),
            ka=(ka if ka != 1.0 else None),
            Fa=Fa, Fg=Fg, Peff=Peff, Vmax=Vmax, Km=Km,
            dose_mg=dose_mg, duration_h=duration_h,
            n_doses=n_doses, interval_h=interval_h,
            body_weight=body_weight, age=age,
            route=route, sex=sex,
        )

        # --- Build model (with transporters if provided) ---
        model = PBPKModel(compound, phys,
                          kp_override=kp_override,
                          transporters=transporter_dict if transporter_dict else None)
        dosing = DosingProtocol(
            dose_mg=dose_mg,
            route=Route(route),
            n_doses=n_doses,
            interval_h=interval_h,
            infusion_duration_h=infusion_duration_h,
        )
        # Parse gut CYP CLint — direct or derived from liver fm
        from core.validation import parse_cyp_dict
        parsed_gut_cyp = None
        if gut_cyp_clint:
            # Direct gut CYP CLint input (L/h per CYP); strict parsing
            parsed_gut_cyp = parse_cyp_dict(gut_cyp_clint,
                                            parameter_name="gut_cyp_clint")

        elif fm_per_cyp and (CLint_vitro_hlm or CL_int > 0 or cl_int_resolved > 0):
            # Auto-derive gut CLint from liver CLint + fm per CYP
            from core.fg_prediction import scale_gut_clint_per_cyp
            fm_dict = parse_cyp_dict(fm_per_cyp, parameter_name="fm_per_cyp")
            # Use HLM CLint if available, otherwise back-calculate from in vivo
            if CLint_vitro_hlm:
                hlm_clint = CLint_vitro_hlm
            else:
                # Back-calculate approximate HLM CLint from in vivo
                # CLint_vivo = CLint_hlm / fu_inc * MPPGL * LW * 60/1e6
                # → CLint_hlm ≈ CLint_vivo * 1e6 / (60 * MPPGL * LW)
                hlm_clint = cl_int_resolved * 1e6 / (60.0 * 40.0 * 1800.0) if cl_int_resolved > 0 else 0
            parsed_gut_cyp = scale_gut_clint_per_cyp(hlm_clint, fm_dict)
            ivive_info += f"\n**Gut CLint (auto):** " + ", ".join(
                f"{k}={v:.3f} L/h" for k, v in parsed_gut_cyp.items())

        sim_config = SimulationConfig(
            duration_h=duration_h,
            distribution_model=DistributionModel(distribution_model),
            absorption_model=absorption_model,
            gut_cyp_clint=parsed_gut_cyp,
        )

        # Run simulation
        result = model.simulate(dosing, sim_config)

        # Post-simulation mass-balance assertion (Mass-balance invariant).
        # Failure aborts with explicit error rather than emitting a warning.
        from core.invariants import check_dose_recovery as _check_dose_recovery
        _dose_violation = _check_dose_recovery(
            result=result, model=model,
            dose_mg=dose_mg, n_doses=n_doses, route=route,
            tolerance=0.01,
        )
        if _dose_violation is not None:
            from core.invariants import raise_on_violations
            raise_on_violations([_dose_violation])

        # Calculate PK parameters
        # For multi-dose, AUC(0-t) covers all administered doses; pass total dose
        # so CL/F = total_dose / total_AUC under linear PK (Vss still correct).
        is_iv = route in ("iv_bolus", "iv_infusion")
        effective_dose_for_nca = dose_mg * max(n_doses, 1)
        pk = calculate_pk_parameters(
            result.time,
            result.venous_plasma,
            dose_mg=effective_dose_for_nca,
            is_iv=is_iv,
        )

        # Generate plot
        plot_path = _generate_plot(result, pk, compound.name)

        # Format output
        output = []
        output.append(f"# PBPK Simulation — {compound.name}")
        # Surface input warnings prominently, immediately after the title
        if soft_warnings:
            output.append(format_warnings_block(soft_warnings))
        output.append(f"\n**Dose:** {result.dose_info}")
        output.append(f"**Subject:** {body_weight} kg {sex}, {age}y")
        output.append(f"**Model:** {distribution_model.replace('_', '-')}")
        output.append(f"**Duration:** {duration_h} h")
        output.append(f"**Kp method:** {kp_method}")
        if kp_warning:
            output.append(kp_warning)
        if ivive_info:
            output.append(f"**Clearance:** {ivive_info}")
        if transporter_dict:
            t_names = []
            for org, ot in transporter_dict.items():
                for t in ot.influx + ot.efflux:
                    t_names.append(f"{t.gene}({org})")
            output.append(f"**Transporters:** {', '.join(t_names)}")
        output.append("")

        output.append(pk.to_markdown(compound.name, dose_mg))

        # Key organ concentrations at Tmax
        tmax_idx = int(np.argmin(np.abs(result.time - pk.Tmax)))
        output.append(f"\n### Tissue Concentrations at Tmax ({pk.Tmax:.1f} h)\n")
        output.append("| Tissue | C_tissue (mg/L) | C_tissue/C_plasma |")
        output.append("|--------|----------------|------------------|")
        c_plasma = result.venous_plasma[tmax_idx] if result.venous_plasma[tmax_idx] > 0 else 1e-10
        for organ in Organ:
            c_t = result.concentrations.get(organ.value, result.concentrations.get(organ.value, None))
            if c_t is not None:
                c_val = c_t[tmax_idx]
                ratio = c_val / c_plasma if c_plasma > 0 else 0
                output.append(f"| {organ.value.capitalize()} | {c_val:.4g} | {ratio:.2f} |")

        # Kp values used
        output.append(f"\n### Partition Coefficients Used\n")
        output.append("| Tissue | Kp | Kp:blood |")
        output.append("|--------|----|----------|")
        for organ in Organ:
            kp = model.kp[organ]
            kpb = model.kpb[organ]
            output.append(f"| {organ.value.capitalize()} | {kp:.3f} | {kpb:.3f} |")

        if plot_path:
            output.append(f"\n**Plot saved:** `{plot_path}`")

        # --- Structured "what I did / did not" footer ---
        # Forces the tool to enumerate defaults used and skipped validations
        # so a downstream LLM cannot silently elide the model's limits.
        defaults_used = []
        if not in_library_match:
            if compound.fu_p == 1.0:
                defaults_used.append("fu_p=1.0 (sentinel; provide measured value)")
            if compound.R_bp == 1.0:
                defaults_used.append("R_bp=1.0 (sentinel; provide measured or predicted value)")
            if cl_int_resolved == 0 and CL_renal == 0:
                defaults_used.append("CL_int=0, CL_renal=0 (no elimination)")
            if not Peff:
                defaults_used.append("Peff not provided — Fg may be unreliable")
        skipped = []
        if cl_int_resolved > 0 and clearance_source == "hlm" and not in_library_match:
            skipped.append("UGT/SULT/esterase metabolism (HLM measures CYP only — switch to "
                           "clearance_source='hepatocyte' if non-CYP pathways contribute)")
        if not transporter_dict:
            skipped.append("Active transport (OATP/MRP2/OCT2/MATE1/P-gp) — provide "
                           "Km/Vmax pairs and set distribution_model='permeability_limited' to enable")

        output.append("\n### Modelling Provenance")
        output.append("> Tag every parameter as **M**=measured / **L**=literature / **P**=predicted "
                      "/ **D**=default in your interpretation.")
        if defaults_used:
            output.append("\n**Defaults used (no user / library value):**")
            for d in defaults_used:
                output.append(f"- {d}")
        if skipped:
            output.append("\n**Mechanisms NOT modelled:**")
            for s in skipped:
                output.append(f"- {s}")
        if not defaults_used and not skipped:
            output.append("\n_No suspicious defaults; all critical params resolved._")

        # --- Audit log ---
        from core.audit import log_simulation
        try:
            fp = log_simulation(
                tool_name="run_pbpk_simulation",
                inputs=_audit_inputs,
                resolved={
                    "compound_name": compound.name,
                    "in_library": in_library_match,
                    "cl_int_resolved_L_per_h": cl_int_resolved,
                    "kp_method_used": kp_method,
                    "transporters_active": list(transporter_dict.keys()),
                    "ehc": False,
                },
                warnings=soft_warnings,
                summary={
                    "Cmax": pk.Cmax, "Tmax": pk.Tmax,
                    "AUC_0_inf": pk.AUC_0_inf, "t_half": pk.t_half,
                    "CL_F": pk.CL_F, "Vss": pk.Vss,
                },
            )
            output.append(f"\n_Audit fingerprint: `{fp}`_")
        except Exception:
            pass  # never let audit failure break a simulation

        return "\n".join(output)

    @mcp.tool()
    def predict_hepatic_clearance(
        name: str = "",
        CL_int: float = 0.0,
        fu_p: float = 1.0,
        R_bp: float = 1.0,
        body_weight: float = 73.0,
        sex: str = "male",
    ) -> str:
        """
        Predict hepatic clearance using the well-stirred liver model.

        CL_h = (Q_h * fu_b * CL_int) / (Q_h + fu_b * CL_int)

        Where fu_b = fu_p / R_bp (fraction unbound in blood).

        Args:
            name: Library compound name (e.g. "midazolam") — overrides
                  CL_int/fu_p/R_bp if found.
            CL_int: Intrinsic clearance (L/h).
            fu_p: Fraction unbound in plasma.
            R_bp: Blood:plasma ratio.
            body_weight: Body weight (kg).
            sex: "male" or "female".

        Returns:
            Hepatic clearance parameters in markdown.
        """
        # If a library compound name is provided, pull its values
        if name and name.lower() in COMPOUND_LIBRARY:
            c = COMPOUND_LIBRARY[name.lower()]
            CL_int = c.CL_int
            fu_p = c.fu_p
            R_bp = c.R_bp

        # Refuse a CL=0 simulation — that's the silent fallback when
        # the user calls predict_hepatic_clearance() with no args
        # (CL_int=0, fu_p=1.0, R_bp=1.0 → CL_h=0). Returning a
        # plausible-looking zero is misleading.
        if CL_int <= 0:
            raise ValueError(
                f"predict_hepatic_clearance requires CL_int > 0 (was "
                f"{CL_int}). Either pass `name=` of a library compound "
                f"(midazolam, diazepam, warfarin, theophylline, "
                f"caffeine, metformin) or provide CL_int explicitly. "
                f"Calling with all defaults returns CL_h=0, which is "
                f"not a meaningful prediction."
            )

        phys = get_physiology(body_weight=body_weight, sex=Sex(sex))
        Q_h = phys.Q_liver_total  # L/h
        fu_b = fu_p / R_bp

        CL_h = (Q_h * fu_b * CL_int) / (Q_h + fu_b * CL_int) if (Q_h + fu_b * CL_int) > 0 else 0.0
        E_h = CL_h / Q_h if Q_h > 0 else 0.0
        F_h = 1.0 - E_h

        lines = [
            "## Well-Stirred Hepatic Clearance Model\n",
            "| Parameter | Value | Unit |",
            "|-----------|-------|------|",
            f"| Q_h (hepatic blood flow) | {Q_h:.1f} | L/h |",
            f"| fu_p | {fu_p:.4f} | — |",
            f"| fu_b (fu_p/R_bp) | {fu_b:.4f} | — |",
            f"| CL_int | {CL_int:.2f} | L/h |",
            f"| **CL_h** | **{CL_h:.2f}** | **L/h** |",
            f"| E_h (extraction ratio) | {E_h:.3f} | — |",
            f"| F_h (hepatic bioavailability) | {F_h:.3f} | — |",
            "",
            f"CL_h = (Q_h × fu_b × CL_int) / (Q_h + fu_b × CL_int)",
            f"     = ({Q_h:.1f} × {fu_b:.4f} × {CL_int:.2f}) / ({Q_h:.1f} + {fu_b:.4f} × {CL_int:.2f})",
            f"     = {CL_h:.2f} L/h",
        ]
        return "\n".join(lines)

    @mcp.tool()
    def list_physiology(
        body_weight: float = 73.0,
        sex: str = "male",
    ) -> str:
        """
        Display physiological parameters for a virtual individual.

        Shows organ volumes, blood flows, and tissue composition.

        Args:
            body_weight: Body weight in kg.
            sex: "male" or "female".

        Returns:
            Markdown tables with physiological data.
        """
        phys = get_physiology(body_weight=body_weight, sex=Sex(sex))

        lines = [
            f"## Physiological Parameters — {body_weight} kg {sex}\n",
            f"Cardiac output: {phys.cardiac_output:.1f} L/h "
            f"({phys.cardiac_output / 60:.1f} L/min)\n",
            "### Organ Volumes and Blood Flows\n",
            "| Organ | Volume (L) | %BW | Blood Flow (L/h) | %CO |",
            "|-------|-----------|-----|------------------|-----|",
        ]

        for organ in Organ:
            vol = phys.organ_volumes[organ]
            pct_bw = vol / body_weight * 100
            flow = phys.blood_flows.get(organ, 0)
            pct_co = flow / phys.cardiac_output * 100 if phys.cardiac_output > 0 else 0
            lines.append(
                f"| {organ.value.capitalize()} | {vol:.3f} | {pct_bw:.1f} | "
                f"{flow:.1f} | {pct_co:.1f} |"
            )

        lines.extend([
            f"| *Arterial blood* | {phys.V_arterial:.3f} | "
            f"{phys.V_arterial / body_weight * 100:.1f} | — | — |",
            f"| *Venous blood* | {phys.V_venous:.3f} | "
            f"{phys.V_venous / body_weight * 100:.1f} | — | — |",
            "",
            f"Portal vein flow: {phys.Q_portal:.1f} L/h "
            f"(gut {phys.blood_flows[Organ.GUT]:.1f} + spleen {phys.blood_flows[Organ.SPLEEN]:.1f})",
            f"Hepatic artery: {phys.Q_hepatic_artery:.1f} L/h",
            f"Total liver flow: {phys.Q_liver_total:.1f} L/h",
        ])

        lines.extend([
            "\n### Tissue Composition (Rodgers & Rowland)\n",
            "| Tissue | f_EW | f_IW | f_NL | f_NP | f_AP | pH_IW |",
            "|--------|------|------|------|------|------|-------|",
        ])
        for organ in Organ:
            tc = TISSUE_COMPOSITION[organ]
            lines.append(
                f"| {organ.value.capitalize()} | "
                f"{tc['f_EW']:.3f} | {tc['f_IW']:.3f} | {tc['f_NL']:.3f} | "
                f"{tc['f_NP']:.4f} | {tc['f_AP']:.5f} | {tc['pH_IW']:.1f} |"
            )

        return "\n".join(lines)

    @mcp.tool()
    def list_compounds() -> str:
        """
        List available reference compounds in the built-in library.

        Returns:
            Markdown table with compound properties.
        """
        lines = [
            "## Built-in Compound Library\n",
            "| Name | MW | logP | pKa | fu_p | Type | R_bp | ka | CL_int | CL_renal |",
            "|------|----|----|-----|------|------|------|----|----|------|",
        ]
        for key, c in COMPOUND_LIBRARY.items():
            lines.append(
                f"| {c.name} | {c.mw} | {c.logP} | {c.pKa} | "
                f"{c.fu_p} | {c.compound_type.value} | {c.R_bp} | "
                f"{c.ka} | {c.CL_int} | {c.CL_renal} |"
            )

        lines.extend([
            "",
            "Use the compound name (lowercase) in `run_pbpk_simulation` "
            "or `predict_kp` to use these reference values.",
            "",
            "You can also provide custom physicochemical properties directly.",
        ])
        return "\n".join(lines)

    @mcp.tool()
    def plot_concentration(
        name: str = "",
        dose_mg: float = 100.0,
        route: str = "oral",
        duration_h: float = 24.0,
        organs: str = "plasma,liver,kidney,gut,brain",
        body_weight: float = 73.0,
        sex: str = "male",
        logP: float = 0.0,
        pKa: float = 7.0,
        fu_p: float = 1.0,
        compound_type: str = "neutral",
        R_bp: float = 1.0,
        mw: float = 300.0,
        ka: float = 1.0,
        Fa: float = 1.0,
        Fg: float = 1.0,
        CL_int: float = 0.0,
        CL_renal: float = 0.0,
        log_scale: bool = True,
    ) -> str:
        """
        Generate a concentration-time plot for selected organs.

        Args:
            name: Compound name (from library) or custom name.
            dose_mg: Dose in mg.
            route: "oral", "iv_bolus", or "iv_infusion".
            duration_h: Simulation duration in hours.
            organs: Comma-separated organ names to plot.
                    "plasma" shows venous plasma concentration.
                    Available: plasma, adipose, bone, brain, gut, heart,
                    kidney, liver, lung, muscle, pancreas, skin, spleen, rest.
            body_weight: Body weight in kg.
            sex: "male" or "female".
            logP, pKa, fu_p, compound_type, R_bp, mw: Compound properties.
            ka, Fa, Fg, CL_int, CL_renal: PK parameters.
            log_scale: Use log scale for y-axis (default True).

        Returns:
            Path to the saved plot image.
        """
        # Build compound
        if name.lower() in COMPOUND_LIBRARY:
            compound = COMPOUND_LIBRARY[name.lower()]
        else:
            compound = CompoundSpec(
                name=name or "Custom Compound",
                mw=mw, logP=logP, pKa=pKa,
                fu_p=fu_p, compound_type=CompoundType(compound_type),
                R_bp=R_bp, ka=ka, Fa=Fa, Fg=Fg,
                CL_int=CL_int, CL_renal=CL_renal,
            )

        phys = get_physiology(body_weight=body_weight, sex=Sex(sex))
        model = PBPKModel(compound, phys)
        dosing_proto = DosingProtocol(dose_mg=dose_mg, route=Route(route))
        sim_config = SimulationConfig(duration_h=duration_h)
        result = model.simulate(dosing_proto, sim_config)

        # Parse organ list
        organ_list = [o.strip().lower() for o in organs.split(",")]

        plot_path = _generate_organ_plot(result, organ_list, compound.name, log_scale)

        if plot_path:
            return f"Plot saved: `{plot_path}`"
        return "Failed to generate plot (matplotlib not available)"

    # ----------------------------------------------------------------
    # IVIVE tools
    # ----------------------------------------------------------------

    @mcp.tool()
    def ivive_microsomal(
        clint_vitro: float = 10.0,
        fu_inc: Optional[float] = None,
        logP: float = 2.0,
        protein_conc: float = 1.0,
        mppgl: float = 45.0,
        body_weight: float = 73.0,
        sex: str = "male",
    ) -> str:
        """
        Scale microsomal CLint to in vivo CLint (IVIVE).

        CLint_in_vivo = (CLint_vitro / fu_inc) * MPPGL * liver_weight

        Args:
            clint_vitro: Measured CLint (uL/min/mg protein).
            fu_inc: Fraction unbound in incubation. If None, predicted from logP.
            logP: For fu_inc prediction.
            protein_conc: Microsomal protein concentration (mg/mL).
            mppgl: Mg protein per gram liver (default 45).
            body_weight: kg.
            sex: "male" or "female".

        Returns:
            IVIVE result with CLint_in_vivo (L/h).
        """
        result = scale_microsomal_clint(
            clint_vitro, fu_inc=fu_inc, logP=logP,
            protein_conc=protein_conc, mppgl=mppgl,
            body_weight=body_weight, sex=sex,
        )
        return format_ivive_result(result, "microsomal")

    # ----------------------------------------------------------------
    # Fg prediction
    # ----------------------------------------------------------------

    @mcp.tool()
    def predict_fg(
        name: str = "",
        Peff: float = 5.0,
        CLint_gut: float = 0.0,
        fu_gut: float = 1.0,
        Q_villi: float = 18.0,
        method: str = "qgut",
    ) -> str:
        """
        Predict Fg (fraction escaping gut wall metabolism).

        Methods:
          - qgut: Yang et al. (2007) Qgut model (accounts for permeability)
          - wellstirred: Simple well-stirred gut model

        Args:
            name: Library compound name — if a library value for Peff and/or
                  CLint_gut exists, overrides the corresponding args.
            Peff: Human jejunal permeability (x10^-4 cm/s).
            CLint_gut: Gut wall intrinsic clearance (L/h).
            fu_gut: Fraction unbound in enterocytes.
            Q_villi: Villous blood flow (L/h).
            method: "qgut" or "wellstirred".

        Returns:
            Fg prediction with intermediate parameters.
        """
        # Refuse to silently return Fg ≈ 1.0 — that's what happens
        # when a user calls predict_fg() with no args (CLint_gut=0
        # default → Fg=1). The result looks meaningful but isn't.
        in_lib = bool(name and name.lower() in COMPOUND_LIBRARY)
        if not in_lib and CLint_gut <= 0:
            raise ValueError(
                "predict_fg requires either a library compound name "
                "(e.g. name='midazolam') OR CLint_gut > 0 (gut wall "
                "intrinsic clearance, L/h). With CLint_gut=0 the model "
                "trivially returns Fg=1.0, which is not a real "
                "prediction. Provide CLint_gut explicitly or use a "
                "library compound."
            )
        # Library override for known compounds. Derive CLint_gut from library
        # Fg by first computing Qgut from Peff, then inverting the Qgut formula:
        # Fg = Qgut/(Qgut + fu_gut*CLint_gut) ⟹ CLint_gut = Qgut*(1-Fg)/(fu_gut*Fg)
        if name and name.lower() in COMPOUND_LIBRARY:
            c = COMPOUND_LIBRARY[name.lower()]
            if c.Peff is not None:
                Peff = c.Peff
            if c.CLint_gut > 0:
                CLint_gut = c.CLint_gut
            elif c.Fg < 1.0 and CLint_gut == 0.0:
                # Compute Qgut from Peff first (Yang 2007 formulation)
                # Qgut = Q_villi * CLperm / (Q_villi + CLperm)
                # CLperm = Peff * surface_area (using Yang 2007 constant)
                from core.fg_prediction import predict_fg_qgut as _pfg
                baseline = _pfg(Peff, 0.0, fu_gut, Q_villi)
                Qgut_eff = baseline.Qgut
                CLint_gut = Qgut_eff * (1 - c.Fg) / max(fu_gut * c.Fg, 1e-6)

        if method == "qgut":
            result = predict_fg_qgut(Peff, CLint_gut, fu_gut, Q_villi)
        else:
            result = predict_fg_wellstirred(CLint_gut, fu_gut, Q_villi)
        return result.to_markdown()

    # ----------------------------------------------------------------
    # Hepatic model comparison
    # ----------------------------------------------------------------

    @mcp.tool()
    def compare_hepatic_clearance(
        name: str = "",
        CLint: float = 10.0,
        fu_p: float = 0.1,
        R_bp: float = 1.0,
        Q_h: float = 90.0,
    ) -> str:
        """
        Compare 3 hepatic clearance models side by side.

        Models: Well-stirred, Parallel-tube, Dispersion (DN=0.17)

        Args:
            name: Library compound name — overrides CLint/fu_p/R_bp if found.
            CLint: Intrinsic clearance (L/h).
            fu_p: Fraction unbound in plasma.
            R_bp: Blood:plasma ratio.
            Q_h: Hepatic blood flow (L/h).

        Returns:
            Side-by-side comparison table.
        """
        if name and name.lower() in COMPOUND_LIBRARY:
            c = COMPOUND_LIBRARY[name.lower()]
            CLint = c.CL_int
            fu_p = c.fu_p
            R_bp = c.R_bp
        return compare_hepatic_models(CLint, fu_p, R_bp, Q_h)

    # ----------------------------------------------------------------
    # R_bp prediction
    # ----------------------------------------------------------------

    @mcp.tool()
    def predict_blood_plasma_ratio(
        name: str = "",
        logP: float = 2.0,
        pKa: float = 7.0,
        fu_p: float = 0.5,
        compound_type: str = "neutral",
        mw: float = 300.0,
    ) -> str:
        """
        Predict blood:plasma ratio (R_bp) from Rodgers-Rowland
        RBC partitioning equations.

        R_bp = 1 - HCT + HCT * Kp_RBC

        Args:
            name: Compound name (from library) or custom.
            logP, pKa, fu_p, compound_type, mw: Compound properties.

        Returns:
            R_bp prediction with Kp_RBC.
        """
        from core.validation import require_compound_input
        # Note: predict_blood_plasma_ratio uses different sentinel
        # defaults (logP=2.0, fu_p=0.5) than predict_kp — but the
        # require_compound_input check below uses our canonical
        # sentinel set, so we adapt: only the canonical sentinels
        # signal "no input given".
        require_compound_input(
            name=name, library=COMPOUND_LIBRARY,
            logP=logP if logP != 2.0 else 0.0,   # un-shift to canonical sentinel
            pKa=pKa, fu_p=fu_p if fu_p != 0.5 else 1.0, mw=mw,
            tool_name="predict_blood_plasma_ratio",
        )
        if name.lower() in COMPOUND_LIBRARY:
            compound = COMPOUND_LIBRARY[name.lower()]
        else:
            compound = CompoundSpec(
                name=name or "Custom", mw=mw, logP=logP, pKa=pKa,
                fu_p=fu_p, compound_type=CompoundType(compound_type),
            )

        result = predict_rbp(compound)
        lines = [
            f"## R_bp Prediction — {compound.name}\n",
            f"**R_bp = {result['R_bp']:.3f}**\n",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Kp_RBC | {result['Kp_RBC']:.3f} |",
            f"| Hematocrit | {result['HCT']:.2f} |",
            f"| R_bp | {result['R_bp']:.3f} |",
        ]
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # DDI prediction
    # ----------------------------------------------------------------

    @mcp.tool()
    def predict_ddi(
        mechanism: str = "reversible",
        Ki: Optional[float] = None,
        KI: Optional[float] = None,
        kinact: Optional[float] = None,
        Emax: Optional[float] = None,
        EC50: Optional[float] = None,
        I_h_u: float = 1.0,
        fm: float = 0.9,
        kdeg: float = 0.019,
        I_gut: Optional[float] = None,
        Fg_baseline: float = 1.0,
    ) -> str:
        """
        Predict drug-drug interaction (DDI) using static models.

        Mechanisms:
          - reversible: Competitive CYP inhibition (requires Ki)
          - mbi: Mechanism-based inhibition (requires KI, kinact)
          - induction: CYP induction (requires Emax, EC50)
          - net: Combined inhibition + induction

        Args:
            mechanism: DDI mechanism type.
            Ki: Reversible inhibition constant (uM).
            KI: MBI half-maximal concentration (uM).
            kinact: MBI max inactivation rate (1/h).
            Emax: Max induction fold.
            EC50: Induction half-maximal concentration (uM).
            I_h_u: Unbound inhibitor/inducer at liver (uM).
            fm: Fraction metabolized by affected CYP.
            kdeg: CYP degradation rate (1/h). CYP3A4 ~0.019.
            I_gut: Inhibitor concentration in enterocytes (uM).
            Fg_baseline: Baseline Fg of substrate.

        Returns:
            AUC ratio and DDI classification.
        """
        # Validate required parameters per mechanism — reject silent defaults
        missing = []
        if mechanism == "reversible" and not Ki:
            missing.append("Ki (reversible inhibition constant in µM)")
        elif mechanism == "mbi" and (not KI or not kinact):
            missing.extend([m for m, v in [("KI (µM)", KI), ("kinact (1/h)", kinact)] if not v])
        elif mechanism == "induction" and (not Emax or not EC50):
            missing.extend([m for m, v in [("Emax (fold)", Emax), ("EC50 (µM)", EC50)] if not v])
        elif mechanism == "net" and not any([Ki, KI, kinact, Emax, EC50]):
            missing.append("at least one of Ki/KI+kinact/Emax+EC50 for net mechanism")

        if missing:
            return (f"## DDI Prediction — Missing parameters\n\n"
                    f"Mechanism `{mechanism}` requires: {', '.join(missing)}.\n\n"
                    f"Currently passed: I_h_u={I_h_u} µM, fm={fm}. "
                    f"Add the listed parameters and re-run.")

        # Echo what was actually used so user sees their inputs were applied
        echo = (f"\n*Inputs used: mechanism={mechanism}, fm={fm}, "
                f"I_h_u={I_h_u} µM"
                + (f", Ki={Ki}" if Ki else "")
                + (f", KI={KI}, kinact={kinact}" if KI and kinact else "")
                + (f", Emax={Emax}, EC50={EC50}" if Emax and EC50 else "")
                + "*\n")

        if mechanism == "reversible":
            result = ddi_reversible_inhibition(Ki, I_h_u, fm, I_gut, Fg_baseline)
        elif mechanism == "mbi":
            result = ddi_mechanism_based_inhibition(
                KI, kinact, I_h_u, fm, kdeg, I_gut, Fg_baseline)
        elif mechanism == "induction":
            result = ddi_induction(Emax, EC50, I_h_u, fm)
        else:
            result = ddi_net_effect(Ki, KI, kinact, Emax, EC50, I_h_u, fm, kdeg)
        return result.to_markdown() + echo

    # ----------------------------------------------------------------
    # ACAT absorption model
    # ----------------------------------------------------------------

    @mcp.tool()
    def simulate_acat(
        name: str = "",
        dose_mg: float = 100.0,
        Peff: float = 5.0,
        S0: float = 1.0,
        particle_radius_um: float = 25.0,
        mw: float = 300.0,
        pKa: float = 7.0,
        compound_type: str = "neutral",
        CLint_gut: float = 0.0,
        fu_gut: float = 1.0,
        duration_h: float = 24.0,
    ) -> str:
        """
        Run standalone ACAT absorption simulation (GI tract only).

        Predicts Fa, Fg, dissolution profile, and regional absorption
        using a 9-segment GI model with Noyes-Whitney dissolution,
        pH-dependent solubility, and segment-specific CYP3A4 metabolism.

        9 segments: Stomach, Duodenum, Jejunum 1-2, Ileum 1-3, Caecum, Asc Colon

        Args:
            name: Compound name (from library) or custom.
            dose_mg: Oral dose (mg).
            Peff: Human jejunal permeability (x10^-4 cm/s).
            S0: Intrinsic solubility of neutral form (mg/mL).
            particle_radius_um: Mean particle radius (um).
            mw: Molecular weight (g/mol).
            pKa: Dissociation constant.
            compound_type: acid, base, neutral, etc.
            CLint_gut: Total gut wall intrinsic clearance (L/h).
            fu_gut: Fraction unbound in enterocytes.
            duration_h: Simulation duration (h).

        Returns:
            ACAT result with Fa, Fg, dissolution times, regional absorption.
        """
        # Refuse all-default invocation — every other input becomes a
        # sentinel and the result is plausible-looking nonsense.
        in_lib_acat = bool(name and name.lower() in COMPOUND_LIBRARY)
        if not in_lib_acat:
            all_default = (mw == 300.0 and pKa == 7.0
                           and compound_type == "neutral"
                           and S0 == 1.0 and Peff == 5.0)
            if all_default:
                raise ValueError(
                    "simulate_acat requires either a library compound "
                    "name (e.g. name='midazolam') OR explicit drug "
                    "parameters (mw, pKa, compound_type, S0, Peff). "
                    "All-default arguments produce a plausible-looking "
                    "ACAT prediction with no physical meaning."
                )

        if name.lower() in COMPOUND_LIBRARY:
            c = COMPOUND_LIBRARY[name.lower()]
            mw = c.mw
            pKa = c.pKa
            compound_type = c.compound_type.value
            cname = c.name
        else:
            cname = name or "Custom"

        result = simulate_acat_standalone(
            dose_mg=dose_mg, Peff_e4=Peff, mw=mw, pKa=pKa,
            compound_type=compound_type, S0=S0,
            formulation=FormulationSpec(S0=S0, particle_radius_um=particle_radius_um),
            CLint_gut_total=CLint_gut, fu_gut=fu_gut,
            duration_h=duration_h,
        )
        return format_acat_result(result, cname)

    # ----------------------------------------------------------------
    # Lymphatic absorption
    # ----------------------------------------------------------------

    @mcp.tool()
    def predict_lymphatic(
        logP: float = 5.0,
        Fa: float = 1.0,
        Fg: float = 1.0,
        Fh: float = 0.5,
        TG_solubility: Optional[float] = None,
    ) -> str:
        """
        Predict lymphatic absorption fraction and its impact on oral bioavailability.

        For highly lipophilic drugs (logP > 5), lymphatic transport bypasses
        hepatic first-pass metabolism entirely.

        F_oral = Fa * [(1 - F_lymph) * Fg * Fh + F_lymph]

        Args:
            logP: Octanol:water partition coefficient.
            Fa: Fraction absorbed. Fg: Fraction escaping gut wall.
            Fh: Hepatic bioavailability.
            TG_solubility: Triglyceride solubility (mg/g), if known.

        Returns:
            F_oral with and without lymphatic pathway.
        """
        F_lymph = estimate_lymphatic_fraction(logP, TG_solubility)
        result = oral_bioavailability_with_lymph(Fa, Fg, Fh, F_lymph)

        F_portal_only = Fa * Fg * Fh
        lines = [
            f"## Lymphatic Absorption Analysis\n",
            f"logP = {logP}, F_lymph = {F_lymph:.3f}\n",
            "| Pathway | F_oral |",
            "|---------|--------|",
            f"| Portal only (no lymph) | {F_portal_only:.4f} |",
            f"| Portal + lymphatic | **{result['F_oral']:.4f}** |",
            f"| Lymph contribution | {result['F_lymph_pathway']:.4f} ({result['F_lymph_fraction_of_total']*100:.1f}% of total) |",
        ]
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Concentration-dependent binding
    # ----------------------------------------------------------------

    @mcp.tool()
    def predict_binding_profile(
        fu_p: float = 0.01,
        mw: float = 300.0,
        binding_type: str = "single_site",
        fraction_albumin: float = 0.7,
        name: str = "",
    ) -> str:
        """
        Model concentration-dependent protein binding.

        At high drug concentrations, binding sites saturate and fu_p increases.
        Relevant for valproic acid, phenytoin, warfarin at high dose.

        Args:
            fu_p: Fraction unbound at low concentration.
            mw: Molecular weight (for concentration conversion).
            binding_type: "single_site" (albumin only) or "two_site" (albumin + AAG).
            fraction_albumin: For two_site model, fraction of binding due to albumin.
            name: Compound name.

        Returns:
            fu vs concentration profile.
        """
        if binding_type == "two_site":
            model = two_site_albumin_aag(fu_p, fraction_albumin)
        else:
            model = single_site_albumin(fu_p)
        return format_binding_profile(model, name or "Drug", mw)

    # ----------------------------------------------------------------
    # Population simulation
    # ----------------------------------------------------------------

    @mcp.tool()
    def run_population_pbpk(
        name: str = "",
        dose_mg: float = 100.0,
        route: str = "oral",
        duration_h: float = 24.0,
        n_individuals: int = 100,
        proportion_female: float = 0.5,
        weight_mean: float = 73.0,
        logP: float = 0.0,
        pKa: float = 7.0,
        fu_p: float = 1.0,
        compound_type: str = "neutral",
        R_bp: float = 1.0,
        mw: float = 300.0,
        ka: float = 1.0,
        CL_int: float = 0.0,
        CL_renal: float = 0.0,
        seed: int = 42,
        kp_method: str = "rodgers_rowland",
    ) -> str:
        """
        Run population PBPK simulation with inter-individual variability.

        Generates N virtual individuals with varied body weight, CYP abundance,
        and fu_p, then runs PBPK for each.

        Args:
            name: Compound name (from library) or custom.
            dose_mg: Dose (mg). route: "oral" or "iv_bolus".
            duration_h: Simulation duration.
            n_individuals: Number of virtual individuals (50-500).
            proportion_female: Fraction female (0-1).
            weight_mean: Mean body weight (kg).
            logP, pKa, fu_p, compound_type, R_bp, mw, ka, CL_int, CL_renal: Drug properties.
            seed: Random seed for reproducibility.

        Returns:
            Population PK summary (median, 5th, 95th percentiles).
        """
        # Hard validation: kp_method enum (typos previously silently
        # ran with R&R)
        from core.validation import validate_kp_method
        validate_kp_method(kp_method)
        # If a custom compound, range-check the supplied physchem
        in_lib_pop = name and name.lower() in COMPOUND_LIBRARY
        if not in_lib_pop:
            from core.invariants import (
                check_compound_ranges, raise_on_violations,
            )
            raise_on_violations(check_compound_ranges(
                mw=mw if mw != 300.0 else None,
                logP=logP if logP != 0.0 else None,
                pKa=pKa if pKa != 7.0 else None,
                fu_p=fu_p,
                R_bp=R_bp,
                ka=ka,
                CL_int=CL_int if CL_int > 0 else None,
                CL_renal=CL_renal if CL_renal > 0 else None,
            ))
        if name.lower() in COMPOUND_LIBRARY:
            compound = COMPOUND_LIBRARY[name.lower()]
        else:
            compound = CompoundSpec(
                name=name or "Custom", mw=mw, logP=logP, pKa=pKa,
                fu_p=fu_p, compound_type=CompoundType(compound_type),
                R_bp=R_bp, ka=ka, CL_int=CL_int, CL_renal=CL_renal,
            )

        dosing_proto = DosingProtocol(dose_mg=dose_mg, route=Route(route))
        sim_config = SimulationConfig(duration_h=duration_h, n_timepoints=500)

        n = min(max(n_individuals, 10), 500)
        pop_result = run_population_simulation(
            compound, dosing_proto, sim_config,
            n_individuals=n, proportion_female=proportion_female,
            weight_mean=weight_mean, seed=seed,
            kp_method=kp_method,
        )
        return pop_result.to_markdown(compound.name)

    # ----------------------------------------------------------------
    # Pregnancy physiology
    # ----------------------------------------------------------------

    @mcp.tool()
    def pregnancy_physiology(
        gestational_age_weeks: float = 28.0,
    ) -> str:
        """
        Show pregnancy physiological changes at given gestational age.

        Displays multipliers for cardiac output, GFR, CYP enzymes, etc.
        relative to non-pregnant baseline.

        Args:
            gestational_age_weeks: Gestational age (0-40 weeks).

        Returns:
            Table of pregnancy-related physiological changes.
        """
        # Range check: GA must be physiologically possible (0-42 weeks).
        # Previously, GA=100 would silently extrapolate the multiplier
        # curves into nonsense.
        if not (0.0 <= gestational_age_weeks <= 42.0):
            raise ValueError(
                f"gestational_age_weeks={gestational_age_weeks} out of "
                f"range [0, 42]. Pregnancy lasts ~40 weeks (term); 42 "
                f"is the latest viable post-term value. Negative or "
                f">42 is non-physiological."
            )
        return format_pregnancy_profile(gestational_age_weeks)

    # ----------------------------------------------------------------
    # Species comparison + allometric scaling
    # ----------------------------------------------------------------

    @mcp.tool()
    def species_comparison() -> str:
        """
        Compare physiological parameters across species
        (Human, Rat, Mouse, Dog, Monkey).

        Shows reference body weight, organ sizes, GFR, MPPGL for each species.
        Useful for preclinical-to-clinical translation planning.
        """
        return format_species_comparison()

    @mcp.tool()
    def allometric_scaling(
        CL_animal: float = 1.0,
        Vss_animal: float = 0.5,
        BW_animal: float = 0.25,
        species: str = "rat",
        BW_human: float = 73.0,
    ) -> str:
        """
        Allometric scaling of preclinical PK to human prediction.

        CL: BW^0.75, Vss: BW^1.0, t½: BW^0.25

        Args:
            CL_animal: Animal clearance (L/h).
            Vss_animal: Animal Vss (L).
            BW_animal: Animal body weight (kg).
            species: "rat", "mouse", "dog", or "monkey".
            BW_human: Target human BW (kg).
        """
        # Refuse all-default scaling — sentinel CL=1.0 / Vss=0.5 with
        # rat default produces a plausible-looking but meaningless
        # human prediction.
        if CL_animal == 1.0 and Vss_animal == 0.5 and BW_animal == 0.25:
            raise ValueError(
                "allometric_scaling requires non-default CL_animal, "
                "Vss_animal, BW_animal — all three at their sentinel "
                "defaults (1.0, 0.5, 0.25 kg) means no real preclinical "
                "data was supplied. Provide observed animal PK."
            )
        if CL_animal <= 0:
            raise ValueError(
                f"CL_animal must be > 0 (got {CL_animal})."
            )
        result = scale_preclinical_to_human(
            CL_animal, Vss_animal, BW_animal, Species(species), BW_human
        )
        lines = [
            f"## Allometric Scaling — {species.capitalize()} to Human\n",
            "| Parameter | Animal | Human (predicted) |",
            "|-----------|--------|-------------------|",
            f"| BW (kg) | {BW_animal:.2f} | {BW_human:.1f} |",
            f"| CL (L/h) | {CL_animal:.4g} | {result['CL_human_L_per_h']:.4g} |",
            f"| Vss (L) | {Vss_animal:.4g} | {result['Vss_human_L']:.4g} |",
            f"| t½ (h) | — | {result['t_half_human_h']:.2f} |",
            "",
            "_Note: simple BW^0.75 / BW^1.0 scaling assumes conserved fu_p "
            "and Kp across species. For highly-bound drugs (fu_p < 0.05) "
            "with cross-species fu differences, use "
            "`fu_corrected_allometric_scaling`. For ≥3 species data, use "
            "`mahmood_rule_of_exponents`._",
        ]
        return "\n".join(lines)

    @mcp.tool()
    def fu_corrected_allometric_scaling(
        CL_animal: float,
        BW_animal: float,
        fu_animal: float,
        fu_human: float,
        BW_human: float = 73.0,
        exponent: float = 0.75,
    ) -> str:
        """
        Fu-corrected single-species allometric scaling (Tang & Mayersohn 2005).

        For drugs where fu_p differs across species (e.g. warfarin: rat
        fu ~0.04 vs human fu ~0.005, an 8× difference), naive BW^0.75
        scaling is systematically wrong. This tool multiplies the simple
        allometric prediction by the unbound-fraction ratio:

            CL_human = CL_animal × (BW_h/BW_a)^b × (fu_human/fu_animal)

        ANTI-FABRICATION: do not pass placeholder fu values — the
        correction factor scales linearly with them. Provide measured
        or literature fu_p for both species.

        Reference: Tang H, Mayersohn M. Drug Metab Dispos 2005;33:1294.
        """
        result = fu_corrected_allometry(
            CL_animal, BW_animal, fu_animal, fu_human,
            BW_human=BW_human, exponent=exponent,
        )
        return (
            f"## Fu-corrected Allometric Scaling (Tang 2005)\n\n"
            f"| Parameter | Value |\n"
            f"|-----------|-------|\n"
            f"| CL_animal (L/h) | {CL_animal:.4g} |\n"
            f"| BW_animal (kg) | {BW_animal:.3g} |\n"
            f"| fu_animal | {fu_animal:.4g} |\n"
            f"| fu_human | {fu_human:.4g} |\n"
            f"| Allometric exponent | {exponent} |\n"
            f"| Naive CL_human (no fu correction) | "
            f"{result['CL_human_naive_L_per_h']:.4g} L/h |\n"
            f"| **fu correction factor (fu_h/fu_a)** | "
            f"**{result['fu_correction_factor']:.4g}** |\n"
            f"| **Corrected CL_human** | **{result['CL_human_L_per_h']:.4g} L/h** |\n"
            f"\n_Method: {result['method']}_\n"
        )

    @mcp.tool()
    def vertical_allometric_scaling(
        CL_animal: float,
        BW_animal: float,
        species: str = "rat",
        BW_human: float = 73.0,
    ) -> str:
        """
        Vertical allometry with brain weight correction (Mahmood 1996).

        For drugs whose simple-allometry exponent falls in 0.7-1.0
        (typical of moderate-to-high hepatic CL drugs), the Rule of
        Exponents recommends multiplying by brain weight:

            CL_human = CL_animal × (BW_h × BrW_h) / (BW_a × BrW_a)

        Reference brain weights: mouse 0.4 g, rat 1.8 g, dog 72 g,
        monkey 95 g, human 1400 g.

        Use this when the standard `allometric_scaling` over- or
        under-predicts and you suspect brain-weight scaling applies
        (high CL, multi-species data unavailable for full ROE).

        Reference: Mahmood I, Balian JD. Xenobiotica 1996;26:887.
        """
        sp = Species(species)
        result = vertical_allometry_brain_weight(
            CL_animal, BW_animal, sp, BW_human=BW_human,
        )
        return (
            f"## Vertical Allometry × Brain Weight (Mahmood 1996)\n\n"
            f"| Parameter | Value |\n"
            f"|-----------|-------|\n"
            f"| Species | {sp.value} |\n"
            f"| CL_animal (L/h) | {CL_animal:.4g} |\n"
            f"| BW_animal (kg) | {BW_animal:.3g} |\n"
            f"| BrW_animal (g) | {result['BrW_animal_g']:.1f} |\n"
            f"| BW_human (kg) | {BW_human:.1f} |\n"
            f"| BrW_human (g) | {result['BrW_human_g']:.1f} |\n"
            f"| Scaling factor | {result['scaling_factor']:.4g} |\n"
            f"| **Predicted CL_human** | **{result['CL_human_L_per_h']:.4g} L/h** |\n"
            f"\n_Method: {result['method']}_\n"
            f"\n_Note: when ≥3 species data is available, prefer "
            f"`mahmood_rule_of_exponents` — the multi-species fit "
            f"chooses MLP vs brain-weight correction by the empirical "
            f"exponent rather than assuming brain weight a priori._\n"
        )

    @mcp.tool()
    def mahmood_rule_of_exponents_scaling(
        species_data_csv: str,
        BW_human: float = 73.0,
    ) -> str:
        """
        Multi-species allometric scaling with Rule of Exponents
        (Mahmood & Balian 1996).

        Workflow:
          1. Fit log(CL) = log(a) + b·log(BW) across ≥3 species.
          2. Choose correction by exponent b:
               b < 0.55       → simple allometry
               0.55 ≤ b ≤ 0.70 → MLP correction (lifespan)
               0.70 < b ≤ 1.00 → brain-weight correction
               b > 1.00       → INAPPLICABLE (do not use result)
          3. Refit in the corrected coordinate, predict human CL.

        Args:
            species_data_csv: comma-separated triples
                "species:CL_L_per_h:BW_kg, ..."
                e.g. "mouse:0.05:0.025, rat:0.4:0.25, dog:8.0:10"
                Need ≥ 3 entries from {mouse, rat, dog, monkey, human}.
            BW_human: target human BW (kg).

        ANTI-FABRICATION: do not pass made-up CL values. Use actual
        observed clearances from preclinical PK studies; cite each
        in your downstream provenance audit.

        Reference: Mahmood I, Balian JD. Xenobiotica 1996;26:887-895.
        """
        # Parse "species:CL:BW, species:CL:BW" string
        triples: list[tuple[Species, float, float]] = []
        for raw in species_data_csv.split(","):
            entry = raw.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Malformed entry '{entry}'. Expected "
                    f"'species:CL_L_per_h:BW_kg' separated by colons."
                )
            sp_name, cl_s, bw_s = (p.strip() for p in parts)
            try:
                sp = Species(sp_name.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown species '{sp_name}' in entry '{entry}'. "
                    f"Use one of: mouse, rat, dog, monkey, human."
                )
            triples.append((sp, float(cl_s), float(bw_s)))

        if len(triples) < 3:
            raise ValueError(
                f"Mahmood ROE requires ≥3 species; got {len(triples)}. "
                f"Use `fu_corrected_allometric_scaling` or "
                f"`vertical_allometric_scaling` for single-species data."
            )

        result = mahmood_rule_of_exponents(triples, BW_human=BW_human)

        # Build report
        lines = ["## Mahmood Rule of Exponents (multi-species allometry)\n"]
        lines.append("### Input data\n")
        lines.append("| Species | CL (L/h) | BW (kg) | BrW (g) | MLP (y) |")
        lines.append("|---------|----------|---------|---------|---------|")
        for sp, cl, bw in triples:
            lines.append(
                f"| {sp.value} | {cl:.4g} | {bw:.3g} | "
                f"{BRAIN_WEIGHT_G[sp]:.1f} | {MLP_YEARS[sp]:.1f} |"
            )
        lines.append("")
        lines.append("### Fit + Rule of Exponents\n")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Simple-allometry exponent (b) | "
                     f"{result['exponent_simple']:.4f} |")
        lines.append(f"| Simple-allometry intercept (a) | "
                     f"{result['intercept_simple']:.4g} |")
        lines.append(f"| Simple-allometry R² | {result['r2_simple']:.4f} |")
        lines.append(f"| Method chosen | {result['method']} |")
        lines.append(f"| Applicability | {result['applicability']} |")
        lines.append(f"| **Predicted CL_human (L/h)** | "
                     f"**{result['CL_human_L_per_h']:.4g}** |")
        if result["notes"]:
            lines.append(f"\n_{result['notes']}_")
        if result["applicability"] == "fail":
            lines.append("\n⚠️ **Do NOT use this prediction for human dosing.** "
                         "The Rule of Exponents flagged this case as unreliable.")
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Local sensitivity analysis (one-at-a-time)
    # ----------------------------------------------------------------

    @mcp.tool()
    def sensitivity_analysis(
        name: str = "",
        # Custom compound (used if `name` is not in library)
        logP: float = 0.0,
        pKa: float = 7.0,
        fu_p: float = 1.0,
        compound_type: str = "neutral",
        R_bp: float = 1.0,
        mw: float = 300.0,
        ka: float = 1.0,
        Fa: float = 1.0,
        Fg: float = 1.0,
        CL_int: float = 0.0,
        CL_renal: float = 0.0,
        # Dosing for the simulation that drives sensitivity
        dose_mg: float = 100.0,
        route: str = "oral",
        duration_h: float = 24.0,
        body_weight: float = 73.0,
        sex: str = "male",
        age: float = 30.0,
        kp_method: str = "rodgers_rowland",
        # Which parameters to perturb + which PK metric to track
        parameters: str = "CL_int,fu_p,R_bp,ka,Fa,Fg",
        pk_metric: str = "AUC_0_inf",
        perturbation: float = 0.05,
    ) -> str:
        """
        One-at-a-time (OAT) local sensitivity analysis.

        For each parameter in `parameters`, perturb by ±`perturbation`
        fraction (default 5%) and measure the central-difference
        normalized sensitivity of the chosen PK metric:

            S = (ΔPK / PK_base) / (Δp / p_base)

        Magnitude classification (FDA/EMA convention):
          |S| ≥ 1.0 → "high"        (proportional driver)
          0.5–1.0   → "moderate"
          0.1–0.5   → "low"
          < 0.1     → "negligible"

        ANTI-FABRICATION: do not run sensitivity on a sentinel-default
        compound — the result will be physically meaningless. Provide
        either a library `name` or explicit physchem (logP, pKa, fu_p,
        mw at minimum).

        Args:
            name: Library compound name OR empty for custom.
            (other physchem args same as run_pbpk_simulation)
            parameters: Comma-separated list of parameter names to
                perturb. Allowed: CL_int, CL_renal, fu_p, R_bp, ka,
                Fa, Fg, logP, pKa, mw. Default: the six common drivers.
            pk_metric: Which PK metric to track. One of:
                AUC_0_inf, AUC_0_t, Cmax, t_half, CL_F, Vss, Tmax, MRT.
            perturbation: Fractional perturbation (0.01-0.20 typical).

        Returns:
            Markdown table ranked by |S|, with magnitude classification
            and a "drivers" summary listing the parameters that
            dominate the prediction.

        Reference:
            McNally K et al. Front Pharmacol 2020;11:1-15.
        """
        # --- Validate inputs ---
        from core.validation import (
            require_compound_input, validate_kp_method,
        )
        from core.invariants import raise_on_violations, check_compound_ranges
        require_compound_input(
            name=name, library=COMPOUND_LIBRARY,
            logP=logP, pKa=pKa, fu_p=fu_p, mw=mw,
            tool_name="sensitivity_analysis",
        )
        validate_kp_method(kp_method)
        if not (0.005 <= perturbation <= 0.50):
            raise ValueError(
                f"perturbation={perturbation} out of range [0.005, 0.50]. "
                f"Typical values are 0.01-0.10. Smaller is more local; "
                f"larger leaves the linear regime."
            )

        # --- Build base compound ---
        name = name or ""
        if name.lower() in COMPOUND_LIBRARY:
            base_compound = COMPOUND_LIBRARY[name.lower()]
        else:
            base_compound = CompoundSpec(
                name=name or "Custom", mw=mw, logP=logP, pKa=pKa,
                fu_p=fu_p, compound_type=CompoundType(compound_type),
                R_bp=R_bp, ka=ka, Fa=Fa, Fg=Fg,
                CL_int=CL_int, CL_renal=CL_renal,
            )

        # --- Validate which parameters can be perturbed ---
        ALLOWED_PARAMS = {
            "CL_int", "CL_renal", "fu_p", "R_bp", "ka", "Fa", "Fg",
            "logP", "pKa", "mw",
        }
        ALLOWED_METRICS = {
            "AUC_0_inf", "AUC_0_t", "Cmax", "t_half", "CL_F", "Vss",
            "Tmax", "MRT",
        }
        param_list = [p.strip() for p in parameters.split(",") if p.strip()]
        bad_params = [p for p in param_list if p not in ALLOWED_PARAMS]
        if bad_params:
            raise ValueError(
                f"Unknown parameter(s) {bad_params}. Allowed: "
                f"{sorted(ALLOWED_PARAMS)}."
            )
        if pk_metric not in ALLOWED_METRICS:
            raise ValueError(
                f"Unknown pk_metric='{pk_metric}'. Allowed: "
                f"{sorted(ALLOWED_METRICS)}."
            )

        # --- Build base parameter dict for OAT ---
        base_params = {p: getattr(base_compound, p) for p in param_list}
        # Reject zero-base parameters (would make ratio undefined)
        zero_params = [p for p, v in base_params.items() if v == 0]
        if zero_params:
            raise ValueError(
                f"Parameter(s) {zero_params} have base value 0; "
                f"normalized sensitivity is undefined when the base is 0. "
                f"Either set them to a non-zero value or remove from "
                f"the perturbation list."
            )

        # --- Build simulate_fn closure that rebuilds compound and runs sim ---
        phys = get_physiology(body_weight=body_weight, sex=Sex(sex),
                              age_years=age)
        kp_method_enum = KpMethod(kp_method)

        def _simulate(params_dict: dict):
            # Construct a compound with the perturbed values
            kw = {
                "name": base_compound.name,
                "mw": base_compound.mw, "logP": base_compound.logP,
                "pKa": base_compound.pKa, "fu_p": base_compound.fu_p,
                "compound_type": base_compound.compound_type,
                "R_bp": base_compound.R_bp,
                "ka": base_compound.ka, "Fa": base_compound.Fa,
                "Fg": base_compound.Fg,
                "CL_int": base_compound.CL_int,
                "CL_renal": base_compound.CL_renal,
            }
            kw.update(params_dict)
            c = CompoundSpec(**kw)
            kp_override = (predict_kp_all(c, kp_method_enum)
                           if kp_method_enum != KpMethod.RODGERS_ROWLAND
                           else None)
            model = PBPKModel(c, phys, kp_override=kp_override)
            dosing = DosingProtocol(dose_mg=dose_mg, route=Route(route))
            cfg = SimulationConfig(duration_h=duration_h, n_timepoints=500)
            r = model.simulate(dosing, cfg)
            return calculate_pk_parameters(
                r.time, r.venous_plasma,
                dose_mg=dose_mg,
                is_iv=route in ("iv_bolus", "iv_infusion"),
            )

        # --- Run sensitivity ---
        result = local_sensitivity(
            simulate_fn=_simulate,
            base_params=base_params,
            param_names=param_list,
            pk_metric=pk_metric,
            perturbation=perturbation,
        )

        # --- Format output with method header + provenance ---
        out = [
            f"# Sensitivity Analysis — {base_compound.name}\n",
            f"**Method**: One-at-a-time (OAT) local sensitivity, "
            f"central-difference normalized.  ",
            f"**Compound**: {base_compound.name} "
            f"({'library' if name and name.lower() in COMPOUND_LIBRARY else 'custom'})  ",
            f"**Dose / route / duration**: {dose_mg} mg / {route} / {duration_h} h  ",
            f"**Subject**: {body_weight} kg {sex}, {age}y  ",
            f"**Kp method**: {kp_method}\n",
            result.to_markdown(),
        ]
        return "\n".join(out)

    # ----------------------------------------------------------------
    # Disease states
    # ----------------------------------------------------------------

    @mcp.tool()
    def disease_state(
        disease_type: str = "ckd",
        stage: str = "moderate",
    ) -> str:
        """
        Show disease state parameter multipliers vs healthy baseline.

        Disease types:
          - ckd: Chronic Kidney Disease (stages: normal, mild, moderate, severe, esrd)
          - hepatic: Hepatic Impairment (stages: normal, mild_A, moderate_B, severe_C)

        Args:
            disease_type: "ckd" or "hepatic".
            stage: Disease severity stage.
        """
        return format_disease_profile(disease_type, stage)

    # ----------------------------------------------------------------
    # PKSimDB queries
    # ----------------------------------------------------------------

    @mcp.tool()
    def pksim_ontogeny(molecule: str = "CYP3A4") -> str:
        """
        Query PK-Sim database for enzyme/protein ontogeny data.

        Returns age-dependent maturation factors from PKSimDB.sqlite.
        Available: CYP1A2, CYP2C8/9/18/19, CYP2D6, CYP2E1, CYP3A4/5/7,
        UGT1A1/4/6/9, UGT2B4/7, AGP, ALB

        Args:
            molecule: Enzyme name (e.g., "CYP3A4", "UGT2B7", "ALB").
        """
        from core.pksim_db import format_ontogeny
        return format_ontogeny(molecule)

    @mcp.tool()
    def pksim_organ_volumes(
        age: float = 30.0,
        gender: str = "male",
        population: str = "European_ICRP_2002",
    ) -> str:
        """
        Query PK-Sim database for organ volumes at specific age/sex/population.

        Populations: European_ICRP_2002, Asian_Tanaka_1996,
        BlackAmerican_NHANES_1997, WhiteAmerican_NHANES_1997,
        Japanese_Population, Pregnant, Preterm

        Args:
            age: Age in years.
            gender: "male" or "female".
            population: Population identifier.
        """
        from core.pksim_db import format_organ_volumes
        g = 1 if gender.lower() == "male" else 2
        return format_organ_volumes(age, g, population)

    @mcp.tool()
    def pksim_transporters() -> str:
        """List all transporters in PK-Sim database with direction (influx/efflux)."""
        from core.pksim_db import list_transporters as db_list_transporters
        transporters = db_list_transporters()
        lines = [
            "## PK-Sim Transporters\n",
            "| Gene | Name | Direction |",
            "|------|------|-----------|",
        ]
        for t in transporters:
            lines.append(f"| {t['gene']} | {t['name']} | {t['direction']} |")
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Drug property lookup
    # ----------------------------------------------------------------

    @mcp.tool()
    def drug_properties(drug_name: str = "midazolam") -> str:
        """
        Look up drug physicochemical properties.

        Searches offline curated DB (16 common drugs) then ChEMBL API.

        Args:
            drug_name: Drug name (e.g., "midazolam", "atorvastatin").
        """
        result = lookup_drug(drug_name)
        if result:
            return result.to_markdown()
        return f"Drug '{drug_name}' not found in offline DB or ChEMBL."

    # ----------------------------------------------------------------
    # Observed data fitting
    # ----------------------------------------------------------------

    @mcp.tool()
    def fit_to_observed(
        csv_file: str = "",
        compound_name: str = "caffeine",
        dose_mg: float = 200.0,
        route: str = "oral",
    ) -> str:
        """
        Fit PBPK model parameters to observed PK data from CSV file.

        CSV format: two columns (time_h, concentration_mg_L).
        Fits CL_int and ka (oral) to minimize weighted residuals.

        Args:
            csv_file: Path to CSV file with observed data.
            compound_name: Drug name from library.
            dose_mg: Dose (mg).
            route: "oral" or "iv_bolus".
        """
        if not csv_file:
            return "Please provide csv_file path."
        from core.data_fitting import fit_pbpk_to_data
        return fit_pbpk_to_data(csv_file, compound_name, dose_mg, route)

    # ----------------------------------------------------------------
    # PK-Sim XML import
    # ----------------------------------------------------------------

    @mcp.tool()
    def import_pksim_model(filepath: str = "") -> str:
        """
        Import a PK-Sim simulation XML file (.pkml).

        Extracts compound properties, organ volumes, and parameters.

        Args:
            filepath: Path to PK-Sim .pkml or .xml file.
        """
        if not filepath:
            return "Please provide filepath to PK-Sim .pkml file."
        if not os.path.isfile(filepath):
            return f"File not found: {filepath}"
        from core.pksim_import import parse_pksim_xml, format_pksim_import
        try:
            result = parse_pksim_xml(filepath)
        except Exception as e:
            return f"Failed to parse PK-Sim file: {type(e).__name__}: {e}"
        return format_pksim_import(result)

    # ----------------------------------------------------------------
    # Transporter-mediated clearance
    # ----------------------------------------------------------------

    @mcp.tool()
    def transporter_clearance(
        organ: str = "liver",
        profile: str = "statins",
        CLint_met: float = 10.0,
        fu_p: float = 0.02,
        R_bp: float = 1.0,
        Q_h: float = 90.0,
    ) -> str:
        """
        Calculate transporter-mediated organ clearance using extended clearance concept.

        Profiles: "statins" (OATP1B1/1B3 + MRP2/BCRP),
                  "renal" (OAT1/OCT2 + MATE1),
                  "gut_efflux" (P-gp + BCRP)

        Shows: PS_influx, PS_efflux, CLint_overall, Kp_uu, CL_organ

        Args:
            organ: "liver", "kidney", or "gut".
            profile: Transporter profile preset.
            CLint_met: Metabolic intrinsic clearance (L/h).
            fu_p: Fraction unbound in plasma.
            R_bp: Blood:plasma ratio.
            Q_h: Organ blood flow (L/h).
        """
        # Auto-map organ → profile so users can pass organ="kidney" naturally
        organ_to_profile = {"liver": "statins", "kidney": "renal", "gut": "gut_efflux"}
        effective_profile = organ_to_profile.get(organ.lower(), profile)

        if effective_profile == "statins":
            t = statins_liver_transporters()
        elif effective_profile == "renal":
            t = renal_transporters()
        elif effective_profile == "gut_efflux":
            t = gut_efflux_transporters()
        else:
            return f"Unknown profile: {effective_profile}"

        result = format_transporter_profile(t)

        # Extended clearance calculation
        from core.hepatic_models import cl_hepatic_extended
        ext = cl_hepatic_extended(
            PS_inf=t.PS_influx_total, PS_eff=t.PS_efflux_total,
            CLint_met=CLint_met, fu_p=fu_p, R_bp=R_bp, Q_h=Q_h,
        )
        result += f"\n\n### Extended Clearance Result\n"
        result += f"| Parameter | Value |\n|-----------|-------|\n"
        result += f"| CLint_overall | {ext['CLint_overall']:.2f} L/h |\n"
        result += f"| Kp,uu (liver) | {ext['Kp_uu_liver']:.2f} |\n"
        result += f"| CL_organ | {ext['CL_h']:.2f} L/h |\n"
        result += f"| E_h | {ext['E_h']:.3f} |\n"
        return result

    # ----------------------------------------------------------------
    # Dynamic DDI
    # ----------------------------------------------------------------

    @mcp.tool()
    def run_dynamic_ddi(
        # Victim drug
        victim_name: str = "",
        victim_dose_mg: float = 5.0,
        victim_route: str = "oral",
        victim_n_doses: int = 1,
        victim_interval_h: float = 24.0,
        victim_first_dose_h: float = 0.0,
        victim_logP: float = 0.0,
        victim_pKa: float = 7.0,
        victim_fu_p: float = 1.0,
        victim_compound_type: str = "neutral",
        victim_R_bp: float = 1.0,
        victim_mw: float = 300.0,
        victim_ka: float = 1.0,
        victim_CL_int: float = 10.0,
        # Perpetrator drug
        perp_name: str = "",
        perp_dose_mg: float = 200.0,
        perp_route: str = "oral",
        perp_n_doses: int = 1,
        perp_interval_h: float = 24.0,
        perp_first_dose_h: float = 0.0,
        perp_logP: float = 0.0,
        perp_pKa: float = 7.0,
        perp_fu_p: float = 1.0,
        perp_compound_type: str = "neutral",
        perp_R_bp: float = 1.0,
        perp_mw: float = 300.0,
        perp_ka: float = 1.0,
        perp_CL_int: float = 10.0,
        # DDI mechanism
        ddi_mechanism: str = "reversible",
        Ki: Optional[float] = None,
        KI: Optional[float] = None,
        kinact: Optional[float] = None,
        Emax: Optional[float] = None,
        EC50: Optional[float] = None,
        fm: float = 0.9,
        # Simulation
        duration_h: float = 72.0,
        body_weight: float = 73.0,
        sex: str = "male",
        n_liver_segments: int = 5,
    ) -> str:
        """
        Run dynamic DDI simulation with two drugs simultaneously.

        Both drugs are simulated in one ODE system. The perpetrator's
        time-varying liver concentration modulates the victim's CL_int
        via enzyme pool dynamics.

        DDI mechanisms:
          - reversible: instant CL reduction by Ki
          - mbi: enzyme inactivation (KI, kinact) with turnover
          - induction: enzyme synthesis increase (Emax, EC50)
          - combined: all three simultaneously

        Dosing: Each drug has independent dose, route, schedule, start time.
        Example: perpetrator QD×3 days, then victim single dose on day 4.

        Args:
            victim_*: Victim drug properties and dosing.
            perp_*: Perpetrator drug properties and dosing.
            ddi_mechanism: "reversible", "mbi", "induction", or "combined".
            Ki: Reversible inhibition constant (µM).
            KI: MBI half-maximal concentration (µM).
            kinact: MBI max inactivation rate (1/h).
            Emax: Induction max fold. EC50: Induction half-max (µM).
            fm: Fraction of victim metabolism by affected CYP.
            duration_h: Total simulation time (h).
        """
        # Build victim
        # Hard validations (range-check on custom compounds,
        # DDI-mechanism prerequisite check)
        from core.invariants import (
            check_compound_ranges, check_ddi_ranges, raise_on_violations,
        )
        # DDI mechanism prerequisites: each mechanism requires specific params
        if ddi_mechanism in ("reversible", "combined") and Ki is None:
            raise ValueError(
                f"ddi_mechanism='{ddi_mechanism}' requires Ki (reversible "
                f"inhibition constant, µM). Currently Ki=None — provide a value "
                f"or change mechanism."
            )
        if ddi_mechanism in ("mbi", "combined") and (KI is None or kinact is None):
            raise ValueError(
                f"ddi_mechanism='{ddi_mechanism}' requires both KI (µM) and "
                f"kinact (1/h) for mechanism-based inhibition. "
                f"KI={KI}, kinact={kinact}."
            )
        if ddi_mechanism == "induction" and (Emax is None or EC50 is None):
            raise ValueError(
                f"ddi_mechanism='induction' requires Emax (fold) and EC50 (µM). "
                f"Emax={Emax}, EC50={EC50}."
            )
        # DDI parameter ranges
        raise_on_violations(check_ddi_ranges(
            Ki=Ki, KI=KI, kinact=kinact, Emax=Emax, EC50=EC50, fm=fm,
        ))
        # Range-check custom victim/perp physchem
        if not (victim_name and victim_name.lower() in COMPOUND_LIBRARY):
            raise_on_violations(check_compound_ranges(
                mw=victim_mw if victim_mw != 300.0 else None,
                logP=victim_logP if victim_logP != 0.0 else None,
                pKa=victim_pKa if victim_pKa != 7.0 else None,
                fu_p=victim_fu_p, R_bp=victim_R_bp, ka=victim_ka,
                CL_int=victim_CL_int if victim_CL_int > 0 else None,
            ))
        if not (perp_name and perp_name.lower() in COMPOUND_LIBRARY):
            raise_on_violations(check_compound_ranges(
                mw=perp_mw if perp_mw != 300.0 else None,
                logP=perp_logP if perp_logP != 0.0 else None,
                pKa=perp_pKa if perp_pKa != 7.0 else None,
                fu_p=perp_fu_p, R_bp=perp_R_bp, ka=perp_ka,
                CL_int=perp_CL_int if perp_CL_int > 0 else None,
            ))

        if victim_name.lower() in COMPOUND_LIBRARY:
            v_comp = COMPOUND_LIBRARY[victim_name.lower()]
        else:
            v_comp = CompoundSpec(
                name=victim_name or "Victim", mw=victim_mw, logP=victim_logP,
                pKa=victim_pKa, fu_p=victim_fu_p,
                compound_type=CompoundType(victim_compound_type),
                R_bp=victim_R_bp, ka=victim_ka, CL_int=victim_CL_int,
            )

        # Build perpetrator
        if perp_name.lower() in COMPOUND_LIBRARY:
            p_comp = COMPOUND_LIBRARY[perp_name.lower()]
        else:
            p_comp = CompoundSpec(
                name=perp_name or "Perpetrator", mw=perp_mw, logP=perp_logP,
                pKa=perp_pKa, fu_p=perp_fu_p,
                compound_type=CompoundType(perp_compound_type),
                R_bp=perp_R_bp, ka=perp_ka, CL_int=perp_CL_int,
            )

        victim_spec = DDIDrugSpec(v_comp, victim_dose_mg, victim_route,
                                   victim_n_doses, victim_interval_h, victim_first_dose_h)
        perp_spec = DDIDrugSpec(p_comp, perp_dose_mg, perp_route,
                                 perp_n_doses, perp_interval_h, perp_first_dose_h)

        mech = DDIMechanism(fm=fm)
        if Ki: mech.Ki = Ki
        if KI: mech.KI = KI
        if kinact: mech.kinact = kinact
        if Emax: mech.Emax = Emax
        if EC50: mech.EC50 = EC50

        phys = get_physiology(body_weight, Sex(sex))
        result = simulate_ddi(victim_spec, perp_spec, mech, phys, duration_h,
                              n_liver_segments=n_liver_segments)

        output = [
            f"# Dynamic DDI: {v_comp.name} + {p_comp.name}\n",
            f"**Victim:** {v_comp.name} {victim_dose_mg}mg {victim_route} "
            f"(first dose at t={victim_first_dose_h}h)\n",
            f"**Perpetrator:** {p_comp.name} {perp_dose_mg}mg {perp_route} "
            f"(first dose at t={perp_first_dose_h}h)\n",
            f"**Mechanism:** {ddi_mechanism} (fm={fm})\n",
            result.to_markdown(),
        ]
        return "\n".join(output)

    # ----------------------------------------------------------------
    # User guide
    # ----------------------------------------------------------------

    @mcp.tool()
    def pbpk_help() -> str:
        """
        Show the complete PBPK parameter input guide.

        Lists all required, recommended, and optional parameters
        organized by tier, with descriptions and defaults.
        """
        from prompts.user_guide import format_user_guide, count_all_parameters
        guide = format_user_guide()
        counts = count_all_parameters()
        guide += f"\n\n**Total configurable parameters: {counts['total']}** "
        guide += f"(Required: {counts['tier1_required']}, "
        guide += f"Recommended: {counts['tier2_recommended']}, "
        guide += f"Optional: {counts['tier3_optional']})"
        guide += (
            "\n\n## Kp Method Selection Guide\n\n"
            "Choose via `kp_method=` in `run_pbpk_simulation`.\n\n"
            "| Compound Type | Recommended | Rationale |\n"
            "|---|---|---|\n"
            "| Neutral / Weak base (general) | `rodgers_rowland` | Default, validated across classes |\n"
            "| Lipophilic base (logP>3, e.g. Midazolam) | `poulin_theil` | R&R over-predicts adipose for lipophilic bases |\n"
            "| Highly-bound acid (fu_p<0.01, e.g. Warfarin) | `berezhkovskiy` or `pksim_standard` | R&R under-predicts Vss for low-fu acids (Rodgers 2006) |\n"
            "| Hydrophilic (logP<0, e.g. Metformin) | `rodgers_rowland` | Ionic partitioning model adequate |\n"
            "| Membrane-associated, very lipophilic (logP>5) | `kp_membrane` | Membrane-specific partitioning |\n"
            "\n"
            "Use `compare_kp_methods` tool to see side-by-side Kp values before committing.\n"
        )
        return guide


def _generate_plot(
    result: SimulationResult,
    pk,
    compound_name: str,
) -> Optional[str]:
    """Generate a basic concentration-time plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Venous plasma concentration (linear + log)
        ax1.plot(result.time, result.venous_plasma, "b-", linewidth=2, label="Venous Plasma")
        ax1.set_xlabel("Time (h)")
        ax1.set_ylabel("Concentration (mg/L)")
        ax1.set_title(f"{compound_name} — Venous Plasma")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: Log scale with key organs
        organs_to_show = ["liver", "kidney", "gut", "brain", "muscle"]
        colors = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#3498db"]

        ax2.plot(result.time, result.venous_plasma, "b-", linewidth=2,
                 label="Venous Plasma", alpha=0.8)
        for organ_name, color in zip(organs_to_show, colors):
            if organ_name in result.concentrations:
                conc = result.concentrations[organ_name]
                ax2.plot(result.time, conc, color=color, linewidth=1.5,
                         label=organ_name.capitalize(), alpha=0.7)

        ax2.set_yscale("log")
        ax2.set_xlabel("Time (h)")
        ax2.set_ylabel("Concentration (mg/L)")
        ax2.set_title(f"{compound_name} — Tissue Concentrations")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, alpha=0.3)

        # Filter out zero/negative values for log scale
        ymin = min(result.venous_plasma[result.venous_plasma > 0].min() * 0.1
                   if np.any(result.venous_plasma > 0) else 1e-6, 1e-6)
        ax2.set_ylim(bottom=ymin)

        fig.suptitle(f"PBPK Simulation — {compound_name} ({result.dose_info})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        # Save to temp file
        plot_dir = os.path.expanduser("~/Desktop")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(
            plot_dir,
            f"pbpk_{compound_name.lower().replace(' ', '_')}.png"
        )
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return plot_path

    except Exception as e:
        return None


def _generate_organ_plot(
    result: SimulationResult,
    organ_list: list[str],
    compound_name: str,
    log_scale: bool = True,
) -> Optional[str]:
    """Generate a plot for specified organs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.tab10.colors
        color_idx = 0

        for organ_name in organ_list:
            if organ_name == "plasma":
                ax.plot(result.time, result.venous_plasma,
                       color=colors[color_idx % len(colors)],
                       linewidth=2, label="Venous Plasma")
            elif organ_name in result.concentrations:
                conc = result.concentrations[organ_name]
                ax.plot(result.time, conc,
                       color=colors[color_idx % len(colors)],
                       linewidth=1.5, label=organ_name.capitalize())
            color_idx += 1

        if log_scale:
            ax.set_yscale("log")
            # Set sensible y limits
            all_concs = []
            for name in organ_list:
                if name == "plasma":
                    all_concs.extend(result.venous_plasma[result.venous_plasma > 0])
                elif name in result.concentrations:
                    c = result.concentrations[name]
                    all_concs.extend(c[c > 0])
            if all_concs:
                ymin = min(all_concs) * 0.1
                ax.set_ylim(bottom=max(ymin, 1e-10))

        ax.set_xlabel("Time (h)", fontsize=12)
        ax.set_ylabel("Concentration (mg/L)", fontsize=12)
        ax.set_title(f"PBPK — {compound_name} ({result.dose_info})",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_dir = os.path.expanduser("~/Desktop")
        os.makedirs(plot_dir, exist_ok=True)
        organs_str = "_".join(organ_list[:3])
        plot_path = os.path.join(
            plot_dir,
            f"pbpk_{compound_name.lower().replace(' ', '_')}_{organs_str}.png"
        )
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return plot_path

    except Exception as e:
        return None


# Need numpy for plot functions
import numpy as np
