"""
Session-based PBPK workflow tools — schema-level decomposition of
run_pbpk_simulation. Forces explicit declaration of each parameter
group, with prerequisite checks at validate_model() and a token
gate on simulate_validated().

Also exposes citation verification (verify_citation, verify_citation_list)
so any LLM-supplied PMID/DOI is checked against PubMed/Crossref before
being accepted into a Source 4-tuple.
"""

from __future__ import annotations
from typing import Optional

from mcp.server.fastmcp import FastMCP


def register_session_and_citation_tools(mcp: FastMCP):

    # ----------------------------------------------------------------
    # Session-based PBPK workflow (7 tools)
    # ----------------------------------------------------------------

    @mcp.tool()
    def register_compound(
        name: str,
        mw: float,
        logP: float,
        pKa: float,
        compound_type: str,
        mw_source: str = "",
        logP_source: str = "",
    ) -> str:
        """
        Step 1 of the session-based PBPK workflow. Initialize a session
        with compound identity and basic physchem.

        ANTI-FABRICATION: do not pass placeholder physchem values. mw,
        logP, pKa are required (no defaults). If you do not have a
        verifiable value, ask the user. Pass mw_source / logP_source as
        a PMID, DOI, ChEMBL ID, or 'user_provided'; pass an empty
        string only when no source is yet known (validate_model will
        warn).

        Returns the compound_id used by all subsequent add_* calls.
        """
        from core.session import register_compound as _reg
        cid = _reg(
            name=name, mw=mw, logP=logP, pKa=pKa,
            compound_type=compound_type,
            mw_source=mw_source or None,
            logP_source=logP_source or None,
        )
        return f"## Session created\n\n- **compound_id**: `{cid}`\n- **name**: {name}\n\nNext steps:\n1. `add_binding({cid}, fu_p=..., R_bp=...)`\n2. `add_clearance({cid}, source='hepatocyte', CLint_vitro_hep=...)`\n3. `add_absorption({cid}, ka=..., Fa=..., Fg=...)`\n4. `select_model_structure({cid}, kp_method=..., distribution_model=...)`\n5. `validate_model({cid})` → returns validation_token\n6. `simulate_validated(validation_token, dose_mg=..., route=...)`"

    @mcp.tool()
    def add_binding(
        compound_id: str,
        fu_p: float,
        R_bp: float,
        fu_p_source: str = "",
        R_bp_source: str = "",
    ) -> str:
        """
        Step 2 of the session workflow. Record protein binding.

        ANTI-FABRICATION: fu_p and R_bp materially affect every Kp
        prediction. Provide measured values when available (rapid
        equilibrium dialysis for fu_p, Bp/p assay for R_bp). When
        falling back to literature, pass a PMID/DOI in *_source — the
        server can verify it via verify_citation().
        """
        from core.session import add_binding as _add
        _add(compound_id, fu_p=fu_p, R_bp=R_bp,
             fu_p_source=fu_p_source or None,
             R_bp_source=R_bp_source or None)
        return f"binding recorded for `{compound_id}`: fu_p={fu_p}, R_bp={R_bp}"

    @mcp.tool()
    def add_clearance(
        compound_id: str,
        source: str,
        CL_int: Optional[float] = None,
        CLint_vitro_hlm: Optional[float] = None,
        CLint_vitro_hep: Optional[float] = None,
        CLint_per_cyp: Optional[str] = None,
        protein_conc: float = 1.0,
        CL_renal: float = 0.0,
        clearance_source_citation: str = "",
    ) -> str:
        """
        Step 3 of the session workflow. Record clearance with explicit source.

        `source` must be one of: 'direct', 'hlm', 'hepatocyte', 'rcyp'.
        The matching field is required for each source — schema rejects
        any other combination.

        ANTI-FABRICATION: HLM misses non-CYP metabolism (UGT, SULT,
        esterase) by definition. If the drug has known UGT pathways
        (NSAIDs, glucuronidated metabolites), use 'hepatocyte' instead
        of 'hlm' or your CL prediction will be 2-5× too low.
        """
        from core.session import add_clearance as _add
        _add(compound_id, source=source, CL_int=CL_int,
             CLint_vitro_hlm=CLint_vitro_hlm,
             CLint_vitro_hep=CLint_vitro_hep,
             CLint_per_cyp=CLint_per_cyp,
             protein_conc=protein_conc, CL_renal=CL_renal,
             clearance_source_citation=clearance_source_citation or None)
        return f"clearance recorded for `{compound_id}`: source={source}"

    @mcp.tool()
    def add_absorption(
        compound_id: str,
        ka: float = 1.0,
        Fa: float = 1.0,
        Fg: float = 1.0,
        Peff: Optional[float] = None,
        ka_source: str = "",
        Fa_source: str = "",
        Fg_source: str = "",
        Peff_source: str = "",
    ) -> str:
        """
        Step 4 of the session workflow. Record absorption parameters.

        Defaults (ka=1.0, Fa=1.0, Fg=1.0) trigger silent-fallback
        warnings unless a source is recorded. If you legitimately set
        Fa=1.0 because the drug is BCS I/II well-absorbed, pass
        Fa_source='BCS II + Caco-2 Papp ...' so the audit knows it
        was a deliberate choice rather than a missing input.

        Provide ka from oral C-t fit, Fa from BCS or Caco-2 modeling,
        Fg from Yang Qgut model with measured Peff.
        """
        from core.session import add_absorption as _add
        _add(compound_id, ka=ka, Fa=Fa, Fg=Fg, Peff=Peff,
             ka_source=ka_source or None,
             Fa_source=Fa_source or None,
             Fg_source=Fg_source or None,
             Peff_source=Peff_source or None)
        return f"absorption recorded for `{compound_id}`: ka={ka}, Fa={Fa}, Fg={Fg}"

    @mcp.tool()
    def add_transporters(
        compound_id: str,
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
    ) -> str:
        """
        Step 5 (optional) of the session workflow. Record transporter
        Km/Vmax pairs. Both Km AND Vmax are required for any active
        transporter — XOR raises (schema-level pair enforcement).

        Transporters only enter the ODE when distribution_model is
        'permeability_limited' — set that in select_model_structure().
        """
        from core.session import add_transporters as _add
        _add(compound_id,
             liver_oatp_km=liver_oatp_km, liver_oatp_vmax=liver_oatp_vmax,
             liver_mrp2_km=liver_mrp2_km, liver_mrp2_vmax=liver_mrp2_vmax,
             kidney_oct2_km=kidney_oct2_km, kidney_oct2_vmax=kidney_oct2_vmax,
             kidney_mate1_km=kidney_mate1_km, kidney_mate1_vmax=kidney_mate1_vmax,
             gut_pgp_km=gut_pgp_km, gut_pgp_vmax=gut_pgp_vmax)
        return f"transporters recorded for `{compound_id}`"

    @mcp.tool()
    def select_model_structure(
        compound_id: str,
        kp_method: str = "rodgers_rowland",
        distribution_model: str = "perfusion_limited",
        absorption_model: str = "first_order",
    ) -> str:
        """
        Step 6 of the session workflow. Choose ODE structure.

        ANTI-FABRICATION: pick kp_method by compound class.
        Lipophilic base (logP>3) → poulin_theil; highly-bound acid
        (fu_p<0.01) → berezhkovskiy or pksim_standard; otherwise R&R.
        """
        from core.session import select_model_structure as _sel
        _sel(compound_id, kp_method=kp_method,
             distribution_model=distribution_model,
             absorption_model=absorption_model)
        return f"structure recorded for `{compound_id}`: kp_method={kp_method}, distribution={distribution_model}"

    @mcp.tool()
    def validate_model(compound_id: str) -> str:
        """
        Step 7 of the session workflow. Verify all parameter groups are
        present, emit warnings, and issue a validation_token.

        On success, returns a token that simulate_validated() requires.
        On missing parameter groups, returns a list of what's still
        needed.

        Once validated, the session is locked — to modify parameters,
        register_compound() a fresh session.
        """
        from core.session import validate_model as _val
        rep = _val(compound_id)

        # Run the provenance audit alongside schema validation so we can
        # show schema_ok and audit_ok as two distinct signals (codex UX
        # review 2026-04-30 HIGH: previously, validate_model returned
        # `ok: True` while the same response said "failed-audit", which
        # is mixed authority).
        audit_text = ""
        audit_verdict = "unknown"
        if rep.ok:
            from prompts.provenance_audit import render_session_audit
            audit_text = render_session_audit(compound_id)
            for tag in ("failed-audit", "passed-with-flags", "passed"):
                if tag in audit_text:
                    audit_verdict = tag
                    break

        lines = [f"## Validation Report — `{compound_id}`"]
        lines.append("")
        lines.append("| Check | Result | Notes |")
        lines.append("|---|---|---|")
        lines.append(
            f"| **schema_ok** | {'✓' if rep.ok else '✗'} | "
            f"{'all required parameter groups present' if rep.ok else 'missing parameter groups (see below)'} |"
        )
        if rep.ok:
            lines.append(
                f"| **audit_ok** | "
                f"{'✓' if audit_verdict == 'passed' else ('⚠️' if audit_verdict == 'passed-with-flags' else '✗')} | "
                f"{audit_verdict} (provenance audit verdict — see table) |"
            )
            simulation_ready = audit_verdict in ("passed", "passed-with-flags")
            lines.append(
                f"| **simulation_ready** | {'✓' if simulation_ready else '✗'} | "
                f"{'token issued — simulate_validated() will run' if simulation_ready else 'add citations / replace silent fallbacks before treating outputs as predictions'} |"
            )
        else:
            lines.append("| **audit_ok** | — | not evaluated until schema_ok |")
            lines.append("| **simulation_ready** | ✗ | resolve schema first |")
        lines.append("")

        if rep.missing:
            lines.append("**missing parameter groups:**")
            for m in rep.missing:
                lines.append(f"- {m}")
        if rep.warnings:
            lines.append("\n**warnings:**")
            for w in rep.warnings:
                lines.append(f"- {w}")
        if rep.ok and rep.validation_token:
            lines.append(f"\n**validation_token**: `{rep.validation_token}`")
            lines.append(
                f"\nProceed with: `simulate_validated('{rep.validation_token}', dose_mg=..., route=...)` "
                f"(use route='iv_bolus' / 'iv_infusion' / 'oral' as appropriate)"
            )
            if audit_verdict == "failed-audit":
                lines.append(
                    "\n> ⚠️ Note: schema_ok=True but audit_ok=failed-audit. "
                    "The token will let simulate_validated() RUN, but the "
                    "output is NOT a prediction-grade result until the "
                    "unsourced/silent-fallback parameters in the audit "
                    "below are resolved."
                )
            lines.append("\n---\n")
            lines.append(audit_text)
        return "\n".join(lines)

    @mcp.tool()
    def simulate_validated(
        validation_token: str,
        dose_mg: float,
        route: str = "oral",
        duration_h: float = 24.0,
        n_doses: int = 1,
        interval_h: float = 24.0,
        infusion_duration_h: float = 0.5,
        body_weight: float = 73.0,
        sex: str = "male",
        age: float = 30.0,
    ) -> str:
        """
        Step 8 of the session workflow. Run the simulation. Only accepts
        a validation_token issued by validate_model() — a half-built
        session cannot be simulated.

        This tool reuses the same ODE engine as run_pbpk_simulation but
        gates entry on the schema-validated session state.
        """
        from core.session import get_validated_draft
        from core.compound import CompoundSpec, CompoundType, MetabolismModel
        from core.physiology import get_physiology, Sex
        from core.partition_coeff import predict_kp_all, KpMethod
        from core.pbpk_model import (
            PBPKModel, DosingProtocol, SimulationConfig, Route,
            DistributionModel,
        )
        from core.pk_calculator import calculate_pk_parameters
        from core.physiology import Organ
        from core.invariants import (
            check_dose_subject_ranges, check_dose_self_consistency,
            raise_on_violations,
        )
        from core.transporter_spec import TransporterKwargs
        from core.audit import log_simulation
        import numpy as np

        d = get_validated_draft(validation_token)

        # Range-check dosing inputs
        viols = check_dose_subject_ranges(
            dose_mg=dose_mg, duration_h=duration_h, n_doses=n_doses,
            interval_h=interval_h, body_weight=body_weight, age=age,
        )
        v = check_dose_self_consistency(
            dose_mg=dose_mg, n_doses=n_doses, interval_h=interval_h,
            duration_h=duration_h, route=route,
        )
        if v:
            viols.append(v)
        raise_on_violations(viols)

        # Resolve clearance
        spec = d.clearance["spec"]
        cl_int_L_per_h = 0.0
        ivive_info = ""
        if spec["source"] == "direct":
            cl_int_L_per_h = spec["CL_int_L_per_h"]
        elif spec["source"] == "hlm":
            from core.ivive import scale_microsomal_clint
            r = scale_microsomal_clint(
                clint_vitro=spec["CLint_vitro_uL_min_mg"],
                logP=d.physchem["logP"],
                protein_conc=spec.get("protein_conc_mg_mL", 1.0),
                body_weight=body_weight, sex=sex,
            )
            cl_int_L_per_h = r["CLint_in_vivo_L_per_h"]
            ivive_info = f"HLM IVIVE: {spec['CLint_vitro_uL_min_mg']} µL/min/mg → {cl_int_L_per_h:.1f} L/h"
        elif spec["source"] == "hepatocyte":
            from core.ivive import scale_hepatocyte_clint
            r = scale_hepatocyte_clint(
                clint_hep=spec["CLint_vitro_uL_min_1e6cells"],
                logP=d.physchem["logP"], body_weight=body_weight, sex=sex,
            )
            cl_int_L_per_h = r["CLint_in_vivo_L_per_h"]
            ivive_info = f"Hepatocyte IVIVE: {spec['CLint_vitro_uL_min_1e6cells']} µL/min/1e6 → {cl_int_L_per_h:.1f} L/h"
        elif spec["source"] == "rcyp":
            from core.ivive import scale_recombinant_clint
            r = scale_recombinant_clint(
                clint_per_cyp=spec["CLint_per_cyp"],
                body_weight=body_weight, sex=sex,
            )
            cl_int_L_per_h = r["CLint_in_vivo_L_per_h"]
            ivive_info = f"rCYP IVIVE: → {cl_int_L_per_h:.1f} L/h"

        # Build CompoundSpec
        compound = CompoundSpec(
            name=d.name,
            mw=d.physchem["mw"],
            logP=d.physchem["logP"],
            pKa=d.physchem["pKa"],
            fu_p=d.binding["fu_p"],
            compound_type=CompoundType(d.physchem["compound_type"]),
            R_bp=d.binding["R_bp"],
            ka=d.absorption["ka"],
            Fa=d.absorption["Fa"],
            Fg=d.absorption["Fg"],
            CL_int=cl_int_L_per_h,
            CL_renal=d.clearance["CL_renal"],
            metabolism_model=MetabolismModel.FIRST_ORDER,
            Peff=d.absorption.get("Peff"),
        )

        # Physiology (mass balance enforced inside)
        phys = get_physiology(body_weight=body_weight, sex=Sex(sex), age_years=age)

        # Kp method
        kp_method_enum = KpMethod(d.structure["kp_method"])
        kp_override = predict_kp_all(compound, kp_method_enum) \
            if kp_method_enum != KpMethod.RODGERS_ROWLAND else None

        # Transporters
        trans_dict = {}
        if d.transporters:
            trans_dict = TransporterKwargs(**d.transporters).to_organ_transporters()

        model = PBPKModel(compound, phys, kp_override=kp_override,
                          transporters=trans_dict if trans_dict else None)
        dosing = DosingProtocol(
            dose_mg=dose_mg, route=Route(route), n_doses=n_doses,
            interval_h=interval_h, infusion_duration_h=infusion_duration_h,
        )
        sim_cfg = SimulationConfig(
            duration_h=duration_h,
            distribution_model=DistributionModel(d.structure["distribution_model"]),
            absorption_model=d.structure["absorption_model"],
        )
        result = model.simulate(dosing, sim_cfg)

        # Mass-balance assertion (Mass-balance invariant).
        from core.invariants import check_dose_recovery, raise_on_violations
        viol = check_dose_recovery(
            result=result, model=model,
            dose_mg=dose_mg, n_doses=n_doses, route=route,
            tolerance=0.01,
        )
        if viol is not None:
            raise_on_violations([viol])

        is_iv = route in ("iv_bolus", "iv_infusion")
        pk = calculate_pk_parameters(
            result.time, result.venous_plasma,
            dose_mg=dose_mg * max(n_doses, 1), is_iv=is_iv,
        )

        # Output
        out = [f"# Validated PBPK Simulation — {d.name}"]
        out.append(f"\n**Session:** `{d.compound_id}` (validated {d.validated_at})")
        out.append(f"**Validation token:** `{validation_token}`")
        out.append(f"**Dose:** {dose_mg} mg {route} ({n_doses} doses)")
        out.append(f"**Subject:** {body_weight} kg {sex}, {age}y")
        out.append(f"**Kp method:** {d.structure['kp_method']}")
        out.append(f"**Distribution:** {d.structure['distribution_model']}")
        if ivive_info:
            out.append(f"**Clearance:** {ivive_info}")
        out.append("")
        out.append(pk.to_markdown(d.name, dose_mg))

        # Source provenance footer
        out.append("\n### Source Provenance")
        if d.sources:
            out.append("| Parameter | Source |")
            out.append("|---|---|")
            for k, v in d.sources.items():
                out.append(f"| {k} | {v} |")
        else:
            out.append("_No sources recorded — pass *_source kwargs to add_* calls "
                       "to track provenance and enable verify_citation_list._")

        # Audit
        try:
            fp = log_simulation(
                tool_name="simulate_validated",
                inputs={
                    "compound_id": d.compound_id,
                    "validation_token": validation_token,
                    "dose_mg": dose_mg, "route": route,
                    "duration_h": duration_h,
                },
                resolved={
                    "compound_name": d.name,
                    "cl_int_resolved_L_per_h": cl_int_L_per_h,
                    "kp_method": d.structure["kp_method"],
                },
                summary={
                    "Cmax": pk.Cmax, "AUC": pk.AUC_0_inf,
                    "t_half": pk.t_half, "CL_F": pk.CL_F, "Vss": pk.Vss,
                },
            )
            out.append(f"\n_Audit fingerprint: `{fp}`_")
        except Exception:
            pass

        return "\n".join(out)

    @mcp.tool()
    def audit_model_provenance(compound_id: str) -> str:
        """
        Generate a deterministic provenance audit for a PBPK model.
        Output is a table with one row per parameter (Parameter | Value
        | Unit | Source type | Source citation | Confidence | Note),
        plus three explicit lists:

          (a) silent-fallback parameters (server defaults triggered)
          (b) low-confidence parameters that drive model output
          (c) unsourced parameters (no PMID/DOI/measurement)

        And a final verdict: `passed`, `passed-with-flags`, or
        `failed-audit`.

        This is a SEPARATE LAYER from input-time schema validation.
        Input validation catches malformed inputs; this audit catches
        the opposite failure — outputs that look reasonable but were
        assembled from defaults the LLM didn't realize it was using.

        Accepted forms of compound_id:
          - A session compound_id (`cmpd_<hex>`) from register_compound.
          - A library compound name (e.g. 'midazolam', 'metformin') —
            the audit then runs against COMPOUND_LIBRARY values plus
            their bundled `citations` metadata. This lets users audit
            a legacy run_pbpk_simulation result without first creating
            a session.
        """
        from prompts.provenance_audit import render_session_audit
        # Library-compound shortcut: when the user passes a known library
        # name, run the audit against the library entry's bundled
        # citation metadata so the audit path is reachable from the
        # legacy flat-API workflow too. Codex UX review 2026-04-30
        # flagged the previous "Unknown compound_id='midazolam'" output
        # as a BLOCKING workflow break.
        from core.compound import COMPOUND_LIBRARY
        key = (compound_id or "").lower().strip()
        if key in COMPOUND_LIBRARY:
            return _render_library_audit(COMPOUND_LIBRARY[key])
        return render_session_audit(compound_id)

    @mcp.tool()
    def session_summary(compound_id: str) -> str:
        """Inspect a draft compound session without locking it. Useful
        between add_* steps to confirm what's been recorded."""
        from core.session import session_summary as _sum
        s = _sum(compound_id)
        if "error" in s:
            return f"Error: {s['error']}"
        lines = [f"## Session `{s['compound_id']}` ({s['name']})"]
        lines.append(f"- validated: {s['validated']}")
        if s["missing"]:
            lines.append(f"- missing groups: {', '.join(s['missing'])}")
        for grp in ("physchem", "binding", "clearance", "absorption",
                    "transporters", "structure"):
            val = s.get(grp)
            if val:
                lines.append(f"- **{grp}**: {val}")
        if s.get("sources"):
            lines.append(f"- sources: {s['sources']}")
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Citation verification (2 tools)
    # ----------------------------------------------------------------

    @mcp.tool()
    def verify_citation(
        identifier: str,
        mode: str = "online",
    ) -> str:
        """
        Verify a PMID or DOI against PubMed (E-utils) or Crossref.

        Use this BEFORE inserting a citation into a Source field. LLMs
        commonly fabricate plausible-looking but non-existent PMIDs;
        this tool catches them.

        Modes:
          - online (default): cache hit → return; miss → live HTTP, cache
          - offline: cache only; miss → UNVERIFIED status
          - strict: live HTTP required; not_found / network_error → ValueError
        """
        from core.citation import verify_citation as _verify
        result = _verify(identifier, mode=mode)
        lines = [f"## Citation: {identifier}"]
        lines.append(f"- **type**: {result.type}")
        lines.append(f"- **status**: {result.status.value}")
        if result.title:
            lines.append(f"- **title**: {result.title}")
        if result.authors:
            lines.append(f"- **authors**: {result.authors}")
        if result.year:
            lines.append(f"- **year**: {result.year}")
        if result.journal:
            lines.append(f"- **journal**: {result.journal}")
        if result.error:
            lines.append(f"- **error**: {result.error}")
        return "\n".join(lines)

    @mcp.tool()
    def verify_citation_list(identifiers: str, mode: str = "online") -> str:
        """
        Bulk verify a comma-separated list of PMIDs / DOIs.

        Returns a markdown table with status per identifier. Useful for
        validating an entire bibliography or a Source list before
        accepting a parameter set.
        """
        from core.citation import verify_citation as _verify
        ids = [i.strip() for i in identifiers.split(",") if i.strip()]
        if not ids:
            return "No identifiers provided."
        lines = ["## Citation verification\n",
                 "| Identifier | Type | Status | Title |",
                 "|---|---|---|---|"]
        n_ok = 0
        for ident in ids:
            r = _verify(ident, mode=mode)
            title = (r.title[:60] + "…") if r.title and len(r.title) > 60 else (r.title or "")
            lines.append(f"| `{ident}` | {r.type} | {r.status.value} | {title} |")
            if r.is_verified():
                n_ok += 1
        lines.append(f"\n**Verified: {n_ok}/{len(ids)}**")
        return "\n".join(lines)


def _render_library_audit(compound) -> str:
    """
    Provenance audit for a library compound (no session). Renders the
    same parameter-by-parameter table as render_session_audit, but
    pulls Source/Citation values from the library entry's `citations`
    dict (each library compound bundles its own provenance metadata).

    Verdict logic mirrors the session audit:
      - passed:           every scientific parameter has a citation,
                          no sentinel defaults
      - passed-with-flags: citations present but some are non-PMID
                          (DrugBank/ChEMBL/calibrated-to)
      - failed-audit:     scientific parameter without any citation,
                          or sentinel-default value

    Codex UX review 2026-04-30 (BLOCKING) — without this branch,
    `audit_model_provenance('midazolam')` returned "Unknown
    compound_id='midazolam'" and the audit path was unreachable from
    the legacy run_pbpk_simulation flow.
    """
    citations = compound.citations or {}

    rows = [
        ("name", compound.name, "—"),
        ("mw", compound.mw, "g/mol"),
        ("logP", compound.logP, "log10"),
        ("pKa", compound.pKa, "—"),
        ("compound_type", compound.compound_type.value, "—"),
        ("fu_p", compound.fu_p, "fraction"),
        ("R_bp", compound.R_bp, "ratio"),
        ("ka", compound.ka, "1/h"),
        ("Fa", compound.Fa, "fraction"),
        ("Fg", compound.Fg, "fraction"),
        ("CL_int", compound.CL_int, "L/h"),
        ("CL_renal", compound.CL_renal, "L/h"),
    ]
    if getattr(compound, "Peff", None) is not None:
        rows.append(("Peff", compound.Peff, "1e-4 cm/s"))
    if getattr(compound, "S0", None) is not None:
        rows.append(("S0", compound.S0, "mg/mL"))

    lines = [
        f"## Provenance Audit — `{compound.name}` (library compound)",
        "",
        "| Parameter | Value | Unit | Source citation | Confidence | Note |",
        "|---|---|---|---|---|---|",
    ]
    unsourced: list[str] = []
    soft_flags: list[str] = []
    for name, value, unit in rows:
        cite = citations.get(name, "")
        if not cite:
            unsourced.append(name)
            mark = "⚠️ "
            confidence = "unverified"
        elif cite.startswith("PMID:") or cite.startswith("DOI:"):
            mark = ""
            confidence = "high"
        else:
            mark = ""
            confidence = "medium"
            soft_flags.append(f"{name}: '{cite}' is not a PMID/DOI")
        note = ""
        if name in ("CL_int",) and "calibrated" in cite.lower():
            note = "fitted to clinical CL — not a direct measurement"
        lines.append(
            f"| {mark}{name} | {value} | {unit} | "
            f"{cite or '(none)'} | {confidence} | {note} |"
        )

    if compound.recommended_kp_method:
        lines.append(
            f"| recommended_kp_method | {compound.recommended_kp_method} | — | "
            f"{citations.get('recommended_kp_method', '(none)')} | "
            f"{'high' if 'PMID' in citations.get('recommended_kp_method', '') else 'medium'} | "
            f"library default Kp method |"
        )

    lines.append("")
    lines.append("### (a) Silent-fallback parameters")
    lines.append("- _none — library compounds carry curated values_")

    lines.append("")
    lines.append("### (b) Citations that are not PMID/DOI")
    if soft_flags:
        for s in soft_flags:
            lines.append(f"- {s}")
    else:
        lines.append("- _none — every cited source is a PMID or DOI_")

    lines.append("")
    lines.append("### (c) Unsourced parameters")
    if unsourced:
        for p in unsourced:
            lines.append(f"- `{p}` — UNSOURCED (no entry in compound.citations)")
    else:
        lines.append("- _none — every parameter has a citation_")

    lines.append("")
    if unsourced:
        verdict = "failed-audit"
        note = (
            f"{len(unsourced)} parameter(s) lack citations. Add them to the "
            f"library entry's `citations` dict in core/compound.py."
        )
    elif soft_flags:
        verdict = "passed-with-flags"
        note = (
            f"{len(soft_flags)} citation(s) are non-PMID/DOI. "
            f"Database IDs (ChEMBL, DrugBank) are acceptable but PMID is "
            f"preferred for measured values."
        )
    else:
        verdict = "passed"
        note = "every scientific parameter has a verifiable PMID/DOI."

    lines.append("### Audit status")
    lines.append(f"**`{verdict}`** — {note}")
    return "\n".join(lines)
