"""
Session-based PBPK model construction.

Decomposes the legacy `run_pbpk_simulation` (47 flat parameters) into
discrete, prerequisite-checked steps. Forces the LLM to declare each
parameter group explicitly; missing groups make `validate_model()`
fail with a list of specifics.

Workflow:
    register_compound(physchem) → compound_id
    add_binding(compound_id, ...)
    add_clearance(compound_id, ...)
    add_absorption(compound_id, ...)
    add_transporters(compound_id, ...)        # optional
    select_model_structure(compound_id, ...)
    validate_model(compound_id) → validated_model_id
    simulate_validated(validated_model_id, dosing) → result

`simulate_validated` only accepts a token issued by `validate_model`,
so a half-built session cannot be simulated.

State lives in process memory (no persistence). Sessions are GC'd
after 30 minutes of inactivity to bound memory.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

@dataclass
class CompoundDraft:
    """Mutable container that accumulates parameter groups."""
    compound_id: str
    name: str
    physchem: Optional[dict] = None        # {mw, logP, pKa, compound_type}
    binding: Optional[dict] = None          # {fu_p, R_bp, fu_inc?, fu_hep?}
    clearance: Optional[dict] = None        # ClearanceSpec dict + CL_renal
    absorption: Optional[dict] = None       # {ka, Fa, Fg, Peff?, S0?, ...}
    transporters: Optional[dict] = None     # TransporterKwargs dict
    structure: Optional[dict] = None        # {kp_method, distribution_model, absorption_model}
    sources: dict = field(default_factory=dict)   # {param: Source}
    created_at: float = field(default_factory=time.time)
    validated: bool = False
    validated_at: Optional[float] = None
    validation_token: Optional[str] = None


_SESSIONS: dict[str, CompoundDraft] = {}
_VALIDATION_TOKENS: dict[str, str] = {}    # token → compound_id
_TTL_SECONDS = 30 * 60


def _gc():
    """Remove sessions older than TTL."""
    now = time.time()
    stale = [cid for cid, d in _SESSIONS.items() if now - d.created_at > _TTL_SECONDS]
    for cid in stale:
        d = _SESSIONS.pop(cid, None)
        if d and d.validation_token:
            _VALIDATION_TOKENS.pop(d.validation_token, None)


def _new_id() -> str:
    return f"cmpd_{uuid.uuid4().hex[:12]}"


def _new_token() -> str:
    return f"vmodel_{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------
# Step 1: register_compound
# ---------------------------------------------------------------------

def register_compound(name: str, mw: float, logP: float, pKa: float,
                      compound_type: str,
                      mw_source: Optional[str] = None,
                      logP_source: Optional[str] = None) -> str:
    """Initialize a session and store basic identity + physchem.
    Returns compound_id. Subsequent calls use this ID."""
    from .invariants import check_compound_ranges, raise_on_violations
    raise_on_violations(check_compound_ranges(mw=mw, logP=logP, pKa=pKa))
    if compound_type not in ("strong_base", "moderate_base", "weak_base",
                             "acid", "neutral", "zwitterion"):
        raise ValueError(
            f"compound_type='{compound_type}' invalid. Choose one of: "
            f"strong_base, moderate_base, weak_base, acid, neutral, zwitterion."
        )
    _gc()
    cid = _new_id()
    draft = CompoundDraft(
        compound_id=cid, name=name,
        physchem={"mw": mw, "logP": logP, "pKa": pKa, "compound_type": compound_type},
    )
    if mw_source:
        draft.sources["mw"] = mw_source
    if logP_source:
        draft.sources["logP"] = logP_source
    _SESSIONS[cid] = draft
    return cid


def _get(compound_id: str) -> CompoundDraft:
    if compound_id not in _SESSIONS:
        raise ValueError(
            f"Unknown compound_id='{compound_id}'. "
            f"Call register_compound() first, or check that the session "
            f"hasn't expired (TTL: {_TTL_SECONDS // 60} min)."
        )
    if _SESSIONS[compound_id].validated:
        raise ValueError(
            f"Compound '{compound_id}' is already validated and locked. "
            f"To modify, register_compound() a fresh session."
        )
    return _SESSIONS[compound_id]


# ---------------------------------------------------------------------
# Step 2-5: add_*
# ---------------------------------------------------------------------

def add_binding(compound_id: str, fu_p: float, R_bp: float,
                fu_p_source: Optional[str] = None,
                R_bp_source: Optional[str] = None) -> None:
    from .invariants import check_compound_ranges, raise_on_violations
    raise_on_violations(check_compound_ranges(fu_p=fu_p, R_bp=R_bp))
    d = _get(compound_id)
    d.binding = {"fu_p": fu_p, "R_bp": R_bp}
    if fu_p_source:
        d.sources["fu_p"] = fu_p_source
    if R_bp_source:
        d.sources["R_bp"] = R_bp_source


def add_clearance(compound_id: str, *, source: str,
                  CL_int: Optional[float] = None,
                  CLint_vitro_hlm: Optional[float] = None,
                  CLint_vitro_hep: Optional[float] = None,
                  CLint_per_cyp: Optional[str] = None,
                  protein_conc: float = 1.0,
                  CL_renal: float = 0.0,
                  clearance_source_citation: Optional[str] = None) -> None:
    from .clearance_spec import parse_clearance_from_legacy_args
    from .invariants import check_compound_ranges, raise_on_violations
    raise_on_violations(check_compound_ranges(CL_renal=CL_renal if CL_renal > 0 else None))
    spec = parse_clearance_from_legacy_args(
        clearance_source=source, CL_int=CL_int or 0.0,
        CLint_vitro_hlm=CLint_vitro_hlm,
        CLint_vitro_hep=CLint_vitro_hep,
        CLint_per_cyp=CLint_per_cyp,
        protein_conc=protein_conc,
    )
    if spec is None:
        raise ValueError(
            f"add_clearance(source='{source}') produced no clearance spec. "
            f"Provide one of: CL_int (direct), CLint_vitro_hlm, "
            f"CLint_vitro_hep, CLint_per_cyp."
        )
    d = _get(compound_id)
    d.clearance = {
        "spec": spec.model_dump(), "CL_renal": CL_renal,
    }
    if clearance_source_citation:
        d.sources["clearance"] = clearance_source_citation


def add_absorption(compound_id: str, ka: float = 1.0, Fa: float = 1.0,
                   Fg: float = 1.0, Peff: Optional[float] = None,
                   ka_source: Optional[str] = None) -> None:
    from .invariants import check_compound_ranges, raise_on_violations
    raise_on_violations(check_compound_ranges(ka=ka, Fa=Fa, Fg=Fg, Peff=Peff))
    d = _get(compound_id)
    d.absorption = {"ka": ka, "Fa": Fa, "Fg": Fg, "Peff": Peff}
    if ka_source:
        d.sources["ka"] = ka_source


def add_transporters(compound_id: str, **transporter_pairs) -> None:
    """Pass transporter Km/Vmax pairs as kwargs (liver_oatp_km=..., etc.).
    XOR pair raises (see TransporterKwargs.from_legacy_kwargs)."""
    from .transporter_spec import TransporterKwargs
    spec = TransporterKwargs.from_legacy_kwargs(**transporter_pairs)
    d = _get(compound_id)
    d.transporters = spec.model_dump(exclude_none=True)


def select_model_structure(compound_id: str,
                           kp_method: str = "rodgers_rowland",
                           distribution_model: str = "perfusion_limited",
                           absorption_model: str = "first_order") -> None:
    from .validation import validate_kp_method
    validate_kp_method(kp_method)
    if distribution_model not in ("perfusion_limited", "permeability_limited"):
        raise ValueError(f"Invalid distribution_model='{distribution_model}'")
    d = _get(compound_id)
    d.structure = {
        "kp_method": kp_method,
        "distribution_model": distribution_model,
        "absorption_model": absorption_model,
    }


# ---------------------------------------------------------------------
# Step 6: validate_model — issues token, locks session
# ---------------------------------------------------------------------

@dataclass
class ValidationReport:
    compound_id: str
    ok: bool
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation_token: Optional[str] = None


def validate_model(compound_id: str) -> ValidationReport:
    """Verify all required parameter groups are present and emit
    soft warnings. On success, locks the session and issues a token
    that simulate_validated() requires."""
    if compound_id not in _SESSIONS:
        return ValidationReport(
            compound_id=compound_id, ok=False,
            missing=[f"unknown compound_id (call register_compound first)"],
        )
    d = _SESSIONS[compound_id]
    if d.validated:
        return ValidationReport(
            compound_id=compound_id, ok=True,
            validation_token=d.validation_token,
            warnings=["already validated"],
        )

    missing: list[str] = []
    if not d.physchem:    missing.append("physchem (call register_compound)")
    if not d.binding:     missing.append("binding (call add_binding)")
    if not d.clearance:   missing.append("clearance (call add_clearance)")
    if not d.absorption:  missing.append("absorption (call add_absorption)")
    if not d.structure:   missing.append("structure (call select_model_structure)")

    warnings: list[str] = []
    if d.binding:
        if d.binding.get("fu_p", 1.0) >= 1.0:
            warnings.append("fu_p=1.0 — provide measured value, sentinel default")
        if d.binding.get("R_bp", 1.0) == 1.0:
            warnings.append("R_bp=1.0 — provide measured or predicted value")
    if d.clearance and d.clearance.get("CL_renal", 0.0) == 0.0:
        spec = d.clearance.get("spec", {})
        if spec.get("source") == "direct" and spec.get("CL_int_L_per_h", 0) == 0:
            warnings.append("zero clearance — drug will not be eliminated")
    if d.transporters and d.structure and \
       d.structure["distribution_model"] == "perfusion_limited":
        warnings.append(
            "transporters provided but distribution_model='perfusion_limited' "
            "(transporters only fire in permeability_limited)"
        )
    # Source provenance check
    critical = ["fu_p", "R_bp", "logP", "clearance"]
    for key in critical:
        if key not in d.sources:
            warnings.append(
                f"no source recorded for '{key}' — pass *_source argument "
                f"on the corresponding add_* call to track provenance"
            )

    if missing:
        return ValidationReport(
            compound_id=compound_id, ok=False,
            missing=missing, warnings=warnings,
        )

    # Issue token
    token = _new_token()
    d.validated = True
    d.validated_at = time.time()
    d.validation_token = token
    _VALIDATION_TOKENS[token] = compound_id
    return ValidationReport(
        compound_id=compound_id, ok=True,
        warnings=warnings, validation_token=token,
    )


def get_validated_draft(token: str) -> CompoundDraft:
    """Resolve a validation token to its draft. Raises if not issued by us."""
    if token not in _VALIDATION_TOKENS:
        raise ValueError(
            f"Invalid or expired validation_token='{token}'. "
            f"Call validate_model() first. simulate_validated() only "
            f"accepts tokens issued by a successful validate_model() call."
        )
    cid = _VALIDATION_TOKENS[token]
    if cid not in _SESSIONS:
        raise ValueError(f"Session for token '{token}' was garbage-collected")
    return _SESSIONS[cid]


def session_summary(compound_id: str) -> dict:
    """Inspect session state without locking."""
    if compound_id not in _SESSIONS:
        return {"error": "unknown compound_id"}
    d = _SESSIONS[compound_id]
    return {
        "compound_id": d.compound_id, "name": d.name,
        "physchem": d.physchem, "binding": d.binding,
        "clearance": d.clearance, "absorption": d.absorption,
        "transporters": d.transporters, "structure": d.structure,
        "sources": d.sources, "validated": d.validated,
        "validation_token": d.validation_token,
        "missing": [
            grp for grp, val in [
                ("physchem", d.physchem), ("binding", d.binding),
                ("clearance", d.clearance), ("absorption", d.absorption),
                ("structure", d.structure),
            ] if not val
        ],
    }
