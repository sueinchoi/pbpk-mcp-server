"""
Input validation and silent-fallback prevention for PBPK tools.

Surfaces user mistakes that would otherwise produce a successful but
misleading simulation. Two severity levels:

  - ERROR: raise ValueError with an actionable message. Use for cases
    where the simulation cannot be physically meaningful (zero clearance,
    invalid enum, mismatched clearance source).

  - WARNING: returned in a list, surfaced in the tool's markdown output
    under a "⚠️ Warnings" section. Use for cases where the simulation
    runs but the user probably didn't get what they asked for (library
    name silently overrides custom params; transporters silently
    ignored in perfusion-limited mode; suspicious defaults).

Every PBPK tool should call `validate_run_pbpk_inputs` and prepend any
returned warnings to its output. Errors propagate as exceptions.
"""

from __future__ import annotations
from typing import Optional


KP_METHOD_NAMES = (
    "rodgers_rowland", "lukacova", "schmitt", "poulin_theil",
    "berezhkovskiy", "pksim_standard", "kp_membrane",
)

CLEARANCE_SOURCES = ("direct", "hlm", "hepatocyte", "rcyp")

DISTRIBUTION_MODELS = ("perfusion_limited", "permeability_limited")

ROUTES = ("oral", "iv_bolus", "iv_infusion")

ABSORPTION_MODELS = ("first_order", "cat", "acat")


def validate_absorption_model(absorption_model: str) -> str:
    """Strict enum check for absorption_model. Previously, anything not
    'acat' silently fell back to first_order — including typos."""
    if absorption_model in ABSORPTION_MODELS:
        return absorption_model
    suggestions = [m for m in ABSORPTION_MODELS
                   if m.replace("_", "") == absorption_model.replace("-", "").replace("_", "")]
    suggestion_msg = f" Did you mean `{suggestions[0]}`?" if suggestions else ""
    raise ValueError(
        f"Invalid absorption_model='{absorption_model}'.{suggestion_msg} "
        f"Valid options: {', '.join(ABSORPTION_MODELS)}. "
        f"Use underscores, not hyphens."
    )


def parse_cyp_dict(s: str, *, parameter_name: str) -> dict[str, float]:
    """
    Parse a 'CYP3A4:0.5,CYP2C9:0.1' string into {CYP_NAME: float}.

    Previously, malformed entries were silently skipped — a typo
    (CYP3A4=0.5 with '=' instead of ':') silently dropped the gut
    metabolism term. Now we raise with the offending entry.
    """
    if not s:
        return {}
    out: dict[str, float] = {}
    bad: list[str] = []
    for raw in s.split(","):
        pair = raw.strip()
        if not pair:
            continue
        if pair.count(":") != 1:
            bad.append(pair)
            continue
        cyp, val = pair.split(":")
        cyp = cyp.strip()
        try:
            out[cyp] = float(val.strip())
        except ValueError:
            bad.append(pair)
    if bad:
        raise ValueError(
            f"Malformed entries in {parameter_name}: {bad}. "
            f"Expected 'CYP_NAME:VALUE,CYP_NAME:VALUE,...' "
            f"using a colon (:) as the separator. Examples: "
            f"'CYP3A4:0.8,CYP2C9:0.15' (fm) or "
            f"'CYP3A4:0.5,CYP2C9:0.1' (CLint per pmol-rCYP)."
        )
    return out


def require_compound_input(
    *,
    name: str,
    library: dict,
    logP: float,
    pKa: float,
    fu_p: float,
    mw: float,
    tool_name: str = "tool",
) -> None:
    """
    Refuse to run a Kp / binding / clearance prediction when neither
    a library name NOR meaningful custom physchem inputs were supplied.
    Previously, a bare call (e.g. predict_kp()) returned a plausible
    Kp table built from sentinel defaults — that's the silent fallback
    this guard exists to prevent.

    Allowed:
      - name in library (any combination of physchem)
      - name not in library AND user actually supplied non-default
        logP / pKa / fu_p / mw

    The threshold for "supplied" is "any of the four is non-default".
    Sentinel defaults: logP=0, pKa=7.0, fu_p=1.0, mw=300.0.
    """
    # Defensive: name may be passed as None (vs the empty-string default).
    # `None.lower()` would raise AttributeError; coerce to empty string.
    if name is None:
        name = ""
    if name and name.lower() in library:
        return
    all_default = (
        logP == 0.0 and pKa == 7.0 and fu_p == 1.0 and mw == 300.0
    )
    if all_default:
        raise ValueError(
            f"`{tool_name}` requires either a library compound name "
            f"(e.g. name='midazolam') OR explicit physchem inputs "
            f"(logP, pKa, fu_p, mw — at least one must be non-default). "
            f"Calling with no arguments would return a Kp / binding / "
            f"clearance table built from sentinel defaults "
            f"(logP=0, fu_p=1.0, mw=300) — that result has no physical "
            f"meaning. Provide a real compound or library name."
        )


def validate_subject_sentinel(
    body_weight: float, sex: str, age: float,
) -> Optional[str]:
    """
    The default subject (73 kg, male, 30 y) is reasonable for many
    drugs but wrong for pediatric / pregnant / elderly / female
    studies. Detect the *exact* sentinel triple — if all three are
    at default values, it likely means the user did not specify the
    subject at all.

    Returns a soft-warning string, or None.
    """
    if body_weight == 73.0 and sex == "male" and age == 30.0:
        return (
            "Subject defaults used: 73 kg male, age 30 years. If you "
            "are simulating a pediatric, female, elderly, pregnant, or "
            "non-typical subject, set body_weight, sex, and age "
            "explicitly. Default subject is silently applied otherwise."
        )
    return None


def validate_kp_method(kp_method: str) -> str:
    """Strict enum validation. Raises with the closest valid option."""
    if kp_method in KP_METHOD_NAMES:
        return kp_method
    # Suggest closest match
    suggestions = [m for m in KP_METHOD_NAMES if m.replace("_", "") == kp_method.replace("-", "").replace("_", "")]
    suggestion_msg = f" Did you mean `{suggestions[0]}`?" if suggestions else ""
    raise ValueError(
        f"Invalid kp_method='{kp_method}'.{suggestion_msg} "
        f"Valid options: {', '.join(KP_METHOD_NAMES)}. "
        f"Note: use underscores, not hyphens (e.g. 'poulin_theil', not 'poulin-theil')."
    )


def validate_clearance_source_mismatch(
    clearance_source: str,
    CL_int: float,
    CLint_vitro_hlm: Optional[float],
    CLint_vitro_hep: Optional[float],
    CLint_per_cyp: Optional[str],
) -> None:
    """
    Detect the case where the user explicitly chose a clearance_source
    other than 'direct' but provided a different IVIVE input field
    (e.g. clearance_source='hlm' with only CLint_vitro_hep supplied).
    Library-compound users typically leave clearance_source='direct'
    and provide nothing — that case is allowed (library carries CL_int).

    Raises only on an unambiguous user mistake; the "no clearance at
    all" case is handled later by validate_suspicious_defaults after
    library lookup has resolved cl_int_resolved.
    """
    if clearance_source not in CLEARANCE_SOURCES:
        raise ValueError(
            f"Invalid clearance_source='{clearance_source}'. "
            f"Valid options: {CLEARANCE_SOURCES}."
        )
    if clearance_source == "direct":
        return  # nothing to mismatch — library or direct CL_int both fine

    expected_field = {
        "hlm": ("CLint_vitro_hlm", CLint_vitro_hlm),
        "hepatocyte": ("CLint_vitro_hep", CLint_vitro_hep),
        "rcyp": ("CLint_per_cyp", CLint_per_cyp),
    }[clearance_source]
    name, value = expected_field
    expected_present = value is not None and (
        value > 0 if isinstance(value, (int, float)) else len(value) > 0
    )
    if expected_present:
        return  # consistent

    other_provided = [
        (n, v) for n, v in [
            ("CLint_vitro_hlm", CLint_vitro_hlm),
            ("CLint_vitro_hep", CLint_vitro_hep),
            ("CLint_per_cyp", CLint_per_cyp),
        ] if v and (n, v) != (name, value)
    ]
    if other_provided:
        suggested = {
            "CLint_vitro_hlm": "hlm",
            "CLint_vitro_hep": "hepatocyte",
            "CLint_per_cyp": "rcyp",
        }[other_provided[0][0]]
        raise ValueError(
            f"clearance_source='{clearance_source}' expects {name}, but you "
            f"provided {other_provided[0][0]} instead. Set "
            f"clearance_source='{suggested}' to match, or supply {name}."
        )
    raise ValueError(
        f"clearance_source='{clearance_source}' requires {name}, but it was "
        f"not provided. Either supply {name} or set clearance_source='direct' "
        f"and provide CL_int directly."
    )


def validate_distribution_and_transporters(
    distribution_model: str,
    has_transporters: bool,
) -> list[str]:
    """
    Warn if transporter parameters were provided but the perfusion-limited
    model is active (transporters are silently dropped — see CLAUDE.md).
    """
    if distribution_model not in DISTRIBUTION_MODELS:
        raise ValueError(
            f"Invalid distribution_model='{distribution_model}'. "
            f"Valid options: {DISTRIBUTION_MODELS}."
        )
    warnings = []
    if has_transporters and distribution_model == "perfusion_limited":
        warnings.append(
            "Transporter parameters were provided but `distribution_model="
            "\"perfusion_limited\"` (default). Active transport is only "
            "evaluated in the permeability-limited model. Set "
            "`distribution_model=\"permeability_limited\"` to enable them, "
            "or remove the transporter inputs to silence this warning."
        )
    return warnings


def validate_library_override(
    name: str,
    library: dict,
    user_params: dict,
) -> list[str]:
    """
    When the user passes a library compound name AND custom physicochemical
    parameters, the library values currently win. Warn that the user's
    custom values are being ignored, and tell them the explicit override
    path (use a non-library name, or pass kp_override).

    user_params: dict of {param_name: (user_value, default_value)} —
                 only flag parameters the user actually changed from default.
    """
    if not name or name.lower() not in library:
        return []
    overridden = [
        p for p, (val, default) in user_params.items() if val != default
    ]
    if not overridden:
        return []
    return [
        f"Library compound `{name}` was matched, so custom values for "
        f"{overridden} were IGNORED (library values used instead). "
        f"To use your measured values, either: (a) call with a different "
        f"`name=` (e.g. `name='Midazolam_custom'`) so the library lookup "
        f"misses, or (b) modify `core/compound.py` `COMPOUND_LIBRARY` "
        f"and re-import."
    ]


def validate_suspicious_defaults(
    in_library: bool,
    fu_p: float,
    R_bp: float,
    cl_int_resolved: float,
    CL_renal: float,
    compound_type: str,
) -> list[str]:
    """
    Custom (non-library) compounds that left key physicochemical defaults
    untouched probably aren't what the user intended. The defaults are
    sentinels (fu_p=1.0, R_bp=1.0) that produce non-physical predictions
    for most real drugs.
    """
    warnings = []
    if in_library:
        return warnings  # Library values were validated when curated
    if fu_p >= 1.0:
        warnings.append(
            "`fu_p=1.0` (100% unbound) is the default sentinel and is "
            "almost never realistic. Most drugs have fu_p in 0.001-0.5. "
            "Provide a measured or literature value."
        )
    if R_bp == 1.0:
        warnings.append(
            "`R_bp=1.0` is the default sentinel. Acids typically have "
            "R_bp ≈ 0.55, neutrals ≈ 0.7-1.0, lipophilic bases ≈ 1.0-2.0. "
            "Provide a measured or predicted value (predict_blood_plasma_ratio)."
        )
    if cl_int_resolved <= 0 and CL_renal <= 0:
        warnings.append(
            "No hepatic clearance and no renal clearance. The drug will "
            "not be eliminated. Set CL_int, an IVIVE input "
            "(CLint_vitro_hlm/hep), or CL_renal."
        )
    if compound_type == "neutral" and fu_p < 0.01:
        warnings.append(
            f"compound_type='neutral' with fu_p={fu_p:.4f} (highly bound). "
            f"Highly-bound drugs are usually acids (e.g. NSAIDs, warfarin). "
            f"Verify compound_type — it controls Kp prediction."
        )
    return warnings


def validate_run_pbpk_inputs(
    *,
    name: str,
    library: dict,
    distribution_model: str,
    kp_method: str,
    clearance_source: str,
    CL_int: float,
    CLint_vitro_hlm: Optional[float],
    CLint_vitro_hep: Optional[float],
    CLint_per_cyp: Optional[str],
    CL_renal: float,
    has_transporters: bool,
    fu_p: float,
    R_bp: float,
    cl_int_resolved: float,
    compound_type: str,
    user_overrides: Optional[dict] = None,
    # Range-checked fields (None to skip)
    mw: Optional[float] = None,
    logP: Optional[float] = None,
    pKa: Optional[float] = None,
    ka: Optional[float] = None,
    Fa: Optional[float] = None,
    Fg: Optional[float] = None,
    Peff: Optional[float] = None,
    Vmax: Optional[float] = None,
    Km: Optional[float] = None,
    dose_mg: Optional[float] = None,
    duration_h: Optional[float] = None,
    n_doses: Optional[int] = None,
    interval_h: Optional[float] = None,
    body_weight: Optional[float] = None,
    age: Optional[float] = None,
    route: Optional[str] = None,
    sex: Optional[str] = None,
) -> list[str]:
    """
    Run all checks. Raises on hard errors (invalid enum, mismatched
    clearance source, out-of-range physical parameters). Returns a list
    of soft-warning strings to be surfaced in the tool's markdown output.
    """
    from .invariants import (
        check_compound_ranges, check_dose_subject_ranges,
        check_dose_self_consistency, raise_on_violations,
    )

    # Hard errors (raised at call site for fail-fast, here as defense in depth)
    validate_kp_method(kp_method)
    validate_clearance_source_mismatch(
        clearance_source, CL_int, CLint_vitro_hlm,
        CLint_vitro_hep, CLint_per_cyp,
    )

    # Range invariants — only check values the user actually supplied
    # (library lookups go through curated values that should already be in range).
    in_library = bool(name and name.lower() in library)
    range_violations: list = []
    if not in_library:
        range_violations.extend(check_compound_ranges(
            mw=mw, logP=logP, pKa=pKa, fu_p=fu_p, R_bp=R_bp,
            ka=ka, Fa=Fa, Fg=Fg, Peff=Peff,
            CL_int=CL_int if CL_int > 0 else None,
            CL_renal=CL_renal if CL_renal > 0 else None,
            CLint_vitro_hlm=CLint_vitro_hlm,
            CLint_vitro_hep=CLint_vitro_hep,
            Vmax=Vmax, Km=Km,
        ))
    range_violations.extend(check_dose_subject_ranges(
        dose_mg=dose_mg, duration_h=duration_h,
        n_doses=n_doses, interval_h=interval_h,
        body_weight=body_weight, age=age,
    ))
    if (dose_mg is not None and n_doses is not None
            and interval_h is not None and duration_h is not None
            and route is not None):
        v = check_dose_self_consistency(
            dose_mg=dose_mg, n_doses=n_doses, interval_h=interval_h,
            duration_h=duration_h, route=route,
        )
        if v:
            range_violations.append(v)
    raise_on_violations(range_violations)

    # Soft warnings
    warnings: list[str] = []
    warnings.extend(validate_distribution_and_transporters(
        distribution_model, has_transporters))
    warnings.extend(validate_library_override(
        name, library, user_overrides or {}))
    warnings.extend(validate_suspicious_defaults(
        in_library=in_library,
        fu_p=fu_p, R_bp=R_bp,
        cl_int_resolved=cl_int_resolved,
        CL_renal=CL_renal,
        compound_type=compound_type,
    ))
    if body_weight is not None and sex is not None and age is not None:
        sentinel_warning = validate_subject_sentinel(
            body_weight=body_weight, sex=sex, age=age,
        )
        if sentinel_warning:
            warnings.append(sentinel_warning)
    return warnings


def format_warnings_block(warnings: list[str]) -> str:
    """Render warnings as a markdown callout for the tool output."""
    if not warnings:
        return ""
    lines = ["", "> ⚠️ **Input warnings** (simulation ran, but check these):"]
    for w in warnings:
        lines.append(f"> - {w}")
    lines.append("")
    return "\n".join(lines)
