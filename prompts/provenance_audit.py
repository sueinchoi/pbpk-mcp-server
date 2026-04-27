"""
Provenance audit — separate prompt layer for silent-fallback detection.

Input-time schema validation catches malformed inputs. This audit
catches the OPPOSITE failure mode: outputs that look reasonable but
were assembled from defaults the LLM didn't realize it was using.
The technique is to force the LLM to write a row for every parameter,
mark each row as user / literature / default / inferred, and refuse to
fabricate citations.

Two components:
  - PROVENANCE_AUDIT_PROMPT  — generic audit any model can be passed through
  - render_session_audit(compound_id) — server-side deterministic audit
    using session state; the LLM cannot lie about what was actually
    stored in the session
"""


PROVENANCE_AUDIT_PROMPT = """You are auditing a PBPK model. For EVERY
parameter used in the model — physiological constants, drug-specific
inputs, formulation parameters, AND model-structure choices (Kp method,
distribution model, dosing schedule, simulation tolerance) — output one
row in the table below.

| Parameter | Value | Unit | Source type | Source citation | Confidence |

`Source type` MUST be one of:
- `user_provided`    — given by the user in this conversation
- `measurement`      — laboratory measurement supplied by the user
                        (RED, Bp/p assay, Caco-2 papp, hepatocyte
                        depletion, etc.)
- `literature`       — extracted from a paper or database — citation
                        REQUIRED (PMID, DOI, ChEMBL ID, DrugBank ID)
- `library`          — bundled compound from `core/compound.py`
                        COMPOUND_LIBRARY (treat as "literature" with
                        the curator's documented references)
- `default`          — server-default sentinel (fu_p=1.0, R_bp=1.0,
                        ka=1.0, etc.) was used because no input was
                        supplied
- `inferred`         — derived from another parameter (e.g. fu_inc
                        predicted from logP via Austin 2002, R_bp
                        predicted from RBC partitioning, gut CLint
                        scaled from liver)

`Confidence` MUST be one of: `high`, `medium`, `low`, `unverified`.

## Rules (non-negotiable)

1. If a row's source type is `default`, prefix it with ⚠️ and add a
   line explaining WHY the default was triggered:
   - Missing input from the user?
   - Source extraction failed?
   - Citation could not be verified (cache miss + no network)?
   Refusing to flag `default` rows is a silent fallback — this is the
   exact bug this audit exists to catch.

2. If a row's source type is `inferred`, write the inference logic in
   one line ("Austin 2002 from logP=4.5 → fu_inc=0.19") and one
   sentence on what would change if the inference is wrong.

3. If you cannot find a source for a row, write `UNSOURCED` literally —
   do NOT invent a plausible-looking PMID. Fabricated citations are a
   harder bug to detect than `UNSOURCED`.

4. Vague citations are flagged: any row whose Source citation is
   "literature value", "typical value", "commonly used", "Smith et al.
   (no year)", or empty must be marked `UNSOURCED` and flagged.

5. Every PMID and DOI must be verified via `verify_citation()` before
   you write it. If verification returned `unverified` (cache miss +
   network error), set Confidence to `unverified` and note it.

## After the table

List three explicit sets:

(a) **Silent-fallback parameters** — every row whose source type is
    `default`. If this list is empty, state "no defaults triggered".
    Either way, do not omit this section.

(b) **Low-confidence parameters that drive model output** — rows with
    Confidence ∈ {low, unverified} that are also in the model's
    sensitivity list (CL_int, fu_p, Kp method, ka are typically the
    drivers; brain Kp typically is not).

(c) **Unsourced parameters** — rows whose Source citation is
    `UNSOURCED` or vague.

Refuse to summarize the model as "validated" if any of these three
lists are non-empty. State the model's status explicitly: `passed`,
`passed-with-flags`, or `failed-audit`.
"""


def get_audit_prompt() -> str:
    """The full audit prompt for any PBPK model."""
    return PROVENANCE_AUDIT_PROMPT


# ---------------------------------------------------------------------
# Deterministic server-side audit (session-backed)
# ---------------------------------------------------------------------

def _classify_source(key: str, source_str: str | None) -> tuple[str, str, str]:
    """
    Classify a source string into (source_type, citation, confidence).
    Empty / missing source → ('UNSOURCED', '', 'unverified').
    Distinguishes PMID/DOI/measurement/library labels heuristically.
    """
    if not source_str:
        return ("UNSOURCED", "", "unverified")
    s = source_str.strip()
    sl = s.lower()
    if sl.startswith("pmid:") or sl.startswith("pmid "):
        return ("literature", s, "medium")
    if sl.startswith("10.") and "/" in sl:
        return ("literature", s, "medium")
    if sl.startswith("doi:"):
        return ("literature", s, "medium")
    if "chembl" in sl:
        return ("literature", s, "high")
    if "drugbank" in sl:
        return ("literature", s, "high")
    if "pubchem" in sl:
        return ("literature", s, "high")
    if "user_provided" in sl or "in_house" in sl or "measured" in sl or "measurement" in sl:
        return ("measurement", s, "high")
    if "library" in sl or "compound_library" in sl:
        return ("library", s, "medium")
    if "austin" in sl or "predicted" in sl or "qsar" in sl or "inferred" in sl:
        return ("inferred", s, "low")
    if any(v in sl for v in ("typical", "literature value", "commonly used", "approx", "estimate")):
        # Vague — flag as unsourced
        return ("UNSOURCED", s, "unverified")
    return ("literature", s, "low")    # has *some* string but unclear → low confidence


def render_session_audit(compound_id: str) -> str:
    """
    Generate a deterministic provenance audit from a session draft.
    The output is structured Markdown matching the prompt's required
    format — useful (a) for the LLM to consume as ground-truth, (b) as
    the artifact a reviewer reads.

    Distinguishes:
      - parameters explicitly given (in d.physchem/binding/etc.)
      - parameters with a recorded source (in d.sources)
      - parameters left at server defaults (sentinel detection)
      - inferred parameters (clearance IVIVE → in vivo CL_int)
    """
    from core.session import _SESSIONS

    if compound_id not in _SESSIONS:
        return f"Unknown compound_id='{compound_id}'."
    d = _SESSIONS[compound_id]

    rows: list[tuple[str, str, str, str, str, str, str]] = []
    # Each row: (parameter, value, unit, source_type, citation, confidence, note)

    def add_row(param: str, val, unit: str, source_key: str | None = None,
                inferred_from: str | None = None, note: str = ""):
        src_str = d.sources.get(source_key, "") if source_key else ""
        has_source = bool(src_str)
        if inferred_from and not src_str:
            stype, cite, conf = ("inferred", f"derived from {inferred_from}", "low")
        else:
            stype, cite, conf = _classify_source(source_key or param, src_str)
            # Sentinel default detection — ONLY when no source is recorded.
            # If the user has explicitly cited a source for the value, trust
            # them: Fa=1.0 with Fa_source="BCS II + Caco-2" is not a sentinel.
            if not has_source:
                if param == "fu_p" and val == 1.0:
                    stype, cite, conf, note = "default", "", "unverified", \
                        "fu_p=1.0 sentinel — no measurement / no library entry"
                elif param == "R_bp" and val == 1.0:
                    stype, cite, conf, note = "default", "", "unverified", \
                        "R_bp=1.0 sentinel — provide measured or predicted value"
                elif param in ("ka", "Fa", "Fg") and val == 1.0:
                    stype, cite, conf, note = "default", "", "unverified", \
                        f"{param}=1.0 sentinel — record a source if intended"
        flag = "⚠️ " if stype == "default" or stype == "UNSOURCED" else ""
        rows.append((flag + param, str(val), unit, stype,
                     cite or ("(none)" if stype != "inferred" else f"derived from {inferred_from}"),
                     conf, note))

    # --- physchem ---
    if d.physchem:
        add_row("name", d.name, "—", source_key=None)
        add_row("mw", d.physchem["mw"], "g/mol", source_key="mw")
        add_row("logP", d.physchem["logP"], "log10", source_key="logP")
        add_row("pKa", d.physchem["pKa"], "—", source_key="pKa")
        add_row("compound_type", d.physchem["compound_type"], "—", source_key=None)

    # --- binding ---
    if d.binding:
        add_row("fu_p", d.binding["fu_p"], "fraction", source_key="fu_p")
        add_row("R_bp", d.binding["R_bp"], "ratio", source_key="R_bp")

    # --- clearance ---
    if d.clearance:
        spec = d.clearance.get("spec", {})
        src = spec.get("source")
        if src == "direct":
            add_row("CL_int", spec["CL_int_L_per_h"], "L/h",
                    source_key="clearance")
        elif src == "hlm":
            add_row("CLint_vitro_hlm", spec["CLint_vitro_uL_min_mg"],
                    "uL/min/mg", source_key="clearance")
            add_row("CL_int (resolved)", "via IVIVE",
                    "L/h", inferred_from="HLM IVIVE (Austin fu_inc)")
        elif src == "hepatocyte":
            add_row("CLint_vitro_hep", spec["CLint_vitro_uL_min_1e6cells"],
                    "uL/min/1e6 cells", source_key="clearance")
            add_row("CL_int (resolved)", "via IVIVE", "L/h",
                    inferred_from="Hepatocyte IVIVE (Austin fu_hep)")
        elif src == "rcyp":
            add_row("CLint_per_cyp", spec["CLint_per_cyp"],
                    "uL/min/pmol-rCYP", source_key="clearance")
        add_row("CL_renal", d.clearance.get("CL_renal", 0), "L/h",
                source_key=None)

    # --- absorption ---
    if d.absorption:
        add_row("ka", d.absorption["ka"], "1/h", source_key="ka")
        add_row("Fa", d.absorption["Fa"], "fraction", source_key="Fa")
        add_row("Fg", d.absorption["Fg"], "fraction", source_key="Fg")
        if d.absorption.get("Peff") is not None:
            add_row("Peff", d.absorption["Peff"], "1e-4 cm/s",
                    source_key="Peff")

    # --- structure ---
    if d.structure:
        add_row("kp_method", d.structure["kp_method"], "—",
                source_key=None,
                note="model structure choice — affects all 13 Kp values")
        add_row("distribution_model", d.structure["distribution_model"],
                "—", source_key=None)
        add_row("absorption_model", d.structure["absorption_model"],
                "—", source_key=None)

    # --- transporters (if any) ---
    if d.transporters:
        for k, v in d.transporters.items():
            if v:
                add_row(f"{k}_Km", v.get("Km_uM"), "uM", source_key=None)
                add_row(f"{k}_Vmax", v.get("Vmax"), "pmol/min/pmol",
                        source_key=None)

    # --- physiology (always defaults from ICRP 89 unless user overrode) ---
    add_row("organ_volumes", "ICRP 89 (Valentin 2002)", "L",
            inferred_from="ICRP 89 + body weight scaling",
            note="physiology table; user can override get_physiology(...)")
    add_row("blood_flows", "ICRP 89 + Williams Leggett 1989",
            "L/h fraction", inferred_from="ICRP 89 + cardiac output allometry")

    # --- assemble markdown ---
    out: list[str] = []
    out.append(f"## Provenance Audit — `{compound_id}` ({d.name})\n")
    out.append("| Parameter | Value | Unit | Source type | Source citation | Confidence | Note |")
    out.append("|---|---|---|---|---|---|---|")
    silent_fallbacks: list[str] = []
    unsourced: list[str] = []
    low_conf: list[str] = []
    for (param, val, unit, stype, cite, conf, note) in rows:
        out.append(f"| {param} | {val} | {unit} | {stype} | {cite} | {conf} | {note} |")
        if stype == "default":
            silent_fallbacks.append(param.replace("⚠️ ", ""))
        if stype == "UNSOURCED":
            unsourced.append(param.replace("⚠️ ", ""))
        if conf in ("low", "unverified") and any(
            k in param for k in ("CL_int", "fu_p", "kp_method", "ka", "Fg")
        ):
            low_conf.append(param.replace("⚠️ ", ""))

    out.append("\n### (a) Silent-fallback parameters")
    if silent_fallbacks:
        for p in silent_fallbacks:
            out.append(f"- ⚠️ `{p}` — server default used; no user value or library entry")
    else:
        out.append("- _no defaults triggered_")

    out.append("\n### (b) Low-confidence parameters that drive model output")
    if low_conf:
        for p in low_conf:
            out.append(f"- `{p}` — confidence low/unverified; affects model output")
    else:
        out.append("- _none_")

    out.append("\n### (c) Unsourced parameters")
    if unsourced:
        for p in unsourced:
            out.append(f"- `{p}` — UNSOURCED")
    else:
        out.append("- _none_")

    # --- Status verdict ---
    out.append("\n### Audit status")
    if not (silent_fallbacks or unsourced):
        if not low_conf:
            out.append("**`passed`** — no defaults, no unsourced parameters, "
                       "no low-confidence drivers")
        else:
            out.append("**`passed-with-flags`** — low-confidence parameters "
                       "drive model output (see section b); model results "
                       "should be interpreted with caveat")
    else:
        out.append("**`failed-audit`** — silent fallbacks or unsourced "
                   "parameters present (see sections a, c). "
                   "Resolve before treating results as model predictions.")

    return "\n".join(out)
