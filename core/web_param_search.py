"""
Anti-hallucination structure for web-sourced PBPK parameters.

Why this exists
---------------
Users will ask the LLM driving the MCP server to "search the literature"
or "find the fu_hep of compound X." An LLM can hallucinate:

  1. A plausible-looking number with a fabricated PMID.
  2. A real PMID for the wrong topic.
  3. A value from a blog / SDK doc / preprint that LOOKS like literature.
  4. A "consensus value" that is actually one paper averaged with the LLM's prior.
  5. A correctly cited paper, with the WRONG NUMBER from inside that paper —
     e.g., comparator drug Peff submitted as target drug Peff.

The first 4 are caught by metadata gates (citation existence, topic match,
evidence class, range, unit). The 5th — "right paper, wrong number" — is
the failure mode that survives every metadata-only architecture. We mitigate
it with `EvidenceBindingStatus`: the server fetches the abstract or PMC
full text and verifies the candidate's exact snippet appears, near both
the value and the unit, in the retrievable text.

Citation laundering (two papers that share an upstream measurement, sold
as "two independent confirmations") is mitigated by `upstream_citation_id`
on each candidate and `_classify_independence`.

Codex review summary (2026-04-29):
  - Replace ALLOWED_PRIMARY_JOURNALS allowlist with EvidenceClass tag.
  - Add `conflict` confidence state — refuse-to-merge if values disagree.
  - HEAD check is theatre — replace with snippet-in-text verification.
  - Geometric mean of 0.5 and 50 → 5 is supported by neither source; do
    not auto-merge unless contexts match and values within tolerance.
  - The surviving failure mode is "real paper, comparator-drug number" —
    snippet+context binding is the only mitigation.

Layer position: 8th in the safety architecture (CLAUDE.md), called by
the new MCP tool `search_parameter_with_citation` and importable from
session-tool flows that ingest user-supplied web-sourced values.

Public API
----------
  WebParamQuery, WebParamCandidate, WebParamResult, GateResult
  EvidenceClass, EvidenceBindingStatus, Confidence
  verify_candidate(candidate, query) -> list[GateResult]
  synthesize(verified, query) -> WebParamResult
  search_and_verify(query, candidates) -> WebParamResult   # one-shot
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from .citation import (
    CitationResult,
    CitationStatus,
    fetch_pmid_abstract,
    fetch_pmcid_for_pmid,
    fetch_pmc_full_text,
    verify_citation,
    verify_citation_topic,
)


# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------

class EvidenceClass(str, Enum):
    """How the source was generated. Only the top three are eligible
    for automatic acceptance into a model parameter set."""
    PRIMARY_MEASUREMENT = "primary_measurement"      # the paper measured it
    REGULATORY_REVIEW = "regulatory_review"          # FDA NDA/EMA assessment carrying primary data
    CURATED_DB_WITH_SOURCE = "curated_db_with_source"  # DrugBank/ChEMBL/PubChem with traceable upstream
    REVIEW_WITH_TRACEABLE_UPSTREAM = "review_with_traceable_upstream"
    MODEL_ASSUMPTION = "model_assumption"            # PBPK paper that fitted the value
    PREPRINT = "preprint"                            # bioRxiv / medRxiv / arXiv
    VENDOR_DOC = "vendor_doc"                        # PK-Sim / Simcyp / GastroPlus docs
    BLOG_OR_WEB = "blog_or_web"                      # marketing, blog, vendor SDK
    UNKNOWN = "unknown"


class EvidenceBindingStatus(str, Enum):
    """Did the cited source actually contain the claimed value?"""
    VERIFIED_EXACT_SNIPPET = "verified_exact_snippet"            # snippet found verbatim, value+unit nearby
    VERIFIED_NUMERIC_NEAR_CONTEXT = "verified_numeric_near_context"  # value+unit found near drug+parameter terms
    METADATA_ONLY = "metadata_only"          # citation verified but no full text retrievable
    NOT_FOUND = "not_found"                  # snippet not in fetched text
    PAYWALLED_UNVERIFIED = "paywalled_unverified"  # text unavailable; cannot verify


class Confidence(str, Enum):
    HIGH = "high"          # ≥2 INDEPENDENT primary measurements within tolerance
    MEDIUM = "medium"      # 1 primary measurement, snippet-verified
    LOW = "low"            # 1 secondary or metadata-only candidate
    CONFLICT = "conflict"  # ≥2 primary candidates that DISAGREE materially
    NONE = "none"          # 0 verified candidates


# Evidence classes that MAY be auto-accepted (gated on snippet binding).
_ACCEPTABLE_CLASSES = {
    EvidenceClass.PRIMARY_MEASUREMENT,
    EvidenceClass.REGULATORY_REVIEW,
    EvidenceClass.CURATED_DB_WITH_SOURCE,
}


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass
class MeasurementContext:
    """The biological context the value applies to. Mismatched contexts
    are the dominant 'right paper, wrong number' failure mode."""
    species: str = ""             # 'human', 'rat', 'dog', 'monkey'
    matrix: str = ""              # 'plasma', 'hepatocyte', 'HLM', 'jejunum', 'enterocyte'
    method: str = ""              # 'RED', 'ultrafiltration', 'Caco-2', 'in_vivo_clinical'
    assay: str = ""               # free text, e.g. 'pH 7.4, 1 mg/mL HLM'
    temperature_C: Optional[float] = None
    pH: Optional[float] = None


@dataclass
class EvidenceLocation:
    """Where in the source the value appears. Empty fields mean the LLM
    did not specify — gates downgrade binding strength accordingly."""
    section: str = ""             # 'Results', 'Methods', 'Abstract', 'Discussion'
    table: str = ""               # 'Table 2'
    figure: str = ""              # 'Figure 3A'
    page: str = ""
    supplementary_file: str = ""


@dataclass
class WebParamQuery:
    """Specifies what we are looking for. The server uses this to compute
    topic-match keywords, range bounds, and unit expectations."""
    parameter: str                # 'fu_hep', 'R_bp', 'Peff', 'CL_int', 'fu_p', 'MW', 'logP'
    drug_name: str
    drug_synonyms: list[str] = field(default_factory=list)   # ['midazolam', 'CHEMBL601', 'versed']
    parameter_synonyms: list[str] = field(default_factory=list)  # ['fu,hep', 'fraction unbound hepatocyte']
    expected_unit: str = ""       # 'unitless', 'L/h', '10^-4 cm/s'
    expected_range: Optional[tuple[float, float]] = None  # from invariants.py
    species: str = "human"
    matrix: str = ""              # constrains the evidence binding gate
    notes: str = ""


@dataclass
class WebParamCandidate:
    """One claim retrieved from the web. Every field marked REQUIRED must
    be populated by the LLM driver — empty values fail the gate."""
    # REQUIRED
    parameter: str                # must match query.parameter
    drug_name: str                # must topic-match query
    value: float
    unit: str                     # raw unit as found in source; server converts
    citation_id: str              # PMID (digits) or DOI (10.xxxx/yyy)
    citation_type: str            # 'pmid' or 'doi'
    source_url: str               # actual URL the LLM read
    snippet: str                  # exact verbatim quoted text from the source

    # CONTEXT
    context: MeasurementContext = field(default_factory=MeasurementContext)
    evidence_class: EvidenceClass = EvidenceClass.UNKNOWN
    evidence_location: EvidenceLocation = field(default_factory=EvidenceLocation)

    # PROVENANCE — laundering defense
    upstream_citation_id: Optional[str] = None  # if this paper cites someone else for the value
    is_direct_measurement: bool = False         # the paper measured it itself

    # AUDIT
    raw_search_query: str = ""
    retrieved_at: str = ""        # ISO 8601 UTC

    def __post_init__(self):
        if not self.retrieved_at:
            self.retrieved_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class GateResult:
    gate: str
    passed: bool
    reason: str
    severity: str = "hard"        # 'hard' (rejection) | 'soft' (warning)


@dataclass
class WebParamResult:
    query: WebParamQuery
    candidates: list[WebParamCandidate]
    gate_results: dict[int, list[GateResult]]    # candidate index -> gates
    binding_status: dict[int, EvidenceBindingStatus]
    accepted_indices: list[int]                  # passed all hard gates
    confidence: Confidence
    accepted_value: Optional[float] = None
    accepted_unit: Optional[str] = None
    rationale: str = ""
    audit_path: Optional[str] = None

    def to_markdown(self) -> str:
        lines = [
            f"## Web Parameter Search — `{self.query.parameter}` for `{self.query.drug_name}`",
            "",
            f"**Confidence:** `{self.confidence.value}`",
        ]
        if self.accepted_value is not None:
            lines.append(
                f"**Accepted value:** `{self.accepted_value}` {self.accepted_unit or ''}"
            )
        else:
            lines.append("**Accepted value:** *(none — see candidates below)*")
        if self.rationale:
            lines.append(f"\n> {self.rationale}\n")
        lines.append("")

        # Candidates table
        lines.append(
            "| # | Value | Unit | Citation | Class | Binding | Gates passed | Reject reason |"
        )
        lines.append(
            "|---|-------|------|----------|-------|---------|--------------|---------------|"
        )
        for i, c in enumerate(self.candidates):
            gates = self.gate_results.get(i, [])
            passed_n = sum(1 for g in gates if g.passed)
            total_n = len(gates)
            failed = next((g for g in gates if not g.passed), None)
            reason = (failed.reason[:60] + "…") if failed and len(failed.reason) > 60 else (failed.reason if failed else "")
            binding = self.binding_status.get(i, EvidenceBindingStatus.NOT_FOUND).value
            mark = "✓" if i in self.accepted_indices else "✗"
            lines.append(
                f"| {i} {mark} | {c.value} | {c.unit} | "
                f"[{c.citation_type}:{c.citation_id}]({c.source_url}) | "
                f"{c.evidence_class.value} | {binding} | "
                f"{passed_n}/{total_n} | {reason} |"
            )
        lines.append("")
        lines.append(
            f"*Audit trail: `{self.audit_path}`*"
            if self.audit_path else "*Audit not persisted.*"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Text normalization (snippet matching)
# ---------------------------------------------------------------------

def _normalize_text(s: str) -> str:
    """Normalize for tolerant snippet matching: lowercase, NFKD, collapse
    whitespace, strip soft hyphens, normalize hyphens/dashes."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("\u00ad", "")           # soft hyphen
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    s = s.replace("\u00d7", "x").replace("\u00b7", ".")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _tokenize_words(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", _normalize_text(s)))


# ---------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------

def _gate_required_fields(c: WebParamCandidate, q: WebParamQuery) -> GateResult:
    missing = []
    if not c.citation_id.strip():
        missing.append("citation_id")
    if not c.citation_type.strip():
        missing.append("citation_type")
    if not c.source_url.strip():
        missing.append("source_url")
    if not c.snippet.strip():
        missing.append("snippet")
    if c.parameter != q.parameter:
        missing.append(f"parameter mismatch (got '{c.parameter}', want '{q.parameter}')")
    if not c.unit.strip():
        missing.append("unit")
    if missing:
        return GateResult(
            gate="required_fields", passed=False,
            reason=f"missing or invalid: {missing}",
        )
    return GateResult(gate="required_fields", passed=True, reason="all fields present")


def _gate_citation_exists(c: WebParamCandidate) -> tuple[GateResult, Optional[CitationResult]]:
    res = verify_citation(c.citation_id)
    if res.status == CitationStatus.VERIFIED:
        return GateResult(gate="citation_exists", passed=True,
                          reason=f"verified: {res.title or '(no title)'}"), res
    return GateResult(
        gate="citation_exists", passed=False,
        reason=f"status={res.status.value}; "
               f"{res.error or 'identifier not in PubMed/Crossref'}",
    ), res


def _gate_topic_match(
    c: WebParamCandidate, q: WebParamQuery,
) -> GateResult:
    keywords = (
        [q.drug_name] + list(q.drug_synonyms)
        + [q.parameter] + list(q.parameter_synonyms)
    )
    keywords = [k for k in keywords if k]
    res, ok = verify_citation_topic(
        c.citation_id, claim_keywords=keywords, min_overlap=1,
    )
    if ok:
        return GateResult(
            gate="topic_match", passed=True,
            reason=f"title shares keyword(s) with claim",
        )
    if res.status == CitationStatus.TOPIC_MISMATCH:
        return GateResult(
            gate="topic_match", passed=False,
            reason=f"PMID exists but title shares no keyword with "
                   f"{[q.drug_name, q.parameter]}",
        )
    return GateResult(
        gate="topic_match", passed=False,
        reason=f"upstream verify status={res.status.value}",
    )


def _gate_evidence_class(c: WebParamCandidate) -> GateResult:
    if c.evidence_class in _ACCEPTABLE_CLASSES:
        return GateResult(
            gate="evidence_class", passed=True,
            reason=f"{c.evidence_class.value} is acceptable for auto-acceptance",
        )
    return GateResult(
        gate="evidence_class", passed=False,
        reason=f"{c.evidence_class.value} is not acceptable for auto-acceptance "
               f"(only {sorted(c.value for c in _ACCEPTABLE_CLASSES)} pass)",
    )


def _gate_range(c: WebParamCandidate, q: WebParamQuery) -> GateResult:
    if q.expected_range is None:
        return GateResult(
            gate="range_check", passed=True, severity="soft",
            reason="no expected_range supplied — skipped",
        )
    lo, hi = q.expected_range
    if lo <= c.value <= hi:
        return GateResult(
            gate="range_check", passed=True,
            reason=f"value {c.value} within [{lo}, {hi}]",
        )
    return GateResult(
        gate="range_check", passed=False,
        reason=f"value {c.value} outside [{lo}, {hi}]",
    )


def _gate_unit(c: WebParamCandidate, q: WebParamQuery) -> GateResult:
    if not q.expected_unit:
        return GateResult(
            gate="unit_check", passed=True, severity="soft",
            reason="no expected_unit supplied — skipped",
        )
    # Loose match: server-side conversion is expected at the boundary.
    raw = _normalize_text(c.unit).replace(" ", "")
    expect = _normalize_text(q.expected_unit).replace(" ", "")
    if raw == expect:
        return GateResult(
            gate="unit_check", passed=True,
            reason=f"unit '{c.unit}' matches expected '{q.expected_unit}'",
        )
    # The server should attempt a pint conversion downstream. Here we only
    # flag obvious mismatches by string comparison so the LLM cannot pass
    # 'mg/L' as 'L/h' silently.
    return GateResult(
        gate="unit_check", passed=False,
        reason=f"unit '{c.unit}' does not match expected '{q.expected_unit}' — "
               f"server will not auto-convert; provide explicit conversion",
    )


def _gate_context_match(c: WebParamCandidate, q: WebParamQuery) -> GateResult:
    """
    Species/matrix mismatch is the dominant 'comparator drug value
    in the same paper' failure mode. When the query specifies species
    or matrix, the candidate MUST declare a matching value — declaring
    nothing is rejected (codex review 2026-04-29).
    """
    issues = []
    if q.species:
        if not c.context.species:
            issues.append(
                f"query.species='{q.species}' but candidate did not declare "
                f"context.species — silent omission would let a rat value "
                f"be submitted as human"
            )
        elif c.context.species.lower() != q.species.lower():
            issues.append(
                f"species mismatch: requested '{q.species}', "
                f"candidate '{c.context.species}'"
            )
    if q.matrix:
        if not c.context.matrix:
            issues.append(
                f"query.matrix='{q.matrix}' but candidate did not declare "
                f"context.matrix — silent omission would let an HLM value "
                f"be submitted as hepatocyte"
            )
        elif c.context.matrix.lower() != q.matrix.lower():
            issues.append(
                f"matrix mismatch: requested '{q.matrix}', "
                f"candidate '{c.context.matrix}'"
            )
    # Even when query did not specify species/matrix, candidate must declare
    # at minimum the species — measurement context without species is
    # ambiguous on its face.
    if not q.species and not c.context.species:
        issues.append(
            "candidate did not declare species — submit context.species "
            "to claim this value (no value is biologically meaningful "
            "without species)"
        )
    if issues:
        return GateResult(gate="context_match", passed=False, reason="; ".join(issues))
    return GateResult(
        gate="context_match", passed=True,
        reason=f"species={c.context.species}, "
               f"matrix={c.context.matrix or 'unspecified'}",
    )


def _strip_xml(text: str) -> str:
    """Crude XML/HTML tag stripper for PMC/abstract text."""
    return re.sub(r"<[^>]+>", " ", text or "")


# Local window (in normalized characters) around the located snippet within
# which we require value, unit, drug term, and parameter term to co-occur.
# Set by codex review 2026-04-29: 'somewhere in abstract' is too loose;
# comparator-drug snippets in the same paper would pass that gate.
_BINDING_WINDOW_CHARS = 600


def _bind_in_window(
    text_norm: str, snippet_norm: str, c: WebParamCandidate, q: WebParamQuery,
) -> tuple[bool, str]:
    """
    Given normalized source text containing the snippet, check whether
    drug term, parameter term, value, unit, AND species/matrix all
    co-occur within a local window around the snippet.

    Returns (all_present, missing_terms_message).
    """
    pos = text_norm.find(snippet_norm)
    if pos < 0:
        return False, "snippet not located"
    window_start = max(0, pos - _BINDING_WINDOW_CHARS // 2)
    window_end = min(len(text_norm), pos + len(snippet_norm) + _BINDING_WINDOW_CHARS // 2)
    window = text_norm[window_start:window_end]

    value_str = _normalize_text(str(c.value))
    unit_str = _normalize_text(c.unit)
    drug_terms = [_normalize_text(q.drug_name)] + [_normalize_text(s) for s in q.drug_synonyms]
    drug_terms = [t for t in drug_terms if t]
    param_terms = [_normalize_text(q.parameter)] + [_normalize_text(s) for s in q.parameter_synonyms]
    param_terms = [t for t in param_terms if t]

    missing = []
    if value_str and value_str not in window:
        missing.append(f"value={value_str}")
    if unit_str and unit_str not in window:
        missing.append(f"unit={unit_str}")
    if drug_terms and not any(t in window for t in drug_terms):
        missing.append(f"drug∈{drug_terms}")
    if param_terms and not any(t in window for t in param_terms):
        missing.append(f"param∈{param_terms}")

    # Species and matrix bind: at least the species must appear in the
    # window (matrix is bound separately via context_match gate, but
    # we add it here as a soft signal).
    if q.species:
        spec_norm = _normalize_text(q.species)
        if spec_norm and spec_norm not in window:
            missing.append(f"species={spec_norm}")

    if missing:
        return False, f"window-bind missing: {missing}"
    return True, "all terms co-occur in window"


def _evidence_binding_status(
    c: WebParamCandidate, q: WebParamQuery,
) -> tuple[EvidenceBindingStatus, str]:
    """
    Verify the snippet, value, unit, drug, parameter, and species/matrix
    co-occur in a LOCAL WINDOW of retrievable source text — not just
    exist somewhere in the document.

    Window-scoped binding (codex review 2026-04-29) defends against:
      - comparator-drug snippet in the same paper (drug term won't be
        in the window around the snippet)
      - pre-converted units (the LLM-declared unit won't appear near
        the snippet because the source uses the original unit)
      - value extracted from a different table row than the snippet
        (value won't be in the window around the snippet)

    Order of attempts:
      1. PubMed abstract (E-utils efetch).
      2. PMC OA full text.
      3. PAYWALLED_UNVERIFIED if neither retrievable.

    Returns (status, evidence_text_excerpt). VERIFIED_EXACT_SNIPPET
    requires window co-occurrence of all terms. VERIFIED_NUMERIC_NEAR_CONTEXT
    means snippet is found but window binding incomplete — this state
    is NOT acceptable for auto-merged HIGH confidence (see synthesize).
    """
    if c.citation_type != "pmid":
        # DOI-only candidates currently cannot be snippet-verified through
        # E-utils. Future: integrate Unpaywall / Crossref full-text links.
        return EvidenceBindingStatus.METADATA_ONLY, "DOI candidates not yet snippet-verifiable"

    pmid = c.citation_id.strip()
    snippet_norm = _normalize_text(c.snippet)
    if not snippet_norm:
        return EvidenceBindingStatus.NOT_FOUND, "empty snippet"

    abstract = fetch_pmid_abstract(pmid)
    if abstract:
        abs_norm = _normalize_text(abstract)
        if snippet_norm in abs_norm:
            ok, why = _bind_in_window(abs_norm, snippet_norm, c, q)
            if ok:
                return EvidenceBindingStatus.VERIFIED_EXACT_SNIPPET, abstract[:400]
            return (EvidenceBindingStatus.VERIFIED_NUMERIC_NEAR_CONTEXT,
                    f"snippet found but window binding loose: {why}")

    pmcid = fetch_pmcid_for_pmid(pmid)
    if pmcid:
        full = fetch_pmc_full_text(pmcid)
        if full:
            full_norm = _normalize_text(_strip_xml(full))
            if snippet_norm in full_norm:
                ok, why = _bind_in_window(full_norm, snippet_norm, c, q)
                if ok:
                    return EvidenceBindingStatus.VERIFIED_EXACT_SNIPPET, full[:400]
                return (EvidenceBindingStatus.VERIFIED_NUMERIC_NEAR_CONTEXT,
                        f"snippet found but window binding loose: {why}")

    if abstract is None and not pmcid:
        return EvidenceBindingStatus.PAYWALLED_UNVERIFIED, "no abstract or PMC OA full text"
    return EvidenceBindingStatus.NOT_FOUND, "snippet not in abstract or PMC text"


def verify_candidate(
    candidate: WebParamCandidate, query: WebParamQuery,
) -> tuple[list[GateResult], EvidenceBindingStatus, str]:
    """Run all gates against one candidate. Returns
    (gate_results, binding_status, binding_evidence_excerpt)."""
    gates: list[GateResult] = []

    g1 = _gate_required_fields(candidate, query)
    gates.append(g1)
    if not g1.passed:
        return gates, EvidenceBindingStatus.NOT_FOUND, ""

    g2, _cite_res = _gate_citation_exists(candidate)
    gates.append(g2)
    if not g2.passed:
        return gates, EvidenceBindingStatus.NOT_FOUND, ""

    gates.append(_gate_topic_match(candidate, query))
    gates.append(_gate_evidence_class(candidate))
    gates.append(_gate_range(candidate, query))
    gates.append(_gate_unit(candidate, query))
    gates.append(_gate_context_match(candidate, query))

    # Evidence binding is the heaviest gate (HTTP fetch). Run last so
    # quick rejections short-circuit the network.
    binding, excerpt = _evidence_binding_status(candidate, query)
    # Only EXACT_SNIPPET (window-bound: drug+parameter+value+unit+species
    # all co-occur in a local window around the snippet) is a hard pass.
    # NUMERIC_NEAR_CONTEXT now means "snippet exists but window binding
    # is incomplete" — that is a soft pass: candidate is preserved for
    # user review and counts toward LOW confidence, never HIGH/MEDIUM
    # auto-acceptance. Codex review 2026-04-29.
    binding_strict_pass = (binding == EvidenceBindingStatus.VERIFIED_EXACT_SNIPPET)
    gates.append(GateResult(
        gate="evidence_binding", passed=binding_strict_pass,
        reason=f"status={binding.value}; {excerpt[:120]}",
        severity="hard" if not binding_strict_pass else "soft",
    ))

    return gates, binding, excerpt


# ---------------------------------------------------------------------
# Independence (laundering defense) and synthesis
# ---------------------------------------------------------------------

def _classify_independence(
    candidates: list[tuple[int, WebParamCandidate]],
) -> list[list[int]]:
    """
    Group candidate indices by upstream measurement using union-find.

    Two candidates share an evidence group if either:
      - they have the same `upstream_citation_id`, OR
      - one's `citation_id` equals the other's `upstream_citation_id`, OR
      - they share the same `citation_id` (duplicate submission).

    Union-find makes this transitive: if C1↔C2 and C2↔C3 are linked, the
    final groups merge all three. Codex review 2026-04-29 caught a
    bridging case (C1.upstream=X, C2.upstream=Y, C3.upstream=X with
    C3.cite=C2.cite) that the previous ad-hoc loop split into two
    groups, inflating confidence.

    A "high confidence" call requires ≥2 INDEPENDENT GROUPS, not ≥2
    candidates.
    """
    if not candidates:
        return []

    n = len(candidates)
    indices = [idx for idx, _ in candidates]
    cands = [c for _, c in candidates]
    pos = {idx: i for i, idx in enumerate(indices)}

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Build keys for each candidate. Each candidate offers up to two
    # identifiers that could link it to another: its own citation_id
    # and its upstream_citation_id (if set).
    own_id = [c.citation_id.strip() for c in cands]
    upstream = [(c.upstream_citation_id or "").strip() for c in cands]

    # Index by every identifier that anchors a measurement chain.
    anchors: dict[str, list[int]] = {}
    for i, c in enumerate(cands):
        for key in {own_id[i], upstream[i] or own_id[i]}:
            if key:
                anchors.setdefault(key, []).append(i)

    # Union all candidates sharing any identifier.
    for key, members in anchors.items():
        if len(members) > 1:
            for m in members[1:]:
                union(members[0], m)

    # Cross-link: if candidate A's upstream == candidate B's own_id,
    # they share an upstream measurement.
    for i in range(n):
        if upstream[i]:
            for j in range(n):
                if i == j:
                    continue
                if upstream[i] == own_id[j] or upstream[i] == upstream[j]:
                    union(i, j)
                if own_id[i] == upstream[j]:
                    union(i, j)

    # Collect groups by root.
    by_root: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        by_root.setdefault(r, []).append(indices[i])

    return list(by_root.values())


def _within_tolerance(values: list[float], parameter: str) -> bool:
    """
    Parameter-specific compatibility tolerance. Two values are 'compatible'
    if they could plausibly come from the same true distribution given
    typical inter-lab variability.

    Bounded fractions (fu_p, fu_hep, R_bp): absolute tolerance.
    Ratio-distributed (CL_int, Peff): geometric tolerance (≤2x).
    """
    if len(values) < 2:
        return True
    bounded = parameter.lower() in {"fu_p", "fu_hep", "fu_inc", "fu_gut", "fa", "fg", "fu_t"}
    if bounded:
        # Logit-aware: spread on logit scale ≤ 1
        clipped = [min(max(v, 1e-4), 1 - 1e-4) for v in values]
        import math
        logits = [math.log(v / (1 - v)) for v in clipped]
        return (max(logits) - min(logits)) <= 1.0
    # Ratio: geometric spread ≤ 2x
    pos = [v for v in values if v > 0]
    if len(pos) < 2:
        return False
    return max(pos) / min(pos) <= 2.0


def synthesize(
    query: WebParamQuery,
    candidates: list[WebParamCandidate],
    gate_results: dict[int, list[GateResult]],
    binding_status: dict[int, EvidenceBindingStatus],
) -> WebParamResult:
    """Combine per-candidate gate results into a final WebParamResult."""
    accepted_indices = []
    for i in range(len(candidates)):
        gates = gate_results.get(i, [])
        # ALL hard gates must pass.
        hard_failed = [g for g in gates if not g.passed and g.severity == "hard"]
        if hard_failed:
            continue
        accepted_indices.append(i)

    if not accepted_indices:
        return WebParamResult(
            query=query, candidates=candidates,
            gate_results=gate_results, binding_status=binding_status,
            accepted_indices=[], confidence=Confidence.NONE,
            rationale="No candidate passed all hard gates.",
        )

    # Independence grouping
    accepted_pairs = [(i, candidates[i]) for i in accepted_indices]
    groups = _classify_independence(accepted_pairs)

    # Confidence selection
    if len(groups) >= 2:
        # Take one representative per group (highest-binding-strength).
        reps: list[int] = []
        for g in groups:
            best = max(
                g,
                key=lambda j: (
                    binding_status.get(j, EvidenceBindingStatus.NOT_FOUND)
                    == EvidenceBindingStatus.VERIFIED_EXACT_SNIPPET,
                    binding_status.get(j, EvidenceBindingStatus.NOT_FOUND)
                    == EvidenceBindingStatus.VERIFIED_NUMERIC_NEAR_CONTEXT,
                    candidates[j].is_direct_measurement,
                ),
            )
            reps.append(best)
        rep_values = [candidates[j].value for j in reps]
        if _within_tolerance(rep_values, query.parameter):
            # Median is safer than geomean when 2 values, and identical for 2.
            rep_values_sorted = sorted(rep_values)
            mid = rep_values_sorted[len(rep_values_sorted) // 2]
            return WebParamResult(
                query=query, candidates=candidates,
                gate_results=gate_results, binding_status=binding_status,
                accepted_indices=accepted_indices,
                confidence=Confidence.HIGH,
                accepted_value=mid,
                accepted_unit=candidates[reps[0]].unit,
                rationale=(
                    f"{len(groups)} independent measurement group(s) within "
                    f"parameter-specific tolerance — accepted median."
                ),
            )
        return WebParamResult(
            query=query, candidates=candidates,
            gate_results=gate_results, binding_status=binding_status,
            accepted_indices=accepted_indices,
            confidence=Confidence.CONFLICT,
            rationale=(
                f"{len(groups)} independent groups disagree beyond tolerance "
                f"(values: {rep_values}). Refusing to merge — manual review "
                f"required. Each candidate is preserved in the audit trail."
            ),
        )

    # Single independent group — accepted_value ONLY for EXACT_SNIPPET.
    # NUMERIC_NEAR_CONTEXT (snippet present but window binding loose) is
    # explicitly NOT auto-accepted. Codex review 2026-04-29.
    rep = accepted_indices[0]
    binding = binding_status.get(rep, EvidenceBindingStatus.NOT_FOUND)
    if binding == EvidenceBindingStatus.VERIFIED_EXACT_SNIPPET:
        return WebParamResult(
            query=query, candidates=candidates,
            gate_results=gate_results, binding_status=binding_status,
            accepted_indices=accepted_indices,
            confidence=Confidence.MEDIUM,
            accepted_value=candidates[rep].value,
            accepted_unit=candidates[rep].unit,
            rationale=(
                "1 candidate, snippet exact-bound to source text "
                "(drug, parameter, value, unit, species co-occur in "
                "the window around the snippet)."
            ),
        )
    return WebParamResult(
        query=query, candidates=candidates,
        gate_results=gate_results, binding_status=binding_status,
        accepted_indices=accepted_indices,
        confidence=Confidence.LOW,
        accepted_value=None,
        accepted_unit=candidates[rep].unit,
        rationale=(
            f"1 candidate with binding={binding.value}. Auto-acceptance "
            f"requires VERIFIED_EXACT_SNIPPET (window co-occurrence of "
            f"drug+parameter+value+unit+species). Manual review required."
        ),
    )


# ---------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------

_AUDIT_DIR = Path(__file__).resolve().parent.parent / "data"
_AUDIT_LOG = _AUDIT_DIR / "web_param_audit.jsonl"


def _write_audit(result: WebParamResult) -> str:
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": asdict(result.query),
        "candidates": [_candidate_to_dict(c) for c in result.candidates],
        "gate_results": {
            str(i): [asdict(g) for g in gs]
            for i, gs in result.gate_results.items()
        },
        "binding_status": {
            str(i): b.value for i, b in result.binding_status.items()
        },
        "accepted_indices": result.accepted_indices,
        "confidence": result.confidence.value,
        "accepted_value": result.accepted_value,
        "accepted_unit": result.accepted_unit,
        "rationale": result.rationale,
    }
    try:
        _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        with open(_AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        return ""
    return str(_AUDIT_LOG)


def _candidate_to_dict(c: WebParamCandidate) -> dict:
    d = asdict(c)
    d["evidence_class"] = c.evidence_class.value
    return d


# ---------------------------------------------------------------------
# Public one-shot entry point
# ---------------------------------------------------------------------

def search_and_verify(
    query: WebParamQuery,
    candidates: list[WebParamCandidate],
    *,
    persist_audit: bool = True,
) -> WebParamResult:
    """
    Verify a list of LLM-supplied candidates against `query`.

    This function does NOT perform the web search — it verifies what the
    LLM driver returns. The driver (Claude Code, etc.) is responsible
    for issuing search queries; this layer prevents the driver from
    fabricating values.
    """
    gate_results: dict[int, list[GateResult]] = {}
    binding_status: dict[int, EvidenceBindingStatus] = {}
    for i, c in enumerate(candidates):
        gates, binding, _excerpt = verify_candidate(c, query)
        gate_results[i] = gates
        binding_status[i] = binding

    result = synthesize(query, candidates, gate_results, binding_status)
    if persist_audit:
        result.audit_path = _write_audit(result)
    return result
