"""
Anti-hallucination layer tests for core/web_param_search.py.

Run as:
    python -m tests.test_web_param_search

The tests are network-aware: gates that need PubMed E-utils (citation
verification, abstract fetch) use monkeypatched stubs so the suite is
deterministic and offline-runnable. A few "live" tests are clearly
marked and skip if requests is not installed or no network.
"""
from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import core.citation as citation_mod
import core.web_param_search as wps
from core.web_param_search import (
    EvidenceClass, EvidenceBindingStatus, Confidence,
    WebParamQuery, WebParamCandidate, MeasurementContext,
    EvidenceLocation, GateResult,
    verify_candidate, synthesize, search_and_verify,
    _classify_independence, _within_tolerance,
)


PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def test(name: str):
    def decorator(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  ✓ {name}")
        except AssertionError as e:
            FAILED.append((name, str(e)))
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            FAILED.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
        return fn
    return decorator


# ---------------------------------------------------------------------
# Stubs — replace network calls so the suite is deterministic
# ---------------------------------------------------------------------

class _StubRegistry:
    """Drives the stubbed citation behavior per-PMID."""
    def __init__(self):
        # pmid -> CitationResult-shaped dict
        self.pmids: dict[str, dict] = {}
        # pmid -> abstract text
        self.abstracts: dict[str, str] = {}
        # pmid -> pmcid (or None)
        self.pmcids: dict[str, str] = {}
        # pmcid -> full text
        self.pmc_text: dict[str, str] = {}

    def add_pmid(
        self, pmid: str, *, title: str = "", status: str = "verified",
        abstract: str = "", pmcid: str = "", pmc_text: str = "",
    ):
        self.pmids[pmid] = {
            "identifier": pmid, "type": "pmid", "status": status,
            "title": title, "authors": "", "year": 2020, "journal": "",
        }
        if abstract:
            self.abstracts[pmid] = abstract
        if pmcid:
            self.pmcids[pmid] = pmcid
            if pmc_text:
                self.pmc_text[pmcid] = pmc_text


_STUBS = _StubRegistry()


def _stub_verify_citation(identifier, *, mode="online", timeout=5.0):
    from core.citation import CitationResult, CitationStatus
    rec = _STUBS.pmids.get(identifier.strip())
    if rec is None:
        return CitationResult(identifier, "pmid", CitationStatus.NOT_FOUND)
    return CitationResult(
        identifier=rec["identifier"], type=rec["type"],
        status=CitationStatus(rec["status"]),
        title=rec["title"], authors=rec["authors"],
        year=rec["year"], journal=rec["journal"],
    )


def _stub_fetch_abstract(pmid, timeout=5.0):
    return _STUBS.abstracts.get(pmid.strip())


def _stub_fetch_pmcid(pmid, timeout=5.0):
    return _STUBS.pmcids.get(pmid.strip())


def _stub_fetch_pmc(pmcid, timeout=10.0):
    return _STUBS.pmc_text.get(pmcid.strip())


# Install stubs (citation_mod functions are imported by web_param_search
# at module-load, so we patch web_param_search's references too)
citation_mod.verify_citation = _stub_verify_citation
citation_mod.fetch_pmid_abstract = _stub_fetch_abstract
citation_mod.fetch_pmcid_for_pmid = _stub_fetch_pmcid
citation_mod.fetch_pmc_full_text = _stub_fetch_pmc
wps.verify_citation = _stub_verify_citation
wps.fetch_pmid_abstract = _stub_fetch_abstract
wps.fetch_pmcid_for_pmid = _stub_fetch_pmcid
wps.fetch_pmc_full_text = _stub_fetch_pmc


# verify_citation_topic uses verify_citation internally — re-import so the
# already-bound reference picks up the stub. We re-define it locally to
# guarantee the stub flows through.
def _stub_verify_citation_topic(identifier, *, claim_keywords, mode="online", min_overlap=1):
    from core.citation import _tokenize, CitationStatus
    res = _stub_verify_citation(identifier)
    if res.status != CitationStatus.VERIFIED:
        return res, False
    title_tokens = _tokenize(res.title or "")
    claim_tokens = {k.lower() for k in claim_keywords if k}
    if len(title_tokens & claim_tokens) >= min_overlap:
        return res, True
    res.status = CitationStatus.TOPIC_MISMATCH
    return res, False


citation_mod.verify_citation_topic = _stub_verify_citation_topic
wps.verify_citation_topic = _stub_verify_citation_topic


# ---------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------

def _midaz_query(parameter="fu_hep") -> WebParamQuery:
    return WebParamQuery(
        parameter=parameter,
        drug_name="midazolam",
        drug_synonyms=["midazolam", "versed"],
        parameter_synonyms=["fu,hep", "fraction unbound hepatocyte"],
        expected_unit="unitless",
        expected_range=(0.001, 1.0),
        species="human",
        matrix="hepatocyte",
    )


def _good_candidate(pmid="11111", value=0.05, unit="unitless",
                     species="human", matrix="hepatocyte",
                     evidence_class=EvidenceClass.PRIMARY_MEASUREMENT,
                     snippet="The fraction unbound in human hepatocytes "
                             "(fu_hep) was 0.05 in midazolam"):
    return WebParamCandidate(
        parameter="fu_hep", drug_name="midazolam",
        value=value, unit=unit,
        citation_id=pmid, citation_type="pmid",
        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        snippet=snippet,
        context=MeasurementContext(species=species, matrix=matrix, method="RED"),
        evidence_class=evidence_class,
        is_direct_measurement=True,
        raw_search_query="midazolam fu hepatocyte",
    )


# ---------------------------------------------------------------------
# Tests — required fields gate
# ---------------------------------------------------------------------
print("\n## required_fields gate")


@test("missing citation_id rejects with 'missing'")
def t():
    c = _good_candidate()
    c.citation_id = ""
    gates, _binding, _ = verify_candidate(c, _midaz_query())
    failed = [g for g in gates if not g.passed]
    assert any("citation_id" in g.reason for g in failed), failed


@test("missing snippet rejects")
def t():
    c = _good_candidate()
    c.snippet = ""
    gates, _, _ = verify_candidate(c, _midaz_query())
    assert any("snippet" in g.reason for g in gates if not g.passed)


@test("parameter mismatch rejects")
def t():
    c = _good_candidate()
    c.parameter = "fu_p"  # query asks for fu_hep
    gates, _, _ = verify_candidate(c, _midaz_query())
    assert any("parameter mismatch" in g.reason for g in gates if not g.passed)


# ---------------------------------------------------------------------
# Tests — citation existence + topic match
# ---------------------------------------------------------------------
print("\n## citation_exists + topic_match gates")


@test("fabricated PMID rejects with status=not_found")
def t():
    _STUBS.pmids.clear()  # nothing exists
    c = _good_candidate(pmid="99999999")
    gates, binding, _ = verify_candidate(c, _midaz_query())
    cite_gate = next(g for g in gates if g.gate == "citation_exists")
    assert not cite_gate.passed
    assert "not_found" in cite_gate.reason


@test("real PMID with unrelated title rejects topic_match")
def t():
    _STUBS.pmids.clear()
    _STUBS.add_pmid(
        "9106794", title="Dietary glucose transport in rat jejunum",
    )
    c = _good_candidate(pmid="9106794")
    gates, _, _ = verify_candidate(c, _midaz_query())
    topic = next(g for g in gates if g.gate == "topic_match")
    assert not topic.passed
    assert "no keyword" in topic.reason or "TOPIC_MISMATCH" in topic.reason or "topic" in topic.reason.lower()


@test("real PMID with on-topic title passes citation+topic")
def t():
    _STUBS.pmids.clear()
    _STUBS.add_pmid(
        "11111", title="Midazolam hepatocyte fu_hep determination",
        abstract="The fraction unbound in human hepatocytes (fu_hep) "
                 "was 0.05 in midazolam incubation. Methods: RED.",
    )
    c = _good_candidate(pmid="11111")
    gates, _, _ = verify_candidate(c, _midaz_query())
    cite = next(g for g in gates if g.gate == "citation_exists")
    topic = next(g for g in gates if g.gate == "topic_match")
    assert cite.passed and topic.passed, gates


# ---------------------------------------------------------------------
# Tests — evidence_class gate
# ---------------------------------------------------------------------
print("\n## evidence_class gate")


@test("blog_or_web class rejects")
def t():
    _STUBS.pmids.clear()
    _STUBS.add_pmid("22222", title="Midazolam absorption review")
    c = _good_candidate(pmid="22222",
                        evidence_class=EvidenceClass.BLOG_OR_WEB)
    gates, _, _ = verify_candidate(c, _midaz_query())
    ec = next(g for g in gates if g.gate == "evidence_class")
    assert not ec.passed


@test("vendor_doc class rejects")
def t():
    _STUBS.pmids.clear()
    _STUBS.add_pmid("33333", title="midazolam fu_hep value")
    c = _good_candidate(pmid="33333",
                        evidence_class=EvidenceClass.VENDOR_DOC)
    gates, _, _ = verify_candidate(c, _midaz_query())
    ec = next(g for g in gates if g.gate == "evidence_class")
    assert not ec.passed


@test("primary_measurement passes class gate")
def t():
    _STUBS.pmids.clear()
    _STUBS.add_pmid(
        "44444", title="midazolam fu_hep primary measurement",
        abstract="fu_hep value 0.05 unitless midazolam human hepatocyte RED",
    )
    c = _good_candidate(pmid="44444")
    gates, _, _ = verify_candidate(c, _midaz_query())
    ec = next(g for g in gates if g.gate == "evidence_class")
    assert ec.passed


# ---------------------------------------------------------------------
# Tests — range + unit + context match
# ---------------------------------------------------------------------
print("\n## range/unit/context gates")


@test("value above expected_range rejects")
def t():
    _STUBS.pmids.clear()
    _STUBS.add_pmid("55555", title="midazolam fu_hep",
                    abstract="fu_hep midazolam value 99 unitless human hepatocyte")
    c = _good_candidate(pmid="55555", value=99.0)
    gates, _, _ = verify_candidate(c, _midaz_query())
    rg = next(g for g in gates if g.gate == "range_check")
    assert not rg.passed


@test("unit mismatch rejects")
def t():
    _STUBS.pmids.clear()
    _STUBS.add_pmid("66666", title="midazolam fu_hep")
    c = _good_candidate(pmid="66666", unit="L/h")  # query expects unitless
    gates, _, _ = verify_candidate(c, _midaz_query())
    ug = next(g for g in gates if g.gate == "unit_check")
    assert not ug.passed


@test("species mismatch rejects (rat submitted as human)")
def t():
    _STUBS.pmids.clear()
    _STUBS.add_pmid("77777", title="midazolam fu_hep", abstract="rat hepatocyte fu 0.05")
    c = _good_candidate(pmid="77777", species="rat")
    gates, _, _ = verify_candidate(c, _midaz_query())
    cg = next(g for g in gates if g.gate == "context_match")
    assert not cg.passed
    assert "species" in cg.reason


@test("matrix mismatch rejects (HLM submitted as hepatocyte)")
def t():
    _STUBS.pmids.clear()
    _STUBS.add_pmid("88888", title="midazolam fu_inc HLM",
                    abstract="midazolam HLM fu_inc 0.05")
    c = _good_candidate(pmid="88888", matrix="HLM")
    gates, _, _ = verify_candidate(c, _midaz_query())
    cg = next(g for g in gates if g.gate == "context_match")
    assert not cg.passed
    assert "matrix" in cg.reason


@test("missing species rejects context_match")
def t():
    _STUBS.pmids.clear()
    _STUBS.add_pmid("99999", title="midazolam fu_hep", abstract="midazolam fu 0.05")
    c = _good_candidate(pmid="99999", species="")
    gates, _, _ = verify_candidate(c, _midaz_query())
    cg = next(g for g in gates if g.gate == "context_match")
    assert not cg.passed


# ---------------------------------------------------------------------
# Tests — evidence binding (the "right paper, wrong number" defense)
# ---------------------------------------------------------------------
print("\n## evidence_binding gate")


@test("snippet not in abstract → binding=NOT_FOUND, gate fails")
def t():
    _STUBS.pmids.clear()
    _STUBS.add_pmid(
        "11000",
        title="Midazolam hepatocyte fu_hep determination",
        abstract="Different unrelated text about hepatocytes that does NOT contain the candidate snippet at all.",
    )
    c = _good_candidate(pmid="11000")
    gates, binding, _ = verify_candidate(c, _midaz_query())
    eb = next(g for g in gates if g.gate == "evidence_binding")
    assert not eb.passed
    assert binding == EvidenceBindingStatus.NOT_FOUND, binding


@test("snippet in abstract + value+unit → binding=VERIFIED_EXACT_SNIPPET")
def t():
    _STUBS.pmids.clear()
    snippet = "the fraction unbound in human hepatocytes (fu_hep) was 0.05 in midazolam"
    _STUBS.add_pmid(
        "11001",
        title="Midazolam hepatocyte fu_hep",
        abstract=f"Methods: RED. Results: {snippet}. Conclusion: clear unitless reading.",
    )
    c = _good_candidate(pmid="11001", snippet=snippet)
    gates, binding, _ = verify_candidate(c, _midaz_query())
    eb = next(g for g in gates if g.gate == "evidence_binding")
    assert eb.passed
    assert binding == EvidenceBindingStatus.VERIFIED_EXACT_SNIPPET, binding


@test("snippet in abstract but value/unit missing → VERIFIED_NUMERIC_NEAR_CONTEXT")
def t():
    _STUBS.pmids.clear()
    snippet = "the fraction unbound in human hepatocytes was determined"
    # abstract has snippet but lacks value and unit
    _STUBS.add_pmid(
        "11002",
        title="Midazolam hepatocyte fu",
        abstract=f"Methods: RED. {snippet}. Result reported elsewhere.",
    )
    c = _good_candidate(pmid="11002", snippet=snippet)
    gates, binding, _ = verify_candidate(c, _midaz_query())
    assert binding == EvidenceBindingStatus.VERIFIED_NUMERIC_NEAR_CONTEXT, binding


@test("paywalled (no abstract, no PMC) → PAYWALLED_UNVERIFIED")
def t():
    _STUBS.pmids.clear()
    _STUBS.add_pmid("11003", title="Midazolam hepatocyte fu_hep")
    c = _good_candidate(pmid="11003")
    gates, binding, _ = verify_candidate(c, _midaz_query())
    assert binding == EvidenceBindingStatus.PAYWALLED_UNVERIFIED, binding


# ---------------------------------------------------------------------
# Tests — independence + confidence
# ---------------------------------------------------------------------
print("\n## independence + confidence")


@test("two candidates sharing upstream_citation_id collapse to 1 group")
def t():
    pairs = [
        (0, _good_candidate(pmid="A", value=0.05)),
        (1, _good_candidate(pmid="B", value=0.052)),
    ]
    pairs[0][1].upstream_citation_id = "upstream-X"
    pairs[1][1].upstream_citation_id = "upstream-X"
    groups = _classify_independence(pairs)
    assert len(groups) == 1, f"expected 1 group, got {len(groups)}: {groups}"


@test("two candidates with distinct upstream → 2 independent groups")
def t():
    pairs = [
        (0, _good_candidate(pmid="A", value=0.05)),
        (1, _good_candidate(pmid="B", value=0.06)),
    ]
    groups = _classify_independence(pairs)
    assert len(groups) == 2, f"expected 2 groups: {groups}"


@test("_within_tolerance: bounded fraction logit ≤ 1 → compatible")
def t():
    assert _within_tolerance([0.05, 0.07], "fu_hep") is True


@test("_within_tolerance: fu_hep 0.05 vs 0.5 → incompatible")
def t():
    assert _within_tolerance([0.05, 0.5], "fu_hep") is False


@test("_within_tolerance: ratio Peff 1.0 vs 1.5 → compatible (within 2x)")
def t():
    assert _within_tolerance([1.0, 1.5], "Peff") is True


@test("_within_tolerance: ratio Peff 1.0 vs 50.0 → incompatible")
def t():
    assert _within_tolerance([1.0, 50.0], "Peff") is False


# ---------------------------------------------------------------------
# Tests — synthesize end-to-end
# ---------------------------------------------------------------------
print("\n## synthesize confidence states")


def _full_run(stubs_setup, candidates, query=None):
    """Run search_and_verify with audit disabled."""
    stubs_setup()
    q = query or _midaz_query()
    return search_and_verify(q, candidates, persist_audit=False)


@test("end-to-end: no candidates → confidence=NONE")
def t():
    res = _full_run(lambda: None, [])
    assert res.confidence == Confidence.NONE


@test("end-to-end: 1 verified primary candidate → MEDIUM")
def t():
    def setup():
        _STUBS.pmids.clear()
        snippet = "midazolam human hepatocyte fu_hep 0.05 unitless RED"
        _STUBS.add_pmid("MED1", title="midazolam hepatocyte fu_hep RED",
                        abstract=snippet)
    c = _good_candidate(pmid="MED1",
                        snippet="midazolam human hepatocyte fu_hep 0.05 unitless")
    res = _full_run(setup, [c])
    assert res.confidence == Confidence.MEDIUM, res.confidence
    assert res.accepted_value == 0.05


@test("end-to-end: 2 independent compatible candidates → HIGH")
def t():
    def setup():
        _STUBS.pmids.clear()
        snip = "midazolam fu_hep 0.05 unitless human hepatocyte"
        _STUBS.add_pmid("HI1", title="midazolam fu_hep paper one", abstract=snip)
        snip2 = "midazolam fu_hep 0.07 unitless human hepatocyte"
        _STUBS.add_pmid("HI2", title="midazolam fu_hep paper two", abstract=snip2)
    c1 = _good_candidate(pmid="HI1", value=0.05,
                         snippet="midazolam fu_hep 0.05 unitless human hepatocyte")
    c2 = _good_candidate(pmid="HI2", value=0.07,
                         snippet="midazolam fu_hep 0.07 unitless human hepatocyte")
    res = _full_run(setup, [c1, c2])
    assert res.confidence == Confidence.HIGH, res.confidence
    assert res.accepted_value is not None


@test("end-to-end: 2 candidates that disagree → CONFLICT, no accepted_value")
def t():
    def setup():
        _STUBS.pmids.clear()
        snip = "midazolam fu_hep 0.05 unitless human hepatocyte"
        _STUBS.add_pmid("CF1", title="midazolam fu_hep paper one", abstract=snip)
        snip2 = "midazolam fu_hep 0.5 unitless human hepatocyte"
        _STUBS.add_pmid("CF2", title="midazolam fu_hep paper two", abstract=snip2)
    c1 = _good_candidate(pmid="CF1", value=0.05,
                         snippet="midazolam fu_hep 0.05 unitless human hepatocyte")
    c2 = _good_candidate(pmid="CF2", value=0.5,
                         snippet="midazolam fu_hep 0.5 unitless human hepatocyte")
    res = _full_run(setup, [c1, c2])
    assert res.confidence == Confidence.CONFLICT, res.confidence
    assert res.accepted_value is None


@test("end-to-end: 2 verified BUT same upstream → MEDIUM not HIGH (laundering defended)")
def t():
    def setup():
        _STUBS.pmids.clear()
        snip = "midazolam fu_hep 0.05 unitless human hepatocyte"
        _STUBS.add_pmid("L1", title="midazolam fu_hep review one", abstract=snip)
        _STUBS.add_pmid("L2", title="midazolam fu_hep review two", abstract=snip)
    c1 = _good_candidate(pmid="L1", value=0.05,
                         snippet="midazolam fu_hep 0.05 unitless human hepatocyte")
    c1.upstream_citation_id = "UPSTREAM_SMITH_2007"
    c1.is_direct_measurement = False
    c2 = _good_candidate(pmid="L2", value=0.05,
                         snippet="midazolam fu_hep 0.05 unitless human hepatocyte")
    c2.upstream_citation_id = "UPSTREAM_SMITH_2007"
    c2.is_direct_measurement = False
    res = _full_run(setup, [c1, c2])
    assert res.confidence != Confidence.HIGH, \
        f"laundering not detected: {res.confidence}"
    assert res.confidence in (Confidence.MEDIUM, Confidence.LOW)


# ---------------------------------------------------------------------
# Tests — MCP tool wiring
# ---------------------------------------------------------------------
print("\n## MCP tool wiring")


@test("search_parameter_with_citation tool exists in registered tools")
def t():
    from mcp.server.fastmcp import FastMCP
    from tools.pbpk_tools import register_pbpk_tools
    m = FastMCP("test")
    register_pbpk_tools(m)
    tools = m._tool_manager._tools
    assert "search_parameter_with_citation" in tools, list(tools.keys())[:5]


@test("tool rejects non-JSON candidates_json")
def t():
    import core.web_param_search as _wps
    # Bypass server import — call the tool function via registration
    from mcp.server.fastmcp import FastMCP
    from tools.pbpk_tools import register_pbpk_tools
    m = FastMCP("test")
    register_pbpk_tools(m)
    fn = m._tool_manager._tools["search_parameter_with_citation"].fn
    try:
        fn(parameter="fu_hep", drug_name="midazolam",
           candidates_json="not valid {{{")
    except ValueError as e:
        assert "JSON" in str(e), str(e)
        return
    raise AssertionError("expected ValueError on bad JSON")


@test("tool with empty candidates returns NONE confidence markdown")
def t():
    from mcp.server.fastmcp import FastMCP
    from tools.pbpk_tools import register_pbpk_tools
    m = FastMCP("test")
    register_pbpk_tools(m)
    fn = m._tool_manager._tools["search_parameter_with_citation"].fn
    out = fn(parameter="fu_hep", drug_name="midazolam",
             candidates_json="[]", expected_unit="unitless")
    assert "Confidence" in out and "none" in out.lower()


# ============================================================
print("\n" + "=" * 60)
print(f"Passed: {len(PASSED)}/{len(PASSED) + len(FAILED)}")
if FAILED:
    print(f"\nFAILED ({len(FAILED)}):")
    for name, err in FAILED:
        print(f"  ✗ {name}")
        print(f"      {err}")
    sys.exit(1)
else:
    print("All web-param-search tests passed.")
