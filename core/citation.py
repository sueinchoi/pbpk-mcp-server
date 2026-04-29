"""
Citation verification against PubMed (PMID) and Crossref (DOI).

Why this exists: parameters in PBPK models traditionally cite literature
(R&R 2006, Austin 2002, Yang 2007). LLM-generated parameter sets often
cite plausible-sounding but non-existent PMIDs or DOIs. This module
verifies the citation actually exists with one HTTP call per
identifier, caches the result, and exposes a Pydantic `Citation` model
for use in `Source` 4-tuples.

Modes:
  - online (default): live HTTP to NCBI E-utils and Crossref
  - offline: cache hits only; cache miss returns Unverified status
  - strict: cache miss raises ValueError (use in production /
    publication-grade workflows)

The cache is JSONL at `data/citation_cache.jsonl` so repeated calls
to the same PMID don't hit the network.
"""

from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
_CACHE_FILE = _CACHE_DIR / "citation_cache.jsonl"


class CitationStatus(str, Enum):
    VERIFIED = "verified"             # PMID/DOI exists in PubMed/Crossref
    NOT_FOUND = "not_found"
    UNVERIFIED = "unverified"        # offline mode + cache miss
    NETWORK_ERROR = "network_error"
    INVALID_FORMAT = "invalid_format"
    TOPIC_MISMATCH = "topic_mismatch"  # PMID exists but title unrelated to claim


@dataclass
class CitationResult:
    identifier: str           # PMID or DOI as given
    type: str                 # "pmid" or "doi"
    status: CitationStatus
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    error: Optional[str] = None

    def is_verified(self) -> bool:
        return self.status == CitationStatus.VERIFIED


_PMID_RE = re.compile(r"^\d{1,9}$")
_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$")


# ---------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------

def _load_cache() -> dict[str, dict]:
    """Load the entire JSONL cache into a dict keyed by 'type:identifier'."""
    out: dict[str, dict] = {}
    if not _CACHE_FILE.exists():
        return out
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = f"{rec['type']}:{rec['identifier']}"
                    out[key] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    except OSError:
        pass
    return out


def _append_cache(record: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------
# Live verification
# ---------------------------------------------------------------------

def _verify_pmid_online(pmid: str, timeout: float = 5.0) -> CitationResult:
    if not _HAS_REQUESTS:
        return CitationResult(pmid, "pmid", CitationStatus.NETWORK_ERROR,
                              error="requests not installed")
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "json"}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        result = data.get("result", {})
        if pmid not in result or "error" in result.get(pmid, {}):
            return CitationResult(pmid, "pmid", CitationStatus.NOT_FOUND)
        rec = result[pmid]
        title = rec.get("title", "").strip()
        authors_list = rec.get("authors", [])
        authors = ", ".join(a.get("name", "") for a in authors_list[:3])
        if len(authors_list) > 3:
            authors += " et al."
        pubdate = rec.get("pubdate", "")
        year_m = re.search(r"(\d{4})", pubdate)
        year = int(year_m.group(1)) if year_m else None
        journal = rec.get("source", "")
        return CitationResult(
            identifier=pmid, type="pmid", status=CitationStatus.VERIFIED,
            title=title, authors=authors, year=year, journal=journal,
        )
    except requests.exceptions.RequestException as e:
        return CitationResult(pmid, "pmid", CitationStatus.NETWORK_ERROR, error=str(e))
    except (ValueError, KeyError) as e:
        return CitationResult(pmid, "pmid", CitationStatus.NETWORK_ERROR, error=str(e))


def fetch_pmid_abstract(pmid: str, timeout: float = 5.0) -> Optional[str]:
    """
    Retrieve the abstract text for a PMID via NCBI E-utils efetch.

    Used by the evidence-binding gate in `core/web_param_search.py` —
    a candidate's quoted snippet must appear in the abstract (or PMC
    full text, when available) for the candidate to be auto-accepted.

    Returns None on network/parse error so callers must distinguish
    "abstract unavailable" (network) from "abstract empty" (no result).
    """
    if not _HAS_REQUESTS:
        return None
    if not _PMID_RE.match(pmid.strip()):
        return None
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "text"}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        text = r.text or ""
        # E-utils returns the full record; the abstract is typically the
        # bulk of the body. We return it raw — callers normalize before
        # snippet matching.
        return text.strip() or None
    except requests.exceptions.RequestException:
        return None


def fetch_pmcid_for_pmid(pmid: str, timeout: float = 5.0) -> Optional[str]:
    """
    Translate a PMID to its PMCID (PubMed Central ID) via NCBI E-utils
    elink, when an open-access full-text version is available.

    Returns the PMCID (e.g. "PMC1234567") or None if no PMC record
    exists for this PMID.
    """
    if not _HAS_REQUESTS:
        return None
    if not _PMID_RE.match(pmid.strip()):
        return None
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    params = {
        "dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json",
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        linksets = data.get("linksets", [])
        if not linksets:
            return None
        for ldb in linksets[0].get("linksetdbs", []):
            if ldb.get("dbto") == "pmc":
                links = ldb.get("links", [])
                if links:
                    return f"PMC{links[0]}"
        return None
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return None


def fetch_pmc_full_text(pmcid: str, timeout: float = 10.0) -> Optional[str]:
    """
    Retrieve PMC open-access full-text XML/text for a PMCID. Returns
    raw response text — callers strip XML tags before snippet matching.

    Only works for PMC open-access subset (PMC OA). Returns None for
    paywalled / non-OA records.
    """
    if not _HAS_REQUESTS:
        return None
    pmcid = pmcid.strip().upper()
    if not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc", "id": pmcid.removeprefix("PMC"), "rettype": "xml",
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        text = r.text or ""
        return text or None
    except requests.exceptions.RequestException:
        return None


def _verify_doi_online(doi: str, timeout: float = 5.0) -> CitationResult:
    if not _HAS_REQUESTS:
        return CitationResult(doi, "doi", CitationStatus.NETWORK_ERROR,
                              error="requests not installed")
    url = f"https://api.crossref.org/works/{doi}"
    try:
        r = requests.get(url, timeout=timeout, headers={
            "User-Agent": "pbpk-mcp/1.7 (citation verification)"
        })
        if r.status_code == 404:
            return CitationResult(doi, "doi", CitationStatus.NOT_FOUND)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        title_list = msg.get("title", [])
        title = title_list[0] if title_list else ""
        authors_list = msg.get("author", [])
        authors = ", ".join(
            f"{a.get('family', '')} {a.get('given', '')[:1]}." for a in authors_list[:3]
        ).strip()
        if len(authors_list) > 3:
            authors += " et al."
        year = None
        date_parts = msg.get("issued", {}).get("date-parts", [[None]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]
        journal_list = msg.get("container-title", [])
        journal = journal_list[0] if journal_list else ""
        return CitationResult(
            identifier=doi, type="doi", status=CitationStatus.VERIFIED,
            title=title, authors=authors, year=year, journal=journal,
        )
    except requests.exceptions.RequestException as e:
        return CitationResult(doi, "doi", CitationStatus.NETWORK_ERROR, error=str(e))
    except (ValueError, KeyError) as e:
        return CitationResult(doi, "doi", CitationStatus.NETWORK_ERROR, error=str(e))


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

VerificationMode = str  # "online", "offline", "strict"


def verify_citation(
    identifier: str,
    *,
    mode: VerificationMode = "online",
    timeout: float = 5.0,
) -> CitationResult:
    """
    Verify a PMID or DOI. Auto-detects type from string format.

    Parameters
    ----------
    identifier : str
        PMID (digits only) or DOI ('10.xxxx/yyyy').
    mode : "online" | "offline" | "strict"
        - online: try cache, then live HTTP, append result to cache
        - offline: cache only; cache miss → UNVERIFIED
        - strict: cache miss with no live verification → ValueError
    """
    identifier = identifier.strip()
    if _PMID_RE.match(identifier):
        ident_type = "pmid"
    elif _DOI_RE.match(identifier):
        ident_type = "doi"
    else:
        return CitationResult(
            identifier=identifier, type="unknown",
            status=CitationStatus.INVALID_FORMAT,
            error=f"Not a valid PMID (digits only) or DOI (10.xxxx/yyyy)",
        )

    cache = _load_cache()
    key = f"{ident_type}:{identifier}"
    if key in cache:
        rec = cache[key]
        cached_status = CitationStatus(rec["status"])
        # NOT_FOUND records have a 24h TTL — a transient network failure
        # could have produced a NOT_FOUND that the API would now return
        # as VERIFIED. Verified records are kept indefinitely (PMIDs and
        # DOIs are stable identifiers).
        cached_at = rec.get("cached_at", 0)
        age_h = (time.time() - cached_at) / 3600
        if cached_status == CitationStatus.NOT_FOUND and age_h > 24:
            pass  # fall through to re-verify
        else:
            return CitationResult(
                identifier=rec["identifier"], type=rec["type"],
                status=cached_status,
                title=rec.get("title"), authors=rec.get("authors"),
                year=rec.get("year"), journal=rec.get("journal"),
            )

    if mode == "offline":
        return CitationResult(identifier, ident_type, CitationStatus.UNVERIFIED,
                              error="cache miss in offline mode")
    if mode == "strict":
        # Try once online, but a network error becomes a hard fail
        result = (_verify_pmid_online(identifier, timeout) if ident_type == "pmid"
                  else _verify_doi_online(identifier, timeout))
        if result.status not in (CitationStatus.VERIFIED, CitationStatus.NOT_FOUND):
            raise ValueError(
                f"Strict mode: could not verify {ident_type} '{identifier}' "
                f"({result.status.value}): {result.error}"
            )
        if result.status == CitationStatus.NOT_FOUND:
            raise ValueError(
                f"Strict mode: {ident_type} '{identifier}' not found in "
                f"{'PubMed' if ident_type == 'pmid' else 'Crossref'}"
            )
        _append_cache(_result_to_dict(result))
        return result

    # online (default)
    result = (_verify_pmid_online(identifier, timeout) if ident_type == "pmid"
              else _verify_doi_online(identifier, timeout))
    # Cache only definitive answers (verified / not_found), not transient
    # network errors — those should be retryable on next call.
    if result.status in (CitationStatus.VERIFIED, CitationStatus.NOT_FOUND):
        _append_cache(_result_to_dict(result))
    return result


def _result_to_dict(r: CitationResult) -> dict:
    return {
        "identifier": r.identifier, "type": r.type, "status": r.status.value,
        "title": r.title, "authors": r.authors, "year": r.year,
        "journal": r.journal, "error": r.error,
        "cached_at": time.time(),
    }


def cache_path() -> Path:
    return _CACHE_FILE


# ---------------------------------------------------------------------
# Topic-match verification (defends against existing-but-wrong PMIDs)
# ---------------------------------------------------------------------
#
# A PMID can be VERIFIED (exists in PubMed) yet still be the wrong citation
# for a given parameter — e.g. a PMID about dietary glucose transport cited
# as "diclofenac plasma binding". This gap was how fabricated citations
# survived the v2.6 audit. `verify_citation_topic` requires the cached
# title to share at least one keyword with the claimed parameter context.

# Common stopwords + domain-irrelevant high-frequency terms
_STOPWORDS = {
    "a", "an", "and", "the", "of", "in", "on", "for", "to", "with", "from",
    "by", "at", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "their", "his", "her",
    "or", "but", "not", "no", "if", "than", "so", "such", "into", "onto",
    "study", "studies", "analysis", "review", "data", "results", "method",
    "methods", "based", "using", "during", "between", "among", "after",
    "before", "human", "humans", "subject", "subjects", "patient", "patients",
}

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{2,}")


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return {t.lower() for t in _TOKEN_RE.findall(text)
            if t.lower() not in _STOPWORDS}


def verify_citation_topic(
    identifier: str,
    *,
    claim_keywords: list[str],
    mode: VerificationMode = "online",
    min_overlap: int = 1,
) -> tuple[CitationResult, bool]:
    """
    Verify a PMID/DOI exists AND its title topic-matches the claimed
    parameter keywords.

    Why: the v2.6 audit removed PMIDs that did not exist. But a PMID
    can exist yet describe an unrelated topic — e.g. PMID 9106794
    is dietary glucose transport, not diclofenac binding. Existence
    verification is necessary but not sufficient.

    Parameters
    ----------
    identifier : str
        PMID or DOI.
    claim_keywords : list[str]
        Keywords the cited paper should contain (e.g. for a Midazolam
        Vss claim: ['midazolam', 'distribution'] or ['midazolam', 'PBPK']).
        Matching is case-insensitive on title only (abstract is not fetched).
    mode : "online" | "offline" | "strict"
    min_overlap : int
        Minimum number of claim_keywords that must appear in the cached
        title. Default 1.

    Returns
    -------
    (CitationResult, topic_ok)
        result.status == VERIFIED + topic_ok=True   → cite confidently
        result.status == VERIFIED + topic_ok=False  → existing-but-wrong;
                                                      result.status is
                                                      mutated to TOPIC_MISMATCH
    """
    result = verify_citation(identifier, mode=mode)
    if result.status != CitationStatus.VERIFIED:
        return result, False

    title_tokens = _tokenize(result.title or "")
    claim_tokens = {k.lower() for k in claim_keywords if k}
    overlap = title_tokens & claim_tokens

    if len(overlap) < min_overlap:
        # Existing-but-wrong: surface as a distinct status so callers
        # cannot accept it as "verified" by checking is_verified() alone.
        result.status = CitationStatus.TOPIC_MISMATCH
        result.error = (
            f"PMID/DOI exists but title '{result.title or ''}' shares no "
            f"keyword overlap with the claim {sorted(claim_tokens)}. "
            f"Title tokens: {sorted(title_tokens)[:20]}. This is the "
            f"'existing-but-wrong citation' failure mode."
        )
        return result, False
    return result, True


def audit_citation_cache(
    *, claim_keywords_per_id: dict[str, list[str]],
) -> dict[str, str]:
    """
    Re-audit existing cache entries for topic match. Pass a mapping of
    {identifier: [keywords]} where keywords describe what the citation
    is supposed to support.

    Returns {identifier: verdict} where verdict is one of
    'topic_match', 'topic_mismatch', 'not_in_cache', 'not_verified'.
    """
    cache = _load_cache()
    out: dict[str, str] = {}
    for ident, keywords in claim_keywords_per_id.items():
        # Try both PMID and DOI cache keys
        rec = cache.get(f"pmid:{ident}") or cache.get(f"doi:{ident}")
        if rec is None:
            out[ident] = "not_in_cache"
            continue
        if rec.get("status") != CitationStatus.VERIFIED.value:
            out[ident] = "not_verified"
            continue
        title_tokens = _tokenize(rec.get("title") or "")
        claim_tokens = {k.lower() for k in keywords if k}
        if title_tokens & claim_tokens:
            out[ident] = "topic_match"
        else:
            out[ident] = "topic_mismatch"
    return out
