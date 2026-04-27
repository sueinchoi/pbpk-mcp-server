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
    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    UNVERIFIED = "unverified"        # offline mode + cache miss
    NETWORK_ERROR = "network_error"
    INVALID_FORMAT = "invalid_format"


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
        return CitationResult(
            identifier=rec["identifier"], type=rec["type"],
            status=CitationStatus(rec["status"]),
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
