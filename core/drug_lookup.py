"""
Drug property lookup from external databases.

Queries ChEMBL REST API to retrieve physicochemical and PK properties
for a drug by name or ChEMBL ID.

Falls back to a curated offline database for common PBPK drugs.

PROVENANCE NOTE
---------------
The OFFLINE_DB is a hardcoded snapshot, not a live ChEMBL response.
Values were transcribed by hand and have not been verified against
the live ChEMBL endpoint with response hashes. Therefore offline
hits are tagged source='curated_unverified' and emit a stderr
warning the first time they are returned. To clear this, run
`refresh_offline_db_against_chembl()` and replace OFFLINE_DB with
the result.
"""

import hashlib
import json
import sys
import time
from typing import Optional
from dataclasses import dataclass, field

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


@dataclass
class DrugProperties:
    """Retrieved drug properties."""
    name: str
    chembl_id: str = ""
    mw: float = 0.0
    logP: float = 0.0
    pKa: Optional[float] = None
    hbd: int = 0            # H-bond donors
    hba: int = 0            # H-bond acceptors
    psa: float = 0.0        # Polar surface area
    ro5_violations: int = 0
    source: str = "unknown"
    # Provenance fields populated for live ChEMBL fetches
    fetched_at: Optional[str] = None       # ISO 8601 UTC of fetch
    response_hash: Optional[str] = None    # sha256[:16] of raw response

    def to_markdown(self) -> str:
        source_label = self.source
        if self.source == "curated_unverified":
            source_label += (
                "  (⚠️ hardcoded snapshot, not validated against live ChEMBL — "
                "do NOT cite without re-fetch)"
            )
        elif self.source == "ChEMBL" and self.fetched_at:
            source_label += f" (fetched {self.fetched_at}, hash {self.response_hash})"
        lines = [
            f"## Drug Properties — {self.name}\n",
            f"Source: {source_label}\n",
            "| Property | Value |",
            "|----------|-------|",
            f"| ChEMBL ID | {self.chembl_id} |",
            f"| MW | {self.mw:.2f} g/mol |",
            f"| logP | {self.logP:.2f} |",
            f"| pKa | {self.pKa if self.pKa else 'N/A'} |",
            f"| HBD | {self.hbd} |",
            f"| HBA | {self.hba} |",
            f"| PSA | {self.psa:.1f} Å² |",
            f"| Ro5 violations | {self.ro5_violations} |",
        ]
        return "\n".join(lines)


def search_chembl(drug_name: str) -> Optional[DrugProperties]:
    """
    Search ChEMBL for drug properties by name. Records the fetch
    timestamp and a response hash so callers can audit whether the
    values came from a live response (vs the offline snapshot).

    Uses the ChEMBL REST API (no API key required).

    Returns None on:
      - urllib unavailable
      - HTTP error (caught and logged to stderr — do not silently swallow)
      - empty molecule list
      - malformed response

    Distinguishing "no result" (None) from "live empty fields" (zeros)
    is important: a returned `DrugProperties` with mw=0 means ChEMBL
    actually returned 0 / null for that property, not that the lookup
    failed. Callers must check `mw > 0` before using values.
    """
    if not HAS_URLLIB:
        return None

    url = f"{CHEMBL_API}/molecule/search.json?q={drug_name}&limit=1"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            data = json.loads(raw.decode())
    except Exception as exc:  # noqa: BLE001  - urllib raises many subclasses
        # Surface the failure mode to stderr so a caching layer or audit
        # can distinguish "ChEMBL down" from "drug not in ChEMBL".
        print(f"[drug_lookup] ChEMBL fetch failed for '{drug_name}': "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return None

    if not data.get("molecules"):
        return None

    mol = data["molecules"][0]
    props = mol.get("molecule_properties", {}) or {}

    response_hash = hashlib.sha256(raw).hexdigest()[:16]
    fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return DrugProperties(
        name=mol.get("pref_name", drug_name) or drug_name,
        chembl_id=mol.get("molecule_chembl_id", ""),
        mw=float(props.get("full_mwt", 0) or 0),
        logP=float(props.get("alogp", 0) or 0),
        hbd=int(props.get("hbd", 0) or 0),
        hba=int(props.get("hba", 0) or 0),
        psa=float(props.get("psa", 0) or 0),
        ro5_violations=int(props.get("num_ro5_violations", 0) or 0),
        source="ChEMBL",
        fetched_at=fetched_at,
        response_hash=response_hash,
    )


# ===================================================================
# Offline curated database for common PBPK drugs
# ===================================================================

OFFLINE_DB = {
    "midazolam": DrugProperties("Midazolam", "CHEMBL601", 325.8, 3.89, 6.2, 0, 3, 30.2, 0, "curated_unverified"),
    "caffeine": DrugProperties("Caffeine", "CHEMBL113", 194.2, -0.07, 10.4, 0, 3, 58.4, 0, "curated_unverified"),
    "metformin": DrugProperties("Metformin", "CHEMBL1431", 129.2, -1.43, 12.4, 2, 3, 91.5, 0, "curated_unverified"),
    "theophylline": DrugProperties("Theophylline", "CHEMBL190", 180.2, -0.02, 8.6, 1, 3, 69.3, 0, "curated_unverified"),
    "diazepam": DrugProperties("Diazepam", "CHEMBL12", 284.7, 2.82, 3.4, 0, 3, 32.7, 0, "curated_unverified"),
    "warfarin": DrugProperties("Warfarin", "CHEMBL1464", 308.3, 2.60, 5.0, 1, 3, 63.6, 0, "curated_unverified"),
    "ibuprofen": DrugProperties("Ibuprofen", "CHEMBL521", 206.3, 3.97, 4.91, 1, 1, 37.3, 0, "curated_unverified"),
    "omeprazole": DrugProperties("Omeprazole", "CHEMBL1503", 345.4, 2.23, 4.77, 1, 5, 96.3, 0, "curated_unverified"),
    "atorvastatin": DrugProperties("Atorvastatin", "CHEMBL1487", 558.6, 4.46, 4.33, 4, 5, 111.8, 1, "curated_unverified"),
    "metoprolol": DrugProperties("Metoprolol", "CHEMBL13", 267.4, 1.88, 9.56, 2, 4, 50.7, 0, "curated_unverified"),
    "propranolol": DrugProperties("Propranolol", "CHEMBL27", 259.3, 3.48, 9.42, 2, 3, 41.5, 0, "curated_unverified"),
    "ketoconazole": DrugProperties("Ketoconazole", "CHEMBL75", 531.4, 4.35, 6.51, 0, 7, 69.1, 1, "curated_unverified"),
    "rifampin": DrugProperties("Rifampin", "CHEMBL374478", 822.9, 3.71, 1.7, 6, 12, 220.2, 2, "curated_unverified"),
    "carbamazepine": DrugProperties("Carbamazepine", "CHEMBL108", 236.3, 2.45, 13.9, 1, 2, 46.3, 0, "curated_unverified"),
    "phenytoin": DrugProperties("Phenytoin", "CHEMBL16", 252.3, 2.47, 8.33, 2, 2, 58.2, 0, "curated_unverified"),
    "verapamil": DrugProperties("Verapamil", "CHEMBL6966", 454.6, 3.79, 8.92, 0, 6, 64.0, 0, "curated_unverified"),
}


_OFFLINE_WARNED: set[str] = set()


def lookup_drug(
    drug_name: str,
    try_online: bool = True,
    *,
    prefer_live: bool = True,
) -> Optional[DrugProperties]:
    """
    Look up drug properties.

    Order of resolution:
      1. If prefer_live=True (default) and try_online=True, try ChEMBL first.
         A live response is preferred over the unverified offline snapshot.
      2. Otherwise (or on ChEMBL failure), fall back to OFFLINE_DB.

    The first time an OFFLINE_DB hit is returned in a process, a stderr
    warning is emitted documenting that the value is unverified. The set
    `_OFFLINE_WARNED` deduplicates per-name so the warning fires once per
    drug, not on every lookup.
    """
    key = drug_name.lower().strip()

    if try_online and prefer_live:
        live = search_chembl(drug_name)
        if live is not None and live.mw > 0:
            return live

    if key in OFFLINE_DB:
        if key not in _OFFLINE_WARNED:
            print(
                f"[drug_lookup] returning offline snapshot for '{key}' — "
                f"source='curated_unverified'. Values were transcribed by "
                f"hand and have not been validated against live ChEMBL with "
                f"a response hash. Re-fetch via ChEMBL before publication.",
                file=sys.stderr,
            )
            _OFFLINE_WARNED.add(key)
        return OFFLINE_DB[key]

    if try_online and not prefer_live:
        return search_chembl(drug_name)

    return None
