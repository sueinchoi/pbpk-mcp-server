"""
Drug property lookup from external databases.

Queries ChEMBL REST API to retrieve physicochemical and PK properties
for a drug by name or ChEMBL ID.

Falls back to a curated offline database for common PBPK drugs.
"""

import json
from typing import Optional
from dataclasses import dataclass

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

    def to_markdown(self) -> str:
        lines = [
            f"## Drug Properties — {self.name}\n",
            f"Source: {self.source}\n",
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
    Search ChEMBL for drug properties by name.

    Uses the ChEMBL REST API (no API key required).
    """
    if not HAS_URLLIB:
        return None

    try:
        # Search by molecule name
        url = f"{CHEMBL_API}/molecule/search.json?q={drug_name}&limit=1"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        if not data.get("molecules"):
            return None

        mol = data["molecules"][0]
        props = mol.get("molecule_properties", {}) or {}

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
        )
    except Exception:
        return None


# ===================================================================
# Offline curated database for common PBPK drugs
# ===================================================================

OFFLINE_DB = {
    "midazolam": DrugProperties("Midazolam", "CHEMBL601", 325.8, 3.89, 6.2, 0, 3, 30.2, 0, "curated"),
    "caffeine": DrugProperties("Caffeine", "CHEMBL113", 194.2, -0.07, 10.4, 0, 3, 58.4, 0, "curated"),
    "metformin": DrugProperties("Metformin", "CHEMBL1431", 129.2, -1.43, 12.4, 2, 3, 91.5, 0, "curated"),
    "theophylline": DrugProperties("Theophylline", "CHEMBL190", 180.2, -0.02, 8.6, 1, 3, 69.3, 0, "curated"),
    "diazepam": DrugProperties("Diazepam", "CHEMBL12", 284.7, 2.82, 3.4, 0, 3, 32.7, 0, "curated"),
    "warfarin": DrugProperties("Warfarin", "CHEMBL1464", 308.3, 2.60, 5.0, 1, 3, 63.6, 0, "curated"),
    "ibuprofen": DrugProperties("Ibuprofen", "CHEMBL521", 206.3, 3.97, 4.91, 1, 1, 37.3, 0, "curated"),
    "omeprazole": DrugProperties("Omeprazole", "CHEMBL1503", 345.4, 2.23, 4.77, 1, 5, 96.3, 0, "curated"),
    "atorvastatin": DrugProperties("Atorvastatin", "CHEMBL1487", 558.6, 4.46, 4.33, 4, 5, 111.8, 1, "curated"),
    "metoprolol": DrugProperties("Metoprolol", "CHEMBL13", 267.4, 1.88, 9.56, 2, 4, 50.7, 0, "curated"),
    "propranolol": DrugProperties("Propranolol", "CHEMBL27", 259.3, 3.48, 9.42, 2, 3, 41.5, 0, "curated"),
    "ketoconazole": DrugProperties("Ketoconazole", "CHEMBL75", 531.4, 4.35, 6.51, 0, 7, 69.1, 1, "curated"),
    "rifampin": DrugProperties("Rifampin", "CHEMBL374478", 822.9, 3.71, 1.7, 6, 12, 220.2, 2, "curated"),
    "carbamazepine": DrugProperties("Carbamazepine", "CHEMBL108", 236.3, 2.45, 13.9, 1, 2, 46.3, 0, "curated"),
    "phenytoin": DrugProperties("Phenytoin", "CHEMBL16", 252.3, 2.47, 8.33, 2, 2, 58.2, 0, "curated"),
    "verapamil": DrugProperties("Verapamil", "CHEMBL6966", 454.6, 3.79, 8.92, 0, 6, 64.0, 0, "curated"),
}


def lookup_drug(drug_name: str, try_online: bool = True) -> Optional[DrugProperties]:
    """
    Look up drug properties. Tries offline DB first, then ChEMBL.
    """
    key = drug_name.lower().strip()

    # Offline first
    if key in OFFLINE_DB:
        return OFFLINE_DB[key]

    # Try ChEMBL
    if try_online:
        result = search_chembl(drug_name)
        if result:
            return result

    return None
