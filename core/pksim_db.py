"""
PKSimDB.sqlite query interface.

Provides direct access to PK-Sim's built-in physiological database:
  - Age/sex/population-specific organ volumes and blood flows
  - CYP/UGT/transporter ontogeny data (294 data points)
  - Species-specific anatomy (10 species)
  - Population distributions for virtual population generation
  - Kp calculation method formulas

Source: Open Systems Pharmacology PK-Sim v12
  https://github.com/Open-Systems-Pharmacology/PK-Sim
"""

import sqlite3
import os
from typing import Optional

# Path to the database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "PKSimDB.sqlite")


def _get_conn():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"PKSimDB.sqlite not found at {DB_PATH}")
    return sqlite3.connect(DB_PATH)


# ===================================================================
# Ontogeny
# ===================================================================

def get_ontogeny(molecule: str, species: str = "Human") -> list[dict]:
    """
    Get ontogeny (maturation) data for a molecule.

    Returns list of {PostmenstrualAge_years, OntogenyFactor, Deviation(GSD)}.

    Available molecules: CYP1A2, CYP2C8, CYP2C9, CYP2C18, CYP2C19,
    CYP2D6, CYP2E1, CYP3A4, CYP3A5, CYP3A7, UGT1A1, UGT1A4, UGT1A6,
    UGT1A9, UGT2B4, UGT2B7, AGP, ALB
    """
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT PostmenstrualAge, OntogenyFactor, Deviation, GroupName
        FROM VIEW_ONTOGENIES
        WHERE MoleculeName = ? AND SpeciesName = ?
        ORDER BY PostmenstrualAge
    """, (molecule, species))
    results = []
    for row in cur.fetchall():
        results.append({
            "PMA_years": row[0],
            "factor": row[1],
            "GSD": row[2],
            "group": row[3],
        })
    conn.close()
    return results


def list_ontogeny_molecules() -> list[str]:
    """List all molecules with ontogeny data."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT MoleculeName FROM VIEW_ONTOGENIES ORDER BY MoleculeName")
    result = [r[0] for r in cur.fetchall()]
    conn.close()
    return result


# ===================================================================
# Parameter Distributions (age/sex/population)
# ===================================================================

def get_organ_parameter(
    organ: str,
    parameter: str,
    age: float = 30.0,
    gender: int = 1,  # 1=male, 2=female
    population: str = "European_ICRP_2002",
) -> Optional[dict]:
    """
    Get organ parameter distribution from PK-Sim database.

    Args:
        organ: e.g., "Liver", "Kidney", "Brain"
        parameter: e.g., "Volume", "Specific blood flow rate"
        age: Age in years
        gender: 1=male, 2=female
        population: Population name

    Returns:
        dict with Mean, Deviation, Distribution type, Dimension
    """
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT ContainerName, ParameterName, Mean, Deviation, Distribution, Dimension, Age
        FROM VIEW_PARAMETER_DISTRIBUTIONS
        WHERE ContainerName = ? AND ParameterName = ?
          AND Gender = ? AND Population = ?
          AND ABS(Age - ?) < 1
        ORDER BY ABS(Age - ?)
        LIMIT 1
    """, (organ, parameter, gender, population, age, age))
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "organ": row[0],
            "parameter": row[1],
            "mean": row[2],
            "deviation": row[3],
            "distribution": row[4],
            "dimension": row[5],
            "age": row[6],
        }
    return None


def get_all_organ_volumes(
    age: float = 30.0,
    gender: int = 1,
    population: str = "European_ICRP_2002",
) -> dict:
    """Get all organ volumes at given age/sex/population."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT ContainerName, Mean, Deviation, Distribution, Dimension
        FROM VIEW_PARAMETER_DISTRIBUTIONS
        WHERE ParameterName = 'Volume'
          AND Gender = ? AND Population = ?
          AND ABS(Age - ?) < 1
          AND ContainerType IN ('ORGAN', 'ORGANISM')
        ORDER BY ContainerName
    """, (gender, population, age))
    results = {}
    for row in cur.fetchall():
        results[row[0]] = {
            "mean": row[1], "deviation": row[2],
            "distribution": row[3], "dimension": row[4],
        }
    conn.close()
    return results


# ===================================================================
# Species
# ===================================================================

def list_species() -> list[dict]:
    """List all available species."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT species, display_name, is_human FROM tab_species ORDER BY sequence")
    results = [{"id": r[0], "name": r[1], "is_human": bool(r[2])} for r in cur.fetchall()]
    conn.close()
    return results


def list_populations() -> list[dict]:
    """List all available populations."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT population, species, display_name, is_age_dependent
        FROM tab_populations ORDER BY sequence
    """)
    results = [{"id": r[0], "species": r[1], "name": r[2], "age_dependent": bool(r[3])}
               for r in cur.fetchall()]
    conn.close()
    return results


# ===================================================================
# Transporters
# ===================================================================

def list_transporters() -> list[dict]:
    """List all known transporters in PK-Sim."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT gene, species, transport_type
        FROM tab_known_transporters
        ORDER BY gene
    """)
    results = [{"gene": r[0], "name": r[0], "direction": r[2], "species": r[1]} for r in cur.fetchall()]
    conn.close()
    return results


# ===================================================================
# Formatting
# ===================================================================

def format_ontogeny(molecule: str) -> str:
    """Format ontogeny data as markdown."""
    data = get_ontogeny(molecule)
    if not data:
        return f"No ontogeny data for '{molecule}'"

    lines = [
        f"## PK-Sim Ontogeny — {molecule}\n",
        f"Source: PKSimDB.sqlite ({len(data)} data points)\n",
        "| PMA (years) | Age (postnatal) | Factor | GSD |",
        "|-------------|-----------------|--------|-----|",
    ]
    for d in data:
        pma = d["PMA_years"]
        postnatal = max(pma - 40/52, 0)
        lines.append(f"| {pma:.2f} | {postnatal:.2f}y | {d['factor']:.4f} | {d['GSD']:.2f} |")
    return "\n".join(lines)


def format_organ_volumes(age: float = 30, gender: int = 1, population: str = "European_ICRP_2002") -> str:
    """Format organ volumes as markdown."""
    vols = get_all_organ_volumes(age, gender, population)
    sex_str = "Male" if gender == 1 else "Female"
    lines = [
        f"## PK-Sim Organ Volumes — {population}, {sex_str}, Age {age}\n",
        "| Organ | Mean (L) | SD/GSD | Distribution |",
        "|-------|---------|--------|-------------|",
    ]
    for organ, data in sorted(vols.items()):
        mean_l = data["mean"]
        if data["dimension"] and "Volume" in str(data["dimension"]):
            mean_l = data["mean"]
        dist_type = "Normal" if data["distribution"] == 1 else "LogNormal" if data["distribution"] == 2 else str(data["distribution"])
        lines.append(f"| {organ} | {mean_l:.4f} | {data['deviation']:.4f} | {dist_type} |")
    return "\n".join(lines)
