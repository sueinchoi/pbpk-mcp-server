"""
PK-Sim XML/PKML model file importer.

Reads PK-Sim simulation XML exports to extract:
  - Compound physicochemical properties
  - Individual physiology parameters
  - Dosing protocol
  - Organ volumes and blood flows

PK-Sim export format: .pkml (XML-based)
"""

import xml.etree.ElementTree as ET
import os
from typing import Optional


def parse_pksim_xml(filepath: str) -> dict:
    """
    Parse a PK-Sim simulation XML file.

    Args:
        filepath: Path to .pkml or .xml file.

    Returns:
        Dict with extracted model information.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    tree = ET.parse(filepath)
    root = tree.getroot()

    result = {
        "compound": {},
        "individual": {},
        "dosing": {},
        "organs": {},
        "parameters": {},
    }

    # Walk all elements looking for parameter containers
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        name = elem.get("name", elem.get("Name", ""))
        value = elem.get("value", elem.get("Value", ""))

        # Extract compound properties
        if "Compound" in str(elem.get("containerType", "")) or "COMPOUND" in str(elem.get("containerType", "")):
            if name and value:
                try:
                    result["compound"][name] = float(value)
                except ValueError:
                    result["compound"][name] = value

        # Extract parameter values with paths
        if tag in ("Parameter", "parameter") and name and value:
            path = _build_path(elem, root)
            try:
                result["parameters"][path + "/" + name] = float(value)
            except ValueError:
                result["parameters"][path + "/" + name] = value

    # Try to extract specific known compound parameters
    result["compound_summary"] = _extract_compound_summary(result["parameters"])
    result["physiology_summary"] = _extract_physiology_summary(result["parameters"])

    return result


def _build_path(elem, root) -> str:
    """Build parameter path from XML element."""
    # Simplified path building
    parent = elem
    parts = []
    for _ in range(10):  # max depth
        parent_name = parent.get("name", parent.get("Name", ""))
        if parent_name:
            parts.insert(0, parent_name)
        parent = parent.find("..")
        if parent is None or parent == root:
            break
    return "/".join(parts[-3:])  # last 3 levels


def _extract_compound_summary(params: dict) -> dict:
    """Extract compound properties from parameter paths."""
    summary = {}
    key_mappings = {
        "Molecular weight": "MW",
        "Lipophilicity": "logP",
        "Fraction unbound": "fu_p",
        "Is small molecule": "is_small_molecule",
        "Plasma protein binding partner": "binding_partner",
    }
    for path, value in params.items():
        for key, label in key_mappings.items():
            if key.lower() in path.lower():
                summary[label] = value
                break
    return summary


def _extract_physiology_summary(params: dict) -> dict:
    """Extract physiology parameters from parameter paths."""
    summary = {}
    organs = ["Liver", "Kidney", "Brain", "Heart", "Lung", "Muscle",
              "Fat", "Skin", "Bone", "Spleen", "Pancreas",
              "SmallIntestine", "LargeIntestine", "Stomach"]
    for path, value in params.items():
        for organ in organs:
            if organ.lower() in path.lower() and "volume" in path.lower():
                if organ not in summary:
                    summary[organ] = {}
                summary[organ]["Volume"] = value
    return summary


def format_pksim_import(result: dict) -> str:
    """Format imported PK-Sim data as markdown."""
    lines = ["## PK-Sim Model Import\n"]

    if result["compound_summary"]:
        lines.append("### Compound Properties\n")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        for k, v in result["compound_summary"].items():
            lines.append(f"| {k} | {v} |")

    if result["physiology_summary"]:
        lines.append("\n### Organ Volumes\n")
        lines.append("| Organ | Volume |")
        lines.append("|-------|--------|")
        for organ, data in sorted(result["physiology_summary"].items()):
            vol = data.get("Volume", "N/A")
            lines.append(f"| {organ} | {vol} |")

    n_params = len(result["parameters"])
    lines.append(f"\n*Total parameters extracted: {n_params}*")

    return "\n".join(lines)
