"""
Unit-aware parameter handling using pint.

Most PBPK parameter mistakes that survive schema validation come from
unit confusion: HLM CLint in µL/min/mg (in vitro) versus L/h (in vivo),
Km in µM versus mg/L, fu_p as a fraction versus a percent. This module
provides:

  - A canonical unit table for every PBPK parameter
  - parse_quantity() — accept floats (assumed canonical unit) OR
    strings with explicit units ('70 uL/min/mg'), convert to canonical
  - require_unit() — raise if value is given in an incompatible unit

Convention: every numeric parameter the tool stores is in its CANONICAL
unit (defined in CANONICAL_UNITS below). User-facing parsing converts
on entry, never inside the ODE. This keeps the integration math
unitless-numeric (fast) while making boundary errors loud.
"""

from __future__ import annotations
from typing import Optional, Union
import pint

# Single application-wide registry — sharing prevents "unit from
# different registry" errors when objects cross module boundaries.
ureg = pint.UnitRegistry()
Q = ureg.Quantity


# ---------------------------------------------------------------------
# Canonical units (the unit each parameter is STORED as internally)
# ---------------------------------------------------------------------

CANONICAL_UNITS: dict[str, str] = {
    # Physicochemical
    "mw":               "g/mol",
    "logP":             "dimensionless",
    "pKa":              "dimensionless",
    "fu_p":             "dimensionless",        # fraction 0-1
    "R_bp":             "dimensionless",        # ratio
    "Peff":             "1e-4 cm/s",            # human jejunal Peff convention
    # Absorption / disposition
    "ka":               "1/h",
    "Fa":               "dimensionless",
    "Fg":               "dimensionless",
    # Clearance — in vivo (canonical)
    "CL_int":           "L/h",
    "CL_renal":         "L/h",
    "Vmax":             "mg/h",
    "Km":               "mg/L",
    # Clearance — in vitro (per-source canonical)
    "CLint_vitro_hlm":  "uL/min/mg",
    "CLint_vitro_hep":  "uL/min/(1e6 cells)",
    # DDI
    "Ki":               "uM",
    "KI":               "uM",
    "kinact":           "1/h",
    "EC50":             "uM",
    # Subject / dose
    "dose_mg":          "mg",
    "duration_h":       "h",
    "interval_h":       "h",
    "infusion_duration_h": "h",
    "body_weight":      "kg",
    "age":              "year",
}


# Compatibility groups — units in the same group are inter-convertible.
# A parameter in canonical "L/h" rejects any input in "mg" but accepts
# any volume/time pair (mL/min, L/h, ...).
_COMPAT_GROUPS = {
    "volume_per_time":   {"L/h", "mL/min", "mL/h", "L/min", "L/s", "uL/min"},
    "mass_per_time":     {"mg/h", "g/h", "ug/h", "mg/min", "ug/min"},
    "mass_per_volume":   {"mg/L", "ug/mL", "ng/mL", "g/L", "kg/m^3"},
    "concentration_uM":  {"uM", "umol/L", "nM", "nmol/L", "mM", "mmol/L"},
    "vitro_clint_hlm":   {"uL/min/mg", "mL/min/mg", "L/h/mg"},
    "vitro_clint_hep":   {"uL/min/(1e6 cells)", "mL/min/(1e6 cells)"},
    "permeability":      {"1e-4 cm/s", "cm/s", "1e-6 cm/s"},
}


def parse_quantity(
    value: Union[float, int, str, None],
    parameter: str,
) -> Optional[float]:
    """
    Convert a user-supplied value into the canonical unit for `parameter`.

    Accepts:
      - None → returns None
      - float / int → assumed already in canonical unit, returned as-is
      - str like "70 uL/min/mg" → parsed, converted, magnitude returned

    Raises ValueError on incompatible units (e.g. CLint given in mg).
    """
    if value is None:
        return None
    if parameter not in CANONICAL_UNITS:
        raise ValueError(
            f"Unknown parameter '{parameter}' — add it to CANONICAL_UNITS "
            f"in core/units.py before unit-checking."
        )
    canonical = CANONICAL_UNITS[parameter]

    # Plain number → assume canonical
    if isinstance(value, (int, float)):
        return float(value)

    # String must include a unit
    if isinstance(value, str):
        try:
            q = Q(value)
        except (pint.UndefinedUnitError, pint.errors.PintError, ValueError) as e:
            raise ValueError(
                f"Could not parse '{value}' for parameter '{parameter}' "
                f"(canonical unit '{canonical}'): {e}"
            )
        # Convert to canonical
        try:
            converted = q.to(canonical)
        except pint.DimensionalityError as e:
            raise ValueError(
                f"Unit mismatch for '{parameter}': got '{q.units}', "
                f"expected something convertible to '{canonical}'. {e}"
            )
        return float(converted.magnitude)

    raise TypeError(
        f"Unsupported type {type(value).__name__} for parameter '{parameter}'"
    )


def require_unit_compatible(value_with_unit: str, canonical_unit: str) -> None:
    """
    Sanity-check that a unit string is dimensionally compatible with the
    canonical unit, without performing the conversion. Useful for early
    rejection in schema construction.
    """
    try:
        q1 = Q(f"1 {value_with_unit}")
        q2 = Q(f"1 {canonical_unit}")
    except pint.PintError as e:
        raise ValueError(f"Cannot parse units: {e}")
    if q1.dimensionality != q2.dimensionality:
        raise ValueError(
            f"Dimensionality mismatch: '{value_with_unit}' vs canonical "
            f"'{canonical_unit}' ({q1.dimensionality} vs {q2.dimensionality})"
        )


def format_quantity(magnitude: float, parameter: str, *, precision: int = 4) -> str:
    """Format a canonical-unit value with its unit, for display."""
    if parameter not in CANONICAL_UNITS:
        return f"{magnitude:.{precision}g}"
    unit = CANONICAL_UNITS[parameter]
    if unit == "dimensionless":
        return f"{magnitude:.{precision}g}"
    return f"{magnitude:.{precision}g} {unit}"
