"""
Discriminated union for hepatic clearance source.

The previous run_pbpk_simulation accepted 5+ parallel parameters
(CL_int, CLint_vitro_hlm, CLint_vitro_hep, CLint_per_cyp, fm_per_cyp)
where any combination would silently dispatch through clearance_source.
A user passing the wrong field for the chosen source got a no-elimination
simulation.

This module replaces that with a Pydantic discriminated union: choose
ONE clearance variant; required fields per variant are enforced at
schema level.
"""

from __future__ import annotations
from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


class DirectClearance(BaseModel):
    """User supplies in vivo CL_int directly (e.g. from clinical CL fit)."""
    source: Literal["direct"] = "direct"
    CL_int_L_per_h: float = Field(..., gt=0, le=1.0e6,
        description="Hepatic intrinsic clearance referenced to plasma (L/h).")


class HLMClearance(BaseModel):
    """In vitro human liver microsomes — HLM substrate depletion or metabolite formation."""
    source: Literal["hlm"] = "hlm"
    CLint_vitro_uL_min_mg: float = Field(..., gt=0, le=10000,
        description="HLM intrinsic clearance (µL/min/mg microsomal protein).")
    protein_conc_mg_mL: float = Field(1.0, gt=0, le=20,
        description="Microsomal protein concentration in the incubation (mg/mL).")
    fu_inc_measured: Optional[float] = Field(None, gt=0, le=1.0,
        description="Measured unbound fraction in HLM incubation. If None, "
                    "predicted from logP (Austin 2002). Measured value strongly preferred.")


class HepatocyteClearance(BaseModel):
    """In vitro suspended/plated human hepatocytes — preserves CYP + UGT + transporters."""
    source: Literal["hepatocyte"] = "hepatocyte"
    CLint_vitro_uL_min_1e6cells: float = Field(..., gt=0, le=1000,
        description="Hepatocyte intrinsic clearance (µL/min/10⁶ cells).")
    fu_hep_measured: Optional[float] = Field(None, gt=0, le=1.0,
        description="Measured unbound fraction in hepatocyte incubation. "
                    "If None, predicted from logP (Austin 2002). Measured value strongly preferred.")


class RecombinantCYPClearance(BaseModel):
    """Per-CYP recombinant enzyme CLint (rCYP), scaled by ISEF + CYP abundance."""
    source: Literal["rcyp"] = "rcyp"
    CLint_per_cyp: dict[str, float] = Field(...,
        description="Per-CYP intrinsic clearance, e.g. {'CYP3A4': 0.5, 'CYP2C9': 0.1} "
                    "in µL/min/pmol-rCYP. At least one CYP required.")

    @model_validator(mode="after")
    def at_least_one_cyp(self):
        if not self.CLint_per_cyp:
            raise ValueError("CLint_per_cyp must contain at least one CYP entry")
        for cyp, val in self.CLint_per_cyp.items():
            if val <= 0 or val > 1000:
                raise ValueError(
                    f"CLint for {cyp}={val} out of range (0, 1000] µL/min/pmol-rCYP"
                )
        return self


# Discriminated union — the `source` field chooses the variant
ClearanceSpec = Annotated[
    Union[DirectClearance, HLMClearance, HepatocyteClearance, RecombinantCYPClearance],
    Field(discriminator="source"),
]


def parse_clearance_from_legacy_args(
    *,
    clearance_source: str,
    CL_int: float,
    CLint_vitro_hlm: Optional[float],
    CLint_vitro_hep: Optional[float],
    CLint_per_cyp: Optional[str],
    protein_conc: float,
) -> Union[DirectClearance, HLMClearance, HepatocyteClearance, RecombinantCYPClearance, None]:
    """
    Convert the flat tool-call kwargs into a discriminated ClearanceSpec.
    Returns None if no clearance is specified (caller will check CL_renal
    or library compound CL_int).

    Raises ValidationError on mismatched / out-of-range inputs — replaces
    the ad-hoc validate_clearance_source_mismatch from validation.py.
    """
    if clearance_source == "direct":
        if CL_int and CL_int > 0:
            return DirectClearance(CL_int_L_per_h=CL_int)
        return None
    if clearance_source == "hlm":
        if CLint_vitro_hlm is None:
            raise ValueError(
                "clearance_source='hlm' requires CLint_vitro_hlm. "
                "Other IVIVE inputs you provided will not be used in this mode."
            )
        return HLMClearance(
            CLint_vitro_uL_min_mg=CLint_vitro_hlm,
            protein_conc_mg_mL=protein_conc,
        )
    if clearance_source == "hepatocyte":
        if CLint_vitro_hep is None:
            raise ValueError(
                "clearance_source='hepatocyte' requires CLint_vitro_hep. "
                "Other IVIVE inputs you provided will not be used in this mode."
            )
        return HepatocyteClearance(CLint_vitro_uL_min_1e6cells=CLint_vitro_hep)
    if clearance_source == "rcyp":
        if not CLint_per_cyp:
            raise ValueError(
                "clearance_source='rcyp' requires CLint_per_cyp "
                "(format: 'CYP3A4:0.5,CYP2C9:0.1')."
            )
        cyp_dict = {}
        for pair in CLint_per_cyp.split(","):
            parts = pair.strip().split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed CLint_per_cyp entry '{pair}'. "
                    f"Expected 'CYP_NAME:VALUE,CYP_NAME:VALUE,...'"
                )
            cyp_dict[parts[0].strip()] = float(parts[1].strip())
        return RecombinantCYPClearance(CLint_per_cyp=cyp_dict)
    raise ValueError(
        f"Unknown clearance_source='{clearance_source}'. "
        f"Valid: direct, hlm, hepatocyte, rcyp."
    )
