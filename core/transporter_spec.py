"""
Schema-level pairing of transporter Km/Vmax pairs.

The legacy run_pbpk_simulation accepted 10 separate float | None parameters
(liver_oatp_km, liver_oatp_vmax, liver_mrp2_km, ... gut_pgp_km, gut_pgp_vmax).
Providing only one of a pair silently dropped the transporter from the ODE.

This module wraps the raw kwargs into a Pydantic model that requires
both Km and Vmax for any transporter the user activates. parse_transporter_kwargs
returns the dict consumed by PBPKModel(transporters=...) or raises with a
specific message if a pair is incomplete.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, model_validator

from .transporters import (
    OrganTransporters, TransporterSpec, TransportDirection,
)


class TransporterPair(BaseModel):
    """A single transporter (Km, Vmax) pair. Both are required if either is set."""
    Km_uM: float = Field(..., gt=0, le=1.0e5,
        description="Michaelis-Menten constant (µM).")
    Vmax: float = Field(..., gt=0, le=1.0e6,
        description="Max rate (pmol/min/pmol or pre-scaled to L/h).")


class TransporterKwargs(BaseModel):
    """All transporter pairs as a single nested object — schema enforces pairing."""
    liver_oatp:  Optional[TransporterPair] = None
    liver_mrp2:  Optional[TransporterPair] = None
    kidney_oct2: Optional[TransporterPair] = None
    kidney_mate1: Optional[TransporterPair] = None
    gut_pgp:     Optional[TransporterPair] = None

    @classmethod
    def from_legacy_kwargs(
        cls,
        *,
        liver_oatp_km: Optional[float] = None,
        liver_oatp_vmax: Optional[float] = None,
        liver_mrp2_km: Optional[float] = None,
        liver_mrp2_vmax: Optional[float] = None,
        kidney_oct2_km: Optional[float] = None,
        kidney_oct2_vmax: Optional[float] = None,
        kidney_mate1_km: Optional[float] = None,
        kidney_mate1_vmax: Optional[float] = None,
        gut_pgp_km: Optional[float] = None,
        gut_pgp_vmax: Optional[float] = None,
    ) -> "TransporterKwargs":
        """Convert flat kwargs to nested. Raises if a pair is incomplete."""
        pairs = {
            "liver_oatp":  (liver_oatp_km,  liver_oatp_vmax),
            "liver_mrp2":  (liver_mrp2_km,  liver_mrp2_vmax),
            "kidney_oct2": (kidney_oct2_km, kidney_oct2_vmax),
            "kidney_mate1": (kidney_mate1_km, kidney_mate1_vmax),
            "gut_pgp":     (gut_pgp_km,     gut_pgp_vmax),
        }
        out = {}
        for name, (km, vmax) in pairs.items():
            km_set = km is not None
            vmax_set = vmax is not None
            if km_set ^ vmax_set:
                raise ValueError(
                    f"Transporter '{name}' has only "
                    f"{'Km' if km_set else 'Vmax'} set. Both Km and Vmax are required "
                    f"to activate a transporter — providing one is silently ignored "
                    f"in the legacy schema. Provide both or neither."
                )
            if km_set and vmax_set:
                out[name] = TransporterPair(Km_uM=km, Vmax=vmax)
        return cls(**out)

    def has_any(self) -> bool:
        return any(getattr(self, n) is not None for n in
                   ("liver_oatp", "liver_mrp2", "kidney_oct2", "kidney_mate1", "gut_pgp"))

    def to_organ_transporters(self) -> dict[str, OrganTransporters]:
        """Build the dict consumed by PBPKModel(transporters=...)."""
        out: dict[str, OrganTransporters] = {}
        # Liver
        liver_inf = []
        liver_eff = []
        if self.liver_oatp:
            liver_inf.append(TransporterSpec(
                "OATP1B1", "liver",
                TransportDirection.INFLUX_PLASMA_TO_INTERSTITIAL,
                self.liver_oatp.Km_uM, self.liver_oatp.Vmax,
            ))
        if self.liver_mrp2:
            liver_eff.append(TransporterSpec(
                "MRP2", "liver",
                TransportDirection.EXCRETION_BILE,
                self.liver_mrp2.Km_uM, self.liver_mrp2.Vmax,
            ))
        if liver_inf or liver_eff:
            out["liver"] = OrganTransporters("liver", liver_inf, liver_eff)
        # Kidney
        kid_inf = []
        kid_eff = []
        if self.kidney_oct2:
            kid_inf.append(TransporterSpec(
                "OCT2", "kidney",
                TransportDirection.INFLUX_INT_TO_CELL,
                self.kidney_oct2.Km_uM, self.kidney_oct2.Vmax,
            ))
        if self.kidney_mate1:
            kid_eff.append(TransporterSpec(
                "MATE1", "kidney",
                TransportDirection.EXCRETION_KIDNEY,
                self.kidney_mate1.Km_uM, self.kidney_mate1.Vmax,
            ))
        if kid_inf or kid_eff:
            out["kidney"] = OrganTransporters("kidney", kid_inf, kid_eff)
        # Gut
        if self.gut_pgp:
            out["gut"] = OrganTransporters("gut", [], [TransporterSpec(
                "P-gp", "gut",
                TransportDirection.EFFLUX_CELL_TO_LUMEN,
                self.gut_pgp.Km_uM, self.gut_pgp.Vmax,
            )])
        return out
