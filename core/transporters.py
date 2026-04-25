"""
Transporter-mediated active transport for PBPK model.

Implements Michaelis-Menten active transport at organ-specific membranes:
  - Hepatic uptake: OATP1B1/1B3 (sinusoidal influx)
  - Hepatic efflux: MRP2/BCRP/P-gp (canalicular), MRP3/4 (sinusoidal)
  - Renal secretion: OAT1/3, OCT2 (basolateral), MATE1/2 (apical)
  - Intestinal efflux: P-gp, BCRP (apical, back to lumen)
  - BBB efflux: P-gp (brain)

Each transporter has: Km (uM), Vmax (pmol/min/pmol transporter or scaled),
organ location, membrane direction.

Integration with ODE: adds active transport flux terms to the
permeability-limited or extended perfusion-limited model.

References:
  - Shitara Y et al. Pharmacol Ther 2006;112:57-76
  - Giacomini KM et al. Nat Rev Drug Discov 2010;9:215-236
  - ITC whitepaper: Giacomini et al. Clin Pharmacol Ther 2010;88:e-pub
  - PK-Sim PKSimDB transporter database
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TransportDirection(str, Enum):
    """Membrane transport direction (PK-Sim convention)."""
    INFLUX_PLASMA_TO_INTERSTITIAL = "influx_pls_int"    # sinusoidal uptake
    EFFLUX_INTERSTITIAL_TO_PLASMA = "efflux_int_pls"    # sinusoidal efflux
    INFLUX_INT_TO_CELL = "influx_int_cell"              # cell membrane uptake
    EFFLUX_CELL_TO_INT = "efflux_cell_int"              # cell membrane efflux
    EFFLUX_CELL_TO_LUMEN = "efflux_cell_lumen"          # apical efflux (gut, kidney)
    INFLUX_LUMEN_TO_CELL = "influx_lumen_cell"          # apical uptake
    EXCRETION_BILE = "excretion_bile"                    # canalicular (liver)
    EXCRETION_KIDNEY = "excretion_kidney"                # tubular (kidney)


@dataclass
class TransporterSpec:
    """Single transporter specification for a compound."""
    gene: str                    # e.g., "OATP1B1", "P-gp", "OCT2"
    organ: str                   # e.g., "liver", "kidney", "gut", "brain"
    direction: TransportDirection
    Km: float                    # Michaelis constant (uM)
    Vmax: float                  # Maximum rate (pmol/min/mg protein or scaled to L/h)
    is_scaled: bool = False      # True if Vmax is already in L/h (whole organ)

    @property
    def CLint_transport(self) -> float:
        """Intrinsic transport clearance at low concentration: Vmax/Km."""
        return self.Vmax / self.Km if self.Km > 0 else 0.0


@dataclass
class OrganTransporters:
    """Collection of transporters for an organ."""
    organ: str
    influx: list[TransporterSpec] = field(default_factory=list)   # uptake into cell
    efflux: list[TransporterSpec] = field(default_factory=list)   # efflux from cell

    @property
    def PS_influx_total(self) -> float:
        """Total influx PS product (sum of Vmax/Km for all uptake transporters)."""
        return sum(t.CLint_transport for t in self.influx)

    @property
    def PS_efflux_total(self) -> float:
        """Total efflux PS product."""
        return sum(t.CLint_transport for t in self.efflux)


def compute_transport_rate(
    transporter: TransporterSpec,
    C_unbound: float,
) -> float:
    """
    Michaelis-Menten transport rate (amount/time).

    Rate = Vmax * C_u / (Km + C_u)

    Args:
        transporter: Transporter specification.
        C_unbound: Unbound drug concentration at the transporter site (uM or mg/L).

    Returns:
        Transport rate (same units as Vmax * C_u / Km).
    """
    if C_unbound <= 0 or transporter.Km <= 0:
        return 0.0
    return transporter.Vmax * C_unbound / (transporter.Km + C_unbound)


def compute_organ_transport_fluxes(
    organ_transporters: OrganTransporters,
    C_u_blood: float,
    C_u_cell: float,
) -> dict:
    """
    Compute net transport fluxes for an organ.

    Returns dict with:
      influx_rate: total uptake rate (blood/interstitial → cell)
      efflux_rate: total efflux rate (cell → blood/interstitial or bile/urine)
      net_rate: influx - efflux (positive = net uptake)
    """
    influx = sum(compute_transport_rate(t, C_u_blood) for t in organ_transporters.influx)
    efflux = sum(compute_transport_rate(t, C_u_cell) for t in organ_transporters.efflux)

    return {
        "influx_rate": influx,
        "efflux_rate": efflux,
        "net_rate": influx - efflux,
    }


# ===================================================================
# Common transporter profiles for known substrates
# ===================================================================

def statins_liver_transporters(
    Km_oatp1b1: float = 5.0,
    Vmax_oatp1b1: float = 100.0,
    Km_oatp1b3: float = 10.0,
    Vmax_oatp1b3: float = 50.0,
    Km_mrp2: float = 50.0,
    Vmax_mrp2: float = 30.0,
    Km_bcrp: float = 20.0,
    Vmax_bcrp: float = 20.0,
) -> OrganTransporters:
    """
    Typical liver transporter profile for statins.
    OATP1B1/1B3 uptake + MRP2/BCRP biliary efflux.
    """
    return OrganTransporters(
        organ="liver",
        influx=[
            TransporterSpec("OATP1B1", "liver", TransportDirection.INFLUX_PLASMA_TO_INTERSTITIAL,
                          Km_oatp1b1, Vmax_oatp1b1),
            TransporterSpec("OATP1B3", "liver", TransportDirection.INFLUX_PLASMA_TO_INTERSTITIAL,
                          Km_oatp1b3, Vmax_oatp1b3),
        ],
        efflux=[
            TransporterSpec("MRP2", "liver", TransportDirection.EXCRETION_BILE,
                          Km_mrp2, Vmax_mrp2),
            TransporterSpec("BCRP", "liver", TransportDirection.EXCRETION_BILE,
                          Km_bcrp, Vmax_bcrp),
        ],
    )


def renal_transporters(
    Km_oat1: float = 20.0,
    Vmax_oat1: float = 50.0,
    Km_oct2: float = 100.0,
    Vmax_oct2: float = 80.0,
    Km_mate1: float = 50.0,
    Vmax_mate1: float = 40.0,
) -> OrganTransporters:
    """
    Typical kidney transporter profile.
    OAT1/3 or OCT2 basolateral uptake + MATE1/2 apical efflux.
    """
    return OrganTransporters(
        organ="kidney",
        influx=[
            TransporterSpec("OAT1", "kidney", TransportDirection.INFLUX_INT_TO_CELL,
                          Km_oat1, Vmax_oat1),
            TransporterSpec("OCT2", "kidney", TransportDirection.INFLUX_INT_TO_CELL,
                          Km_oct2, Vmax_oct2),
        ],
        efflux=[
            TransporterSpec("MATE1", "kidney", TransportDirection.EXCRETION_KIDNEY,
                          Km_mate1, Vmax_mate1),
        ],
    )


def gut_efflux_transporters(
    Km_pgp: float = 10.0,
    Vmax_pgp: float = 50.0,
    Km_bcrp: float = 20.0,
    Vmax_bcrp: float = 30.0,
) -> OrganTransporters:
    """
    Intestinal efflux transporter profile.
    P-gp and BCRP on apical membrane (back to lumen).
    """
    return OrganTransporters(
        organ="gut",
        influx=[],
        efflux=[
            TransporterSpec("P-gp", "gut", TransportDirection.EFFLUX_CELL_TO_LUMEN,
                          Km_pgp, Vmax_pgp),
            TransporterSpec("BCRP", "gut", TransportDirection.EFFLUX_CELL_TO_LUMEN,
                          Km_bcrp, Vmax_bcrp),
        ],
    )


def format_transporter_profile(organ_t: OrganTransporters) -> str:
    """Format transporter profile as markdown."""
    lines = [
        f"## Transporter Profile — {organ_t.organ.capitalize()}\n",
        f"PS_influx = {organ_t.PS_influx_total:.1f} (Vmax/Km sum)",
        f"PS_efflux = {organ_t.PS_efflux_total:.1f}\n",
        "| Transporter | Direction | Km (uM) | Vmax | CLint (Vmax/Km) |",
        "|-------------|-----------|---------|------|----------------|",
    ]
    for t in organ_t.influx:
        lines.append(f"| {t.gene} | Uptake | {t.Km:.1f} | {t.Vmax:.1f} | {t.CLint_transport:.2f} |")
    for t in organ_t.efflux:
        lines.append(f"| {t.gene} | Efflux | {t.Km:.1f} | {t.Vmax:.1f} | {t.CLint_transport:.2f} |")

    # Extended clearance concept
    from .hepatic_models import cl_hepatic_extended
    if organ_t.organ == "liver":
        lines.append(f"\n### Extended Clearance")
        lines.append(f"CLint_overall and Kp,uu can be computed with `compare_hepatic_clearance`")
        lines.append(f"using PS_inf={organ_t.PS_influx_total:.1f} and PS_eff={organ_t.PS_efflux_total:.1f}")

    return "\n".join(lines)
