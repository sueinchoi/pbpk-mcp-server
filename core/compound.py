"""
Drug compound specification for PBPK modeling.

Defines physicochemical properties required for partition coefficient
prediction and PBPK simulation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CompoundType(str, Enum):
    """Classification per Rodgers & Rowland (2005, 2006)."""
    STRONG_BASE = "strong_base"      # pKa >= 7, Type 1
    MODERATE_BASE = "moderate_base"  # pKa 4-7, Type 2
    WEAK_BASE = "weak_base"          # pKa < 4, Type 2
    ACID = "acid"                    # Type 2
    NEUTRAL = "neutral"              # Type 2
    ZWITTERION = "zwitterion"        # Type 2


class AbsorptionModel(str, Enum):
    FIRST_ORDER = "first_order"
    CAT = "cat"          # Compartmental Absorption Transit (7-segment)
    ACAT = "acat"        # Advanced CAT (9-segment, dissolution, precipitation)


class MetabolismModel(str, Enum):
    FIRST_ORDER = "first_order"       # CL_int * fu * C
    MICHAELIS_MENTEN = "michaelis_menten"  # Vmax * C_u / (Km + C_u)


@dataclass
class CompoundSpec:
    """Drug compound physicochemical and PK properties."""

    name: str
    mw: float                     # Molecular weight (g/mol)
    logP: float                   # Octanol:water partition coefficient (log, unionized)
    pKa: float                    # Dissociation constant
    fu_p: float                   # Fraction unbound in plasma
    compound_type: CompoundType   # Acid/base/neutral classification
    R_bp: float = 1.0            # Blood:plasma ratio

    # Absorption parameters
    ka: float = 1.0              # Absorption rate constant (1/h)
    Fa: float = 1.0              # Fraction absorbed
    Fg: float = 1.0              # Fraction escaping gut wall metabolism

    # ACAT / dissolution parameters (used when absorption_model = ACAT)
    Peff: Optional[float] = None     # Human jejunal permeability (10^-4 cm/s)
    S0: Optional[float] = None       # Intrinsic solubility (mg/mL)
    particle_radius_um: float = 25.0 # Mean particle radius (um)
    CLint_gut: float = 0.0           # Total gut wall intrinsic clearance (L/h)

    # Hepatic clearance
    CL_int: float = 0.0         # Intrinsic clearance (L/h)
    metabolism_model: MetabolismModel = MetabolismModel.FIRST_ORDER
    Vmax: Optional[float] = None  # Vmax for Michaelis-Menten (mg/h)
    Km: Optional[float] = None    # Km for Michaelis-Menten (mg/L)

    # Renal clearance
    CL_renal: float = 0.0       # Renal clearance (L/h), referenced to plasma

    # Optional: additional pKa for zwitterions
    pKa2: Optional[float] = None

    # Optional: per-organ Kp scaling factor to correct known prediction biases.
    # Rodgers-Rowland and related methods systematically over-predict adipose Kp
    # for lipophilic bases (logP > 3): see Jansson et al. 2008, Graham et al. 2012.
    # Example override to fix Midazolam Vss: {"adipose": 0.15}
    # Keys are organ names (lowercase, matching Organ enum values). Missing keys = 1.0.
    kp_scale: Optional[dict] = None

    # Recommended Kp method for this compound based on literature validation.
    # If None, default (rodgers_rowland) is used. When the user picks a
    # different method, tools may surface this recommendation as a hint.
    recommended_kp_method: Optional[str] = None

    # Per-parameter citation map. Keys are parameter names ("logP", "fu_p",
    # "CL_int", etc.) or "general"; values are PMID, DOI, or descriptive
    # source strings. Library curators populate this; user-built compounds
    # can leave it empty (the audit will warn).
    citations: Optional[dict] = None

    def __post_init__(self):
        if isinstance(self.compound_type, str):
            self.compound_type = CompoundType(self.compound_type)
        if isinstance(self.metabolism_model, str):
            self.metabolism_model = MetabolismModel(self.metabolism_model)
        self._validate()

    def _validate(self):
        if not 0 < self.fu_p <= 1.0:
            raise ValueError(f"fu_p must be in (0, 1.0], got {self.fu_p}")
        if not 0 < self.R_bp <= 5.0:
            raise ValueError(f"R_bp must be in (0, 5.0], got {self.R_bp}")
        if self.mw <= 0:
            raise ValueError(f"MW must be positive, got {self.mw}")
        if self.ka < 0:
            raise ValueError(f"ka must be non-negative, got {self.ka}")
        if not 0 <= self.Fa <= 1.0:
            raise ValueError(f"Fa must be in [0, 1.0], got {self.Fa}")
        if not 0 <= self.Fg <= 1.0:
            raise ValueError(f"Fg must be in [0, 1.0], got {self.Fg}")
        if self.CL_int < 0:
            raise ValueError(f"CL_int must be non-negative, got {self.CL_int}")
        if self.CL_renal < 0:
            raise ValueError(f"CL_renal must be non-negative, got {self.CL_renal}")
        if self.metabolism_model == MetabolismModel.MICHAELIS_MENTEN:
            if self.Vmax is None or self.Km is None:
                raise ValueError("Vmax and Km required for Michaelis-Menten metabolism")

    @property
    def has_elimination(self) -> bool:
        """Check if any elimination pathway is defined."""
        return self.CL_int > 0 or self.CL_renal > 0 or self.CLint_gut > 0 or (
            self.metabolism_model == MetabolismModel.MICHAELIS_MENTEN
            and self.Vmax is not None and self.Vmax > 0
        )

    @property
    def Kpb_factor(self) -> float:
        """Kp:blood = Kp:plasma / R_bp. This factor converts Kp to Kpb."""
        return 1.0 / self.R_bp

    @classmethod
    def classify_compound(cls, pKa: float, acid_or_base: str) -> CompoundType:
        """Auto-classify compound type from pKa and acid/base designation."""
        if acid_or_base.lower() == "acid":
            return CompoundType.ACID
        elif acid_or_base.lower() == "neutral":
            return CompoundType.NEUTRAL
        elif acid_or_base.lower() == "zwitterion":
            return CompoundType.ZWITTERION
        elif acid_or_base.lower() == "base":
            if pKa >= 7.0:
                return CompoundType.STRONG_BASE
            elif pKa >= 4.0:
                return CompoundType.MODERATE_BASE
            else:
                return CompoundType.WEAK_BASE
        raise ValueError(f"Unknown acid_or_base: {acid_or_base}")


# --- Well-known compound library (reference values) ---

# Reference compound library — verified against DrugBank, PK-Sim, clinical PK reviews
# CL_int values are calibrated so well-stirred CL_h matches published clinical CL
COMPOUND_LIBRARY = {
    "midazolam": CompoundSpec(
        name="Midazolam", mw=325.8, logP=3.89, pKa=6.2,
        fu_p=0.032, compound_type=CompoundType.MODERATE_BASE,
        R_bp=0.66, ka=4.16, Fa=0.88, Fg=0.57,
        CL_int=700.0,  # gives CL_h ~27 L/h (lit: 18-30 L/h, Thummel 1996)
        citations={
            # Verified database identifiers (preferred when PMIDs are uncertain).
            "logP":  "ChEMBL CHEMBL601",
            "pKa":   "DrugBank DB00683",
            # Free-text references — author+year+journal only. The
            # provenance audit will flag these as UNSOURCED until a
            # verified PMID is added (use verify_citation() before
            # inserting any PMID here).
            "fu_p":  "Thummel et al. 1996, J Pharmacol Exp Ther",
            "CL_int": "calibrated to Thummel et al. 1996 CL 18-30 L/h",
            "Fa":    "Greenblatt et al. 1984 midazolam PK review",
            "Fg":    "Thummel et al. 1996 first-pass extraction",
            "kp_scale": "Bjorkman 2001 rat tissue distribution",
            "recommended_kp_method": "Jansson 2008 / Graham 2012 — R&R adipose over-prediction",
        },
        # Empirical Kp corrections from Björkman 2001 in vivo rat data
        # (tissue:plasma total concentration), scaled to normalize R&R prediction
        # to clinical Vss 0.6-1.5 L/kg (Greenblatt 1984).
        kp_scale={
            "adipose": 0.17,
            "muscle": 0.42,
            "brain": 0.16,
            "liver": 0.38,
            "lung": 0.49,
            "kidney": 0.65,
            "skin": 0.25,
            "bone": 0.30,
            "rest": 0.30,
        },
        recommended_kp_method="poulin_theil",
    ),
    "metformin": CompoundSpec(
        name="Metformin", mw=129.16, logP=-1.43, pKa=12.4,
        fu_p=1.0, compound_type=CompoundType.STRONG_BASE,
        R_bp=1.0, ka=1.26, Fa=0.55,
        # CL_renal is INTRINSIC renal clearance (extraction-limited by Q_kidney).
        # Observed CL = Q_kid * CL_renal / (Q_kid + CL_renal).
        # 50 L/h intrinsic → 28.2 L/h observed (matches Pentikäinen 1979 ~30 L/h).
        CL_renal=50.0,
        citations={
            "logP":  "DrugBank DB00331",
            "pKa":   "DrugBank DB00331 (biguanide)",
            "fu_p":  "Graham et al. 2011 metformin clinical PK review — negligible plasma binding",
            "Fa":    "Pentikainen et al. 1979, Eur J Clin Pharmacol — ~50-60% absorbed",
            "CL_renal": "calibrated to Pentikainen et al. 1979 CL ~30 L/h",
        },
    ),
    "theophylline": CompoundSpec(
        name="Theophylline", mw=180.16, logP=-0.02, pKa=8.6,
        fu_p=0.40, compound_type=CompoundType.NEUTRAL,  # was ACID — corrected
        R_bp=0.82, ka=1.5, Fa=1.0,
        CL_int=5.0,  # gives CL_h ~2.3 L/h (lit: 2-4 L/h, Ogilvie 1978)
        citations={
            "logP": "DrugBank DB00277",
            "pKa": "DrugBank DB00277",
            "fu_p": "Ogilvie 1978, Clin Pharmacokinet theophylline review",
            "CL_int": "calibrated to Ogilvie 1978 CL 2-4 L/h",
        },
    ),
    "diazepam": CompoundSpec(
        name="Diazepam", mw=284.7, logP=2.82, pKa=3.4,
        fu_p=0.021, compound_type=CompoundType.WEAK_BASE,  # fu_p corrected
        R_bp=0.58, ka=1.68, Fa=1.0,
        CL_int=45.0,  # gives CL_h ~1.5 L/h (lit: 1.3-2.2 L/h, Greenblatt 1980)
        recommended_kp_method="poulin_theil",
        citations={
            "logP": "DrugBank DB00829",
            "pKa": "DrugBank DB00829",
            "fu_p": "Greenblatt et al. 1980 diazepam PK review",
            "CL_int": "calibrated to Greenblatt et al. 1980 CL 1.3-2.2 L/h",
            "recommended_kp_method": "Jansson 2008 — R&R lipophilic-base adipose over-prediction",
        },
    ),
    "warfarin": CompoundSpec(
        name="Warfarin", mw=308.3, logP=2.6, pKa=5.0,
        fu_p=0.005, compound_type=CompoundType.ACID,
        R_bp=0.55, ka=1.0, Fa=0.93,
        CL_int=22.0,  # gives CL_h ~0.18 L/h (lit: 0.1-0.3 L/h, Holford 1986)
        # RECOMMENDED Kp method: "berezhkovskiy" or "pksim_standard".
        # Rodgers-Rowland systematically under-predicts Vss for highly-bound
        # acids (fu_p<0.01); R&R 2006 itself notes this limitation.
        # Berezhkovskiy correction for albumin handling gives Vss 0.118 L/kg
        # (clinical target 0.09-0.18, Holford 1986) vs R&R 0.055 L/kg.
        recommended_kp_method="berezhkovskiy",
        citations={
            "logP": "DrugBank DB00682",
            "pKa": "DrugBank DB00682",
            "fu_p": "Holford 1986, Clin Pharmacokinet warfarin review",
            "CL_int": "calibrated to Holford 1986 CL 0.1-0.3 L/h",
            "recommended_kp_method": "Berezhkovskiy 2004, J Pharm Sci — albumin-corrected PT",
        },
    ),
    "caffeine": CompoundSpec(
        name="Caffeine", mw=194.2, logP=-0.07, pKa=10.4,
        fu_p=0.64, compound_type=CompoundType.NEUTRAL,
        R_bp=1.0, ka=2.3, Fa=1.0, CL_int=6.0,
        citations={
            "logP": "DrugBank DB00201",
            "pKa": "DrugBank DB00201",
            "fu_p": "Bonati et al. 1985 caffeine PK review",
            "CL_int": "calibrated to Bonati et al. 1985 PK data",
        },
    ),
}
