# PBPK MCP Server

Whole-body physiologically-based pharmacokinetic (PBPK) modeling exposed as an
MCP (Model Context Protocol) server. Designed for use with Claude Code, Claude
Desktop, or any MCP-capable client.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v1.6-blue.svg)](CHANGELOG.md)
[![MCP](https://img.shields.io/badge/MCP-stdio-green.svg)](https://modelcontextprotocol.io)

## Highlights

- **30 MCP tools** covering full PBPK workflow
- **7 partition coefficient methods**: Rodgers-Rowland, Lukacova, Schmitt,
  Poulin-Theil, Berezhkovskiy, PK-Sim, Kp_membrane
- **ACAT 9-segment** absorption with dissolution, pH-dependent solubility,
  paracellular permeability, bile-salt micellar solubilization
- **IVIVE pipeline**: HLM → hepatocyte → recombinant CYP scaling
- **Dynamic DDI** with segmented liver dispersion model (1–N CSTRs in series)
  and inlet-driven enzyme inactivation/induction
- **Population PBPK** with Monte Carlo variability (BW, CL, fu_p, ka, GFR)
- **PKSimDB integration**: 38,326 ICRP/Tanaka population distributions,
  294 ontogeny points, 38 transporters
- **Special populations**: 5 species (human/rat/mouse/dog/monkey), 5 CKD
  stages, 4 Child-Pugh stages, pregnancy (GA 0–40 weeks)
- **Validated** against 6 reference compounds (Midazolam, Diazepam, Warfarin,
  Theophylline, Caffeine, Metformin) and DDI cases (Keto+Midaz 13.9x,
  Rifampin+Midaz 0.034x, both within literature ranges).

## Quick start

### Install
```bash
git clone https://github.com/sueinchoi/pbpk-mcp-server.git
cd pbpk-mcp-server
python3 -m venv .venv && source .venv/bin/activate
pip install mcp numpy scipy matplotlib
```

### Register with Claude Code
Add to `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "pbpk": {
      "command": "/path/to/pbpk-mcp-server/.venv/bin/python",
      "args": ["/path/to/pbpk-mcp-server/server.py"],
      "env": {}
    }
  }
}
```
Restart Claude Code; the server's 30 tools become available.

### First call (example)
```
> run_pbpk_simulation(name="midazolam", dose_mg=7.5, route="oral",
                      kp_method="poulin_theil", duration_h=24)
```
Returns a markdown report with NCA parameters (Cmax, AUC, t½, Vss, CL/F),
tissue concentrations at Tmax, and the partition coefficients used. A
concentration-time plot is auto-saved.

## Documentation

| File | Purpose |
|---|---|
| [TUTORIAL.md](TUTORIAL.md) | 5 worked scenarios end-to-end |
| [MODEL_DESCRIPTION.md](MODEL_DESCRIPTION.md) | Mathematical model + 32 references |
| [CHANGELOG.md](CHANGELOG.md) | Version history v1.0–v1.6 |
| [NOTICE.md](NOTICE.md) | Third-party attributions |

The `pbpk_help` MCP tool also returns the parameter input guide and Kp
method selection rules at runtime.

## Tool catalog (30 tools)

| Category | Tools |
|---|---|
| Compound info | `list_compounds`, `drug_properties`, `predict_kp`, `compare_kp_methods`, `predict_tissue_binding`, `predict_blood_plasma_ratio`, `predict_binding_profile` |
| Physiology | `list_physiology`, `pregnancy_physiology`, `species_comparison`, `disease_state` |
| Clearance | `predict_hepatic_clearance`, `compare_hepatic_clearance`, `ivive_microsomal`, `predict_fg`, `transporter_clearance` |
| Absorption | `simulate_acat`, `predict_lymphatic` |
| Simulation | `run_pbpk_simulation`, `plot_concentration`, `run_population_pbpk` |
| DDI | `predict_ddi`, `run_dynamic_ddi` |
| Allometric | `allometric_scaling` |
| PKSimDB | `pksim_ontogeny`, `pksim_organ_volumes`, `pksim_transporters` |
| Data I/O | `fit_to_observed`, `import_pksim_model` |
| Help | `pbpk_help` |

## Validation snapshot (v1.6)

| Compound | Method | Vss (L/kg) | CL (L/h) | t½ (h) | Status |
|---|---|---|---|---|---|
| Midazolam | Poulin-Theil | 0.90 | 15.4 | 5.3 | ✓ |
| Diazepam | Poulin-Theil | 1.20 | 0.91 | 67 | ✓ |
| Warfarin | Berezhkovskiy | 0.12 | 0.11 | 52 | ✓ |
| Theophylline | R&R | 0.28 | 1.93 | 7.2 | ✓ |
| Caffeine | R&R | 0.42 | 3.65 | 5.7 | ✓ |
| Metformin | R&R | 1.17 | 27.2 | 2.6* | △ |

*Metformin t½ underpredicted because library uses passive `CL_renal`. To
reproduce biphasic PK, use `distribution_model="permeability_limited"` with
OCT2/MATE1 transporter parameters.

## License

This codebase is released under the [MIT License](LICENSE).
The bundled `data/PKSimDB.sqlite` is GPLv2-licensed third-party data — see
[NOTICE.md](NOTICE.md) for attribution and obligations.

## Citation

If this server contributes to your research, please cite the underlying
methods (Rodgers & Rowland 2006, Poulin & Theil 2002, Yang 2007 Qgut, etc.;
full reference list in `MODEL_DESCRIPTION.md`).
