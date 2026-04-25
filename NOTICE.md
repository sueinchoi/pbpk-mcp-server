# NOTICE — Third-Party Attributions

## PK-Sim Database (`data/PKSimDB.sqlite`)

The file `data/PKSimDB.sqlite` is bundled, unmodified, from the **Open Systems
Pharmacology (OSP) Suite — PK-Sim**:

- **Source**: https://github.com/Open-Systems-Pharmacology/PK-Sim
- **Copyright**: © Open Systems Pharmacology Community
- **License**: GNU General Public License v2.0 (GPLv2)
- **License text**: https://github.com/Open-Systems-Pharmacology/PK-Sim/blob/develop/License.md

The OSP Suite FAQ explicitly states:
> "The OSP Suite is open-source software released under the GPLv2 License.
> It is free for everyone, including commercial use."
> (https://www.open-systems-pharmacology.org/faq/)

### What this database contains
- 38,326 ICRP/Tanaka population parameter distributions
- 294 ontogeny data points (CYP, UGT, ALB, etc.)
- 38 transporter records (gene + direction)

### Modifications
None. The file is redistributed as obtained from PK-Sim source.

### Implications for users of this MCP server
- This server can be used commercially.
- If you redistribute this server (or fork it), you must:
  - Preserve this NOTICE.
  - Either remove `data/PKSimDB.sqlite` and document how to obtain it
    separately, OR comply with GPLv2 obligations for that file.
  - Make any modifications to PKSimDB.sqlite available under GPLv2.

---

## Other dependencies

| Package | License | Use |
|---|---|---|
| numpy | BSD-3-Clause | Numerical arrays |
| scipy | BSD-3-Clause | ODE solver (BDF), optimization |
| matplotlib | PSF/BSD | Plot generation |
| mcp (Anthropic) | MIT | MCP server framework |
| pharmpy-core | MIT | Optional NONMEM utilities |
