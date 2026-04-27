"""
Fail-fast test suite for the PBPK MCP server.

Runs as plain Python (no pytest dependency):
    python -m tests.test_silent_fallback

Each test asserts EITHER that a tool raised on a malformed input,
OR that a tool surfaced a soft warning, OR that a known-good case
runs cleanly. Any silent fallback (success on garbage / no warning
on suspicious defaults) is a failure.
"""

from __future__ import annotations
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcp.server.fastmcp import FastMCP
from tools.pbpk_tools import register_pbpk_tools


def _server():
    m = FastMCP("test")
    register_pbpk_tools(m)
    return m._tool_manager._tools


PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def test(name: str):
    def decorator(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  ✓ {name}")
        except AssertionError as e:
            FAILED.append((name, str(e)))
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            FAILED.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
        return fn
    return decorator


def expect_raises(fn, exc_type=Exception, contains: str = ""):
    """Assert that `fn()` raises `exc_type`. If `contains` given, message must include it."""
    try:
        fn()
    except exc_type as e:
        if contains and contains.lower() not in str(e).lower():
            raise AssertionError(f"raised {exc_type.__name__} but message did not contain '{contains}': {e}")
        return
    raise AssertionError(f"expected {exc_type.__name__}, got no exception")


# ============================================================
# Section 1: Hard errors — invalid enums / mismatched sources
# ============================================================
print("\n## Hard errors — invalid enums + mismatched clearance sources")

@test("invalid kp_method (typo with hyphen) raises with suggestion")
def t():
    tools = _server()
    expect_raises(
        lambda: tools["run_pbpk_simulation"].fn(
            name="midazolam", dose_mg=7.5, kp_method="poulin-theil",
        ),
        ValueError, "poulin_theil",
    )

@test("clearance_source='hlm' with hep input → ValueError")
def t():
    tools = _server()
    expect_raises(
        lambda: tools["run_pbpk_simulation"].fn(
            name="custom", logP=4.0, pKa=4.0, fu_p=0.005, R_bp=0.55,
            compound_type="acid", clearance_source="hlm",
            CLint_vitro_hep=120.0, dose_mg=50.0,
        ),
        ValueError, "CLint_vitro_hlm",
    )

@test("clearance_source='hepatocyte' with no input → ValueError")
def t():
    tools = _server()
    expect_raises(
        lambda: tools["run_pbpk_simulation"].fn(
            name="custom", logP=4.0, pKa=4.0, fu_p=0.005, R_bp=0.55,
            compound_type="acid", clearance_source="hepatocyte",
            dose_mg=50.0,
        ),
        ValueError, "CLint_vitro_hep",
    )

@test("invalid distribution_model raises")
def t():
    tools = _server()
    expect_raises(
        lambda: tools["run_pbpk_simulation"].fn(
            name="midazolam", distribution_model="quantum_limited",
        ),
        ValueError,
    )

# ============================================================
# Section 2: Range invariants — physiological plausibility
# ============================================================
print("\n## Range invariants — physiological plausibility")

@test("fu_p > 1.0 rejected")
def t():
    tools = _server()
    expect_raises(
        lambda: tools["run_pbpk_simulation"].fn(
            name="custom", logP=2.0, pKa=7.0, fu_p=1.5,
            compound_type="neutral", R_bp=1.0, CL_int=10.0, dose_mg=50.0,
        ),
        ValueError, "fu_p",
    )

@test("fu_p < 0 rejected")
def t():
    tools = _server()
    expect_raises(
        lambda: tools["run_pbpk_simulation"].fn(
            name="custom", logP=2.0, pKa=7.0, fu_p=-0.1,
            compound_type="neutral", R_bp=1.0, CL_int=10.0, dose_mg=50.0,
        ),
        ValueError,
    )

@test("logP=20 (impossible) rejected")
def t():
    tools = _server()
    expect_raises(
        lambda: tools["run_pbpk_simulation"].fn(
            name="custom", logP=20.0, pKa=7.0, fu_p=0.5,
            compound_type="neutral", R_bp=1.0, CL_int=10.0, dose_mg=50.0,
        ),
        ValueError, "logP",
    )

@test("MW=10 (too small) rejected")
def t():
    tools = _server()
    expect_raises(
        lambda: tools["run_pbpk_simulation"].fn(
            name="custom", logP=2.0, pKa=7.0, fu_p=0.5, mw=10.0,
            compound_type="neutral", R_bp=1.0, CL_int=10.0, dose_mg=50.0,
        ),
        ValueError, "mw",
    )

@test("dose_mg=0 rejected (definitionally)")
def t():
    tools = _server()
    expect_raises(
        lambda: tools["run_pbpk_simulation"].fn(
            name="midazolam", dose_mg=0.0,
        ),
        ValueError, "dose_mg",
    )

@test("body_weight=500 kg rejected")
def t():
    tools = _server()
    expect_raises(
        lambda: tools["run_pbpk_simulation"].fn(
            name="midazolam", dose_mg=7.5, body_weight=500.0,
        ),
        ValueError, "body_weight",
    )

@test("multi-dose interval beyond duration rejected")
def t():
    tools = _server()
    expect_raises(
        lambda: tools["run_pbpk_simulation"].fn(
            name="midazolam", dose_mg=7.5, n_doses=5, interval_h=24.0,
            duration_h=12.0,
        ),
        ValueError, "duration_h",
    )

# ============================================================
# Section 3: Schema-level — transporter pair completeness
# ============================================================
print("\n## Schema-level — transporter pair completeness")

@test("liver_oatp_km without vmax raises")
def t():
    tools = _server()
    expect_raises(
        lambda: tools["run_pbpk_simulation"].fn(
            name="custom", logP=4.5, pKa=4.0, fu_p=0.05, R_bp=0.55,
            compound_type="acid", CL_int=20.0, dose_mg=40.0,
            liver_oatp_km=5.0,  # missing liver_oatp_vmax
        ),
        ValueError, "liver_oatp",
    )

@test("kidney_mate1_vmax without km raises")
def t():
    tools = _server()
    expect_raises(
        lambda: tools["run_pbpk_simulation"].fn(
            name="custom", logP=2.0, pKa=7.0, fu_p=0.1, R_bp=1.0,
            compound_type="neutral", CL_int=10.0, dose_mg=50.0,
            kidney_mate1_vmax=100.0,  # missing km
        ),
        ValueError, "kidney_mate1",
    )

# ============================================================
# Section 4: Soft warnings — runs but flags suspicious inputs
# ============================================================
print("\n## Soft warnings — runs but surfaces ⚠️ block")

@test("transporter + perfusion-limited → warning")
def t():
    tools = _server()
    out = tools["run_pbpk_simulation"].fn(
        name="custom", logP=4.5, pKa=4.0, fu_p=0.05, R_bp=0.55,
        compound_type="acid", CL_int=20.0, dose_mg=40.0,
        kidney_oct2_km=10.0, kidney_oct2_vmax=50.0,
        # default distribution_model = perfusion_limited
    )
    assert "⚠️" in out, "no warning surfaced"
    assert "perfusion_limited" in out.lower() or "permeability_limited" in out.lower(), \
        "warning didn't mention model mismatch"

@test("library + custom params → override warning")
def t():
    tools = _server()
    out = tools["run_pbpk_simulation"].fn(
        name="midazolam", dose_mg=7.5, fu_p=0.005, CL_int=50.0,
    )
    assert "IGNORED" in out or "ignored" in out, "no override warning"

@test("custom drug with all defaults → suspicious-default warning")
def t():
    tools = _server()
    out = tools["run_pbpk_simulation"].fn(
        name="MyDrug", dose_mg=50.0, CL_int=10.0,
    )  # fu_p=1.0, R_bp=1.0 are sentinels
    assert "fu_p" in out and ("1.0" in out or "sentinel" in out.lower()), \
        "no fu_p sentinel warning"
    assert "R_bp" in out and "1.0" in out, "no R_bp sentinel warning"

@test("zero clearance everywhere → warning, doesn't crash")
def t():
    tools = _server()
    out = tools["run_pbpk_simulation"].fn(
        name="MyDrug2", logP=2.0, pKa=7.0, fu_p=0.5, R_bp=1.0,
        compound_type="neutral", dose_mg=100.0,
        CL_int=0.0, CL_renal=0.0,
    )
    assert "eliminat" in out.lower() or "no hepatic" in out.lower(), \
        "no zero-clearance warning"

@test("compound_type='neutral' with fu_p<0.01 → warning")
def t():
    tools = _server()
    out = tools["run_pbpk_simulation"].fn(
        name="MisclassDrug", logP=4.0, pKa=4.0, fu_p=0.005, R_bp=0.55,
        compound_type="neutral",  # should probably be 'acid'
        CL_int=20.0, dose_mg=50.0,
    )
    assert "compound_type" in out or "acid" in out.lower(), \
        "no misclassification warning"

# ============================================================
# Section 5: Known-good cases — must run without warnings
# ============================================================
print("\n## Known-good — no warnings on validated workflows")

@test("library Midazolam (Tutorial Scenario A) runs cleanly")
def t():
    tools = _server()
    out = tools["run_pbpk_simulation"].fn(
        name="midazolam", dose_mg=7.5, route="oral",
        duration_h=12.0, kp_method="poulin_theil",
    )
    assert "PBPK Simulation" in out
    # Library compound + valid kp_method → no soft warnings
    assert "⚠️" not in out, f"unexpected warning:\n{out}"

@test("custom Diclofenac + hepatocyte IVIVE (full Scenario F-style) clean")
def t():
    tools = _server()
    out = tools["run_pbpk_simulation"].fn(
        name="Diclofenac_test", logP=4.51, pKa=4.0, fu_p=0.005,
        compound_type="acid", R_bp=0.55, ka=2.0, Fa=1.0, Fg=0.76,
        clearance_source="hepatocyte", CLint_vitro_hep=120.0,
        CL_renal=0.1, dose_mg=50.0, route="oral", duration_h=12.0,
        kp_method="schmitt",
    )
    assert "⚠️" not in out, f"unexpected warning:\n{out}"

@test("30 tools registered")
def t():
    tools = _server()
    assert len(tools) == 30, f"expected 30 tools, got {len(tools)}"

# ============================================================
# Section 6: Determinism — same input → same NCA result
# ============================================================
print("\n## Determinism — identical input → identical output")

@test("two consecutive midazolam runs produce identical NCA")
def t():
    tools = _server()
    args = dict(name="midazolam", dose_mg=7.5, route="oral",
                duration_h=12.0, kp_method="poulin_theil")
    out1 = tools["run_pbpk_simulation"].fn(**args)
    out2 = tools["run_pbpk_simulation"].fn(**args)
    # Strip the audit fingerprint line + plot path (which differ)
    def strip_volatile(s):
        lines = []
        for line in s.split("\n"):
            if line.startswith("_Audit fingerprint") or "Plot saved" in line:
                continue
            lines.append(line)
        return "\n".join(lines)
    assert strip_volatile(out1) == strip_volatile(out2), \
        "non-deterministic output"

@test("audit fingerprint stable across runs with same input")
def t():
    tools = _server()
    args = dict(name="midazolam", dose_mg=7.5, route="oral",
                duration_h=12.0, kp_method="poulin_theil")
    out1 = tools["run_pbpk_simulation"].fn(**args)
    out2 = tools["run_pbpk_simulation"].fn(**args)
    import re
    fp1 = re.search(r"fingerprint: `(\w+)`", out1)
    fp2 = re.search(r"fingerprint: `(\w+)`", out2)
    assert fp1 and fp2 and fp1.group(1) == fp2.group(1), \
        "audit fingerprint differs across identical inputs"

# ============================================================
# Section 7: Physiology mass balance
# ============================================================
print("\n## Physiology mass balance")

@test("get_physiology balances blood flow and volume")
def t():
    from core.physiology import get_physiology
    p = get_physiology(body_weight=70.0)
    # If invariants fail, get_physiology raises; reaching here means OK.
    total_vol = sum(p.organ_volumes.values()) + p.V_arterial + p.V_venous
    assert 50 < total_vol < 100, f"organ volumes total {total_vol} L for 70 kg"

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
total = len(PASSED) + len(FAILED)
print(f"Passed: {len(PASSED)}/{total}")
if FAILED:
    print(f"\nFailed:")
    for name, err in FAILED:
        print(f"  ✗ {name}\n    {err}")
    sys.exit(1)
else:
    print("All tests passed.")
    sys.exit(0)
