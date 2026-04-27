# PBPK MCP Server — Changelog

## v1.9 (2026-04-22) — Five invariants in MCP system prompt

The FastMCP `instructions` field now declares five hard invariants
that the server enforces. Together with the v1.7 + v1.8 schema /
validator architecture, this binds LLM behavior at both the prompt
and runtime layers.

### Invariants declared in system prompt (`server.py`)
1. **Refuse-to-default** — never silently substitute a default for
   a missing required parameter; ask the user or return a structured
   missing-parameter error.
2. **Cite-or-abstain** — every literature value carries a verifiable
   identifier (PMID, DOI, ChEMBL ID, etc.); use `verify_citation()`
   before inserting into a Source field.
3. **Unit-explicit** — pass canonical-unit floats or unit-bearing
   strings ('70 uL/min/mg'); pint validators reject incompatible
   units.
4. **Range-check** — out-of-range values are REJECTED, not clipped.
5. **Mass-balance** — every simulation runs a post-hoc dose-recovery
   check. Failure ABORTS, never warns.

### New
- `core/invariants.py::check_dose_recovery` — post-simulation mass
  balance assertion. Computes body burden (Σ V_organ × C_organ +
  blood pools) + cumulative eliminated (∫ CL × C_unbound dt) +
  lumen remaining; compares to total input (dose × Fa × Fg for oral,
  dose × n_doses for IV). Tolerance 1% IV, 5% oral (post-hoc trap
  integration vs BDF agreement).
- `pbpk_modeling_guide` prompt now opens with the five invariants
  as rules the LLM must respect in its reasoning.
- `tests/test_silent_fallback.py` — 2 new mass-balance cases.

### Behavior changes
- `run_pbpk_simulation` and `simulate_validated` both call
  `check_dose_recovery` after the ODE solve. A violation raises
  `ValueError` with the numeric breakdown (input / body /
  eliminated / rel_err) — the simulation does NOT return a result.

### Verification
- 40/40 fail-fast tests pass (was 38)
- Library Midazolam IV/oral, Diclofenac hepatocyte-IVIVE all clear
  the dose-recovery check
- Deliberately corrupted result (×5 concentration) is detected and
  aborted

## v1.8 (2026-04-22) — Units, citations, session-based workflow

Completes the server-side safety architecture with the three remaining
axes from the v1.7 audit: unit-aware parameter parsing, live citation
verification, and decomposition of `run_pbpk_simulation` into a
prerequisite-gated session workflow.

### New
- `core/units.py` — pint-based canonical unit table for every PBPK
  parameter. `parse_quantity('70 uL/min/mg', 'CLint_vitro_hlm')`
  converts to canonical and returns the magnitude. Incompatible units
  (e.g. CL_int passed in mg) raise with a dimensional mismatch error.
- `core/citation.py` — verify a PMID against PubMed E-utils or a DOI
  against Crossref. Three modes: online (cache + live), offline
  (cache only), strict (cache miss → ValueError). Results cached to
  `data/citation_cache.jsonl`.
- `core/session.py` — session-based PBPK workflow. Decomposes the
  47-flat-parameter `run_pbpk_simulation` into 8 prerequisite-checked
  steps (register → binding → clearance → absorption → transporters →
  structure → validate → simulate). `validate_model()` issues a token
  that `simulate_validated()` requires; missing parameter groups are
  enumerated explicitly.
- `tools/session_tools.py` — 11 new MCP tools exposing the session
  workflow and citation verification. Total tool count: 41 (30 PBPK
  + 9 session + 2 citation).

### Changed
- `core/clearance_spec.py` — every CLint field accepts unit-bearing
  strings via pint validators. Dimensional mismatches reject at
  schema construction.
- `tests/test_silent_fallback.py` — 38 cases now (was 24). New
  sections: unit-aware parsing, citation verification, session
  workflow.
- `pyproject.toml` — adds pydantic, pint, requests dependencies.

### Architecture summary (v1.7 + v1.8)
The server now has 4 server-side safety layers:
1. **Schema** — Pydantic discriminated unions (clearance), nested
   pair models (transporters), unit parsing (pint).
2. **Invariants** — physiological ranges per parameter, mass
   balance for physiology tables.
3. **Workflow** — session-based decomposition with token-gated
   simulation; missing groups fail validate_model() loudly.
4. **Audit + provenance** — append-only JSONL, citation cache,
   per-parameter source tracking.

## v1.7 (2026-04-22) — Server-side safety architecture

Schema and invariant enforcement against silent-fallback patterns
discovered during the Diclofenac case study and a structured audit
of all 30 tools.

### New
- `core/clearance_spec.py` — Pydantic discriminated union for the
  five clearance sources (direct, hlm, hepatocyte, rcyp). Required
  fields per variant are now enforced at schema level. Previously
  `clearance_source="hlm"` with `CLint_vitro_hep` provided silently
  ran with `CL_int=0`.
- `core/transporter_spec.py` — pairs Km/Vmax at schema level. XOR
  (only Km or only Vmax) raises with the offending pair name.
- `core/invariants.py` — physiological ranges for every parameter
  (fu_p ∈ [1e-5, 1.0], logP ∈ [-5, 10], etc.) plus mass-balance
  checks for the physiology tables.
- `core/audit.py` — append-only JSONL log at `data/audit.jsonl` with
  input fingerprint, resolved parameters, warnings, NCA summary.
- `tests/test_silent_fallback.py` — 24-case fail-fast suite
  (invalid enums, mismatched clearance source, out-of-range values,
  transporter pair completeness, soft-warning surfacing, known-good
  workflows, determinism). Run via
  `python -m tests.test_silent_fallback`.

### Changed
- `run_pbpk_simulation` rejects (instead of silently coercing):
  - invalid `kp_method` strings (suggests closest valid option)
  - mismatched `clearance_source` vs. supplied IVIVE input
  - any physicochemical value outside its physiological range
  - incomplete transporter Km/Vmax pairs
  - multi-dose regimens whose interval × n_doses exceeds duration
- Output now includes a "Modelling Provenance" footer enumerating
  defaults used, mechanisms not modelled, and an audit fingerprint.
- `get_physiology` runs mass-balance invariants at startup; corrupt
  physiology tables fail with an actionable message.
- Top-tier tool docstrings carry an explicit anti-fabrication
  instruction.

## v1.6 (2026-04-25) — Deployment release

### UX 개선 (3rd-party 평가 후)
- `run_population_pbpk`에 `kp_method` 파라미터 추가 (PT/Berezhkovskiy 등 모든 method 사용 가능)
- `run_dynamic_ddi`에 `kp_method` + `n_liver_segments` 파라미터 노출 (이전 5 하드코드)
- `CompoundSpec.recommended_kp_method` 메타데이터 추가
- `run_pbpk_simulation`이 라이브러리 약물에 sub-optimal Kp 사용 시 자동 ℹ️ Tip 출력
  - Midazolam/Diazepam → poulin_theil 권장
  - Warfarin → berezhkovskiy 권장
- `predict_ddi`가 누락 필수 파라미터 reject + 입력값 echo

## v1.5 (2026-04-20) — Tool API 일관성
- `predict_hepatic_clearance` / `compare_hepatic_clearance`에 `name` 파라미터 추가
- `predict_fg`에 `name` + Qgut 역산출 (라이브러리 Fg 재현)
- `transporter_clearance`에 organ→profile 자동 매핑
- `disease_state`에 hepatic stage alias (mild→mild_A 등)
- `predict_ddi` FDA 2020 분류 재구현 (induction 정확 인식)
- `import_pksim_model` graceful FileNotFoundError 처리

## v1.4 (2026-04-20) — Mass conservation 버그 수정
- **Critical**: Perm-limited liver_in이 pancreas 누락 → 췌장 outflow mass 소실 (inert에서도 70% 손실) 수정
- DDI baseline 불일치 (alone vs DDI 다른 모델 사용) → 동일 segmented ODE로 통일

## v1.3 (2026-04-20) — Permeability-limited 모델 버그 수정
- LUNG 혈류 누락 (self._Q에 LUNG 제외) → Qco 명시적 주입
- CL_renal Kp_kidney 정규화 (perfusion-limited와 일치성)

## v1.2 (2026-04-20) — 라이브러리 보정 + Kp 가이드
- Metformin CL_renal: 26.1 → 50 L/h (intrinsic CL 해석 정정)
- Warfarin Berezhkovskiy 권장 문서화
- pbpk_help에 Kp Method Selection Guide 추가

## v1.1 (2026-04-20) — 정확성 개선
- `CompoundSpec.kp_scale` per-organ 보정 (Björkman 2001)
- Midazolam Vss: 8.6 → 0.90 L/kg (Poulin-Theil + kp_scale)
- Permeability-limited multi-dose + IV infusion 지원
- Dynamic DDI: 5-segment dispersion liver + inlet 농도 기반 enzyme dynamics
- Gut Fg well-stirred 공식 (induction에서 음수 방지)
- RHS 벡터화 (~17% 성능 향상)
- Heavy deps lazy import

## v1.0 (initial) — 기본 PBPK MCP 서버
- 25 코어 모듈, 30 MCP 도구
- 7 Kp methods, 9-segment ACAT, 3 hepatic models, 4 DDI 메커니즘
- 5종 species 생리, 5단계 CKD, 4단계 Child-Pugh
- PKSimDB.sqlite 통합 (38K distributions, 294 ontogeny points, 38 transporters)
