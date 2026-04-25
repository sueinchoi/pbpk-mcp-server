# PBPK MCP Server — Changelog

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
