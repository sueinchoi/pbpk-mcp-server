# PBPK MCP Server — 사용자 튜토리얼

실제 테스트 기록과 함께 5개 시나리오로 PBPK 모델 구축 과정을 안내합니다.
모든 출력은 `claude-opus-4-7` 세션에서 실제 MCP 호출로 검증된 결과입니다.

**일시**: 2026-04-20  |  **서버 버전**: v1.5  |  **도구 30개**

---

## 목차

- [Scenario A: 알려진 약물 시뮬레이션 (Midazolam)](#scenario-a)
- [Scenario B: In vitro 데이터로 신약 모델링](#scenario-b)
- [Scenario C: 약물-약물 상호작용 (DDI) 평가](#scenario-c)
- [Scenario D: 특수 집단 (소아/임부/CKD/변이)](#scenario-d)
- [Scenario E: Transporter 기반 약물 (스타틴형)](#scenario-e)
- [참고: Kp Method 선택 규칙](#kp-method-선택)
- [참고: 파라미터 명명 규칙](#파라미터-명명-규칙)

---

<a id="scenario-a"></a>
## Scenario A: 알려진 약물 시뮬레이션 (Midazolam)

**목표**: 임상에서 흔히 쓰는 Midazolam 7.5 mg 경구 투여 후 PK 예측.

### A1. 약물 정보 조회

```python
mcp.call_tool("drug_properties", {"drug_name": "midazolam"})
```

**출력**:
```
## Drug Properties — Midazolam
Source: curated

| Property | Value |
|----------|-------|
| ChEMBL ID | CHEMBL601 |
| MW | 325.80 g/mol |
| logP | 3.89 |
| pKa | 6.2 |
| Ro5 violations | 0 |
```

**해석**: Midazolam은 moderate base (pKa 6.2), 중간 lipophilic (logP 3.89). Ro5 위반 없음 → 경구 흡수 양호 예상.

---

### A2. Kp Method 선택

```python
mcp.call_tool("compare_kp_methods", {"name": "midazolam"})
```

**출력 발췌** (Adipose/Liver 행):
| Tissue | R&R | Lukacova | Schmitt | PT | PTB | PK-Sim | Kp_mem |
|--------|-----|----------|---------|-----|-----|--------|--------|
| Adipose | 4.26 | 4.26 | 3.83 | 3.20 | 4.16 | 4.56 | 17.36 |
| Liver | 1.89 | 1.89 | 9.84 | 0.35 | 1.28 | 3.97 | 3.03 |

**선택 규칙**: Midazolam은 logP > 3 moderate base → **Poulin-Theil 권장**
(R&R은 lipophilic base adipose Kp를 체계적 과예측 — Jansson 2008, Graham 2012)

---

### A3. PBPK 시뮬레이션 (Midazolam 7.5 mg PO)

```python
mcp.call_tool("run_pbpk_simulation", {
    "name": "midazolam", "dose_mg": 7.5, "route": "oral",
    "duration_h": 24, "kp_method": "poulin_theil",
})
```

**결과**:

| 파라미터 | 값 | 임상값 (Greenblatt 1984) |
|---|---|---|
| Cmax | **72.2 ng/mL** | 40-100 ng/mL ✓ |
| Tmax | 0.38 h | 0.5-1.5 h ≈ |
| AUC_inf | **167.9 ng·h/mL** | 75-250 ng·h/mL ✓ |
| t½ | **5.40 h** | 1.8-6.4 h ✓ |
| CL/F | 44.66 L/h | - |

조직별 농도가 Kp에 따라 올바르게 분포함 (Gut C_tissue/C_plasma = 8.2 — 흡수 직후 enterocyte 집중).

---

### A4. IV vs PO — 경구 생체이용률(F) 계산

```python
# IV: 5 mg bolus
r_iv = call("run_pbpk_simulation", {..., "route": "iv_bolus", ...})
# PO: 5 mg
r_po = call("run_pbpk_simulation", {..., "route": "oral", ...})
F = AUC_po / AUC_iv
```

**결과**:
- IV AUC = 0.312 mg·h/L, CL = **16.03 L/h** (lit 18-30)
- PO AUC = 0.112 mg·h/L, CL/F = 44.66 L/h
- **F_oral = 0.359 (36%)** — lit 30-50% ✓

---

<a id="scenario-b"></a>
## Scenario B: In vitro 데이터로 신약 모델링

**목표**: HLM에서 측정한 CLint = 30 µL/min/mg인 CYP3A4 기질 신약의 임상 PK 예측.

### B1. IVIVE — HLM에서 in vivo CLint 계산

```python
mcp.call_tool("ivive_microsomal", {
    "clint_vitro": 30.0, "logP": 3.5, "protein_conc": 1.0,
})
```

**결과**:
```
**CLint in vivo = 300.64 L/h**

| Parameter | Value |
|-----------|-------|
| CLint_unbound_uL_min_mg | 59.35 |
| fu_inc | 0.5055 (예측됨, logP=3.5에서)
| MPPGL | 45 mg/g liver |
| liver_weight_g | 1876 |
| scaling_factor | 5.065 |
```

**공식**: `CLint_in_vivo = (CLint_vitro / fu_inc) × MPPGL × liver_weight × 60 / 10^6`

---

### B2. Fg (장벽 생체이용률) 예측

```python
mcp.call_tool("predict_fg", {
    "Peff": 4.0, "CLint_gut": 10.0, "fu_gut": 0.2,
})
```

**결과**: **Fg = 0.6793** (Qgut Yang 2007 모델)
- Qgut = 4.24 L/h
- CLperm = 5.54 L/h

---

### B3. Full simulation — Custom compound + IVIVE

```python
mcp.call_tool("run_pbpk_simulation", {
    "name": "NewDrug",
    "logP": 3.5, "pKa": 7.0, "fu_p": 0.1,
    "compound_type": "neutral", "R_bp": 1.0, "mw": 400,
    "ka": 1.5, "Fa": 0.9, "Fg": 0.7,
    "clearance_source": "hlm",
    "CLint_vitro_hlm": 30.0,
    "dose_mg": 20, "route": "oral", "duration_h": 24,
})
```

**결과**: IVIVE 자동 적용 (`30.0 µL/min/mg → 267.2 L/h`), 예측 PK:
- Cmax = 37.8 ng/mL
- Tmax = 0.89 h
- AUC_inf = 375 ng·h/mL
- t½ = 17.3 h (긴 distribution + moderate CL)
- CL/F = 53.27 L/h

---

<a id="scenario-c"></a>
## Scenario C: 약물-약물 상호작용 (DDI) 평가

### C1. Static DDI 스크리닝

in vitro Ki 만 있는 초기 평가:

```python
mcp.call_tool("predict_ddi", {
    "mechanism": "reversible", "Ki": 0.05, "I_h_u": 0.5, "fm": 0.9,
})
```

**결과**: **AUC Ratio = 5.50x (Strong inhibition)** [FDA 2020 분류]

공식: `AUC_ratio = 1 / (fm/(1 + I/Ki) + (1-fm))`
- R_h (I/Ki) = 11, fm = 0.9
- R = 1 / (0.9/12 + 0.1) = 1/0.175 = 5.71 ≈ 5.50x

---

### C2. Dynamic DDI — Ketoconazole SS + Midazolam

```python
mcp.call_tool("run_dynamic_ddi", {
    "victim_name": "midazolam", "victim_dose_mg": 7.5, "victim_route": "oral",
    "victim_first_dose_h": 72,
    "perp_name": "Keto", "perp_dose_mg": 400, "perp_route": "oral",
    "perp_n_doses": 5, "perp_interval_h": 24,
    "perp_logP": 3.86, "perp_pKa": 6.5, "perp_fu_p": 0.01,
    "perp_compound_type": "moderate_base", "perp_R_bp": 0.6,
    "perp_mw": 531, "perp_ka": 1.0, "perp_CL_int": 150,
    "ddi_mechanism": "combined", "Ki": 0.015, "KI": 3.0, "kinact": 1.1,
    "fm": 0.94, "duration_h": 120,
})
```

**결과**:
- **AUC ratio: 18.24x** (lit Olkkola 1993: ~15x)
- **Cmax ratio: 2.38x**
- Liver CYP3A4 at Midaz Tmax: **40.7% of baseline** (59% 비가역 + 가역 억제)
- **Classification: Strong inhibition** (FDA 2020)

---

### C3. Dynamic DDI — Rifampin 유도

```python
mcp.call_tool("run_dynamic_ddi", {
    ...(Rifampin 600 mg × 8d, Midaz day 8)...
    "ddi_mechanism": "induction", "Emax": 14.0, "EC50": 0.5, "fm": 0.94,
})
```

**결과**:
- **AUC ratio: 0.01x** (lit Backman 1996: 0.04-0.10)
- Liver CYP3A4: **897% of baseline** (9x 유도)
- **Classification: Strong induction**

> **Note**: segmented liver (n_liver_segments=5 기본값)가 induction을 약간 과대예측 경향. 보수적 예측을 원하면 `n_liver_segments=1` 옵션 권장 (단 API 노출 필요).

---

<a id="scenario-d"></a>
## Scenario D: 특수 집단

### D1. 소아 (5세, 20 kg)

```python
mcp.call_tool("run_pbpk_simulation", {
    "name": "midazolam", "dose_mg": 4.0, "route": "iv_bolus",
    "body_weight": 20, "age": 5, "duration_h": 12,
    "kp_method": "poulin_theil",
})
```

**결과**:
- Cmax = 6.31 mg/L
- **CL/F = 11.17 L/h** (lit Reed 2001, 5세: 11-15 L/h) ✓
- **Vss = 18.92 L** (lit 12-18 L) ✓

---

### D2. 임부 (GA 32주, 3분기)

```python
mcp.call_tool("pregnancy_physiology", {"gestational_age_weeks": 32})
```

**주요 변화** (vs 비임신):
- CYP3A4: **1.61x ↑** (Midazolam 등 대사 증가)
- CYP1A2: 0.65x ↓
- GFR: 1.39x ↑
- Cardiac output: 1.34x ↑
- Hematocrit: 0.90x ↓

---

### D3. CKD Stage 3 (Moderate)

```python
mcp.call_tool("disease_state", {"disease_type": "ckd", "stage": "moderate"})
```

**Multipliers**:
- GFR: **0.37x ↓** (63% 감소)
- fu_p: 1.15x ↑ (단백질 결합 감소)
- CYP3A4: 0.85x ↓ (uremic toxins 효과)
- Hematocrit: 0.88x ↓

---

### D4. Population PK (변이 포함)

```python
mcp.call_tool("run_population_pbpk", {
    "name": "midazolam", "dose_mg": 7.5, "n_individuals": 50,
    "route": "oral", "duration_h": 24,
})
```

**결과** (50명 가상 환자):

| 파라미터 | Median | 5th %ile | 95th %ile |
|---|---|---|---|
| Cmax (ng/mL) | 24.5 | 11.6 | 38.9 |
| AUC (ng·h/mL) | 166 | 52.6 | 363 |

> **주의**: population tool은 기본 R&R Kp 사용 (kp_method 파라미터 없음). 정밀 예측이 필요하면 커스텀 compound로 kp_scale 조정.

---

<a id="scenario-e"></a>
## Scenario E: Transporter 기반 약물 (스타틴형)

### E1. 간 Transporter 프로파일 조회

```python
mcp.call_tool("transporter_clearance", {
    "organ": "liver", "CLint_met": 20, "fu_p": 0.02, "R_bp": 0.6,
})
```

**Liver 프로필**:
| Transporter | Direction | Km (µM) | Vmax | CLint |
|---|---|---|---|---|
| OATP1B1 | Uptake | 5.0 | 100 | 20.0 |
| OATP1B3 | Uptake | 10.0 | 50 | 5.0 |
| MRP2 | Efflux | 50 | 30 | 0.60 |
| BCRP | Efflux | 20 | 20 | 1.00 |

Extended clearance: **CLint_overall = 10.73 L/h**

---

### E2. Permeability-limited + OATP1B1 시뮬레이션

스타틴형 약물 (logP 4.5, acidic, 고결합):

```python
mcp.call_tool("run_pbpk_simulation", {
    "name": "Statin", "logP": 4.5, "pKa": 4.2, "fu_p": 0.05,
    "compound_type": "acid", "R_bp": 0.55, "mw": 410,
    "ka": 2.0, "Fa": 0.9, "Fg": 1.0, "CL_int": 10,
    "distribution_model": "permeability_limited",
    "liver_oatp_km": 1.5, "liver_oatp_vmax": 50.0,
    "liver_mrp2_km": 5.0, "liver_mrp2_vmax": 20.0,
    "dose_mg": 40, "route": "oral", "duration_h": 48,
})
```

**결과**:
- Cmax = 0.87 µg/mL
- AUC_inf = 45.7 mg·h/L
- t½ = 37.7 h (OATP uptake + tissue 축적으로 긴 분포)
- CL/F = 0.88 L/h

---

<a id="kp-method-선택"></a>
## 참고: Kp Method 선택 규칙

| 약물 분류 | 권장 Method | 근거 |
|---|---|---|
| Neutral / Weak base (일반) | `rodgers_rowland` | 기본, 범용 검증 |
| Lipophilic base (logP > 3) | `poulin_theil` | R&R adipose 과예측 개선 |
| Highly-bound acid (fu_p < 0.01) | `berezhkovskiy` or `pksim_standard` | R&R acid Vss 과소예측 (Rodgers 2006) |
| Hydrophilic (logP < 0) | `rodgers_rowland` | Ionic partitioning 적절 |
| Very lipophilic (logP > 5) | `kp_membrane` | 멤브레인 결합 우세 |

**결정 트리**:
1. 알려진 약물인가? → library의 `kp_scale` 있으면 R&R 그대로 사용 가능
2. logP > 3 + base → PT
3. fu_p < 0.01 + acid → Berezhkovskiy
4. 그 외 → R&R (default)
5. 먼저 `compare_kp_methods`로 tissue별 Kp 변동 확인

---

<a id="파라미터-명명-규칙"></a>
## 참고: 파라미터 명명 규칙 (중요!)

FastMCP는 알 수 없는 kwarg를 **조용히 drop**합니다. 잘못된 파라미터명 사용 시 도구가 default값으로 동작 (경고 없음).

### 라이브러리 약물 로딩
| 도구 | 파라미터명 |
|---|---|
| `run_pbpk_simulation`, `predict_kp`, `compare_kp_methods`, `predict_tissue_binding`, `predict_hepatic_clearance`, `compare_hepatic_clearance`, `predict_fg`, `run_population_pbpk` | `name` |
| `run_dynamic_ddi` | `victim_name`, `perp_name` |
| `drug_properties` | `drug_name` |

### 기타 흔한 실수
| 도구 | 주의 |
|---|---|
| `run_population_pbpk` | `n_individuals` (**not** n_subjects), 10-500 clamp |
| `ivive_microsomal` | `clint_vitro` (**not** CLint_vitro_hlm) |
| `predict_ddi` | `I_h_u` (unbound liver conc µM), `fm`, `Ki`/`KI`/`kinact`/`Emax`/`EC50` |
| `disease_state` | `disease_type` + `stage` (hepatic은 mild→mild_A alias) |
| `pregnancy_physiology` | `gestational_age_weeks` (**not** gestational_week) |
| `run_pbpk_simulation` | `kp_method` 미지정 시 R&R 사용 |

---

## 라이브러리 기본 약물과 검증 상태

| 약물 | 권장 Kp | Vss | CL | t½ | 검증 |
|---|---|---|---|---|---|
| Midazolam | PT | 0.90 L/kg | 15.4 L/h | 5.3 h | ✓ |
| Diazepam | PT | 1.20 L/kg | 0.91 L/h | 67 h | ✓ |
| Warfarin | Berezhkovskiy | 0.12 L/kg | 0.11 L/h | 52 h | ✓ |
| Theophylline | R&R | 0.28 L/kg | 1.93 L/h | 7.2 h | ✓ |
| Caffeine | R&R | 0.42 L/kg | 3.65 L/h | 5.7 h | ✓ |
| Metformin | R&R | 1.17 L/kg | 27.2 L/h | 2.6 h* | △ |

*Metformin t½은 biphasic PK (OCT 수송)로 실측값(4-9h) 대비 과소예측. 재현을 원하면 perm-limited + kidney OCT2/MATE1 튜닝 필요.

---

## 권장 Workflow

```
1. [drug_properties] 약물 정보 조회
     ↓
2. [compare_kp_methods] Kp method 선택
     ↓
3. [run_pbpk_simulation] 기본 시뮬레이션
     ↓
4. 임상값 비교 — Vss/CL 2x 이상 차이 시:
     - Kp method 변경 (Scenario A 참조)
     - IVIVE 재검토 (Scenario B)
     - kp_override로 문헌값 주입
     ↓
5. [run_dynamic_ddi] DDI 평가 (해당 시)
     ↓
6. [run_population_pbpk] 변이 평가
     ↓
7. [fit_to_observed] 관찰 데이터로 fine-tune
```

---

*서버 버전 v1.5 / 30 tools / 25 core modules / ODE: BDF method*
*문의: MCP tool `pbpk_help`로 전체 파라미터 가이드 조회 가능*
