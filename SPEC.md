# TrustED PETs Transformer — Implementation Spec

> Instructions for the coding agent: build the **engine first** as a pure Python package with tests and a CLI, then the Django UI on top. Follow the phase plan in §9. Do not use an LLM for any transformation decision; decisions must be deterministic and reproducible (fixed random seeds).

---

## 1. Goal

A web tool that takes a tabular dataset (synthetic), detects privacy risks, applies the **minimum sufficient** privacy-enhancing transformations for a defined sharing scenario, and proves with numbers that (a) disclosure risk dropped and (b) analytical utility for a stated purpose is preserved.

Core idea: **"Minimum sufficient transformation, with a simulated attacker in the loop."**
Measure re-identification risk with a real (simulated) linkage attack, then greedily apply the transformation step with the best risk-reduction-per-utility-loss until risk falls below the scenario's target. The search path doubles as the privacy/utility trade-off curve.

Mandatory demo outputs (from the challenge brief):
1. Data types identified in the original dataset
2. PETs selected per data type (with reason)
3. Transformed dataset / protected view
4. Utility before vs. after
5. The sharing scenario the protected data is suitable for
Also required: threat model, residual risks, honest limitations.

---

## 2. Tech stack

- **Backend/UI:** Django 5, server-rendered templates, HTML/CSS.
- **Interactivity:** HTMX (no React/SPA needed). Chart.js (or Plotly.js) for charts, loaded from a CDN.
- **Engine:** Python 3.11+, pandas, numpy, scikit-learn, scipy.
- **Free-text PII:** Microsoft Presidio (`presidio-analyzer`, `presidio-anonymizer`) + spaCy `en_core_web_lg` (or `_sm` if install time is an issue), plus custom regex recognizers.
- **Optional LLM:** Anthropic API (key in env var), used only for (a) ambiguous column-role suggestions, (b) plain-language report text. Must be switchable off; app must fully work without it.
- **Storage:** SQLite for job metadata; uploaded/processed files in `MEDIA_ROOT/jobs/<job_id>/`. No Celery; run synchronously (datasets ≤ 50k rows). Show a spinner via HTMX.
- **Tests:** pytest.

---

## 3. Project structure

```
pets_transformer/
  engine/                 # pure Python, no Django imports
    roles.py              # Role & Gate enums, dataclasses
    profiling.py          # column role detection
    gates.py              # default gate assignment
    ladders.py            # transformation ladders per role
    transforms.py         # apply a "level plan" to a DataFrame
    risk.py               # uniqueness, k-anonymity, l-diversity, linkage attack
    utility.py            # fidelity proxy + task-based utility
    search.py             # greedy search over ladder levels
    text_pii.py           # Presidio + regex redaction, ground-truth recall
    dp.py                 # Laplace mechanism for aggregate queries (phase 3)
    report.py             # decision table, JSON/HTML report builder
    llm.py                # optional, isolated
    cli.py                # python -m engine.cli run data.csv --scenario scenario.json
  webapp/                 # Django project + app
  data/
    generate_sample.py    # synthetic sample dataset generator with ground truth
  tests/
```

---

## 4. UI flow (5-step wizard)

A top progress bar: **Upload → Profile → Scenario → Results → Export**.

### Step 1 — Upload
- Upload CSV or XLSX (max ~50k rows), or button **"Load sample dataset"**.
- Show row/column count and first 10 rows.
- Banner: "Use only synthetic or approved data."

### Step 2 — Profile review
- Table, one row per column: name, dtype, % unique, % missing, example values (3), **detected role**, confidence, **gate colour** (red/yellow/green badge), short reason.
- User can override role and gate via dropdowns (HTMX inline update).
- Detected domain shown at top as a hint (e.g., "Looks like: healthcare"), used only to pick defaults.

### Step 3 — Scenario card
Fields:
- **Recipient:** Public release / Trusted research partner / Internal analytics team. Each sets a default risk target (see §5.4).
- **Attacker knowledge:** checkboxes of quasi-identifier columns the attacker is assumed to know (pre-filled with all QI columns).
- **Purpose:** task type (Classification / Regression / Descriptive statistics / Visualization), target column (for ML), key columns to preserve (multi-select; get higher utility weight).
- **Risk target:** pre-filled from recipient, editable (max linkage success rate %, min k).
- Short free-text description of the scenario (shown in the report).

### Step 4 — Results dashboard
- **KPI cards:** linkage attack success before → after; % unique records before → after; min k before → after; task metric (AUC/F1 or R²/MAE) original vs. protected; utility score; rows suppressed %.
- **Trade-off chart:** x = privacy (1 − attack success), y = utility; every greedy step is a point; chosen point highlighted.
- **Decision table:** column, role, gate, PET applied, final level (e.g., "date → month"), reason (numeric: "reduced attack success by X pp at Y utility cost").
- **Before/after preview:** side-by-side first 20 rows, changed cells highlighted.
- **Free-text panel:** example redactions; detection recall if ground truth exists.
- **Residual risk & limitations** box (auto-generated, see §7).

### Step 5 — Export
- Download protected CSV.
- Download JSON report (all metrics, decisions, scenario, seed, config).
- Download HTML report (printable).
- Never export the original data or the pseudonymisation secret.

---

## 5. Engine specification

### 5.1 Roles (`roles.py`)
`DIRECT_ID, QUASI_ID, SENSITIVE, DATE, LOCATION, FREE_TEXT, TECHNICAL_ID, NUMERIC, CATEGORICAL, TARGET, OTHER`
A column can be both `QUASI_ID` and a type (e.g., `DATE`); model this as `role` + `kind` (kind ∈ {date, location, numeric, categorical, text, id}).

### 5.2 Profiling (`profiling.py`)
Rule-based, in this order:
1. Regex on values (≥80% match): email, phone, national-ID-like patterns, IBAN, credit-card-like (Luhn), IPv4/IPv6, MAC, URL, lat/long → DIRECT_ID or TECHNICAL_ID or LOCATION.
2. Column-name keyword lists (name, surname, email, phone, ssn, id, address, zip, postcode, dob, birth, age, gender, sex, city, lat, lon, diagnosis, salary, ip, device, comment, notes, review…).
3. dtype + cardinality: unique ratio > 0.95 and string → likely identifier; parseable dates → DATE; long strings (mean > 40 chars, many tokens) → FREE_TEXT.
4. Default QI set: age/DOB, gender, postcode/zip, city/region, dates of events, ethnicity-like, occupation.
5. SENSITIVE: health/diagnosis, financial amounts, religion, etc. by keyword.
Output: role, kind, confidence (0–1), reason string.
Optional (phase 2): send low-confidence columns' name + 5 sample values to the LLM for a role suggestion; display as suggestion only.

### 5.3 Gates (`gates.py`)
- **Red** (fixed transform, excluded from search): DIRECT_ID → pseudonymise (if needed for joins) or drop; TECHNICAL_ID → pseudonymise or truncate (IP → /24).
- **Yellow** (search space): QUASI_ID columns, DATE, LOCATION, SENSITIVE with rare categories, FREE_TEXT (redaction always applied, not searched).
- **Green** (default unchanged): other numeric/categorical columns. **Rule:** a green column is promoted to yellow if adding it to the attacker's known set increases linkage success by > 2 pp.
Gate colours are both the search constraint and the UI explanation.

### 5.4 Risk metrics (`risk.py`)
Computed on the QI columns the attacker knows.
- **Uniqueness:** % of records whose QI combination is unique.
- **k-anonymity:** min and distribution of equivalence-class sizes.
- **Prosecutor risk:** mean of 1/|class|.
- **l-diversity** for each SENSITIVE column: min distinct sensitive values per class.
- **Linkage attack (headline metric):**
  - Keep a hidden internal row id for every original record.
  - Attacker table = QI values (original, untransformed) of a random 30% sample of records (seeded).
  - Apply the *same* generalisation to the attacker's values (transformations are public/deterministic).
  - For each attacker record, find matching released records on QIs. Success if exactly one match and it is the true record.
  - Attack success rate = successes / attacker records.
Default targets by recipient: Public: success ≤ 1%, k ≥ 10. Trusted partner: ≤ 5%, k ≥ 5. Internal: ≤ 10%, k ≥ 3.

### 5.5 Transformation ladders (`ladders.py`)
Level 0 = unchanged. Each level is deterministic.

| kind | levels |
|---|---|
| date | 0 exact → 1 week → 2 month → 3 quarter → 4 year |
| age / numeric QI | 0 exact → 1 bin width 5 → 2 width 10 → 3 width 20 → 4 top/bottom-coded wide bins |
| other numeric | 0 exact → 1 round to 2 sig. figures → 2 quantile bins (10) → 3 quantile bins (4) |
| location (lat/lon) | 0 exact → 1 3 decimals → 2 2 decimals → 3 1 decimal |
| postcode | 0 full → 1 drop last char → 2 drop last 2 → 3 region prefix only |
| categorical | 0 exact → 1 merge categories < 1% into "Other" → 2 < 5% → 3 use provided hierarchy if any |
| timestamp | 0 exact → 1 hour → 2 day → (then date ladder) |

Plus a global **suppression** move: drop records in classes with size < k_target, capped at 5% of rows.

Fixed (red) transforms:
- **Pseudonymisation:** HMAC-SHA256(secret, value), truncated to 16 hex chars. Secret from env var `PETS_SECRET`, never exported. Same value → same token across tables.
- **Drop** for names/emails/phones unless the user chooses pseudonymise.
- **IP:** truncate to /24 (IPv4) or /48 (IPv6), or pseudonymise.

### 5.6 Greedy search (`search.py`)
```
state = all yellow columns at level 0; red transforms applied
record point(state)
while risk(state) > target and moves exist:
    for each move (one yellow column +1 level, or suppression):
        Δrisk = risk(state) - risk(state+move)
        Δutil = utility_proxy(state) - utility_proxy(state+move)
        score = Δrisk / (Δutil + 1e-6)
    apply best move; record point(state)
return final state, path
```
- `risk()` = linkage attack success (tie-break by uniqueness).
- Use the cheap **utility proxy** (§5.7) inside the loop; compute the expensive task utility only for the final state and ~5 evenly spaced path points (for the chart).
- Every recorded point stores: levels per column, all risk metrics, proxy utility.
- If no move reaches the target: report it explicitly as residual risk, do not hide it.

### 5.7 Utility (`utility.py`)
- **Proxy (fast, used in search):** weighted mean over columns of (1 − normalised information loss), where info loss per column = level / max_level (or normalised bin width); key columns from the scenario get weight 3, others 1. Plus correlation preservation: map transformed values back to numeric (bin midpoints, date midpoints), compute |corr_orig − corr_prot| mean, and combine: `proxy = 0.7 * info_score + 0.3 * (1 − mean_corr_diff)`.
- **Task utility (final):**
  - Split original into train/test (80/20, seeded) **before** transformation.
  - Baseline: model trained on original train, evaluated on original test.
  - Protected: apply the final level plan to train and to test features, train on protected train, evaluate on protected test.
  - Models: GradientBoosting/RandomForest (one fixed choice, seeded). Metrics: classification → ROC-AUC + F1; regression → R² + MAE.
  - Report utility retained = protected metric / baseline metric.
- **Descriptive/visualisation purpose:** relative error of a fixed set of aggregate queries (group means/counts by key columns), and per-column distribution distance (TVD on the protected granularity, Wasserstein on midpoints).

### 5.8 Free text (`text_pii.py`)
- Presidio analyzer with default recognizers + custom regex (Finnish phone formats, Finnish personal identity code pattern, emails, IBAN).
- Replace entities with typed placeholders: `[PERSON]`, `[PHONE]`, `[EMAIL]`, `[LOCATION]`, `[DATE]`, `[ID]`.
- If the dataset has ground-truth spans (sample generator provides them), report precision/recall per entity type. This is evidence for the judges.
- Optional (phase 2): LLM second pass on texts flagged as uncertain. Disclose in limitations that an external API sees the text (acceptable only because data is synthetic).

### 5.9 Differential privacy (`dp.py`, phase 3)
Separate "Aggregate query" panel: user picks group-by column and statistic (count/mean of clipped numeric), sets ε via slider; Laplace mechanism with sensitivity from clipping bounds. Show true vs. noisy values and error vs. ε. This is for aggregate sharing only, not for row-level data.

---

## 6. Sample dataset (`data/generate_sample.py`)

Generate a synthetic healthcare dataset (~5,000 rows, seeded) in case the organisers' starter dataset is not available:
`patient_name, email, phone, national_id, date_of_birth, gender, postcode, city, admission_date, discharge_date, department, diagnosis_code, length_of_stay, num_prior_visits, lab_value_1, lab_value_2, clinician_notes, readmitted_30d (target)`
- `clinician_notes` embeds some names, phones and dates; store ground-truth entity spans in a sidecar JSON.
- Make `readmitted_30d` depend on age, prior visits, length of stay, lab values (so ML utility is meaningful).
The engine must also work on any other uploaded CSV (generic, domain-agnostic).

Demo scenario: *A hospital shares admissions data with an external research partner to build a 30-day readmission model. The attacker is assumed to know age/DOB, gender, postcode and admission date.*

---

## 7. Report & residual risk (`report.py`)

Auto-generate:
- Scenario and threat model statement.
- Decision table (column, role, gate, PET, level, numeric reason).
- Before/after metrics table.
- Residual risks, always listed:
  - Risk measured only against the modelled attacker; stronger background knowledge can raise it.
  - k-anonymity/linkage metrics are empirical, not a formal guarantee (unlike DP).
  - Greedy search is not globally optimal.
  - Free-text redaction recall < 100%; report the measured value.
  - Pseudonyms are linkable across releases using the same secret.
  - Attribute disclosure possible where l-diversity is low (list those classes count).
- Optional LLM: rewrite the report in plain language **from the JSON report only**; the numbers must be copied, not generated.

---

## 8. Non-functional requirements

- Deterministic: global seed in config; same input + scenario → same output.
- No original data in exports, logs or public repo. Secret only from env.
- LLM off by default; when on, never send full datasets, only column names + ≤5 sample values or single text cells.
- Performance target: full run on 5k rows × 20 columns < 30 s.
- Unit tests for: each ladder level, pseudonymisation consistency, linkage attack on a hand-built toy table with known answer, greedy search stops at target, train/test split happens before transformation.

---

## 9. Phase plan

**Phase 1 — MVP (must be done for the demo):**
engine: profiling (rules), gates, ladders, transforms, risk (uniqueness, k, linkage attack), greedy search, utility proxy + task utility, report JSON; sample dataset generator; CLI; Django wizard steps 1–5 with KPI cards, decision table, before/after preview, trade-off chart, CSV + JSON export.

**Phase 2:** Presidio free-text redaction with recall vs. ground truth; l-diversity; green→yellow promotion rule; HTML report; LLM role suggestions and plain-language report.

**Phase 3:** DP aggregate panel; multi-table support (consistent pseudonyms across files); user-provided generalisation hierarchies.

---

## 10. Demo video script (≈3 min)

1. Problem in one sentence; scenario card (recipient, attacker, purpose).
2. Upload → profile table with roles and red/yellow/green gates.
3. Baseline: "An attacker knowing 4 attributes re-identifies X% of patients."
4. Run → trade-off curve; explain the chosen point.
5. After: attack success Y%, AUC retained Z%; decision table with reasons.
6. Free-text redaction example and measured recall.
7. Residual risks and limitations.
