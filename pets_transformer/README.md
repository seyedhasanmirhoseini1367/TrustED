# TrustED PETs Transformer

*Protect the data, keep its value.* Since AI Hackathon 2026 · TrustED challenge (Horizon Europe GA No. 101168467).

A web tool that takes a synthetic tabular dataset, detects privacy risks, applies the **minimum sufficient**
privacy-enhancing transformations for a defined sharing scenario, and proves with numbers that
(a) re-identification risk dropped (a simulated linkage attack) and (b) analytical utility was kept
(model trained on original vs. protected data, aggregate queries, distributions).

**Core idea: minimum sufficient transformation, with a simulated attacker in the loop.** The engine measures
linkage-attack success, then greedily applies the generalisation step with the best risk reduction per unit of
utility loss until the scenario's target (max attack success, min k) is met. The search path is the
privacy/utility trade-off curve shown in the UI.

## Run

```bash
python -m venv .venv && .venv/Scripts/activate          # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python data/generate_sample.py                           # 5,000 synthetic admissions + ground-truth PII spans
python manage.py migrate
python manage.py runserver                               # http://127.0.0.1:8000  → "Load sample dataset"
```

CLI (same engine, no Django):

```bash
python -m engine.cli profile data/sample/health_admissions.csv
python -m engine.cli run data/sample/health_admissions.csv --scenario examples/scenario_partner.json \
       --spans data/sample/health_admissions.spans.json --out out/demo
python -m engine.cli run a.csv b.csv --scenario s.json --out out/multi      # consistent pseudonyms across tables
python -m engine.cli dp data/sample/health_admissions.csv --group department --stat mean --value length_of_stay --eps 1 --curve
python -m pytest                                                             # 24 tests
```

Environment: `PETS_SECRET` (pseudonymisation key; otherwise a local `.pets_secret` file is created and never exported),
`PETS_LLM=1` + Anthropic credentials to enable the optional LLM features.

## Starter kit (TrustED PETs Proving Ground 2026)

Outputs follow the kit's conventions, so the kit's own `pets-eval scorecard` scores them directly:
the table keeps its file name and column names, a birth date coarsened past the year is released as an `age`
band column (`30-39`, `80+`), all labels use notations the kit parses (`2025-W12`, `2025-03`, `201**`), and each
table gets `rowmap_<table>.csv` (`protected_row,original_record_id`; for scoring only, never publish).
The 27 official sharing scenarios are bundled (`engine/kit_data/`, CC BY 4.0, © Since AI ry). Choosing one sets the
recipient, the attacker's knowledge (every catalogued quasi-identifier) and utility requirements: the coarsest level
each column may reach so the scenario's utility tasks stay answerable. Tasks that conflict with k ≥ 5 (hour of
arrival) start as "sacrificed" and can be switched on.

```bash
python -m engine.cli kit-scenarios <kit>/data/health/ed_visits.csv
python -m engine.cli run <kit>/data/health/ed_visits.csv <kit>/data/health/lab_results.csv --kit-scenario HEALTH-S1 --out out/health
pets-eval scorecard --original <kit>/data/health --protected out/health --scenario HEALTH-S1
python -m engine.cli labels <kit>/data/unseen/export_2025_q4.csv --dataset unseen --out unseen_field_labels.csv
```

Kit scorecard, HEALTH-S1 (self-assessment, not the official judging score):

| | privacy | utility | composite |
|---|---|---|---|
| this tool | **1.00** | **0.78** | **0.89** |
| kit baseline B2 / B2-k | 0.36 / 0.52 | 0.79 / 0.40 | — |

All kit risk components are 0 % (direct identifiers, k < 5, linkage re-identification, free-text leakage); tasks
H-U2 1.00, H-U4 0.95, H-U6 0.96, H-U7 0.98; H-U8 (hour-of-day curve) deliberately sacrificed. See
`../KIT_SUMMARY.md` for the full gap analysis and what is still open.

## Wizard

1. **Upload** CSV/XLSX (≤ 50k rows) or the sample dataset.
2. **Profile**: detected role per column, confidence, red/yellow/green gate and reason; override inline (HTMX).
3. **Scenario**: recipient (public ≤ 1 % / k ≥ 10, research partner ≤ 5 % / k ≥ 5, internal ≤ 10 % / k ≥ 3), attacker
   knowledge, purpose (classification / regression / descriptive), target and key columns.
4. **Results**: KPI cards, trade-off chart, decision table with numeric reasons, before/after preview, free-text
   redaction with measured recall, residual risks, DP aggregate panel.
5. **Export**: protected CSV, JSON report, printable HTML report. Never the original or the secret.

## Engine (`engine/`, pure Python)

| Module | What it does |
|---|---|
| `profiling.py` | Rule-based roles: value regexes (email, phone, HETU/SSN, IBAN, Luhn cards, IP, MAC, URL, coordinates) → column-name keywords → type/cardinality → ICD-10 codes → sensitive keywords |
| `gates.py` | Red (drop / HMAC pseudonym / IP truncation), yellow (search space), green (unchanged; promoted if it raises linkage success > 2 pp). Derived attacker knowledge (city follows from postcode); linked columns (event dates, lat/lon) share one level |
| `ladders.py` | Deterministic levels: date (week → month → year), timestamp (hour → day → …), birth date (month → year → age bands, released as `age`), age bins, numeric (2 s.f. → deciles → quartiles), coordinates, postcode masking, rare-category merging, ICD-10 hierarchy (rare codes recoded first) |
| `risk.py` | Uniqueness, k-anonymity, prosecutor risk, l-diversity, linkage attack (30 % attacker sample, same public generalisation, success = unique *and* correct match) |
| `search.py` | Greedy search; skips no-op levels; suppression move capped at 5 % of rows |
| `utility.py` | Proxy (information loss + correlation preservation) inside the loop; RandomForest task utility with the split done on the original data before transformation; aggregate-query error, TVD, Wasserstein |
| `text_pii.py` | Presidio + spaCy NER + Finnish regexes (HETU, phones, IBAN, street addresses) + dataset name/place dictionaries → typed placeholders; precision/recall/F2 vs. ground truth |
| `dp.py` | Laplace counts and clipped means, error vs. ε |
| `kit.py` | Starter-kit scenarios, table recognition, task → column requirements, role mapping to the kit's labels |
| `report.py`, `llm.py`, `cli.py` | Reports; optional Claude helper (role suggestions, plain-language summary from the JSON report only); CLI |

No LLM ever makes a transformation decision; every run is deterministic for a given seed.

## Measured on the sample (research-partner scenario, attacker knows DOB, gender, postcode, admission date)

Attack success 100 % → 0 %, min k 1 → 5, 0.7 % of rows suppressed, ROC-AUC 0.800 → 0.796 (99.6 % retained),
free-text recall 100 % on 19,345 gold spans, ~15 s. On the starter kit's `health/ed_visits` (46k rows): 100 % → 0 %,
k 1 → 5, 0.2 % suppressed, AUC 97.6 % retained, 27 s.

## Limitations

Risk is measured against the modelled attacker only; k-anonymity/linkage are empirical (not a formal guarantee like
DP); greedy search is not globally optimal; redaction recall on unseen text styles can be lower than measured;
pseudonyms are linkable across releases that share a secret. The report lists these for every run.
