# TrustED PETs Transformer

*Protect the data, keep its value.* Since AI Hackathon 2026 · TrustED challenge (Horizon Europe GA No. 101168467).

A tool that takes a synthetic tabular dataset, detects privacy risks, applies the **minimum sufficient**
privacy-enhancing transformations for a defined sharing scenario, and measures that (a) re-identification risk
dropped and (b) analytical utility was kept.

**Core idea: minimum sufficient transformation, with a re-identification test in the loop.** The engine measures
the re-identification rate under the scenario's threat model, then greedily applies the generalisation step with
the best risk reduction per unit of utility loss until the scenario's target (max re-identification rate, min k) is met. The search path is the
privacy/utility trade-off curve shown in the UI.

## Install

```bash
python -m venv .venv && .venv/Scripts/activate          # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt                          # pinned versions; pandas < 3, numpy < 2.3
python -m spacy download en_core_web_sm                  # required
python -m spacy download fi_core_news_md                 # recommended: Finnish free text
python -m pytest                                         # all tests should pass
```

## 1. Starter-kit workflow (TrustED PETs Proving Ground 2026)

`<kit>` is the folder `TRUSTED_PETS_PROVING_GROUND_2026`. Install its scorer once: `pip install -e <kit>/toolkit`.

```bash
python -m engine.cli kit-scenarios <kit>/data/health/ed_visits.csv                 # official scenarios for a table
python -m engine.cli run <kit>/data/health/ed_visits.csv <kit>/data/health/lab_results.csv \
       <kit>/data/health/clinical_notes.csv --kit-scenario HEALTH-S1 --out out/health  # protect (full tables)
pets-eval scorecard --original <kit>/data/health --protected out/health --scenario HEALTH-S1
python -m engine.cli labels <kit>/data/unseen/export_2025_q4.csv --dataset unseen --out unseen_field_labels.csv
pets-eval detect --pred field_labels.csv                                             # after `labels` on data/*
```

Outputs follow the kit's conventions, so `pets-eval` scores them directly: the table keeps its file name and column
names; a birth date coarsened past the year is released as an `age` band column (`30-39`, `80+`); every label uses a
notation the kit parses (`2025-W12`, `2025-03`, `201**`); each table gets `rowmap_<table>.csv`
(`protected_row,original_record_id`) — **for scoring only, never publish it**.

The 27 official sharing scenarios are bundled (`engine/kit_data/`, CC BY 4.0, © Since AI ry). Choosing one sets the
recipient, the threat model's known attributes (every catalogued quasi-identifier, as the kit counts k) and utility
requirements: the coarsest level each column may reach so the scenario's utility tasks stay answerable. Tasks that
conflict with k ≥ 5 (hour of arrival) start as "sacrificed" and can be switched on. Tasks can also require a
stable pseudonym (e.g. `card_pan` for the fraud model) and define the model target. Requirements are mapped for
health (H-U1…H-U8) and finance (FIN-U1…FIN-U6); other domains get the scenario's recipient and threat model only.
FIN-S1 (`card_transactions` + `transfers`) shows the limit honestly: FIN-U1 and FIN-U5 both score 1.00, but they need
hour-level times and near-exact amounts, so k ≥ 5 is not reachable on 398k transactions and the tool reports
*target not met* (kit privacy 0.36; baselines B2 0.29 / B2-k 0.61, utility B2 1.00 / B2-k 0.06). For such scenarios
sacrifice a task or share aggregates instead.

**Threat model and re-identification tests.** The search optimises the re-identification rate under a threat model
with exact knowledge of every catalogued quasi-identifier. For kit tables the report also runs the kit's own attack
specs (`metadata/attack_specs.json`) with tolerant matching (time windows, age ± 3 years). With the same columns a
tolerant re-identification test can never beat the exact one (its candidates include the target's whole equivalence
class; tested), so the exact test stays the objective. HEALTH-S1,
`ed_visits`: news-clips attack 24.4 % → 0 %, social-posts attack 10.2 % → 0 %, public-directory 0.01 % → 0 %.

### Kit scorecard, HEALTH-S1 (self-assessment, not the official judging score)

| | privacy | utility | composite |
|---|---|---|---|
| this tool | **1.00** | **0.78** | **0.89** |
| kit baseline B2 / B2-k | 0.36 / 0.52 | 0.79 / 0.40 | — |

Measured on `ed_visits` + `lab_results` + `clinical_notes`. Tasks: H-U2 1.00, H-U4 0.95, H-U6 0.96, H-U7 0.98;
H-U8 (hour-of-day curve) deliberately sacrificed.
**What "privacy 1.00" means:** all risk components the kit's public scorecard computes are 0 % — direct identifiers
retained, records with k < 5, and linkage re-identification by the kit's **16 public attacks** in
`metadata/attack_specs.json`. The organisers' hidden, stronger attackers are **not** measured here, and k-anonymity is
an empirical property, not a formal guarantee.

### Free-text redaction on the kit's annotated text

Recall = share of gold identifier values no longer recoverable from the redacted text (`pets-eval text --protected`,
case- and separator-insensitive); precision = share of our redacted spans that overlap a gold span
(`pets-eval text --pred`, label-agnostic). 3,000 annotated records per table; names/places from the same domain's
tables are given to both versions.

| table.column | recall before → after | precision before → after | held-out records 10,000–13,000 (recall / precision) |
|---|---|---|---|
| health/clinical_notes.note_text | 89.9 % → **99.8 %** | 0.33 → **0.95** | 99.9 % / 0.94 |
| customer_service/turns.text | 78.5 % → **98.7 %** | 0.68 → **0.95** | 98.5 % / 0.96 |
| public_admin/service_requests.description | 81.0 % → **98.9 %** | 0.66 → **0.90** | 98.4 % / 0.90 |

Without the cross-table name dictionaries the old recognizer reached 57 % / 69 % / 77 % (first measurement).
The recognizers were developed while looking at leaks in the first 3,000 records; the held-out slice shows the same
level. The kit's text is template-generated: expect lower recall on real text. Health mentions are not redacted
in health data (they are its content). The 100 % reported for our own sample (below) is on our generated notes.

### Field detection (`pets-eval detect`, all 475 catalogued columns)

Macro-F1 **0.862** (accuracy 0.87) vs. 0.646 for the kit's rule baseline B1; per role F1: direct identifier 0.93,
quasi-identifier 0.91, sensitive attribute 0.82, technical identifier 0.82, free text 0.81, record key 0.92,
non-sensitive 0.82. Detection uses only column names, values and table structure, never the kit's catalog; the rules
were improved while looking at errors on this public catalog, so the organisers' private `unseen` labels are the
honest test. Our labels for it: `../pets_submission/unseen_field_labels.csv`.

## 2. Web wizard

```bash
python manage.py migrate
python manage.py runserver        # http://127.0.0.1:8000 → upload a file or "Load sample dataset"
```

1. **Upload** CSV/XLSX, or the built-in sample. **Files over 50,000 rows are reduced to a reproducible random sample
   of 50,000 rows (seed 42)** and the page says so; protection and metrics then apply to that sample. Use the CLI
   for full kit tables (the kit's scorecard expects the full table).
2. **Profile**: detected role per column, confidence, red/yellow/green gate and reason; override inline (HTMX).
3. **Scenario**: an official kit scenario (for kit tables), or recipient (public ≤ 1 % / k ≥ 10, research partner
   ≤ 5 % / k ≥ 5, internal ≤ 10 % / k ≥ 3), threat model (known attributes), purpose, target, key columns and
   per-column limits. The step is titled "Scenario: threat model + tasks".
4. **Results**: KPI cards, trade-off chart, decision table with numeric reasons, before/after preview, free-text
   redaction with measured recall, residual risks, DP aggregate panel.
5. **Export**: protected CSV (original file name), row map (internal), JSON report, printable HTML report.
   Never the original or the secret.

## 3. Generic CLI and built-in sample

```bash
python data/generate_sample.py                           # 5,000 synthetic admissions + ground-truth PII spans
python -m engine.cli profile data/sample/health_admissions.csv
python -m engine.cli run data/sample/health_admissions.csv --scenario examples/scenario_partner.json \
       --spans data/sample/health_admissions.spans.json --out out/demo
python -m engine.cli dp data/sample/health_admissions.csv --group department --stat mean --value length_of_stay --eps 1 --curve
```

On the sample (research partner; threat model: DOB, gender, postcode and admission date are known):
re-identification rate 100 % → 0 % (our own exact-match re-identification test), min k 1 → 5, 0.7 % of rows suppressed, ROC-AUC 0.800 → 0.796
(99.6 % retained), free-text recall 100 % on our own generated notes (see above), ~15 s.

Environment: `PETS_SECRET` (pseudonymisation key; otherwise a local `.pets_secret` file is created and never
exported), `PETS_LLM=1` + Anthropic credentials for the optional LLM features, `DJANGO_DEBUG=1` for debug pages
(off by default).

## Engine (`engine/`, pure Python)

| Module | What it does |
|---|---|
| `profiling.py` | Rule-based roles from values and names only (never the kit's catalog): regexes (email, phone, HETU with/without separator, IBAN, Luhn cards, IP, MAC, URL, hashes, coordinates, Finnish numbers/dates) → name keywords → type/cardinality → codes → sensitive keywords |
| `gates.py` | Red (drop / HMAC pseudonym / IP truncation), yellow (search space), green (unchanged; promoted if it raises the re-identification rate > 2 pp). Derived known attributes (city follows from postcode); linked columns share one level |
| `ladders.py` | Deterministic levels: date (week → month → year), timestamp (hour → day → …), birth date (month → year → age bands, released as `age`), age bins, numeric, coordinates, postcode masking, rare-category merging, ICD-10 hierarchy |
| `risk.py` | Uniqueness, k-anonymity, prosecutor risk, l-diversity, re-identification test (exact, or tolerant: age ± years and time windows) |
| `search.py` | Greedy search with utility caps; skips no-op levels; suppression move capped at 5 % of rows |
| `utility.py` | Proxy inside the loop; RandomForest task utility (split on the original data before transformation); aggregate-query error, TVD, Wasserstein |
| `text_pii.py` | Presidio + spaCy (English and Finnish) + Finnish/English regexes + dataset dictionaries → typed placeholders; kit span labels; precision/recall/F2 vs. ground truth |
| `dp.py` | Laplace counts and clipped means, error vs. ε |
| `kit.py` | Starter-kit scenarios, table recognition, task → column requirements, role/label mapping |
| `report.py`, `llm.py`, `cli.py` | Reports; optional Claude helper (role suggestions, plain-language summary from the JSON report only); CLI |

No LLM ever makes a transformation decision; every run is deterministic for a given seed.

## Limitations

Risk is measured against the modelled threat model only (ours and the kit's public attack specs); k-anonymity and
re-identification rates are empirical, not a formal guarantee like DP; greedy search is not globally optimal;
free-text recall is below 100 % (98–99 % on the kit's template-generated text, likely lower on real text);
pseudonyms are linkable across releases that share a secret.
The report lists these for every run. See `../KIT_SUMMARY.md` for the kit analysis and open gaps.
