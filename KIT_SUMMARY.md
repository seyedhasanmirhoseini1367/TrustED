# Starter kit summary and gap analysis

Kit: `TRUSTED_PETS_PROVING_GROUND_2026` (the official starter kit, v1.0.0, 4 Nov 2026, 260 files, sha256 per file in `MANIFEST.json`); paths below are relative to its root.
Our tool: `pets_transformer/` (spec `SPEC.md`), both in this repository.
Measurements below were made on 7 Nov 2026 by running the kit's `pets-eval` 1.0.0 on our engine's output (in a temporary folder; no kit file changed).

## 1. Purpose and rules

- Synthetic data plus self-assessment tools for the TrustED PETs Transformer challenge. 9 labelled domains + 1 unlabelled `unseen` dataset, all from one synthetic population of 100,000 people in Southwest Finland; the same people recur across domains, so cross-dataset linkage is a real risk (`START_HERE.md`).
- Demo must show the 5 brief items; the kit maps them to: `pets-eval detect` (1), `docs/PET_PLAYBOOK.md` (2), protected folder (3), `pets-eval scorecard` (4), `docs/SCENARIOS.md` + residual risks (5).
- Rules: only kit/public/approved data; **never publish `rowmap_*.csv`**; "anonymised"/"differentially private" claims need method + parameters (ε, δ, unit of privacy, contribution bound); read CSVs with `dtype=str`.
- Judging = official Since AI process. `pets-eval` scores are self-assessment only, but judges expect its before/after evidence (`docs/EVALUATION.md`).

## 2. Folders

| Path | Content | Role |
|---|---|---|
| `data/<domain>/` | CSV tables + `README.md` per domain | input to protect |
| `metadata/field_catalog.csv` | 475 columns: `dataset,table,column,privacy_role,data_kind,subtype,art9_special_category,description,traps` | **ground truth for Detect** |
| `metadata/pii_spans/<domain>__<table>__<column>.jsonl` | 10 files, `{"record_id", "spans":[{start,end,label,text}]}` | ground truth for free-text redaction |
| `metadata/scenarios.json` | 27 sharing scenarios (3 per domain) | recipient, threat model, required utility tasks, suggested targets |
| `metadata/attack_specs.json` | 16 linkage attacks + 2 trajectory-unicity specs | config of the scorecard's attacker |
| `metadata/dp_query_workload_health.json` | 254 DP queries on `ed_visits` (answers as `id,value`) | DP task H-DP |
| `metadata/taxonomy.md`, `pii_span_guidelines.md` | label vocabularies, span rules | |
| `adversary/*.csv` | 13 files of attacker background knowledge (**data, not attack code**) | e.g. `public_directory.csv` 43,528 rows (name, street_address, postcode, city, phone) |
| `truth/<dataset>__<table>.csv`, `truth/adversary__<file>.csv` | `record_id,person_id` / `adv_row_id,person_id` | who is who; lets attacks be scored exactly |
| `holdout/<domain>/` | non-member rows with the original schema (e.g. `holdout/health/ed_visits.csv`, 2,514 rows) | membership inference for synthetic data |
| `toolkit/pets_eval/` | Python package + `pets-eval` CLI | all scoring |
| `baselines/` | `protect.py --level b0|b2|b2k`, `b1_detect_rules.py`, `b3_dp_health.py`, `b4_synth_health.py`, `run_all.py` | reference solutions (they read the catalog instead of detecting) |
| `results/` | `b0|b2|b2k|b4/<domain>/scorecard.{json,html}`, `b1_field_labels.csv`, `benchmark.json` | baseline outputs = format examples |
| `examples/quickstart.py` | end-to-end protect + rowmap + scorecard for HEALTH-S1 | **reference for the output format** |
| `tests/` | `test_package.py` (checksums, catalog coverage, spans), `test_toolkit.py` (metrics, attacks, CLI) | |
| `docs/` | DATA_DICTIONARY, SCENARIOS, THREAT_MODEL, PET_PLAYBOOK, EVALUATION, BENCHMARK, GLOSSARY, LIMITATIONS | |

## 3. data/ per domain (rows from line counts; key columns only; examples are synthetic)

| Domain / table | Rows × cols | Notable columns (free text **FT**, dates **D**, locations **L**, technical IDs **T**) |
|---|---|---|
| health/ed_visits | 46,398 × 29 | patient_hetu, patient_name, date_of_birth, sex, phone, home_address, postcode, municipality (L), arrival_ts/discharge_ts (D), chief_complaint (**FT**, short Finnish), icd10_primary/secondary, vitals, attending_physician_id (T), readmit_30d |
| health/lab_results | 130,341 × 9 | visit_id (join key), test_code, value, abnormal_flag, sampled_ts |
| health/clinical_notes | 149,750 × 8 | visit_id, author_staff_id, author_name, note_text (**FT** Finnish) |
| health/published_weekly_stats | 30,169 × 7 | week, municipality, age_band, sex, diagnosis_group, visits (small cells: "what NOT to publish") |
| mobility/taps · cards · bike_trips · stops · bike_stations | 942,781 · 22,619 · 57,831 · 602 · 75 | card_id (T), ts, stop_id; holder_name/birth_date/postcode; user_account (email), start/end lat-lon (L) |
| smart_city/wifi_probes · anpr_parking · footfall_15min · sensors | 585,525 · 48,676 · 26,880 · 20 | mac (T), ssid_probed (may contain surnames); plate, payer_phone; device_count |
| energy/meter_readings_hourly · customers · weather_hourly | 1,635,120 · 757 · 2,160 | metering_point_id (T), ts, kwh; hetu, medical_device_priority (Art. 9), dso_notes (**FT**) |
| finance/card_transactions · transfers · customers · cards | 398,408 · 86,590 · 6,315 · 6,969 | card_pan, ts, amount, merchant, pos_lat/lon, device_id, ip_address, is_fraud; from/to_iban, names, message (**FT**) |
| cyber/proxy_logs · dns_logs · signin_logs · email_gateway · vpn_logs · edr_alerts · identity · incident_tickets · geoip_ranges | 137,024 · 75,606 · 41,252 · 30,254 · 3,804 · 448 · 994 · 315 · 16 | src_ip, upn, url (secrets), hostname, user_agent, attack_label/campaign_id; description (**FT**) |
| education/course_grades · absences · matriculation_results · students · counsellor_notes | 152,026 · 19,561 · 12,649 · 5,161 · 3,820 | student_id (T in child tables), hetu, birth_date, guardian names/phones, special_support; note_text (**FT**) |
| public_admin/service_requests · building_permits | 31,653 · 2,731 | description/response_text (**FT**, neighbours named), reporter_*, lat/lon; applicant_hetu, property_id |
| customer_service/turns · conversations · customers | 163,302 · 11,943 · 9,591 | text (**FT**, spoken numbers, spelled emails, passwords; hidden eval split via `split`), agent_note (**FT**); hetu, iban |
| unseen/export_2025_q4 | 2,818 × 26 | headers `c01..c26`; mixed formats: `200278907W` (HETU without separator), `20.2.1978`, `FI38 9164 …` IBAN, `3 987,96` (Finnish decimals), SHA-1 hash, Finnish address with postcode+city in one field |

Example row (health/ed_visits, synthetic): `ED25-000002, 170292-962B, "Mäntyranta Iida Josefiina", 1992-02-17, F, +358508595635, "Kanervarinne 32 A 2", 20740, Turku, 2025-01-01 00:00:00, walk-in, …, "raskaus, vuotoa ja alavatsakipua", O20.0, …`.
Most tables exceed our 50,000-row limit (we now sample 50k rows on upload).

## 4. metadata/ and truth/ labels

- Column roles (7): `direct_identifier` 76, `quasi_identifier` 128, `sensitive_attribute` 40, `technical_identifier` 38, `free_text` 15, `record_key` 39, `non_sensitive` 139. Plus `data_kind`, 50 `subtype`s (e.g. `NATIONAL_ID`, `BIRTH_DATE`, `MUNICIPALITY`, `HEALTH_DATA`), an `art9_special_category` flag and per-column `traps`.
- 20 span labels: `PERSON, NATIONAL_ID, EMAIL, PHONE, ADDRESS, LOCATION, DATE, AGE, IBAN, PAYMENT_CARD, ACCOUNT_NUMBER, IP_ADDRESS, LICENSE_PLATE, ORG, URL, USERNAME, SECRET, HEALTH, DEVICE_ID, PROPERTY_ID`. Spans are keyed by **`record_id`** (the table's key), not row position.
- No sensitivity *levels*; sensitivity = role + Art. 9 flag. `truth/` gives only record → synthetic person id.

## 5. Attacks (in `toolkit/pets_eval/attacks.py`, configured by `metadata/attack_specs.json`)

- 16 background-knowledge linkage attacks, e.g. `news_clips_health → health.ed_visits` (time window, place, age ±3, sex, name), `public_directory → ed_visits` (name, postcode, city), `receipt_photos_ocr → finance.card_transactions`, `graduation_lists → education.students`. Each match type (`exact`, `time` with window, `age` with tolerance, `postcode`, `geo`, `name`) works on generalised values; QIs whose protected column is missing are skipped.
- Metric: worst re-identification rate (unique + correct match, via `truth/`) or attribute-inference advantage on the `sensitive` column. Also: hash dictionary attack on unsalted MD5/SHA-1/SHA-256, HETU enumeration, trajectory unicity (4 (stop, hour) points for taps; 3 points for bike trips).
- Run: through `pets-eval scorecard` (no separate command). Organisers can also run a **hidden stronger attacker per domain** on submitted tables + row maps.

## 6. Utility evaluation and baselines

- 6–11 tasks per dataset (`pets-eval tasks`). Health: H-U1 weekly ILI by age band, H-U2 median LOS by triage, H-U3 admission model AUC ratio (holdout), H-U4 diagnosis mix top-8 municipalities, H-U5 vital distributions, H-U6 revisit rate by age band, H-U7 CRP vs temperature (needs `lab_results` join), H-U8 hour-of-day arrival curve, H-DP DP workload.
- Each scenario lists its required tasks (HEALTH-S1: H-U2, H-U4, H-U6, H-U7, H-U8). A required task that cannot be answered scores 0.
- Privacy = `1 − (0.5 × worst risk + 0.5 × mean risk)`; utility = mean task score; composite = weighted harmonic mean (privacy 30, utility 25).
- Baselines (from `docs/BENCHMARK.md`): health privacy B0/B2/B2-k 0.28/0.36/0.52; HEALTH-S1 utility 1.00/0.79/0.40. B1 detection macro-F1 0.65. B3 DP: ε=1 → utility 0.69. B4 synthetic ED visits: privacy 1.00, utility 0.38.

## 7. toolkit/ and tests/

- Install: `pip install -e ./toolkit` (Python ≥ 3.10, numpy, pandas, scipy, scikit-learn). CLI `pets-eval` with `scenarios`, `tasks`, `check`, `detect`, `text`, `scorecard`.
- Python API: `pets_eval.privacy` (k-anonymity, l-diversity, t-closeness, uniqueness, small cells, differencing, DCR/NNDR, membership inference, trajectory unicity), `pets_eval.attacks`, `pets_eval.text`, `pets_eval.detect`, `pets_eval.parse` (understands `30-39`, `90+`, `<5`, `201**`, `2025-W12`, `2025-03`), `pets_eval.utility.<domain>`, `pets_eval.io`.
- Tests: `python3 -m pytest -q tests/` from the kit root (package integrity + toolkit). Not run here because they write caches into the kit folder.

## 8. Expected output / submission format

- Protected folder per dataset: **same table file names** (`ed_visits.csv`); **original column names** for kept columns; generalised values in the accepted notations; `rowmap_<table>.csv` (`protected_row` 0-based, `original_record_id`) for each released microdata table, for scoring only.
- No row map and no original ids → table is scored as synthetic data (DCR + membership inference vs `holdout/`).
- DP: `dp_answers.csv` (`id,value`) + mechanism description (ε, δ, unit of privacy, contribution bound).
- Detection: CSV `dataset,table,column,privacy_role,subtype`. Spans: JSONL `record_id, spans[start,end,label]`.
- Optional organiser checks: repo folder `pets_submission/` with `unseen_field_labels.csv`, `customer_service_eval_redacted.csv` (`turn_id,text`) or `customer_service_eval_spans.jsonl`, and one subfolder per dataset of protected tables + row maps.
- **Unclear:** `data/unseen/README.md` asks for `column,privacy_role,subtype`, while `docs/EVALUATION.md` names the file `unseen_field_labels.csv` without columns; adding `dataset,table` as well should satisfy both (ask on Discord).

## 9. Organiser evaluation

Three optional checks you cannot overfit: hidden attackers per domain (data broker, HR, colleagues, merchant, municipal insiders), private labels of `unseen`, hidden gold spans of the `customer_service` eval split. Requested during the event via `pets_submission/`.

## 10. Commands (from the kit root)

```bash
pip install -e ./toolkit
python3 examples/quickstart.py                        # writes results/quickstart/health/scorecard.html
pets-eval scenarios --dataset health ; pets-eval tasks --dataset health
pets-eval detect --pred my_labels.csv
pets-eval text --gold metadata/pii_spans/health__clinical_notes__note_text.jsonl --protected my_out/health/clinical_notes.csv --text-col note_text --id-col note_id
pets-eval check my_out/health
pets-eval scorecard --original data/health --protected my_out/health --scenario HEALTH-S1   # adversary runs inside
python3 baselines/run_all.py --domains health --quick
python3 -m pytest -q tests/
```

## 11. Gap analysis: our tool vs the kit (measured)

**Measured with the kit's tools** (our engine, HEALTH-S1-like scenario: attacker knows DOB, sex, municipality, arrival time; target readmit_30d):

| Check | Our own report | Kit `pets-eval` | Baselines |
|---|---|---|---|
| Detect, all 9 domains | — | macro-F1 **0.63**, accuracy 0.68 | B1 0.65 |
| Scorecard privacy | linkage 100 % → 0 %, k=5 | **0.66** (k<5 on catalogued QIs 53.8 %; linkage 0 %; direct ids 0 %) | B2-k 0.52 |
| Scorecard utility (HEALTH-S1) | AUC 97.6 % retained | **0.20** (H-U2 1.00, H-U4 0.00, H-U6/H-U7/H-U8 not answerable) | B2 0.79, B2-k 0.40 |
| Free-text redaction (kit Finnish text, 3,000 annotated records each) | 100 % on our sample | clinical_notes **57 %**, customer_service turns **69 %**, service_requests **77 %** (values not leaked) | — |

**Status after the 8 Nov changes:** items 1–3 are implemented and item 4 mostly (kit scenarios use every catalogued
QI as attacker knowledge). Re-measured with `pets-eval scorecard` on `ed_visits` + `lab_results`, HEALTH-S1:
privacy **1.00**, utility **0.78** (H-U2 1.00, H-U4 0.95, H-U6 0.96, H-U7 0.98, H-U8 sacrificed), composite **0.89**;
all risk components 0 %.

**Status after the second round (8 Nov):** items 5, 6, 7 and 8 are implemented; 11 is documented (web UI samples
50,000 rows; the CLI processes full tables).
- 5 Free text: Presidio with English + Finnish spaCy models, a Finnish/spoken-number regex layer and the kit's 20 span
  labels. `pets-eval text`, 3,000 annotated records per table, recall before → after: clinical_notes 89.9 → 99.8 %,
  turns 78.5 → 98.7 %, service_requests 81.0 → 98.9 %; precision 0.33/0.68/0.66 → 0.95/0.95/0.90; a held-out slice
  (records 10,000–13,000) gives the same level.
- 6 Detection: `pets-eval detect` macro-F1 0.63 → **0.862** (B1 0.646), using names, values and table structure only.
  `pets_submission/unseen_field_labels.csv` regenerated (HETU with/without separator, Finnish numbers and dates,
  addresses with postcode, hashes, plates).
- 7 Gold spans keyed by `record_id` are accepted (`--spans file.jsonl --spans-id-col note_id`).
- 8 The kit's attack specs run with tolerant matching on every kit run and are reported per attack; the exact attacker
  stays the search objective because it is provably at least as strong on the same columns.
- HEALTH-S1 re-measured on `ed_visits` + `lab_results` + `clinical_notes`: privacy 1.00, utility 0.78, composite 0.89
  (no regression); `clinical_notes` free-text leakage 0 %.
Still open: 10 (DP answers file for H-DP), task requirements for domains other than health and finance.

**Conflicts / what must change (priority order):**

1. **Output format (blocks all kit scoring).** We write `<name>_protected.csv` and no row map. Must: keep the original table file name, write `rowmap_<table>.csv` (internal export only, never public), keep column names. Files: `engine/cli.py` (`cmd_run`), `webapp/wizard/views.py` (`download`, `_run_job`).
2. **Generalised value notation.** Our DOB levels emit `1990-1999` / `1990-1994`; the kit parses those as numbers (age ≈ 1994), not dates, so age-based tasks (H-U1, H-U6) and the age attack break. Must: emit an `age`/`age_band` column (`30-39`, `80+`) instead of a generalised DOB, as in `examples/quickstart.py`. Timestamps: we emit `2025-Q1`, which the kit does not list as accepted; and H-U8 needs a time of day. File: `engine/ladders.py` (DateLadder/TimestampLadder labels).
3. **Utility is scenario-blind.** Our search optimises a generic proxy and drops what the scenario's tasks need: municipality merged to "Other" (H-U4 → 0), arrival time coarsened to quarter (H-U8 n/a), `lab_results` not released (H-U7). Must: load `metadata/scenarios.json`, mark the columns each required task needs as key columns or set a maximum level per column; support releasing the joined tables (multi-table) with consistent pseudonyms. Files: `engine/search.py`, `engine/utility.py`, `engine/roles.py` (Scenario), scenario UI.
4. **k is measured only on the attacker's QIs.** The kit counts k on every catalogued QI we keep (postcode, discharge_ts, …): 53.8 % of our rows have k<5. Must: add an option "k over all QIs" (or include all yellow QIs in the attacker set by default for public/contractual recipients). File: `engine/gates.py` (`attacker_known`), `engine/search.py`.
5. **Free-text redaction for Finnish.** Missing: ages (`54 v`, AGE 2 %), organisations, Finnish date formats, names not in a name column, licence plates, health mentions; `chief_complaint` is not even detected as free text. Must: add Finnish regexes (age, plates, property ids, URLs), a cross-table name dictionary, a lower free-text threshold for short multi-word text, and map our labels to the kit's 20 span labels. Files: `engine/text_pii.py`, `engine/profiling.py`.
6. **Role vocabulary and detection rules.** Our 11 roles must map to the kit's 7 roles + subtypes for `pets-eval detect` and `unseen_field_labels.csv`. Measured errors: person foreign keys (`student_id`, `customer_id`, `visit_id` in child tables) → kit says `technical_identifier`, we say `record_key`; vitals, LOS, disposition, payment flags, kwh → kit says `sensitive_attribute`, we say non-sensitive; card/plate/device IDs sometimes `direct_identifier`. `unseen` needs Finnish number formats (`3 987,96`), HETU without separator and hash detection. Files: `engine/roles.py` (mapping), `engine/profiling.py`, new export in `engine/cli.py`.
7. **Ground-truth span format.** Kit spans are keyed by `record_id`; our loader expects row positions (`engine/text_pii.py` `load_gold`). Must accept `record_id` + key column.
8. **Our attacker vs the kit's attacker.** Ours is a self-made exact-match attack on a 30 % sample; the kit's uses real background files with tolerant matching (time windows, age ±3, names). Not a conflict, but the demo should show the kit's number next to ours. Option: call `pets_eval` scorecard from our pipeline when the input is a kit table (dependency only, no duplication).
9. **Duplication (keep, but label clearly).** Our k/l-diversity, linkage, DP and utility metrics duplicate `pets_eval.privacy/attacks/utility`. Keep ours for unseen/uploaded data (the kit's tasks are table-specific); report the kit's scorecard as the official self-assessment where available.
10. **DP output.** Our DP panel does not produce `dp_answers.csv` for the 254-query health workload (H-DP, HEALTH-S2). Optional.
11. **Row limit.** Most kit tables exceed 50k rows; our upload now samples 50k rows. The kit's scorecard expects the full table: sampled releases score rows that were never released as "withheld". Must: raise the limit for CLI runs (engine handles 46k rows in ~30 s) or document the sample.

Not verified: the hidden-attacker and unseen scoring (organiser-only); `pets-eval text --pred` with our spans (label mapping not done yet).
