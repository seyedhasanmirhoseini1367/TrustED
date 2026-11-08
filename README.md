# TrustED PETs Transformer

*Protect the data, keep its value.* — Since AI Hackathon 2026 · TrustED challenge
(inspired by the EU Horizon Europe project TrustED, grant agreement No. 101168467).

A tool that takes a tabular dataset, detects privacy risks, applies the **minimum sufficient** privacy-enhancing
transformations for a defined sharing scenario, and measures that re-identification risk dropped while analytical
value was kept. A re-identification test runs inside the search: the engine keeps applying the generalisation step
with the best risk reduction per unit of utility loss until the scenario's target is met.

**What it does**

1. **Detect** — classifies every column (direct identifier, quasi-identifier, sensitive attribute, technical
   identifier, free text, …) from names, values and table structure; finds PII inside free text (English and Finnish).
2. **Select** — a red / yellow / green gate per column: fixed transforms (drop, keyed HMAC pseudonym, IP truncation),
   a searched generalisation level, or unchanged.
3. **Transform** — generalisation ladders (dates, birth dates → age bands, postcodes, codes, numbers, coordinates),
   record suppression, typed placeholders in free text, differential privacy for aggregate queries.
4. **Demonstrate** — re-identification rate and k before/after, model and aggregate utility before/after, the
   privacy/utility trade-off curve, a decision table with numeric reasons, and residual risks.

**Results on the official starter kit** (self-assessment with the kit's `pets-eval`, not the official judging score)

| | privacy | utility | composite |
|---|---|---|---|
| HEALTH-S1 (`ed_visits` + `lab_results` + `clinical_notes`) | **1.00** | **0.78** | **0.89** |
| kit baselines B2 / B2-k | 0.36 / 0.52 | 0.79 / 0.40 | — |

- Privacy 1.00 is measured against the kit's 16 public attacks; the organisers' hidden attackers are not measured.
- Free-text redaction on the kit's annotated text: recall 98–99 %, precision 0.90–0.95.
- Field detection: macro-F1 0.86 (kit baseline B1: 0.65).

## Quick start

```bash
cd pets_transformer
python -m venv .venv && .venv/Scripts/activate        # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download fi_core_news_md
python -m pytest                                       # all tests should pass
python manage.py migrate
python manage.py runserver                             # http://127.0.0.1:8000 → "Load sample dataset"
```

Full documentation — the starter-kit workflow, the CLI, how each number was measured, and limitations — is in
[`pets_transformer/README.md`](pets_transformer/README.md).

## Repository

| Path | Content |
|---|---|
| `pets_transformer/engine/` | The engine: pure Python, deterministic, no LLM in any transformation decision |
| `pets_transformer/webapp/` | Django web wizard: Upload → Profile → Scenario → Results → Export |
| `pets_transformer/data/generate_sample.py` | Synthetic sample dataset generator with ground-truth PII spans |
| `pets_transformer/tests/` | Test suite |
| `pets_submission/` | Our column labels for the kit's `unseen` dataset, for optional scoring by the organisers |

Use only synthetic or approved data. The bundled kit metadata (`engine/kit_data/`) is © Since AI ry, CC BY 4.0.
