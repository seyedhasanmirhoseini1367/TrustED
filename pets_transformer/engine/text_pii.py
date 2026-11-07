"""Free-text PII redaction (spec §5.8): Presidio + spaCy with custom regex recognizers.

Falls back to a regex-only engine if Presidio/spaCy is not installed. Entities are replaced with typed
placeholders. If ground-truth spans exist, precision/recall per entity type are reported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

PLACEHOLDER = {"PERSON": "[PERSON]", "PHONE": "[PHONE]", "EMAIL": "[EMAIL]", "LOCATION": "[LOCATION]",
               "DATE": "[DATE]", "ID": "[ID]"}
ENTITY_MAP = {
    "PERSON": "PERSON", "DICT_PERSON": "PERSON",
    "PHONE_NUMBER": "PHONE", "FI_PHONE": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "LOCATION": "LOCATION", "ADDRESS": "LOCATION", "FI_ADDRESS": "LOCATION",
    "DATE_TIME": "DATE", "FI_DATE": "DATE",
    "FI_HETU": "ID", "IBAN_CODE": "ID", "FI_IBAN": "ID", "US_SSN": "ID", "CREDIT_CARD": "ID",
    "IP_ADDRESS": "ID", "MEDICAL_LICENSE": "ID", "FI_PERSONAL_IDENTITY_CODE": "ID",
}
PATTERNS = {
    "FI_HETU": (r"\b\d{6}[-+A-FU-Y]\d{3}[0-9A-FHJ-NPR-Y]\b", 0.95),
    "FI_PHONE": (r"(?<![\w+])(?:\+358|00358|0)[\s-]?\d{1,3}(?:[\s-]?\d{2,4}){2,3}(?![\w])", 0.75),
    "FI_IBAN": (r"\bFI\d{2}(?: ?\d{4}){3} ?\d{2}\b", 0.9),
    "EMAIL_ADDRESS": (r"\b[\w.+'-]+@[\w-]+(?:\.[\w-]+)+\b", 0.95),
    "FI_DATE": (r"\b(?:\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b", 0.7),
    "FI_ADDRESS": (r"\b[A-ZÅÄÖ][a-zåäö]+(?:katu|tie|polku|kuja|ranta|väylä|kaari|puistikko)\s+\d+[A-Za-z]?(?:\s?[A-Z]\s?\d+)?\b", 0.8),
}
CAP_TOKEN = re.compile(r"\b[A-ZÅÄÖ][\w'-]+")
TITLE_NAME = r"\b(?:Dr|Mr|Mrs|Ms|Miss|Nurse|Prof)\.?\s+([A-ZÅÄÖ][a-zåäö]+(?:\s+[A-ZÅÄÖ][a-zåäö]+)?)"


@dataclass
class Span:
    start: int
    end: int
    label: str
    score: float = 1.0

    def to_dict(self):
        return {"start": self.start, "end": self.end, "label": self.label, "score": round(self.score, 3)}


PRIORITY = {"ID": 3, "EMAIL": 3, "PHONE": 2}


def _resolve(spans: list[Span]) -> list[Span]:
    """Merge overlapping spans into their union. The label comes from a pattern-validated type (ID/EMAIL/PHONE beat
    NER guesses such as DATE), otherwise from the longer span, then the higher score."""
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start), -s.score))
    out: list[Span] = []
    for s in spans:
        if out and s.start < out[-1].end:
            last = out[-1]
            rank = lambda x: (PRIORITY.get(x.label, 0), x.end - x.start, x.score)  # noqa: E731
            win = s if rank(s) > rank(last) else last
            out[-1] = Span(last.start, max(last.end, s.end), win.label, win.score)
        else:
            out.append(s)
    return out


def apply_spans(text: str, spans: list[Span]) -> str:
    for s in sorted(spans, key=lambda s: -s.start):
        text = text[:s.start] + PLACEHOLDER.get(s.label, f"[{s.label}]") + text[s.end:]
    return text


class Redactor:
    def __init__(self, extra_names: list[str] | None = None, use_presidio: bool = True, score_threshold: float = 0.35,
                 places: list[str] | None = None):
        self.names = sorted({n for n in (extra_names or []) if len(n) >= 3 and n[:1].isupper()}, key=len, reverse=True)
        self.places = sorted({p for p in (places or []) if len(p) >= 3 and p[:1].isupper()}, key=len, reverse=True)
        self.threshold = score_threshold
        self.engine = "regex + dataset name/place dictionaries"
        self._analyzer = None
        self._batch = None
        if use_presidio:
            self._init_presidio()

    def _init_presidio(self) -> None:
        try:
            import spacy.util
            from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, Pattern, PatternRecognizer
            from presidio_analyzer.nlp_engine import NlpEngineProvider
        except ImportError:
            return
        model = next((m for m in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm") if spacy.util.is_package(m)), None)
        if model is None:
            return
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": model}]})
        analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
        for entity, (rx, score) in PATTERNS.items():
            analyzer.registry.add_recognizer(PatternRecognizer(supported_entity=entity, name=f"custom_{entity}",
                                                               patterns=[Pattern(entity, rx, score)]))
        nlp = analyzer.nlp_engine.nlp["en"]
        # only NER is needed; in the en_core_web_* pipelines NER has its own embedding layer
        for pipe in ("tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "senter"):
            if pipe in nlp.pipe_names and pipe != "ner":
                nlp.disable_pipe(pipe)
        self._analyzer = analyzer
        self._batch = BatchAnalyzerEngine(analyzer_engine=analyzer)
        # PHONE_NUMBER (libphonenumber) is slow and misses Finnish formats; FI_PHONE regex covers phones
        self._entities = sorted(set(ENTITY_MAP) - {"DICT_PERSON", "PHONE_NUMBER"})
        self.engine = f"Presidio + spaCy {model} + custom regex + dataset name/place dictionaries"

    # ------------------------------------------------------------ detection
    def _regex_spans(self, text: str) -> list[Span]:
        spans = []
        for entity, (rx, score) in PATTERNS.items():
            for m in re.finditer(rx, text):
                spans.append(Span(m.start(), m.end(), ENTITY_MAP[entity], score))
        for m in re.finditer(TITLE_NAME, text):
            spans.append(Span(m.start(1), m.end(1), "PERSON", 0.7))
        return spans

    def _dict_spans(self, text: str) -> list[Span]:
        """Capitalised tokens found in the dataset's own name columns (adjacent hits merged), plus its place names."""
        if not hasattr(self, "_name_set"):
            self._name_set = {t for n in self.names for t in n.split()}
        spans: list[Span] = []
        for m in CAP_TOKEN.finditer(text) if self._name_set else ():
            if m.group() in self._name_set:
                if spans and text[spans[-1].end:m.start()] == " ":
                    spans[-1] = Span(spans[-1].start, m.end(), "PERSON", 0.9)
                else:
                    spans.append(Span(m.start(), m.end(), "PERSON", 0.9))
        if self.places:
            if not hasattr(self, "_place_rx"):
                self._place_rx = re.compile(r"\b(?:" + "|".join(map(re.escape, self.places)) + r")\b")
            spans += [Span(m.start(), m.end(), "LOCATION", 0.92) for m in self._place_rx.finditer(text)]
        return spans

    def analyze(self, texts: list[str]) -> list[list[Span]]:
        uniq = list(dict.fromkeys(texts))
        found: dict[str, list[Span]] = {}
        if self._batch is not None:
            results = self._batch.analyze_iterator(uniq, language="en", entities=self._entities,
                                                   score_threshold=self.threshold, batch_size=64)
            for text, res in zip(uniq, results):
                found[text] = [Span(r.start, r.end, ENTITY_MAP[r.entity_type], r.score) for r in res
                               if r.entity_type in ENTITY_MAP]
        for text in uniq:
            spans = found.get(text, []) + self._regex_spans(text) + self._dict_spans(text)
            found[text] = _resolve(spans)
        return [found[t] for t in texts]

    def redact(self, s: pd.Series) -> tuple[pd.Series, dict[int, list[Span]]]:
        mask = s.notna() & (s.astype("string").str.strip() != "")
        texts = s[mask].astype(str).tolist()
        spans = self.analyze(texts)
        out = s.astype("string").copy()
        by_row: dict[int, list[Span]] = {}
        for idx, text, sp in zip(s[mask].index, texts, spans):
            out.loc[idx] = apply_spans(text, sp)
            by_row[int(idx)] = sp
        return out, by_row


# ---------------------------------------------------------------- evaluation
def evaluate(gold: dict[int, list[dict]], pred: dict[int, list[Span]], original: pd.Series, redacted: pd.Series) -> dict:
    """Span precision/recall per entity type (overlap match, same type) and verbatim-leak rate."""
    labels = sorted({g["label"] for spans in gold.values() for g in spans} | {p.label for spans in pred.values() for p in spans})
    stats = {lab: {"gold": 0, "pred": 0, "tp_gold": 0, "tp_pred": 0, "leaked": 0} for lab in labels}
    any_hit = total_gold = leaked = 0
    rows = set(gold) | set(pred)
    for r in rows:
        g_spans = gold.get(r, [])
        p_spans = pred.get(r, [])
        text = str(original.get(r, "")) if r in original.index else ""
        red = str(redacted.get(r, "")) if r in redacted.index else ""
        for g in g_spans:
            st = stats[g["label"]]
            st["gold"] += 1
            total_gold += 1
            if any(p.label == g["label"] and p.start < g["end"] and g["start"] < p.end for p in p_spans):
                st["tp_gold"] += 1
            if any(p.start < g["end"] and g["start"] < p.end for p in p_spans):
                any_hit += 1
            value = text[g["start"]:g["end"]]
            if value and value in red:
                st["leaked"] += 1
                leaked += 1
        for p in p_spans:
            st = stats[p.label]
            st["pred"] += 1
            if any(p.label == g["label"] and p.start < g["end"] and g["start"] < p.end for g in g_spans):
                st["tp_pred"] += 1
    per = {}
    for lab, st in stats.items():
        rec = st["tp_gold"] / st["gold"] if st["gold"] else None
        prec = st["tp_pred"] / st["pred"] if st["pred"] else None
        f2 = None
        if rec is not None and prec is not None and (prec + rec) > 0:
            f2 = 5 * prec * rec / (4 * prec + rec)
        per[lab] = {"gold": st["gold"], "predicted": st["pred"], "precision": prec, "recall": rec, "f2": f2,
                    "leak_rate": st["leaked"] / st["gold"] if st["gold"] else None}
    return {
        "per_label": per,
        "recall_any_label": any_hit / total_gold if total_gold else None,
        "leak_rate": leaked / total_gold if total_gold else None,
        "gold_spans": total_gold,
        "records": len(gold),
    }


def load_gold(obj: dict | list) -> tuple[str | None, dict[int, list[dict]]]:
    """Accepts {"text_column": ..., "records": [{"row": i, "spans": [...]}]} or a bare list of records."""
    if isinstance(obj, dict):
        col, recs = obj.get("text_column"), obj.get("records", [])
    else:
        col, recs = None, obj
    return col, {int(r["row"]): r.get("spans", []) for r in recs}
