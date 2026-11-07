"""Free-text PII redaction (spec §5.8): Presidio + spaCy (English and Finnish) with custom regex recognizers.

Labels follow the starter kit's 20 span labels (metadata/taxonomy.md):
PERSON, NATIONAL_ID, EMAIL, PHONE, ADDRESS, LOCATION, DATE, AGE, IBAN, PAYMENT_CARD, ACCOUNT_NUMBER, IP_ADDRESS,
LICENSE_PLATE, ORG, URL, USERNAME, SECRET, HEALTH, DEVICE_ID, PROPERTY_ID.

Pipeline per text: detect the language (fi / sv / en) -> Presidio NER with the matching spaCy model (Finnish model for
Finnish and Swedish text, English otherwise) -> language-independent regex layer (Finnish formats, spoken and spelled
numbers and e-mails, plates, property ids, ...) -> dataset dictionaries (names and places from the table's own columns)
-> overlapping spans merged -> typed placeholders such as [PERSON]. Without Presidio/spaCy the regex and dictionary
layers still run. Ground-truth spans can be keyed by row position or by record id (the kit's format).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

KIT_LABELS = ("PERSON", "NATIONAL_ID", "EMAIL", "PHONE", "ADDRESS", "LOCATION", "DATE", "AGE", "IBAN", "PAYMENT_CARD",
              "ACCOUNT_NUMBER", "IP_ADDRESS", "LICENSE_PLATE", "ORG", "URL", "USERNAME", "SECRET", "HEALTH",
              "DEVICE_ID", "PROPERTY_ID")
LABEL_ALIASES = {"ID": "NATIONAL_ID"}  # labels used by older ground-truth files
# Presidio entity -> kit label
ENTITY_MAP = {"PERSON": "PERSON", "LOCATION": "LOCATION", "ORGANIZATION": "ORG", "DATE_TIME": "DATE",
              "EMAIL_ADDRESS": "EMAIL", "IP_ADDRESS": "IP_ADDRESS", "CREDIT_CARD": "PAYMENT_CARD", "IBAN_CODE": "IBAN",
              "URL": "URL", "US_SSN": "NATIONAL_ID", "PHONE_NUMBER": "PHONE"}
NER_ENTITIES = {"en": ["PERSON", "LOCATION", "ORGANIZATION", "DATE_TIME", "EMAIL_ADDRESS", "IP_ADDRESS", "CREDIT_CARD",
                       "IBAN_CODE", "URL"],
                "fi": ["PERSON", "LOCATION", "ORGANIZATION", "DATE_TIME"]}
# pattern-validated labels win over NER guesses when spans overlap
PRIORITY = {"NATIONAL_ID": 5, "EMAIL": 5, "IBAN": 5, "PAYMENT_CARD": 4, "URL": 4, "IP_ADDRESS": 4, "PROPERTY_ID": 4,
            "LICENSE_PLATE": 4, "DEVICE_ID": 4, "PHONE": 3, "SECRET": 3, "USERNAME": 3, "AGE": 3, "ACCOUNT_NUMBER": 2,
            "ADDRESS": 2}

I = re.IGNORECASE
_NUMW = (r"(?:nolla|yksi|kaksi|kolme|neljä|viisi|kuusi|seitsemän|kahdeksan|yhdeksän|kymmenen|"
         r"noll|ett|en|två|tre|fyra|fem|sex|sju|åtta|nio|tio|"
         r"zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|\d{1,4})")
# number words that are never ordinary words (no "en", "ett", "one", "oh"): safe in sequences of two
_NUMW_STRICT = (r"(?:nolla|yksi|kaksi|kolme|neljä|viisi|kuusi|seitsemän|kahdeksan|yhdeksän|noll|två|fyra|fem|sju|åtta|"
                r"nio|zero|three|four|five|seven|eight|nine)")
_DOTW = r"(?:dot|piste|punkt|pist)"
_ATW = r"(?:at|ät|ätt|snabel[\s-]?a|kissanhäntä|miukumauku)"
_SEPW = r"(?:dot|piste|punkt|alaviiva|underscore|understreck|viiva|dash|streck|bindestreck|väliviiva)"
_STREET = (r"(?:katu|tie|polku|kuja|kaari|rinne|mäki|raitti|väylä|ranta|puistikko|piha|kulma|tanhua|niitty|pelto|"
           r"vainio|aukio|tori|gatan|vägen|gränd|stigen|backen|torget|road|street)")
# stems, because Finnish inflection changes the end of the word (ostoskeskus -> ostoskeskuksessa)
_PLACE_NOUN = (r"(?:bussipysäk|pysäk|kierrätyspist|pysäköintialu|leikkipuist|uimaran|kauppakesk|ostoskesk|"
               r"rautatieasem|asem|tori|toril|puist|satam|koul|päiväkod|päiväkot|kirjast|terveysasem|uimahall|"
               r"jäähall|urheilukent|busshållplats|parkering|ishall)")
HEALTH_TERMS = (r"(?:diabete\w*|masennu\w*|depressi\w*|dementi\w*|muistisairau\w*|alzheimer\w*|aivoinfarkt\w*|"
                r"aivoverenvuo\w*|sydäninfarkt\w*|sydänkohtau\w*|syöpä\w*|syövä\w*|cancer|kreft|astma\w*|asthma|"
                r"epilepsi\w*|pyörätuoli\w*|wheelchair|rollaattori\w*|näkövamm\w*|sokea|kuulovamm\w*|kuuro|"
                r"huonokuntoi\w*|sairausloma\w*|sjukskriv\w*|sick leave|raskau\w*|raskaana|gravid\w*|pregnan\w*|"
                r"reuma\w*|dialyysi\w*|dialysis|leikkau\w*|surgery|hiv|ms-tauti\w*|parkinson\w*|skitsofreni\w*|"
                r"kaksisuuntai\w*|bipolar|ahdistu\w*|anxiety|ångest|paniikkihäiriö\w*|uupumu\w*|burn-?out|"
                r"cough|yskä\w*|hengenahdistu\w*|mielenterveys\w*|päihde\w*|alkoholi\w*|hoitokoti\w*|omaishoit\w*)")

MONTHS = (r"(?:tammi|helmi|maalis|huhti|touko|kesä|heinä|elo|syys|loka|marras|joulu)kuu\w*|januari|februari|mars|"
          r"april|maj|juni|juli|augusti|september|oktober|november|december|january|february|march|may|june|july|"
          r"august|october|maanantai\w*|tiistai\w*|keskiviikko\w*|torstai\w*|perjantai\w*|lauantai\w*|sunnuntai\w*|"
          r"monday|tuesday|wednesday|thursday|friday|saturday|sunday")
DATE_LIKE = re.compile(rf"\d{{1,2}}[./]\d{{1,2}}|\b(?:19|20)\d{{2}}\b|{MONTHS}", I)

# (label, compiled regex, group to redact, score). Order does not matter; overlaps are merged afterwards.
REGEXES: list[tuple[str, re.Pattern, int, float]] = [
    ("NATIONAL_ID", re.compile(r"\b\d{6}\s*(?:[-+A-FU-Y]|viiva|streck|dash|miinus|plus)\s*\d{3}\s*[0-9A-Y]\b", I), 0, 0.95),
    ("NATIONAL_ID", re.compile(r"(?<![\w-])-\d{3}[0-9a-y]\b", I), 0, 0.5),            # partial HETU end, e.g. "-947x"
    ("NATIONAL_ID", re.compile(r"\b(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])\d{2}\b"), 0, 0.5),  # birth-date part ddmmyy
    ("NATIONAL_ID", re.compile(r"\b[a-fu-y+-]\d{3}[0-9a-y]\b", I), 0, 0.5),          # century sign + individual part
    ("EMAIL", re.compile(r"\b[\w.+'-]+@[\w-]+(?:\.[\w-]+)+\b"), 0, 0.95),
    ("EMAIL", re.compile(rf"\b(?:\w+\s+{_SEPW}\s+)*\w+\s+{_ATW}\s+\w+(?:\s+{_DOTW}\s+\w+)+", I), 0, 0.9),
    ("URL", re.compile(r"\bhttps?://\S+|\bwww\.\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:example|fi|com|net|org)/\S+", I), 0, 0.9),
    ("IBAN", re.compile(r"\bFI\s?\d{2}(?:\s?\d{4}){3}\s?\d{2}\b", I), 0, 0.95),
    ("IBAN", re.compile(rf"\bf\s?i\s+{_NUMW}(?:[\s,]+{_NUMW}){{6,}}", I), 0, 0.9),
    ("PAYMENT_CARD", re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"), 0, 0.9),
    ("PAYMENT_CARD", re.compile(r"(?:päättyy|päättyvä|loppuu|ending(?: in)?|ends in|slutar på)\D{0,25}?(\d{4})\b", I), 1, 0.8),
    ("DEVICE_ID", re.compile(r"\b\d{5}\s?\d{5}\s?\d{5}\b|\b\d{15}\b"), 0, 0.85),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 0, 0.9),
    ("PROPERTY_ID", re.compile(r"\b\d{3}-\d{3}-\d{1,4}-\d{1,4}\b"), 0, 0.9),
    ("LICENSE_PLATE", re.compile(r"\b[A-ZÅÄÖ]{2,3}-\d{1,3}\b", I), 0, 0.8),
    ("PHONE", re.compile(r"(?<![\w+])(?:\+358|00358|0)[\s-]?\d{1,3}(?:[\s-]?\d{2,4}){2,3}(?![\w])"), 0, 0.8),
    ("PHONE", re.compile(rf"\b{_NUMW}(?:[\s,]+{_NUMW}){{3,}}\b", I), 0, 0.6),         # spoken digit sequences
    ("PHONE", re.compile(rf"\b{_NUMW_STRICT}(?:[\s,]+{_NUMW_STRICT})+\b", I), 0, 0.5),  # short unambiguous ones
    ("PHONE", re.compile(r"(?<![\d.,])\b0[1-9]\d\b(?![.,]\d)"), 0, 0.4),                 # operator prefix alone: 040
    ("ACCOUNT_NUMBER", re.compile(r"\b\d{3,4}(?:\s\d{3,4}){1,2}(?:\s\d{1,4})?\b|\b\d{7,12}\b"), 0, 0.5),
    ("DATE", re.compile(rf"\b\d{{1,2}}\.?\s+(?:{MONTHS})\b|\b(?:{MONTHS})\s+\d{{1,2}}\b", I), 0, 0.75),
    ("DATE", re.compile(r"\b\d{1,2}\.\d{1,2}\.(?:\d{2,4})?(?!\d)|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"), 0, 0.75),
    ("AGE", re.compile(r"\b\d{1,3}\s?(?:v\b|-?vuotia\w*|vuotta|vuoden ikäinen|-?år(?:ig\w*)?\b|years?[\s-]old|yo\b|y/o)", I), 0, 0.85),
    ("AGE", re.compile(r"(?:\bikä|\bålder|\baged?|\bage of)\s*:?\s*(\d{1,3})\b", I), 1, 0.8),
    ("AGE", re.compile(r"\((\d{1,3})\s?v?\)"), 1, 0.6),
    ("SECRET", re.compile(r"(?:salasana\w*|password|passcode|lösenord\w*|pin(?:-?koodi)?|tunnusluku|koodi|code)"
                          r"\s*(?:on|is|är|:|=)?\s*['\"]?([^\s,.'\"]{3,})", I), 1, 0.8),
    ("USERNAME", re.compile(r"(?:käyttäjätunnu\w*|käyttäjänimi\w*|username|user name|användarnamn\w*|tunnukseni|"
                            r"tunnus)\s*(?:on|is|är|:)?\s*([\w.@-]{3,})", I), 1, 0.8),
    ("ORG", re.compile(r"\b(?:[A-ZÅÄÖ][\w&-]+\s+){1,3}(?:Oyj|Oy|Ab|Ky|Ay|Tmi|Ltd|Inc|ry|Rf)\b"), 0, 0.85),
    ("ORG", re.compile(r"\b\w+\s+(?:telecom|mobile|net|energia|energy|pankki|bank|vakuutus|insurance|posti|logistics)\b", I), 0, 0.6),
    ("ORG", re.compile(r"\b(?:\w+\s+){1,2}(?:oyj|oy|ab|ky|tmi|regional hospital|hospital|sairaala|upper secondary school|"
                       r"school|lukio|ammattikoulu|akatemia)\b", I), 0, 0.6),
    ("ADDRESS", re.compile(rf"\b[\wåäöÅÄÖ-]*{_STREET}\s+\d{{1,3}}(?:\s?[a-h]\b)?(?:\s?\d{{1,3}}\b)?"
                           rf"(?:,?\s*\d{{5}}(?:\s+[\wåäöÅÄÖ-]+)?)?", I), 0, 0.85),
    ("LOCATION", re.compile(rf"\b(?-i:[A-ZÅÄÖ])[\wåäö-]*n\s+{_PLACE_NOUN}[\wåäö]*", I), 0, 0.7),
    ("LOCATION", re.compile(r"\b[A-ZÅÄÖ][\wåäö]*(?:puisto|halli|hall|kenttä|keskus|kirjasto|uimala|satama|asema|torget|parken)[\wåäö]*"), 0, 0.6),
    ("HEALTH", re.compile(rf"\b{HEALTH_TERMS}\b", I), 0, 0.6),
]
TITLE_NAME = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Miss|Nurse|Prof|tri|lääkäri|hoitaja)\.?\s+([A-ZÅÄÖ][a-zåäö]+(?:\s+[A-ZÅÄÖ][a-zåäö]+)?)")
WORD = re.compile(r"\b[\wåäöÅÄÖ'-]+\b")
STOP = {"fi": {"ja", "on", "ei", "että", "hän", "olen", "minä", "se", "oli", "mutta", "kun", "nyt", "voi", "kiitos",
               "asiakas", "potilas", "joka", "myös", "sitten", "tämä", "mikä", "kanssa"},
        "sv": {"och", "är", "jag", "det", "inte", "att", "har", "på", "med", "en", "som", "för", "kan", "tack", "min"},
        "en": {"the", "and", "is", "to", "my", "i", "you", "of", "a", "it", "for", "with", "this", "that", "can"}}


# words that appear in name columns but are not names ("Tuntematon mies" = unidentified man, etc.)
NAME_STOP = {"tuntematon", "mies", "nainen", "poika", "tyttö", "lapsi", "vauva", "okänd", "kvinna", "pojke", "flicka",
             "unknown", "male", "female", "man", "woman", "boy", "girl", "baby", "patient", "potilas", "asiakas",
             "ole", "olen", "oli", "eli", "ja", "von", "van", "der", "af", "de", "la", "le"}


def plausible_ner(span_text: str, label: str, lowercase_text: bool) -> bool:
    """Drop NER guesses that are clearly not PII: codes, measurements, abbreviations, sentences."""
    t = span_text.strip()
    if len(t) < 3 or ":" in t:
        return False
    if label == "DATE":
        return bool(DATE_LIKE.search(t))
    if any(ch.isdigit() for ch in t):
        return False
    words = t.split()
    if label in ("PERSON", "ORG", "LOCATION"):
        if len(words) > 4:
            return False
        if t.isupper() and len(t) <= 5:          # NRS, LL, GCS, BT ...
            return False
        if not lowercase_text:
            # names and organisations are capitalised word by word; places only need a capitalised head
            # ("Varissuon ostoskeskus", "Turun rautatieasema")
            need = words[:1] if label == "LOCATION" else [w for w in words if w.lower() not in ("of", "and", "och", "ja", "af", "von")]
            if not all(w[:1].isupper() for w in need):
                return False
    if label == "ORG" and len(words) < 2:          # single capitalised words are mostly sentence starts
        return False
    return True


def detect_language(text: str) -> str:
    words = [w.lower() for w in WORD.findall(text)[:60]]
    scores = {lang: sum(w in s for w in words) for lang, s in STOP.items()}
    if re.search(r"[äö]", text) and scores["fi"] >= scores["sv"]:
        scores["fi"] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "fi"


@dataclass
class Span:
    start: int
    end: int
    label: str
    score: float = 1.0

    def to_dict(self):
        return {"start": self.start, "end": self.end, "label": self.label, "score": round(self.score, 3)}


def _resolve(spans: list[Span]) -> list[Span]:
    """Merge overlapping spans into their union. The label comes from a pattern-validated type, otherwise from the
    longer span, then the higher score."""
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start), -s.score))
    out: list[Span] = []
    rank = lambda x: (PRIORITY.get(x.label, 0), x.end - x.start, x.score)  # noqa: E731
    for s in spans:
        if out and s.start < out[-1].end:
            last = out[-1]
            win = s if rank(s) > rank(last) else last
            out[-1] = Span(last.start, max(last.end, s.end), win.label, win.score)
        else:
            out.append(s)
    return out


def apply_spans(text: str, spans: list[Span]) -> str:
    for s in sorted(spans, key=lambda s: -s.start):
        text = text[:s.start] + f"[{s.label}]" + text[s.end:]
    return text


class Redactor:
    def __init__(self, extra_names: list[str] | None = None, use_presidio: bool = True, score_threshold: float = 0.35,
                 places: list[str] | None = None, skip_labels: set[str] | None = None):
        self.names = sorted({n for n in (extra_names or []) if len(n) >= 3}, key=len, reverse=True)
        self.places = sorted({p for p in (places or []) if len(p) >= 3}, key=len, reverse=True)
        self.threshold = score_threshold
        self.languages: list[str] = []
        self.engine = "regex + dataset name/place dictionaries"
        self._batch = None
        self.skip_labels = set(skip_labels or ())   # e.g. HEALTH when the data is itself health data
        self._name_set = {t.lower() for n in self.names for t in re.split(r"[\s,]+", n)
                          if len(t) >= 3 and t.lower() not in NAME_STOP and not any(ch.isdigit() for ch in t)}
        self._place_rx = None
        if self.places:
            # Finnish place names inflect (Turku -> Turussa, Turusta): match the stem plus any ending
            stems = sorted({re.escape(p[:-1] if len(p) > 5 else p) for p in self.places}, key=len, reverse=True)
            self._place_rx = re.compile(r"\b(?:" + "|".join(stems) + r")[\wåäö]*", I)
        if use_presidio:
            self._init_presidio()

    def _init_presidio(self) -> None:
        try:
            import spacy.util
            from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, RecognizerRegistry
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_analyzer.predefined_recognizers import SpacyRecognizer
        except ImportError:
            return
        en = next((m for m in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm") if spacy.util.is_package(m)), None)
        fi = next((m for m in ("fi_core_news_lg", "fi_core_news_md", "fi_core_news_sm") if spacy.util.is_package(m)), None)
        models = [{"lang_code": "en", "model_name": en}] if en else []
        if fi:
            models.append({"lang_code": "fi", "model_name": fi})
        if not models:
            return
        conf = {"nlp_engine_name": "spacy", "models": models,
                "ner_model_configuration": {
                    "model_to_presidio_entity_mapping": {"PERSON": "PERSON", "PER": "PERSON", "GPE": "LOCATION",
                                                         "LOC": "LOCATION", "FAC": "LOCATION", "ORG": "ORGANIZATION",
                                                         "DATE": "DATE_TIME", "TIME": "DATE_TIME", "NORP": "NRP"},
                    "low_confidence_score_multiplier": 0.4, "low_score_entity_names": [],
                    "labels_to_ignore": ["CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT", "PRODUCT", "EVENT",
                                         "WORK_OF_ART", "LAW", "LANGUAGE"]}}
        nlp_engine = NlpEngineProvider(nlp_configuration=conf).create_engine()
        langs = [m["lang_code"] for m in models]
        registry = RecognizerRegistry(supported_languages=langs)
        if "en" in langs:
            registry.load_predefined_recognizers(languages=["en"], nlp_engine=nlp_engine)
        if "fi" in langs:
            registry.add_recognizer(SpacyRecognizer(supported_language="fi",
                                                    supported_entities=["PERSON", "LOCATION", "ORGANIZATION", "DATE_TIME"]))
        for lang in langs:  # only NER is needed from spaCy; the other components cost time
            nlp = nlp_engine.nlp[lang]
            for pipe in ("parser", "lemmatizer", "senter"):
                if pipe in nlp.pipe_names:
                    nlp.disable_pipe(pipe)
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=langs)
        self._batch = BatchAnalyzerEngine(analyzer_engine=analyzer)
        self.languages = langs
        self.engine = (f"Presidio + spaCy {' + '.join(m['model_name'] for m in models)} + regex layer "
                       "(Finnish formats, spoken numbers) + dataset dictionaries")

    # ------------------------------------------------------------ detection
    @staticmethod
    def _regex_spans(text: str) -> list[Span]:
        spans = []
        for label, rx, group, score in REGEXES:
            for m in rx.finditer(text):
                if m.group(group):
                    spans.append(Span(m.start(group), m.end(group), label, score))
        for m in TITLE_NAME.finditer(text):
            spans.append(Span(m.start(1), m.end(1), "PERSON", 0.7))
        return spans

    def _dict_spans(self, text: str) -> list[Span]:
        """Tokens found in the dataset's own name columns (case-insensitive; adjacent hits merged) and its places."""
        spans: list[Span] = []
        if self._name_set:
            for m in WORD.finditer(text):
                if m.group().lower() in self._name_set:
                    if spans and text[spans[-1].end:m.start()] == " ":
                        spans[-1] = Span(spans[-1].start, m.end(), "PERSON", 0.9)
                    else:
                        spans.append(Span(m.start(), m.end(), "PERSON", 0.9))
        if self._place_rx is not None:
            spans += [Span(m.start(), m.end(), "LOCATION", 0.92) for m in self._place_rx.finditer(text)]
        return spans

    def analyze(self, texts: list[str]) -> list[list[Span]]:
        uniq = list(dict.fromkeys(texts))
        found: dict[str, list[Span]] = {t: [] for t in uniq}
        if self._batch is not None:
            by_lang: dict[str, list[str]] = {}
            for t in uniq:
                lang = detect_language(t)
                lang = "fi" if lang in ("fi", "sv") and "fi" in self.languages else lang
                lang = lang if lang in self.languages else self.languages[0]
                by_lang.setdefault(lang, []).append(t)
            for lang, group in by_lang.items():
                results = self._batch.analyze_iterator(group, language=lang, entities=NER_ENTITIES.get(lang),
                                                       score_threshold=self.threshold, batch_size=128)
                for text, res in zip(group, results):
                    low = text == text.lower()
                    found[text] = [Span(r.start, r.end, ENTITY_MAP[r.entity_type], r.score) for r in res
                                   if r.entity_type in ENTITY_MAP
                                   and plausible_ner(text[r.start:r.end], ENTITY_MAP[r.entity_type], low)]
        for text in uniq:
            spans = found[text] + self._regex_spans(text) + self._dict_spans(text)
            found[text] = _resolve([s for s in spans if s.label not in self.skip_labels])
        return [found[t] for t in texts]

    def redact(self, s: pd.Series) -> tuple[pd.Series, dict[int, list[Span]]]:
        mask = (s.notna() & (s.astype("string").str.strip() != "")).fillna(False).to_numpy(dtype=bool)
        idx = s.index[mask]
        texts = s[mask].astype(str).tolist()
        spans = self.analyze(texts)
        out = s.astype("string").copy()
        by_row: dict[int, list[Span]] = {}
        for i, text, sp in zip(idx, texts, spans):
            out.loc[i] = apply_spans(text, sp)
            by_row[int(i)] = sp
        return out, by_row


# ---------------------------------------------------------------- evaluation
_SEP = re.compile(r"[\s\-./()+:,_]")


def _norm(v: str) -> str:
    return _SEP.sub("", str(v).lower())


def evaluate(gold: dict[int, list[dict]], pred: dict[int, list[Span]], original: pd.Series, redacted: pd.Series) -> dict:
    """Span precision/recall per label (overlap match, same label) and value leakage measured like the kit
    (case- and separator-insensitive: a value leaks if it is still recoverable from the redacted text)."""
    norm_gold = {r: [dict(g, label=LABEL_ALIASES.get(g["label"], g["label"])) for g in spans] for r, spans in gold.items()}
    labels = sorted({g["label"] for spans in norm_gold.values() for g in spans} | {p.label for spans in pred.values() for p in spans})
    stats = {lab: {"gold": 0, "pred": 0, "tp_gold": 0, "tp_pred": 0, "leaked": 0, "tested": 0} for lab in labels}
    any_hit = total_gold = leaked = tested = 0
    for r in set(norm_gold) | set(pred):
        g_spans, p_spans = norm_gold.get(r, []), pred.get(r, [])
        text = str(original.get(r, "")) if r in original.index else ""
        red = _norm(redacted.get(r, "")) if r in redacted.index else ""
        for g in g_spans:
            st = stats[g["label"]]
            st["gold"] += 1
            total_gold += 1
            if any(p.label == g["label"] and p.start < g["end"] and g["start"] < p.end for p in p_spans):
                st["tp_gold"] += 1
            if any(p.start < g["end"] and g["start"] < p.end for p in p_spans):
                any_hit += 1
            value = _norm(g.get("text") or text[g["start"]:g["end"]])
            if len(value) >= 3:
                st["tested"] += 1
                tested += 1
                if value in red:
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
        f2 = 5 * prec * rec / (4 * prec + rec) if rec is not None and prec is not None and (prec + rec) > 0 else None
        per[lab] = {"gold": st["gold"], "predicted": st["pred"], "precision": prec, "recall": rec, "f2": f2,
                    "leak_rate": st["leaked"] / st["tested"] if st["tested"] else None}
    return {
        "per_label": per,
        "recall_any_label": any_hit / total_gold if total_gold else None,
        "leak_rate": leaked / tested if tested else None,
        "gold_spans": total_gold,
        "records": len(gold),
    }


def load_gold(obj, id_to_row: dict | None = None) -> tuple[str | None, dict[int, list[dict]]]:
    """Ground-truth spans in either format:
    - ours: {"text_column": ..., "records": [{"row": i, "spans": [...]}]} (or a bare list of such records);
    - the kit's JSONL: one {"record_id": ..., "spans": [...]} per line (pass a path, a list of dicts, or JSONL text).
    Kit records are keyed by record id; `id_to_row` maps them to row positions (records not found are skipped)."""
    if isinstance(obj, (str, Path)) and Path(str(obj)).exists():
        text = Path(str(obj)).read_text(encoding="utf-8")
        obj = json.loads(text) if text.lstrip().startswith(("{\"text_column\"", "[")) and not text.lstrip().startswith(
            "{\"record_id\"") else [json.loads(x) for x in text.splitlines() if x.strip()]
    col = None
    if isinstance(obj, dict):
        col, recs = obj.get("text_column"), obj.get("records", [])
    else:
        recs = obj
    out: dict[int, list[dict]] = {}
    for r in recs:
        if "row" in r:
            out[int(r["row"])] = r.get("spans", [])
        elif "record_id" in r:
            if id_to_row is None:
                raise ValueError("ground truth is keyed by record_id: pass the table's id column")
            row = id_to_row.get(str(r["record_id"]))
            if row is not None:
                out[int(row)] = r.get("spans", [])
    return col, out
