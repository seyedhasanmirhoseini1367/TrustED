"""Synthetic healthcare admissions dataset with ground-truth PII spans (spec §6).

Every person, address, phone number and identity code is invented. Run:

    python data/generate_sample.py [--rows 5000] [--seed 7] [--out data/sample]
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FIRST_F = ["Aino", "Emma", "Helmi", "Sofia", "Aada", "Ella", "Olivia", "Venla", "Lilja", "Eevi", "Maria", "Anna",
           "Liisa", "Kaarina", "Marjatta", "Hannele", "Tuula", "Päivi", "Satu", "Johanna", "Laura", "Minna", "Riitta",
           "Sanna", "Elina", "Katja", "Heidi", "Noora", "Iida", "Kerttu"]
FIRST_M = ["Juhani", "Mikael", "Onni", "Eino", "Väinö", "Leo", "Elias", "Oliver", "Matti", "Timo", "Kari", "Jari",
           "Antti", "Pekka", "Mikko", "Jukka", "Markku", "Heikki", "Ville", "Tuomas", "Janne", "Sami", "Petri",
           "Lauri", "Aleksi", "Joonas", "Eetu", "Niko", "Arttu", "Veikko"]
LAST = ["Korhonen", "Virtanen", "Mäkinen", "Nieminen", "Mäkelä", "Hämäläinen", "Laine", "Heikkinen", "Koskinen",
        "Järvinen", "Lehtonen", "Lehtinen", "Saarinen", "Salminen", "Heinonen", "Niemi", "Heikkilä", "Kinnunen",
        "Salonen", "Turunen", "Salo", "Laitinen", "Tuominen", "Rantanen", "Karjalainen", "Jokinen", "Mattila",
        "Savolainen", "Lahtinen", "Ahonen", "Aaltonen", "Lindholm", "Grönroos", "Nurmi", "Vainio", "Kallio",
        "Suominen", "Leino", "Lundqvist", "Rautio", "Peltonen", "Aalto", "Hiltunen", "Kivelä", "Ojala"]
DOCTORS = ["Lindqvist", "Halonen", "Räsänen", "Mustonen", "Hakala", "Seppälä", "Ketola", "Valtonen"]
STREETS = ["Aurakatu", "Humalistonkatu", "Linnankatu", "Yliopistonkatu", "Uudenmaankatu", "Hämeenkatu",
           "Rauhankatu", "Puistokatu", "Itäinen Pitkäkatu", "Kaskentie", "Ratapihankatu", "Sirkkalankatu"]
# (postcode, city, weight) — Southwest Finland; each postcode belongs to exactly one city
POSTCODES = [("20100", "Turku", 9), ("20200", "Turku", 6), ("20250", "Turku", 5), ("20300", "Turku", 5),
             ("20320", "Turku", 4), ("20360", "Turku", 4), ("20380", "Turku", 3), ("20400", "Turku", 3),
             ("20500", "Turku", 5), ("20520", "Turku", 4), ("20540", "Turku", 4), ("20610", "Turku", 3),
             ("20700", "Turku", 5), ("20720", "Turku", 3), ("20740", "Turku", 3), ("20810", "Turku", 3),
             ("20880", "Turku", 2), ("20900", "Turku", 3), ("20780", "Kaarina", 5), ("20760", "Kaarina", 3),
             ("21200", "Raisio", 5), ("21100", "Naantali", 3), ("21420", "Lieto", 3), ("21250", "Masku", 2),
             ("21600", "Parainen", 2), ("23500", "Uusikaupunki", 2), ("24100", "Salo", 4), ("24240", "Salo", 2),
             ("24800", "Salo", 1), ("32200", "Loimaa", 2)]
DEPARTMENTS = {"Internal Medicine": 0.25, "Surgery": 0.16, "Cardiology": 0.13, "Neurology": 0.08, "Pediatrics": 0.08,
               "Orthopedics": 0.11, "Psychiatry": 0.05, "Oncology": 0.06, "Pulmonology": 0.08}
DIAG = {
    "Internal Medicine": ["E11.9", "N39.0", "A41.9", "K29.7", "E86", "D50.9", "R50.9", "E10.9"],
    "Surgery": ["K35.8", "K80.2", "K40.9", "K56.6", "T81.4", "K57.3"],
    "Cardiology": ["I21.4", "I48.9", "I50.9", "I10", "I20.0", "I25.1"],
    "Neurology": ["I63.9", "G40.9", "G45.9", "G35", "R55"],
    "Pediatrics": ["J21.9", "A08.4", "J06.9", "R56.0", "J45.9"],
    "Orthopedics": ["S72.0", "S82.6", "M17.1", "S52.5", "M16.1", "S42.2"],
    "Psychiatry": ["F32.2", "F20.0", "F10.2", "F31.4", "F41.1"],
    "Oncology": ["C34.9", "C50.9", "C18.7", "C61", "C92.0", "C25.9"],
    "Pulmonology": ["J44.1", "J18.9", "J45.9", "J96.0", "U07.1", "J84.1"],
}
DEPT_EFFECT = {"Internal Medicine": 0.25, "Surgery": -0.1, "Cardiology": 0.35, "Neurology": 0.1, "Pediatrics": -0.6,
               "Orthopedics": -0.3, "Psychiatry": 0.45, "Oncology": 0.7, "Pulmonology": 0.4}
COMPLAINTS = ["chest pain", "shortness of breath", "abdominal pain", "fever", "a fall at home", "confusion",
              "worsening cough", "palpitations", "back pain", "dizziness"]
NEUTRAL = ["Vitals stable overnight.", "Pain managed with paracetamol.", "No known drug allergies.",
           "Discussed discharge plan with the patient.", "Mobilising with a walker.", "Fluids continued.",
           "Blood cultures pending.", "Responding well to treatment.", "Physiotherapy consulted.",
           "ECG shows sinus rhythm."]
RELATIONS = ["Wife", "Husband", "Daughter", "Son", "Sister", "Brother", "Partner"]
HETU_CHECK = "0123456789ABCDEFHJKLMNPRSTUVWXY"


def hetu(dob: date, female: bool, rng: np.random.Generator) -> str:
    century = {18: "+", 19: "-", 20: "A"}[dob.year // 100]
    n = int(rng.integers(2, 900))
    if (n % 2 == 0) != female:
        n += 1 if n < 899 else -1
    base = f"{dob:%d%m}{dob.year % 100:02d}"
    return f"{base}{century}{n:03d}{HETU_CHECK[int(base + f'{n:03d}') % 31]}"


def phone(rng: np.random.Generator) -> str:
    pre = rng.choice(["040", "044", "045", "050", "041"])
    num = f"{int(rng.integers(1000000, 9999999))}"
    if rng.random() < 0.5:
        return f"+358 {pre[1:]} {num[:3]} {num[3:]}"
    return f"{pre} {num[:3]} {num[3:]}"


class Note:
    def __init__(self):
        self.text, self.spans = "", []

    def add(self, *parts):
        """parts: plain strings or (value, label) tuples."""
        for part in parts:
            if isinstance(part, tuple):
                value, label = part
                start = len(self.text)
                self.text += value
                self.spans.append({"start": start, "end": len(self.text), "label": label})
            else:
                self.text += part
        self.text += " "


def fmt_date(d: date, rng) -> str:
    return f"{d.day}.{d.month}.{d.year}" if rng.random() < 0.6 else d.isoformat()


def generate(rows: int = 5000, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    pc_w = np.array([w for *_, w in POSTCODES], float)
    pc_w /= pc_w.sum()
    depts = list(DEPARTMENTS)
    dept_p = np.array(list(DEPARTMENTS.values()))
    dept_p /= dept_p.sum()
    records, gold = [], []
    used_emails = set()
    for i in range(rows):
        female = bool(rng.random() < 0.52)
        first = str(rng.choice(FIRST_F if female else FIRST_M))
        last = str(rng.choice(LAST))
        dept = str(rng.choice(depts, p=dept_p))
        if dept == "Pediatrics":
            age = int(rng.integers(0, 18))
        else:
            age = int(np.clip(rng.normal(64, 18), 18, 99))
        admission = date(2025, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
        dob = admission - timedelta(days=int(age * 365.25 + rng.integers(0, 365)))
        pc, city, _ = POSTCODES[int(rng.choice(len(POSTCODES), p=pc_w))]
        los = int(np.clip(rng.gamma(2.0, 2.2 + 0.03 * age + (2 if dept in ("Oncology", "Neurology") else 0)), 0, 45))
        discharge = admission + timedelta(days=los)
        prior = int(rng.poisson(0.3 + age / 40 + (1.0 if dept in ("Oncology", "Psychiatry") else 0)))
        crp = float(np.round(np.clip(rng.lognormal(2.3 + (0.8 if dept in ("Internal Medicine", "Pulmonology") else 0), 0.9),
                                     0.5, 400), 1))
        hb = float(np.round(np.clip(rng.normal(138 if not female else 128, 14) - 0.12 * max(age - 60, 0), 70, 180), 0))
        diag_list = DIAG[dept]
        weights = np.array([0.5 ** j for j in range(len(diag_list))])
        diag = str(rng.choice(diag_list, p=weights / weights.sum()))
        if rng.random() < 0.015:
            diag = str(rng.choice(["Q21.0", "E84.0", "G12.2", "D66", "E75.2", "M32.1", "B20", "O14.1"]))
        logit = (-3.0 + 0.022 * (age - 55) + 0.42 * prior + 0.07 * los + 0.006 * (crp - 20) - 0.025 * (hb - 132)
                 + DEPT_EFFECT[dept])
        readmit = int(rng.random() < 1 / (1 + np.exp(-logit)))
        email_base = f"{first}.{last}".lower().translate(str.maketrans("äöå", "aoa"))
        email = f"{email_base}{int(rng.integers(1, 999))}@example.fi"
        while email in used_emails:
            email = f"{email_base}{int(rng.integers(1, 9999))}@example.fi"
        used_emails.add(email)
        tel = phone(rng)
        nid = hetu(dob, female, rng)

        note = Note()
        note.add("Patient ", (f"{first} {last}", "PERSON"), f" admitted via ED with {rng.choice(COMPLAINTS)}.")
        extras = rng.permutation(7)[: int(rng.integers(1, 4))]
        for e in extras:
            if e == 0:
                note.add(f"{rng.choice(RELATIONS)} ", (f"{rng.choice(FIRST_F + FIRST_M)}", "PERSON"),
                         " can be reached at ", (phone(rng), "PHONE"), ".")
            elif e == 1:
                note.add("Seen by Dr. ", (str(rng.choice(DOCTORS)), "PERSON"), " on ",
                         (fmt_date(admission + timedelta(days=int(rng.integers(0, 3))), rng), "DATE"), ".")
            elif e == 2:
                note.add("Lives alone at ", (f"{rng.choice(STREETS)} {int(rng.integers(1, 80))}", "ADDRESS"), ", ",
                         (city, "LOCATION"), ".")
            elif e == 3:
                note.add("Identity code ", (nid, "NATIONAL_ID"), " verified at registration.")
            elif e == 4:
                note.add("Follow-up instructions sent to ", (email, "EMAIL"), ".")
            elif e == 5:
                note.add("Previous admission on ", (fmt_date(admission - timedelta(days=int(rng.integers(30, 700))), rng), "DATE"),
                         ".")
            else:
                note.add("Patient asked to call back on ", (tel, "PHONE"), " if symptoms recur.")
        for s in rng.choice(NEUTRAL, size=int(rng.integers(1, 3)), replace=False):
            note.add(str(s))
        note.text = note.text.rstrip()

        records.append({
            "patient_name": f"{first} {last}", "email": email, "phone": tel, "national_id": nid,
            "date_of_birth": dob.isoformat(), "gender": "F" if female else "M", "postcode": pc, "city": city,
            "admission_date": admission.isoformat(), "discharge_date": discharge.isoformat(), "department": dept,
            "diagnosis_code": diag, "length_of_stay": los, "num_prior_visits": prior, "lab_value_1": crp,
            "lab_value_2": hb, "clinician_notes": note.text, "readmitted_30d": readmit,
        })
        gold.append({"row": i, "spans": note.spans})
    df = pd.DataFrame(records)
    meta = {"text_column": "clinician_notes", "seed": seed, "rows": rows,
            "labels": ["PERSON", "PHONE", "EMAIL", "DATE", "LOCATION", "ADDRESS", "NATIONAL_ID"], "records": gold}
    return df, meta


def write(out_dir: str | Path, rows: int = 5000, seed: int = 7) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, meta = generate(rows, seed)
    csv_path, spans_path = out / "health_admissions.csv", out / "health_admissions.spans.json"
    df.to_csv(csv_path, index=False)
    spans_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return csv_path, spans_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).parent / "sample"))
    a = ap.parse_args()
    c, s = write(a.out, a.rows, a.seed)
    print(f"wrote {c} and {s}")
