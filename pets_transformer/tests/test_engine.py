import json
import hashlib

import numpy as np
import pandas as pd
import pytest

from engine import utility
from engine.gates import assign_gates, column_groups, derive_knowledge
from engine.ladders import (AgeLadder, CategoricalLadder, CodeLadder, DateLadder, LatLonLadder, NumericLadder,
                            PostcodeLadder, TimestampLadder)
from engine.pipeline import fit_ladders, run
from engine.profiling import profile_dataframe
from engine.report import render_html, to_json
from engine.risk import k_metrics, l_diversity, linkage_attack
from engine.roles import Gate, Role, Scenario
from engine.search import GreedySearch
from engine.text_pii import Redactor, evaluate
from engine.transforms import Plan, pseudonymise, truncate_ip
from engine.utility import ProxyUtility

SECRET = b"test-secret"


# ---------------------------------------------------------------- ladders
def test_date_ladder_levels():
    s = pd.Series(["2025-03-12", "1987-11-30", None])
    lad = DateLadder("d").fit(s)
    assert lad.apply(s, 0).tolist()[:2] == ["2025-03-12", "1987-11-30"]
    assert lad.apply(s, 1).tolist()[:2] == ["2025-W11", "1987-W49"]
    assert lad.apply(s, 2).tolist()[:2] == ["2025-03", "1987-11"]
    assert lad.apply(s, 3).tolist()[:2] == ["2025", "1987"]          # quarters are not a level: the kit cannot parse them
    assert lad.max_level == 3
    assert pd.isna(lad.apply(s, 3).iloc[2])
    # midpoints lie inside the period
    mid = lad.numeric(s, 2)
    expected = (pd.Timestamp("2025-03-16 12:00") - pd.Timestamp("1970-01-01")).total_seconds() / 86400
    assert mid[0] == pytest.approx(expected, abs=1e-3)


def test_timestamp_ladder_levels():
    s = pd.Series(["2025-03-12 14:37:00"])
    lad = TimestampLadder("t").fit(s)
    assert lad.apply(s, 1).iloc[0] == "2025-03-12 14:00"
    assert lad.apply(s, 2).iloc[0] == "2025-03-12"
    assert lad.apply(s, 3).iloc[0] == "2025-W11"
    assert lad.apply(s, 5).iloc[0] == "2025"


def test_age_ladder_levels():
    s = pd.Series([str(a) for a in range(0, 100)])
    lad = AgeLadder("age").fit(s)
    v = pd.Series(["37"])
    assert lad.apply(v, 1).iloc[0] == "35-39"
    assert lad.apply(v, 2).iloc[0] == "30-39"
    assert lad.apply(v, 3).iloc[0] == "20-39"
    top = lad.apply(pd.Series(["99", "2", "50"]), 4).tolist()
    assert top[0].endswith("+") and top[1].startswith("<") and top[2] == "40-59"
    assert lad.numeric(v, 2).iloc[0] == pytest.approx(34.5)


def test_numeric_ladder_levels():
    s = pd.Series([str(x) for x in np.linspace(0, 1000, 101)])
    lad = NumericLadder("x").fit(s)
    assert lad.apply(pd.Series(["123.4"]), 1).iloc[0] == "120"
    assert lad.apply(s, 2).nunique() == 10
    assert lad.apply(s, 3).nunique() == 4
    assert lad.numeric(s, 3).nunique() == 4


def test_latlon_postcode_categorical_code_ladders():
    s = pd.Series(["60.45123", "22.26789"])
    lad = LatLonLadder("lat").fit(s)
    assert lad.apply(s, 1).tolist() == ["60.451", "22.268"]
    assert lad.apply(s, 3).tolist() == ["60.5", "22.3"]

    pc = pd.Series(["20100", "24240"])
    lad = PostcodeLadder("pc").fit(pc)
    assert lad.apply(pc, 1).tolist() == ["2010*", "2424*"]
    assert lad.apply(pc, 2).tolist() == ["201**", "242**"]
    assert lad.apply(pc, 3).tolist() == ["20***", "24***"]

    cat = pd.Series(["a"] * 90 + ["b"] * 7 + ["c"] * 3)
    lad = CategoricalLadder("c").fit(cat)
    assert set(lad.apply(cat, 1)) == {"a", "b", "c"}          # 3 % > 1 %
    assert set(lad.apply(cat, 2)) == {"a", "b", "Other"}      # 3 % < 5 %
    lad = CategoricalLadder("c", {"a": "vowel", "b": "consonant", "c": "consonant"}).fit(cat)
    assert set(lad.apply(cat, 3)) == {"vowel", "consonant"}

    icd = pd.Series(["J45.9"] * 10 + ["J45.1"] * 2 + ["Q21.0"] * 2)
    lad = CodeLadder("dx", k=5).fit(icd)
    lvl1 = lad.apply(icd, 1)
    assert lvl1.value_counts()["J45.9"] == 10                 # common codes stay exact
    assert set(lvl1[icd == "Q21.0"]) == {"Other"}             # Q21 and Q are still < 5
    assert set(lvl1[icd == "J45.1"]) == {"Other"}             # "J45" would hold only 2 records
    assert set(lad.apply(icd, 2)) == {"J45", "Q21"}
    assert set(lad.apply(icd, 3)) == {"J", "Q"}


# ---------------------------------------------------------------- red transforms
def test_pseudonymisation_consistency():
    a = pseudonymise(pd.Series(["010180-123A", "020290-456B", "010180-123A"]), SECRET)
    b = pseudonymise(pd.Series(["010180-123A"]), SECRET)          # another table, same secret
    c = pseudonymise(pd.Series(["010180-123A"]), b"other-secret")
    assert a[0] == a[2] == b[0]
    assert a[0] != a[1]
    assert a[0] != c[0]
    assert len(a[0]) == 16 and all(ch in "0123456789abcdef" for ch in a[0])
    # not a plain (unsalted) hash of the value
    assert a[0] != hashlib.sha256(b"010180-123A").hexdigest()[:16]


def test_ip_truncation():
    out = truncate_ip(pd.Series(["192.168.17.42", "2001:db8:abcd:12::1", "nope"]))
    assert out[0] == "192.168.17.0/24"
    assert out[1] == "2001:db8:abcd::/48"
    assert pd.isna(out[2])


# ---------------------------------------------------------------- risk
def test_linkage_attack_toy_table():
    # rows 0 and 1 share (30, F); rows 2 and 3 are unique; row 4 unique but suppressed in the release
    released = pd.DataFrame({"age": ["30", "30", "41", "52", "63"], "sex": ["F", "F", "M", "F", "M"]})
    released = released.drop(index=4)
    attacker = pd.DataFrame({"age": ["30", "41", "52", "63"], "sex": ["F", "M", "F", "M"]}, index=[0, 2, 3, 4])
    res = linkage_attack(attacker, released, ["age", "sex"])
    assert res["targets"] == 4
    assert res["successes"] == 2           # rows 2 and 3
    assert res["success_rate"] == 0.5
    assert res["no_match"] == 1            # row 4 was suppressed
    km = k_metrics(released, ["age", "sex"], k_target=2)
    assert km["min_k"] == 1 and km["uniqueness"] == 0.5 and km["below_k"] == 0.5


def test_linkage_attack_wrong_unique_match_is_not_success():
    released = pd.DataFrame({"zip": ["1", "2"]}, index=[10, 11])
    attacker = pd.DataFrame({"zip": ["2"]}, index=[10])      # attacker's value differs from the true record
    assert linkage_attack(attacker, released, ["zip"])["successes"] == 0


def test_l_diversity():
    df = pd.DataFrame({"q": ["a", "a", "b", "b"], "s": ["x", "x", "x", "y"]})
    ld = l_diversity(df, ["q"], ["s"])["s"]
    assert ld["min_l"] == 1 and ld["classes_l1"] == 1 and ld["records_in_l1_classes"] == 2


# ---------------------------------------------------------------- gates / threat model
def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "name": [f"Person {i}" for i in range(n)],
        "date_of_birth": [str(pd.Timestamp("1940-01-01") + pd.Timedelta(days=int(d))) [:10] for d in rng.integers(0, 25000, n)],
        "gender": rng.choice(["F", "M"], n),
        "postcode": rng.choice(["20100", "20200", "20300", "21200", "24100"], n),
        "visit_date": [str(pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(d)))[:10] for d in rng.integers(0, 365, n)],
        "score": rng.normal(50, 10, n).round(1).astype(str),
        "outcome": rng.integers(0, 2, n).astype(str),
    })


def test_profiling_and_gates():
    df = _toy()
    sc = Scenario(recipient="partner", target="outcome")
    profiles = assign_gates(profile_dataframe(df), df, sc)
    by = {p.name: p for p in profiles}
    assert by["name"].gate == Gate.RED
    assert by["date_of_birth"].role == Role.QUASI_ID and by["date_of_birth"].gate == Gate.YELLOW
    assert by["postcode"].ladder == "postcode"
    assert by["score"].gate == Gate.GREEN
    assert by["outcome"].role == Role.TARGET


def test_derived_knowledge_and_groups():
    df = pd.DataFrame({"postcode": ["20100", "20200", "24100"] * 50, "city": ["Turku", "Turku", "Salo"] * 50,
                       "a_date": ["2025-01-01"] * 150, "b_date": ["2025-01-05"] * 150})
    sc = Scenario(recipient="partner")
    profiles = assign_gates(profile_dataframe(df), df, sc)
    derived = derive_knowledge(df, profiles, ["postcode"])
    assert derived and derived[0]["column"] == "city"
    groups = column_groups(profiles)
    assert ["a_date", "b_date"] in groups


# ---------------------------------------------------------------- search / utility
def _search(df, sc):
    profiles = assign_gates(profile_dataframe(df), df, sc)
    ladders = fit_ladders(df, profiles, sc)
    proxy = ProxyUtility(df, profiles, ladders, sc.key_columns)
    known = ["date_of_birth", "gender", "postcode", "visit_date"]
    return GreedySearch(df, profiles, ladders, sc, known, column_groups(profiles), proxy)


def test_greedy_search_stops_at_target():
    df = _toy()
    sc = Scenario(recipient="partner", target="outcome", attacker_qis=["date_of_birth", "gender", "postcode", "visit_date"])
    res = _search(df, sc).run()
    assert res.target_met
    assert res.final["linkage"] <= sc.max_linkage and res.final["min_k"] >= sc.min_k
    # it stops at the first state that meets the target: the previous state did not
    prev = res.path[-2]
    assert not (prev["linkage"] <= sc.max_linkage and prev["min_k"] >= sc.min_k)
    # risk never increases along the path
    links = [p["linkage"] for p in res.path]
    assert all(b <= a + 1e-9 for a, b in zip(links, links[1:]))


def test_search_is_noop_when_target_already_met():
    df = _toy()
    sc = Scenario(recipient="internal", target="outcome", attacker_qis=["gender"], max_linkage=1.0, min_k=1)
    s = _search(df, sc)
    s.known = ["gender"]
    res = s.run()
    assert res.target_met and len(res.path) == 1


def test_train_test_split_happens_before_transformation(monkeypatch):
    df = _toy()
    sc = Scenario(recipient="partner", target="outcome")
    profiles = assign_gates(profile_dataframe(df), df, sc)
    ladders = fit_ladders(df, profiles, sc)
    calls = []
    real_split, real_apply = utility.split_original, utility.apply_plan

    def spy_split(frame, *a, **k):
        calls.append(("split", frame["date_of_birth"].iloc[0]))
        return real_split(frame, *a, **k)

    def spy_apply(frame, *a, **k):
        calls.append(("apply", len(frame)))
        return real_apply(frame, *a, **k)

    monkeypatch.setattr(utility, "split_original", spy_split)
    monkeypatch.setattr(utility, "apply_plan", spy_apply)
    plan = Plan(levels={"date_of_birth": 4, "visit_date": 2})
    out = utility.task_utility(df, profiles, ladders, plan, SECRET, "outcome", "classification", 42)
    assert calls[0][0] == "split"
    assert calls[0][1] == df["date_of_birth"].iloc[0]          # the split saw ORIGINAL values
    assert [c[0] for c in calls[1:]] == ["apply", "apply"]       # then train and test were transformed separately
    assert {c[1] for c in calls[1:]} == {480, 120}
    assert out["train_rows"] == 480 and out["test_rows"] == 120


# ---------------------------------------------------------------- text
def test_redaction_and_evaluation_regex_only():
    texts = pd.Series(["Patient Aino Virtanen, phone 040 123 4567, email aino@example.fi, HETU 010180-123A."])
    gold = {0: [{"start": 8, "end": 21, "label": "PERSON"}, {"start": 29, "end": 41, "label": "PHONE"},
                {"start": 49, "end": 64, "label": "EMAIL"}, {"start": 71, "end": 82, "label": "ID"}]}
    r = Redactor(["Aino Virtanen"], use_presidio=False)
    red, spans = r.redact(texts)
    assert red[0] == "Patient [PERSON], phone [PHONE], email [EMAIL], HETU [ID]."
    ev = evaluate(gold, spans, texts, red)
    assert ev["recall_any_label"] == 1.0 and ev["leak_rate"] == 0.0
    assert all(v["recall"] == 1.0 for v in ev["per_label"].values())


# ---------------------------------------------------------------- end to end
def test_pipeline_deterministic_and_no_original_data_in_exports():
    df = _toy(400)
    df["notes"] = [f"Seen by Dr. Halonen. Patient {n} called from 040 555 {i:04d} about the follow-up visit."
                   for i, n in enumerate(df["name"])]
    sc = Scenario(recipient="partner", target="outcome", attacker_qis=["date_of_birth", "gender", "postcode", "visit_date"])
    r1 = run(df, sc, secret=SECRET, use_presidio=False)
    r2 = run(df, sc, secret=SECRET, use_presidio=False)
    assert r1.protected.equals(r2.protected)
    assert r1.report["metrics"] == r2.report["metrics"]
    exported = to_json(r1.report) + render_html(r1.report) + r1.protected.to_csv()
    for value in list(df["name"].head(50)) + list(df["date_of_birth"].head(50)):
        assert value not in exported
    assert "040 555" not in exported
    assert "test-secret" not in exported
    json.loads(to_json(r1.report))


def test_classification_with_rare_classes_and_bad_targets():
    """Regression: 'y_true contains labels not in parameter labels' when a class is too rare for the train split."""
    df = _toy(500)
    rng = np.random.default_rng(1)
    df["disposition"] = rng.choice(["home", "ward", "icu"], len(df), p=[0.6, 0.3, 0.1]).astype(object)
    df.loc[[3], "disposition"] = "cancelled"          # a single-row class: lands in train OR test only
    df.loc[[7], "disposition"] = "transfer"
    sc = Scenario(recipient="partner", target="disposition", task_type="classification",
                  attacker_qis=["date_of_birth", "gender", "postcode", "visit_date"])
    r = run(df, sc, secret=SECRET, use_presidio=False)
    t = r.report["utility"]["task"]
    assert 0.0 <= t["original"]["roc_auc"] <= 1.0 and 0.0 <= t["protected"]["roc_auc"] <= 1.0

    with pytest.raises(ValueError, match="not a classification label"):
        run(df, Scenario(recipient="partner", target="score", task_type="classification"), secret=SECRET,
            use_presidio=False)


# ---------------------------------------------------------------- starter-kit compatibility
def test_birthdate_ladder_releases_age_bands_the_kit_can_parse():
    from engine.ladders import BirthDateLadder
    s = pd.Series(["1990-06-15", "1941-01-01", None])
    lad = BirthDateLadder("date_of_birth", reference="2025-12-31").fit(s)
    assert lad.apply(s, 1).tolist()[:2] == ["1990-06", "1941-01"]
    assert lad.apply(s, 2).tolist()[:2] == ["1990", "1941"]
    assert lad.apply(s, 4).tolist()[:2] == ["30-39", "80-89"]
    assert lad.apply(s, 5).tolist()[:2] == ["30-39", "80+"]
    assert lad.output_name(2) == "date_of_birth" and lad.output_name(4) == "age"
    assert lad.level_index("age, 10-year bands") == 4 and lad.level_index("year") == 2


def test_utility_caps_are_respected_and_export_follows_kit_conventions():
    df = _toy(600)
    df.insert(0, "visit_id", [f"V{i:05d}" for i in range(len(df))])
    sc = Scenario(recipient="partner", target="outcome", attacker_qis=["date_of_birth", "gender", "postcode", "visit_date"],
                  max_levels={"visit_date": "week", "postcode": 0})
    r = run(df, sc, secret=SECRET, use_presidio=False)
    levels = r.report["plan"]["levels"]
    assert levels["visit_date"] <= 1 and levels["postcode"] == 0
    assert {u["column"] for u in r.report["utility_requirements"]} == {"visit_date", "postcode"}
    out = r.export_table()
    if levels["date_of_birth"] >= 3:
        assert "age" in out.columns and "date_of_birth" not in out.columns
    assert list(r.rowmap.columns) == ["protected_row", "original_record_id"]
    assert len(r.rowmap) == len(out)
    assert r.rowmap["original_record_id"].iloc[0] == df.loc[r.protected.index[0], "visit_id"]


def test_kit_scenario_presets():
    from engine import kit
    cols = list(kit.catalog().query("dataset == 'health' and table == 'ed_visits'")["column"])
    assert kit.match_table(cols) == ("health", "ed_visits")
    ids = [s["id"] for s in kit.scenarios_for(cols)]
    assert ids == ["HEALTH-S1", "HEALTH-S2", "HEALTH-S3"]
    sc = kit.scenario_from_kit("HEALTH-S1", cols)
    assert sc.recipient == "partner" and sc.kit_tasks == ["H-U2", "H-U4", "H-U6", "H-U7"]
    assert sc.sacrificed_tasks == ["H-U8"]
    assert sc.max_levels["date_of_birth"] == "age, 10-year bands"
    assert {"date_of_birth", "sex", "municipality", "arrival_ts"} <= set(sc.attacker_qis)
    assert kit.scenario_from_kit("HEALTH-S1", cols, ["H-U8"]).max_levels == {"arrival_ts": "hour"}
    assert kit.match_table(["a", "b"]) is None
