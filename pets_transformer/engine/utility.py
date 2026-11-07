"""Utility measurement (spec §5.7).

- proxy: fast information-loss + correlation-preservation score used inside the search loop;
- task utility: train/test split of the ORIGINAL data first, then the same plan applied to both parts;
- descriptive utility: aggregate-query relative error and per-column distribution distances.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .ladders import Ladder
from .profiling import parse_dates, to_number
from .roles import ColumnProfile, Gate, Kind, Role
from .transforms import Plan, apply_plan

KEY_WEIGHT = 3.0


# ---------------------------------------------------------------- proxy
class ProxyUtility:
    """Weighted (1 - information loss) over released columns, combined with correlation preservation."""

    def __init__(self, df: pd.DataFrame, profiles: list[ColumnProfile], ladders: dict[str, Ladder], key_columns: list[str]):
        self.ladders = ladders
        self.cols = [p.name for p in profiles if p.gate != Gate.RED and p.role != Role.FREE_TEXT]
        self.weights = {c: (KEY_WEIGHT if c in key_columns else 1.0) for c in self.cols}
        self.num_cols = [c for c in self.cols if c in ladders and ladders[c].numeric_capable]
        target = next((p.name for p in profiles if p.role == Role.TARGET), None)
        self.extra_num = {}
        if target is not None:
            t = to_number(df[target])
            if t.notna().mean() > 0.95 and t.nunique() > 1:
                self.extra_num[target] = t
        self._num_cache: dict[tuple[str, int], pd.Series] = {}
        self.df = df
        self.orig_corr = self._corr({c: 0 for c in self.num_cols}, df.index)

    def numeric(self, col: str, level: int) -> pd.Series:
        key = (col, level)
        if key not in self._num_cache:
            self._num_cache[key] = self.ladders[col].numeric(self.df[col], level)
        return self._num_cache[key]

    def _corr(self, levels: dict[str, int], index: pd.Index) -> np.ndarray | None:
        series = {c: self.numeric(c, levels.get(c, 0)).loc[index] for c in self.num_cols}
        series.update({c: s.loc[index] for c, s in self.extra_num.items()})
        if len(series) < 2:
            return None
        frame = pd.DataFrame(series)
        frame = frame.loc[:, frame.std() > 0] if len(frame) > 1 else frame
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return frame.reindex(columns=list(series)).corr().to_numpy()

    def info_score(self, levels: dict[str, int]) -> float:
        total = sum(self.weights.values()) or 1.0
        score = 0.0
        for c, w in self.weights.items():
            loss = self.ladders[c].info_loss(levels.get(c, 0)) if c in self.ladders else 0.0
            score += w * (1 - loss)
        return score / total

    def corr_diff(self, levels: dict[str, int], index: pd.Index) -> float:
        if self.orig_corr is None:
            return 0.0
        prot = self._corr(levels, index)
        iu = np.triu_indices_from(self.orig_corr, k=1)
        d = np.abs(np.nan_to_num(self.orig_corr[iu]) - np.nan_to_num(prot[iu]))
        return float(d.mean()) if d.size else 0.0

    def __call__(self, levels: dict[str, int], index: pd.Index, suppressed_frac: float) -> dict:
        info = self.info_score(levels)
        cd = self.corr_diff(levels, index)
        proxy = (0.7 * info + 0.3 * (1 - cd)) * (1 - suppressed_frac)
        return {"proxy": float(proxy), "info_score": float(info), "corr_diff": float(cd)}


# ---------------------------------------------------------------- features
def build_features(orig_part: pd.DataFrame, prot_part: pd.DataFrame | None, profiles: list[ColumnProfile],
                   ladders: dict[str, Ladder], levels: dict[str, int]) -> pd.DataFrame:
    """Feature matrix for the original part (prot_part None) or the protected part (aligned on index)."""
    feats = {}
    for p in profiles:
        if p.role in (Role.TARGET, Role.FREE_TEXT) or p.kind == Kind.ID or p.name not in orig_part.columns:
            continue
        if prot_part is not None and (p.gate == Gate.RED or p.name not in prot_part.columns):
            continue
        lad = ladders.get(p.name)
        lvl = levels.get(p.name, 0) if prot_part is not None else 0
        if lad is not None and lad.numeric_capable and p.gate == Gate.YELLOW:
            # numeric view of the generalised value (computed from the original value at the same level)
            feats[p.name] = lad.numeric(orig_part[p.name], lvl).loc[(prot_part if prot_part is not None else orig_part).index]
            continue
        src = (prot_part if prot_part is not None else orig_part)[p.name]
        num = to_number(src)
        if num.notna().mean() > 0.95:
            feats[p.name] = num
        elif p.kind == Kind.DATE or p.ladder in ("date", "timestamp"):
            d = parse_dates(src)
            feats[p.name] = (d - pd.Timestamp("1970-01-01")).dt.days.astype(float)
        else:
            feats[p.name] = src.astype("string").fillna("<NA>")
    return pd.DataFrame(feats, index=(prot_part if prot_part is not None else orig_part).index)


def _encode(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tr, te = train.copy(), test.copy()
    for c in tr.columns:
        # a column is categorical if it is non-numeric in EITHER part (decided jointly so train/test agree)
        if not (pd.api.types.is_numeric_dtype(tr[c]) and pd.api.types.is_numeric_dtype(te[c])):
            tr[c], te[c] = tr[c].astype("string").fillna("<NA>"), te[c].astype("string").fillna("<NA>")
            cats = {v: i for i, v in enumerate(sorted(tr[c].astype(str).unique()))}
            tr[c] = tr[c].astype(str).map(cats).astype(float)
            te[c] = te[c].astype(str).map(cats).fillna(-1).astype(float)
    return tr.fillna(-999).to_numpy(float), te.fillna(-999).to_numpy(float)


def split_original(df: pd.DataFrame, target: str, task: str, seed: int) -> tuple[pd.Index, pd.Index]:
    """80/20 split of the ORIGINAL records, done before any transformation."""
    y = df[target].astype("string").fillna("<NA>")
    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    tr, te = train_test_split(df.index, test_size=0.2, random_state=seed, stratify=strat)
    return pd.Index(np.sort(tr)), pd.Index(np.sort(te))


def _fit_eval(Xtr, ytr, Xte, yte, task: str, seed: int) -> dict:
    if task == "classification":
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        classes = list(model.classes_)
        proba = model.predict_proba(Xte)
        yte = np.asarray(yte)
        # one-vs-rest AUC averaged over the classes the model knows that occur in the test set (with both
        # positives and negatives). Classes seen only in the test split (rare labels) cannot be scored.
        aucs = []
        for i, c in enumerate(classes):
            pos = yte == c
            if 0 < pos.sum() < len(yte):
                aucs.append(roc_auc_score(pos.astype(int), proba[:, i]))
        auc = float(np.mean(aucs)) if aucs else float("nan")
        if len(classes) == 2:
            f1 = f1_score(yte, pred, pos_label=classes[1], labels=classes, average="binary", zero_division=0) \
                if set(np.unique(yte)) <= set(classes) else f1_score(yte, pred, average="macro", zero_division=0)
        else:
            f1 = f1_score(yte, pred, average="macro", zero_division=0)
        return {"roc_auc": auc, "f1": float(f1)}
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {"r2": float(r2_score(yte, pred)), "mae": float(mean_absolute_error(yte, pred))}


def task_utility(df: pd.DataFrame, profiles: list[ColumnProfile], ladders: dict[str, Ladder], plan: Plan,
                 secret: bytes, target: str, task: str, seed: int, baseline: dict | None = None) -> dict:
    """Train on original vs. protected data; evaluate each on its own (original / protected) test split."""
    labelled = df[target].notna() & (df[target].astype("string").str.strip() != "")
    if task == "regression":
        labelled &= to_number(df[target]).notna()
    n_unlabelled = int((~labelled).sum())
    df = df[labelled]
    if task == "classification":
        n_classes = df[target].astype("string").str.strip().nunique()
        if n_classes > 50:
            raise ValueError(f"Target '{target}' has {n_classes} distinct values, which is not a classification label. "
                             "Choose a column with a few classes, or use Regression for a numeric target.")
        if n_classes < 2:
            raise ValueError(f"Target '{target}' has only one class; choose another target column.")
    train_idx, test_idx = split_original(df, target, task, seed)
    orig_train, orig_test = df.loc[train_idx], df.loc[test_idx]
    y_tr = orig_train[target].astype(str).str.strip() if task == "classification" else to_number(orig_train[target])
    y_te = orig_test[target].astype(str).str.strip() if task == "classification" else to_number(orig_test[target])

    if baseline is None:
        Xtr_o, Xte_o = _encode(build_features(orig_train, None, profiles, ladders, {}),
                               build_features(orig_test, None, profiles, ladders, {}))
        base = _fit_eval(Xtr_o, y_tr.to_numpy(), Xte_o, y_te.to_numpy(), task, seed)
        n_feat_orig = int(Xtr_o.shape[1])
    else:
        base, n_feat_orig = baseline["original"], baseline["features_original"]

    prot_train = apply_plan(orig_train, profiles, ladders, plan, secret)        # suppressed rows leave the train set
    test_plan = Plan(levels=plan.levels, suppress=False)                         # every future record is generalised
    prot_test = apply_plan(orig_test, profiles, ladders, test_plan, secret)
    ftr = build_features(orig_train.loc[prot_train.index], prot_train, profiles, ladders, plan.levels)
    fte = build_features(orig_test, prot_test, profiles, ladders, plan.levels)
    Xtr_p, Xte_p = _encode(ftr, fte)
    prot = _fit_eval(Xtr_p, y_tr.loc[prot_train.index].to_numpy(), Xte_p, y_te.to_numpy(), task, seed)

    if task == "classification":
        retained = {"roc_auc": prot["roc_auc"] / base["roc_auc"] if base["roc_auc"] and not np.isnan(base["roc_auc"])
                    and not np.isnan(prot["roc_auc"]) else 0.0,
                    "f1": prot["f1"] / base["f1"] if base["f1"] else 0.0}
        headline = "roc_auc"
    else:
        retained = {"r2": prot["r2"] / base["r2"] if base["r2"] > 0 else 0.0,
                    "mae": base["mae"] / prot["mae"] if prot["mae"] else 1.0}
        headline = "r2"
    return {"task": task, "target": target, "model": "RandomForest (200 trees, seeded)", "original": base,
            "protected": prot, "retained": retained, "headline": headline,
            "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)), "rows_without_label": n_unlabelled,
            "protected_train_rows": int(len(prot_train)), "features_original": n_feat_orig,
            "features_protected": int(Xtr_p.shape[1])}


# ---------------------------------------------------------------- descriptive
def descriptive_utility(df: pd.DataFrame, protected: pd.DataFrame, profiles: list[ColumnProfile],
                        ladders: dict[str, Ladder], levels: dict[str, int], key_columns: list[str]) -> dict:
    """Relative error of fixed aggregate queries + per-column distribution distances."""
    released = [p for p in profiles if p.name in protected.columns and p.role != Role.FREE_TEXT and p.kind != Kind.ID]
    numeric_cols = [p.name for p in released if p.name in ladders and ladders[p.name].numeric_capable
                    and p.ladder in ("numeric", "age") and to_number(df[p.name]).notna().mean() > 0.9]
    numeric_cols += [p.name for p in released if p.role == Role.TARGET and to_number(df[p.name]).notna().mean() > 0.95
                     and p.name not in numeric_cols]
    group_cols = [c for c in key_columns if c in protected.columns] or \
                 [p.name for p in released if p.ladder in ("categorical", "code", "postcode") and p.gate != Gate.RED][:3]

    def num(col, frame_index, prot: bool):
        lad = ladders.get(col)
        if lad is not None and lad.numeric_capable:
            return lad.numeric(df[col], levels.get(col, 0) if prot else 0).loc[frame_index]
        return to_number(df[col]).loc[frame_index]

    queries = []
    for g in group_cols:
        lad = ladders.get(g)
        g_orig = (lad.apply(df[g], levels.get(g, 0)) if lad is not None else df[g]).astype("string").fillna("<NA>")
        g_prot = protected[g].astype("string").fillna("<NA>")
        c_o, c_p = g_orig.value_counts(), g_prot.value_counts()
        for grp, n in c_o.items():
            if n < 5:
                continue
            queries.append({"query": f"count by {g}", "group": grp, "original": float(n), "protected": float(c_p.get(grp, 0))})
        for m in numeric_cols:
            if m == g:
                continue
            mo = num(m, df.index, False).groupby(g_orig).mean()
            mp = num(m, protected.index, True).groupby(g_prot.loc[protected.index]).mean()
            for grp, val in mo.items():
                if c_o.get(grp, 0) < 5 or pd.isna(val):
                    continue
                queries.append({"query": f"mean {m} by {g}", "group": grp, "original": float(val),
                                "protected": float(mp.get(grp, np.nan))})
    rel = []
    for q in queries:
        if pd.isna(q["protected"]):
            q["rel_error"] = 1.0
        else:
            q["rel_error"] = abs(q["protected"] - q["original"]) / max(abs(q["original"]), 1e-9)
        rel.append(min(q["rel_error"], 1.0))
    mre = float(np.mean(rel)) if rel else 0.0

    dist = []
    for p in released:
        col = p.name
        lad = ladders.get(col)
        lvl = levels.get(col, 0)
        o = (lad.apply(df[col], lvl) if lad is not None else df[col]).astype("string").fillna("<NA>")
        pr = protected[col].astype("string").fillna("<NA>")
        po, pp = o.value_counts(normalize=True), pr.value_counts(normalize=True)
        tvd = 0.5 * float(po.subtract(pp, fill_value=0).abs().sum())
        entry = {"column": col, "tvd": tvd}
        if lad is not None and lad.numeric_capable:
            a = lad.numeric(df[col], 0).dropna().to_numpy()
            b = lad.numeric(df[col], lvl).loc[protected.index].dropna().to_numpy()
            sd = float(np.std(a)) or 1.0
            if len(a) and len(b):
                entry["wasserstein_sd"] = float(wasserstein_distance(a, b) / sd)
        dist.append(entry)
    dist_scores = [1 - min(1.0, max(d["tvd"], d.get("wasserstein_sd", 0.0))) for d in dist]
    query_score = 1 - min(1.0, mre / 0.5)
    dist_score = float(np.mean(dist_scores)) if dist_scores else 1.0
    return {
        "query_mean_rel_error": mre,
        "query_score": query_score,
        "distribution_score": dist_score,
        "score": float(0.5 * query_score + 0.5 * dist_score),
        "n_queries": len(queries),
        "worst_queries": sorted(queries, key=lambda q: -q["rel_error"])[:8],
        "distributions": dist,
        "group_columns": group_cols,
        "numeric_columns": numeric_cols,
    }
