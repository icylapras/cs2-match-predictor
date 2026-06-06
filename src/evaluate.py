"""Head-to-head evaluation: ML model vs the raw Elo calculator vs naive.

Answers two questions on a *chronological* (out-of-time) held-out test set:

  1. Does adding the from-scratch, leakage-safe Elo features (built by
     ``src.elo``) lift the model over the raw-stats-only model? (the AUC lift)
  2. Does the trained model actually beat the *model-free* raw Elo calculator
     — the same ``elo_win_probability`` formula the live site shows as its
     FACEIT-Elo bar, applied here to the leakage-safe replayed Elo so it can be
     backtested fairly (using players' *current* FACEIT Elo on historical
     matches would be leakage)?

For every contender we report accuracy, AUC-ROC, log-loss and Brier score so
the comparison covers both ranking (AUC) and calibration (log-loss/Brier).

Usage:
    python -m src.elo --data data/processed/matches.csv   # build elo features first
    python -m src.evaluate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
import datetime
import json

from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from xgboost import XGBClassifier

from src.features import FEATURE_KEYS, elo_win_probability
from src.train import chronological_split

STAT_COLS = [f"{key}_diff" for key in FEATURE_KEYS]
FACEIT_COL = ["faceit_elo_diff"]  # current FACEIT Elo (the user-facing signal)
REPLAYED_COLS = ["elo_diff", "oppelo_diff"]  # leakage-free Elo (rigor cross-check)

DEFAULT_DATA = Path("data/processed/matches.csv")
DEFAULT_FEATURES = Path("data/processed/elo_features.csv")


def _models() -> dict:
    """Fresh model instances (same config as src.train)."""
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def _scores(y_true, proba) -> dict[str, float]:
    """Accuracy / AUC / log-loss / Brier for probabilistic predictions."""
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    preds = (proba >= 0.5).astype(int)
    # AUC is undefined if a contender outputs a single constant probability.
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = float("nan")
    return {
        "acc": accuracy_score(y_true, preds),
        "auc": auc,
        "logloss": log_loss(y_true, proba, labels=[0, 1]),
        "brier": brier_score_loss(y_true, proba),
    }


def _row(name: str, s: dict[str, float]) -> str:
    return f"{name:32s} acc={s['acc']:.3f}  auc={s['auc']:.3f}  logloss={s['logloss']:.3f}  brier={s['brier']:.3f}"


# --- significance tests for "is contender A really better than B on the SAME rows?" ---
# A point AUC gap on a ~hundreds-of-rows test set can be pure noise, so every
# headline comparison is a *paired* test (both contenders score the same matches):
# a paired bootstrap CI on the AUC difference, cross-checked with DeLong's test.


def paired_bootstrap_auc_diff(
    y_true, proba_a, proba_b, *, n_boot: int = 5000, seed: int = 42
) -> dict[str, float]:
    """Bootstrap the AUC difference (A - B) by resampling the test matches.

    Returns the mean difference, its 95% CI, and a two-sided bootstrap p-value
    (the share of resamples on the wrong side of zero, doubled).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():  # a resample with one class has no defined AUC
            continue
        diffs.append(roc_auc_score(yt, proba_a[idx]) - roc_auc_score(yt, proba_b[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(diffs.mean()), "lo": float(lo), "hi": float(hi), "p": float(min(p, 1.0))}


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Mid-ranks of x, averaging ranks within ties (for DeLong)."""
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_auc_test(y_true, proba_a, proba_b) -> dict[str, float]:
    """Fast DeLong test for AUC(A) == AUC(B) on the same samples (Sun & Xu 2014).

    Returns each AUC and the two-sided p-value for their difference.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)  # positives (label 1) first
    m = int(y_true.sum())  # number of positives
    preds = np.vstack((np.asarray(proba_a), np.asarray(proba_b)))[:, order]
    n = preds.shape[1] - m
    k = 2
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(preds[r, :m])
        ty[r] = _compute_midrank(preds[r, m:])
        tz[r] = _compute_midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else float("nan")
    p = 2.0 * norm.sf(abs(z)) if var > 0 else float("nan")
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "p": float(p)}


def _significance(name: str, y_test, probas: dict, a: str, b: str) -> None:
    """Print the paired bootstrap CI + DeLong p-value for contender a vs b."""
    boot = paired_bootstrap_auc_diff(y_test, probas[a], probas[b])
    dl = delong_auc_test(y_test, probas[a], probas[b])
    verdict = "SIGNIFICANT" if boot["hi"] < 0 or boot["lo"] > 0 else "not significant"
    print(f"\n{name}:")
    print(f"  {a}")
    print(f"  vs {b}")
    print(f"  AUC diff = {boot['diff']:+.3f}  95% CI [{boot['lo']:+.3f}, {boot['hi']:+.3f}]"
          f"  (bootstrap p={boot['p']:.3f}, DeLong p={dl['p']:.3f})  -> {verdict} at 5%")


def evaluate(data: Path, features: Path, *, test_size: float = 0.2) -> dict:
    df = pd.read_csv(data)
    if not features.exists():
        raise FileNotFoundError(
            f"{features} not found — build it first with: python -m src.elo --data {data}"
        )
    elo = pd.read_csv(features)
    df = df.merge(elo, on="match_id", how="inner")
    print(f"Matches with Elo features: {len(df)}")

    train_df, test_df = chronological_split(df, "date", test_size)
    y_train, y_test = train_df["label"], test_df["label"].to_numpy()
    print(f"Train: {len(train_df)} | Test: {len(test_df)}\n")

    results: dict[str, dict] = {}
    probas: dict[str, np.ndarray] = {}  # kept for the paired significance tests

    # --- model-free baseline: the FACEIT Elo calculator (the thing to beat) ---
    base_rate = float(y_train.mean())
    probas["naive (train base rate)"] = np.full(len(y_test), base_rate)
    probas["FACEIT Elo calculator (formula)"] = test_df.apply(
        lambda r: elo_win_probability(r["faceit_elo_a"], r["faceit_elo_b"]), axis=1
    ).to_numpy()

    # --- trained models over increasing feature sets ---
    # The headline question: does (stats + FACEIT Elo) beat FACEIT Elo alone?
    feature_sets = {
        "stats": STAT_COLS,
        "FACEIT Elo": FACEIT_COL,
        "stats + FACEIT Elo": STAT_COLS + FACEIT_COL,
        "stats + replayed Elo": STAT_COLS + REPLAYED_COLS,  # leakage-free cross-check
    }
    for set_name, cols in feature_sets.items():
        for mname, model in _models().items():
            model.fit(train_df[cols], y_train)
            probas[f"{mname} [{set_name}]"] = model.predict_proba(test_df[cols])[:, 1]

    for name, proba in probas.items():
        results[name] = _scores(y_test, proba)

    # --- report ---
    print("=" * 84)
    for name, s in results.items():
        print(_row(name, s))
    print("=" * 84)

    def best(set_name: str) -> str:
        return f"{max(_models(), key=lambda m: results[f'{m} [{set_name}]']['auc'])} [{set_name}]"

    calc = "FACEIT Elo calculator (formula)"
    elo_only = best("FACEIT Elo")
    stats_only = best("stats")
    combo = best("stats + FACEIT Elo")
    replayed_combo = best("stats + replayed Elo")

    print(f"\nFACEIT Elo calculator AUC       : {results[calc]['auc']:.3f}")
    print(f"Model: FACEIT Elo only          : {results[elo_only]['auc']:.3f}  ({elo_only})")
    print(f"Model: stats only               : {results[stats_only]['auc']:.3f}  ({stats_only})")
    print(f"Model: stats + FACEIT Elo       : {results[combo]['auc']:.3f}  ({combo})  <-- hypothesis")

    # The two questions that matter for the user's hypothesis.
    _significance("Does (stats + FACEIT Elo) beat the FACEIT Elo calculator?",
                  y_test, probas, combo, calc)
    _significance("Does adding stats help on top of FACEIT Elo? (combo vs Elo-only model)",
                  y_test, probas, combo, elo_only)
    # Leakage-free corroboration: same question with the replayed Elo.
    _significance("Cross-check (leakage-free): stats + replayed Elo  vs  replayed-Elo calculator",
                  y_test, probas, replayed_combo,
                  _add_replayed_calc(probas, results, test_df, y_test))
    return results


def _add_replayed_calc(probas, results, test_df, y_test) -> str:
    """Add the leakage-free replayed-Elo calculator as a contender; return its key."""
    name = "replayed Elo calculator (formula)"
    probas[name] = test_df.apply(
        lambda r: elo_win_probability(r["elo_a"], r["elo_b"]), axis=1
    ).to_numpy()
    results[name] = _scores(y_test, probas[name])
    return name


def _auc(y, p) -> float:
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return float("nan")


def _bootstrap_auc_ci(y, p, *, n_boot: int = 5000, seed: int = 42) -> dict[str, float]:
    """Bootstrap 95% CI for a single predictor's AUC (to test it vs a 0.5 coin flip)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        if yt.min() == yt.max():
            continue
        aucs.append(roc_auc_score(yt, p[idx]))
    aucs = np.array(aucs)
    return {"auc": float(aucs.mean()), "lo": float(np.percentile(aucs, 2.5)),
            "hi": float(np.percentile(aucs, 97.5))}


def balanced_experiment(
    data: Path, features: Path, *, thresholds=(75, 50, 30), test_size: float = 0.2
) -> None:
    """Train AND test only on *balanced* matches (small team Elo gap).

    The hard, honest regime: once both sides are the same rank, the Elo formula is
    near 50/50, so recent-form stats finally have room to matter. For each Elo-gap
    threshold we restrict the whole dataset, re-split chronologically, and ask:
      (1) do stats beat a coin flip?  (stats-only AUC CI vs 0.5)
      (2) do stats add over the residual Elo?  (stats+Elo vs the Elo calculator)
    """
    df = pd.read_csv(data).merge(pd.read_csv(features), on="match_id", how="inner")
    df["date"] = pd.to_datetime(df["date"])
    print("\n" + "=" * 84)
    print("BALANCED-MATCHES EXPERIMENT — train+test only where the team Elo gap is small")
    print("=" * 84)

    for thr in thresholds:
        sub = df[df["faceit_elo_diff"].abs() < thr].copy()
        if len(sub) < 120:
            print(f"\n|Elo gap| < {thr}: n={len(sub)} — too small, skipped")
            continue
        train_df, test_df = chronological_split(sub, "date", test_size)
        y_train = train_df["label"]
        y_test = test_df["label"].to_numpy()

        probas = {
            "calc": test_df.apply(
                lambda r: elo_win_probability(r["faceit_elo_a"], r["faceit_elo_b"]), axis=1
            ).to_numpy()
        }
        for name, cols in {
            "stats": STAT_COLS,
            "elo": FACEIT_COL,
            "stats+elo": STAT_COLS + FACEIT_COL,
        }.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            m.fit(train_df[cols], y_train)
            probas[name] = m.predict_proba(test_df[cols])[:, 1]

        ci = _bootstrap_auc_ci(y_test, probas["stats"])
        sig = paired_bootstrap_auc_diff(y_test, probas["stats+elo"], probas["calc"])
        beats_coin = "YES" if ci["lo"] > 0.5 else "no"
        beats_calc = "YES" if sig["lo"] > 0 else "no"

        print(f"\n|Elo gap| < {thr}:  n={len(sub)} (train {len(train_df)} / test {len(test_df)}, "
              f"base rate {y_test.mean():.2f})")
        print(f"  AUC  calculator={_auc(y_test, probas['calc']):.3f}  "
              f"elo-only={_auc(y_test, probas['elo']):.3f}  "
              f"stats-only={_auc(y_test, probas['stats']):.3f}  "
              f"stats+elo={_auc(y_test, probas['stats+elo']):.3f}")
        print(f"  (1) stats beat coin flip? stats-only AUC {ci['auc']:.3f} "
              f"95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}] -> {beats_coin}")
        print(f"  (2) stats add over Elo?   (stats+elo - calc) {sig['diff']:+.3f} "
              f"95% CI [{sig['lo']:+.3f}, {sig['hi']:+.3f}] (p={sig['p']:.3f}) -> {beats_calc}")


def close_matchup_analysis(
    data: Path, features: Path, *, test_size: float = 0.2
) -> None:
    """Do stats add value where the Elo gap is small (the calculator is ~50/50)?

    The headline AUC is dominated by lopsided blowouts that Elo calls trivially.
    Here we restrict the test set to the *closest* matchups (smallest |Elo gap|)
    and re-ask whether (stats + Elo) beats the Elo calculator there. Models are
    still trained on the full training set; only the *evaluation* is subsetted.
    """
    df = pd.read_csv(data).merge(pd.read_csv(features), on="match_id", how="inner")
    train_df, test_df = chronological_split(df, "date", test_size)
    y_train = train_df["label"]
    y_test = test_df["label"].to_numpy()
    gap = test_df["faceit_elo_diff"].abs().to_numpy()

    # logreg won on the full set; use it for all three feature views here.
    fitted = {}
    for set_name, cols in {
        "calc": None,
        "stats": STAT_COLS,
        "stats+Elo": STAT_COLS + FACEIT_COL,
    }.items():
        if cols is None:
            fitted[set_name] = test_df.apply(
                lambda r: elo_win_probability(r["faceit_elo_a"], r["faceit_elo_b"]),
                axis=1,
            ).to_numpy()
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            m.fit(train_df[cols], y_train)
            fitted[set_name] = m.predict_proba(test_df[cols])[:, 1]

    print("\n" + "=" * 84)
    print("CLOSE-MATCHUP ANALYSIS — does (stats + Elo) beat the Elo calculator when")
    print("the Elo gap is small? (models trained on full data; test set subsetted)")
    print("=" * 84)

    # Subsets: the closest 25% / 50% of test matches, plus an absolute <100 Elo gap.
    q25, q50 = np.quantile(gap, [0.25, 0.50])
    subsets = [
        (f"closest 25% (|gap| < {q25:.0f})", gap <= q25),
        (f"closest 50% (|gap| < {q50:.0f})", gap <= q50),
        ("|gap| < 100 Elo", gap < 100),
        ("all test", np.ones_like(gap, dtype=bool)),
    ]
    for label, mask in subsets:
        n = int(mask.sum())
        yt = y_test[mask]
        if n < 20 or yt.min() == yt.max():
            print(f"\n{label}: n={n} — too few / single-class, skipped")
            continue
        a_calc = _auc(yt, fitted["calc"][mask])
        a_stats = _auc(yt, fitted["stats"][mask])
        a_combo = _auc(yt, fitted["stats+Elo"][mask])
        boot = paired_bootstrap_auc_diff(yt, fitted["stats+Elo"][mask], fitted["calc"][mask])
        verdict = "SIGNIFICANT" if boot["hi"] < 0 or boot["lo"] > 0 else "not significant"
        print(f"\n{label}: n={n}  (base rate {yt.mean():.2f})")
        print(f"  AUC  calculator={a_calc:.3f}  stats-only={a_stats:.3f}  stats+Elo={a_combo:.3f}")
        print(f"  stats+Elo - calculator = {boot['diff']:+.3f}  95% CI "
              f"[{boot['lo']:+.3f}, {boot['hi']:+.3f}]  (p={boot['p']:.3f}) -> {verdict}")


def _ece(y, p, *, n_bins: int = 10) -> float:
    """Expected Calibration Error: avg gap between predicted prob and observed
    frequency, weighted by bin population. 0 = perfectly calibrated."""
    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def calibration_report(
    data: Path, features: Path, *, test_size: float = 0.2, out_png: Path | None = None
) -> None:
    """Quantify (and optionally plot) the model's calibration edge over the raw
    Elo formula, and whether isotonic scaling sharpens it further.

    The model's only proven advantage over the FACEIT Elo calculator is that its
    probabilities are more *honest* (a "70%" really wins ~70% of the time). This
    measures that with log-loss / Brier / ECE and a reliability diagram.

    Isotonic is fit on a chronological calibration slice (the most recent 20% of
    the training data), never on the test set, so there's no leakage.
    """
    cols = STAT_COLS + FACEIT_COL
    df = pd.read_csv(data).merge(pd.read_csv(features), on="match_id", how="inner")
    train_df, test_df = chronological_split(df, "date", test_size)
    fit_df, calib_df = chronological_split(train_df, "date", 0.2)  # calib = newest of train
    y_te = test_df["label"].to_numpy()

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(fit_df[cols], fit_df["label"])
    p_model = model.predict_proba(test_df[cols])[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(model.predict_proba(calib_df[cols])[:, 1], calib_df["label"])
    p_iso = iso.predict(p_model)

    p_calc = test_df.apply(
        lambda r: elo_win_probability(r["faceit_elo_a"], r["faceit_elo_b"]), axis=1
    ).to_numpy()

    contenders = {
        "FACEIT Elo calculator": p_calc,
        "model (stats+Elo)": p_model,
        "model + isotonic": p_iso,
    }
    print("\n" + "=" * 72)
    print(f"CALIBRATION REPORT (test n={len(y_te)}) — lower log-loss/Brier/ECE = better")
    print("=" * 72)
    for name, p in contenders.items():
        s = _scores(y_te, p)
        print(f"  {name:24s} logloss={s['logloss']:.3f}  brier={s['brier']:.3f}  "
              f"ece={_ece(y_te, p):.3f}  auc={s['auc']:.3f}")

    if out_png is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.calibration import calibration_curve
        except ImportError:
            print("\n(matplotlib not installed — skipping reliability diagram)")
            return
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
        for name, p in contenders.items():
            frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, "o-", label=name)
        ax.set_xlabel("Predicted P(Team A wins)")
        ax.set_ylabel("Observed frequency")
        ax.set_title("Reliability diagram — CS2 match predictor")
        ax.legend(loc="upper left", fontsize=9)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"\nReliability diagram -> {out_png}")


def write_metrics(
    data: Path, features: Path, *, test_size: float = 0.2,
    out: Path = Path("data/processed/metrics.json"),
) -> dict:
    """Compute the out-of-time model-vs-FACEIT-Elo comparison and save it to JSON.

    This is what the website displays as its 'track record'. The model here is the
    deployed architecture (isotonic-calibrated logreg on stats+Elo), trained on the
    training split and scored on the held-out test matches it never saw.
    """
    cols = STAT_COLS + FACEIT_COL
    df = pd.read_csv(data).merge(pd.read_csv(features), on="match_id", how="inner")
    df["date"] = pd.to_datetime(df["date"])
    train_df, test_df = chronological_split(df, "date", test_size)
    y_te = test_df["label"].to_numpy()

    model = CalibratedClassifierCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        method="isotonic", cv=5,
    )
    model.fit(train_df[cols], train_df["label"])
    p_model = model.predict_proba(test_df[cols])[:, 1]
    p_elo = test_df.apply(
        lambda r: elo_win_probability(r["faceit_elo_a"], r["faceit_elo_b"]), axis=1
    ).to_numpy()

    def block(p) -> dict:
        s = _scores(y_te, p)
        return {
            "accuracy_pct": round(s["acc"] * 100),
            "auc": round(s["auc"], 3),
            "ece": round(_ece(y_te, p), 3),
        }

    metrics = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_total": len(df),
        "test_from": str(test_df["date"].min().date()),
        "test_to": str(test_df["date"].max().date()),
        "model": block(p_model),
        "faceit_elo": block(p_elo),
        "generated": datetime.date.today().isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\n-> {out}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--calibration", action="store_true",
                        help="run the calibration report + reliability diagram")
    parser.add_argument("--reliability-png", type=Path, default=Path("reports/reliability.png"))
    parser.add_argument("--metrics", action="store_true",
                        help="write data/processed/metrics.json for the website")
    args = parser.parse_args()
    if args.metrics:
        write_metrics(args.data, args.features, test_size=args.test_size)
        return
    if args.calibration:
        calibration_report(args.data, args.features, test_size=args.test_size,
                           out_png=args.reliability_png)
        return
    evaluate(args.data, args.features, test_size=args.test_size)
    close_matchup_analysis(args.data, args.features, test_size=args.test_size)


if __name__ == "__main__":
    main()
