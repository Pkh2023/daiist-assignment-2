"""
Cost model shared by train.py (to pick the operating point) and app.py (to show it live).

Unit of decision: one object on the belt (a potato, a stone or a cut piece).
Positive = should be removed from the line, for one of two reasons with different costs:
  defect - a potato with a rot/blemish spot (dataset class 'bad', plus 'cut')
  stone  - a stone travelling with the potatoes
Every figure below is an assumption argued in REPORT.md; change it here and both the
pipeline and the dashboard follow.
"""

from pathlib import Path

import numpy as np
import pandas as pd

COSTS = {
    "fn_defect": 1.50,   # EUR: rotten potato reaches a 2.5 kg retail bag -> refund/claim on the bag
    "fn_stone": 25.0,    # EUR: stone reaches a bag or the washer/peeler -> complaint or equipment damage
    "fp": 0.08,          # EUR: good potato diverted to the feed/starch stream (lost margin on ~200 g)
    "tp": 0.0,           # a rejected bad item was never sellable; ejection itself is ~free
    "tn": 0.0,
    "review": 0.025,     # EUR per item a person inspects: 3 s at ~EUR 30/h loaded labour
}
HUMAN = {"miss": 0.10, "false_reject": 0.02}   # error rates of a person on the review belt
BASE_RATES = {"defect": 0.01, "stone": 0.002}  # share of objects on the belt (see REPORT.md)
LINE = {"objects_per_hour": 50_000,             # ~10 t/h at ~200 g per potato
        "hours_per_day": 16,                    # two shifts
        "seconds_per_review": 3.0,
        "utilisation": 0.75,                    # a reviewer is not productive every second
        "reviewers": 2}                         # the people already at today's picking table

# fine steps at the bottom, where the scores of defective potatoes actually sit
THRESHOLDS = np.unique(np.round(np.concatenate([[0.001, 0.002, 0.003, 0.005, 0.0075],
                                                 np.arange(0.01, 1.00, 0.01)]), 4))


def _rates(units: pd.DataFrame, t_low, t_high):
    """Share of each truth group that is auto-rejected (score >= t_high) and sent to review
    (t_low <= score < t_high). Phantom units (reject predictions on nothing) count as
    negatives but are normalised by the number of real good objects."""
    arr = _arrays(units)
    out = {}
    for grp in ["defect", "stone", "good"]:
        s, n = arr[grp]
        out[grp] = ((s >= t_high).sum() / n, ((s >= t_low) & (s < t_high)).sum() / n)
    return out


_CACHE = {}


def _arrays(units: pd.DataFrame) -> dict:
    key = id(units)
    if key not in _CACHE or _CACHE[key][0] is not units:
        real = units[~units.phantom]
        arr = {g: (real.loc[real.truth == g, "score"].to_numpy(), max(int((real.truth == g).sum()), 1))
               for g in ["defect", "stone"]}
        # phantoms are negatives, normalised by the number of real good objects
        arr["good"] = (units.loc[units.truth == "good", "score"].to_numpy(),
                       max(int((real.truth == "good").sum()), 1))
        _CACHE.clear()
        _CACHE[key] = (units, arr)
    return _CACHE[key][1]


def expected_cost(units, t_low, t_high, costs=COSTS, base=BASE_RATES, human=HUMAN):
    """Expected EUR per 1,000 objects for a pass / review / reject band.
    t_low == t_high gives the model-alone policy with no review."""
    r = _rates(units, t_low, t_high)
    pi = {"defect": base["defect"], "stone": base["stone"]}
    pi["good"] = 1 - pi["defect"] - pi["stone"]
    fn = {}
    for grp in ["defect", "stone"]:
        auto, rev = r[grp]
        fn[grp] = (1 - auto - rev) + rev * human["miss"]
    fp = r["good"][0] + r["good"][1] * human["false_reject"]
    review_share = sum(pi[g] * r[g][1] for g in pi)
    cost = (pi["defect"] * fn["defect"] * costs["fn_defect"]
            + pi["stone"] * fn["stone"] * costs["fn_stone"]
            + pi["good"] * fp * costs["fp"]
            + (pi["defect"] * (1 - fn["defect"]) + pi["stone"] * (1 - fn["stone"])) * costs["tp"]
            + review_share * costs["review"])
    return {"cost_per_1000": 1000 * cost, "review_share": review_share,
            "recall_defect": 1 - fn["defect"], "recall_stone": 1 - fn["stone"],
            "false_reject_rate": fp,
            "cost_fn_defect": 1000 * pi["defect"] * fn["defect"] * costs["fn_defect"],
            "cost_fn_stone": 1000 * pi["stone"] * fn["stone"] * costs["fn_stone"],
            "cost_fp": 1000 * pi["good"] * fp * costs["fp"],
            "cost_review": 1000 * review_share * costs["review"]}


def sweep_single(units, **kw) -> pd.DataFrame:
    rows = [{"threshold": t, **expected_cost(units, t, t, **kw)} for t in THRESHOLDS]
    return pd.DataFrame(rows)


def best_band(units, max_review_share=None, **kw) -> dict:
    best = None
    for i, lo in enumerate(THRESHOLDS):
        for hi in THRESHOLDS[i:]:
            c = expected_cost(units, lo, hi, **kw)
            if max_review_share is not None and c["review_share"] > max_review_share:
                continue
            if best is None or c["cost_per_1000"] < best["cost_per_1000"] - 1e-12:
                best = {"t_low": float(lo), "t_high": float(hi), **c}
    return best


def alternatives(units, threshold, band, **kw) -> dict:
    costs = kw.get("costs", COSTS); base = kw.get("base", BASE_RATES); human = kw.get("human", HUMAN)
    pi_g = 1 - base["defect"] - base["stone"]
    nothing = 1000 * (base["defect"] * costs["fn_defect"] + base["stone"] * costs["fn_stone"])
    everyone = 1000 * (base["defect"] * human["miss"] * costs["fn_defect"]
                       + base["stone"] * human["miss"] * costs["fn_stone"]
                       + pi_g * human["false_reject"] * costs["fp"] + costs["review"])
    return {"do_nothing": nothing, "person_checks_everything": everyone,
            "person_checks_everything_staff": staffing(1.0),
            "model_alone": expected_cost(units, threshold, threshold, **kw)["cost_per_1000"],
            "model_plus_review": band["cost_per_1000"]}


def sensitivity(units) -> list:
    """Re-pick the model-alone threshold and the review band when one assumption moves."""
    cases = [("as assumed", {}, {})]
    for key, label in [("fn_defect", "missed defect"), ("fn_stone", "missed stone"), ("fp", "good potato thrown out")]:
        for f in (0.25, 4.0):
            cases.append((f"{label} cost x{f:g}", {key: COSTS[key] * f}, {}))
    for grp, vals in [("defect", (0.003, 0.03)), ("stone", (0.0005, 0.01))]:
        for v in vals:
            cases.append((f"{grp} base rate {100 * v:g}%", {}, {grp: v}))
    rows = []
    for label, dc, db in cases:
        costs = dict(COSTS, **dc); base = dict(BASE_RATES, **db)
        sw = sweep_single(units, costs=costs, base=base)
        b = sw.loc[sw.cost_per_1000.idxmin()]
        band = best_band(units, max_review_share=max_review_share(), costs=costs, base=base)
        rows.append({"case": label, "best_threshold": float(b.threshold),
                     "model_alone_cost": float(b.cost_per_1000),
                     "t_low": band["t_low"], "t_high": band["t_high"],
                     "band_cost": float(band["cost_per_1000"]), "review_share": float(band["review_share"])})
    return rows


def max_review_share(line=LINE) -> float:
    """Largest share of objects the available reviewers can inspect."""
    per_hour = line["reviewers"] * 3600 * line["utilisation"] / line["seconds_per_review"]
    return per_hour / line["objects_per_hour"]


def staffing(review_share: float, line=LINE) -> dict:
    per_hour = review_share * line["objects_per_hour"]
    people = per_hour * line["seconds_per_review"] / (3600 * line["utilisation"])
    return {"reviews_per_hour": per_hour, "reviews_per_day": per_hour * line["hours_per_day"],
            "review_hours_per_day": per_hour * line["hours_per_day"] * line["seconds_per_review"] / 3600,
            "people_on_line_at_once": people}


def unit_confusion(units, t_low, t_high) -> pd.DataFrame:
    """Counts on the test belt: truth x (pass, review, reject)."""
    rows = {}
    for grp, sub in [("good", units[units.truth == "good"]),
                     ("defect", units[units.truth == "defect"]),
                     ("stone", units[units.truth == "stone"])]:
        s = sub.score.to_numpy()
        rows[grp] = {"pass": int((s < t_low).sum()),
                     "review": int(((s >= t_low) & (s < t_high)).sum()),
                     "reject": int((s >= t_high).sum())}
    return pd.DataFrame(rows).T[["pass", "review", "reject"]]


def plot_sweep(sweep: pd.DataFrame, chosen: float, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep.threshold, sweep.cost_per_1000, lw=2.2, label="total")
    ax.plot(sweep.threshold, sweep.cost_fn_defect, lw=1, ls="--", label="missed defects")
    ax.plot(sweep.threshold, sweep.cost_fn_stone, lw=1, ls="--", label="missed stones")
    ax.plot(sweep.threshold, sweep.cost_fp, lw=1, ls="--", label="good potatoes thrown out")
    c = float(sweep.loc[(sweep.threshold - chosen).abs().idxmin(), "cost_per_1000"])
    ax.axvline(chosen, color="k", lw=0.8)
    ax.scatter([chosen], [c], color="k", zorder=5)
    ax.annotate(f"t = {chosen:.2f}\nEUR {c:.2f} / 1,000", (chosen, c), xytext=(10, 25),
                textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("reject threshold (model confidence, log scale)")
    ax.set_ylabel("expected EUR per 1,000 objects")
    ax.set_title("Cost against threshold, model deciding alone")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
