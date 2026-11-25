"""Baseline vs LoRA comparison in the paper's table layouts, recomputed from lmms-eval per-sample logs.

Usage:
  python repo/src/lora/compare_lora.py --run lora=results/lora                 # all benchmarks
  python repo/src/lora/compare_lora.py --run lora=results/lora --scannetpp     # VSI-Bench ScanNet++ subset only
Options: --base <dir> (default results/spatialstack), several --run name=dir, --out <file.md>

Tables:  Table 3 (VSI-Bench, 8 categories) · Table 4 (CV-Bench 2D/3D) · Table 2/6 (VSI | SPAR | BLINK-Spatial | CV |
Overall) · plus per-subtask breakdowns for BLINK-Spatial, SPAR-Bench and MMSI-Bench. Every new run gets a Δ row.
Scoring follows the paper/lmms-eval: MRA for numeric answers, accuracy for multiple choice; VSI Avg = mean of
8 categories (RelDir = mean of its easy/medium/hard accuracies); CV Avg = (2D + 3D)/2 with 2D = mean(ADE20K, COCO);
BLINK-Spatial = mean of its 3 subtasks; Table 2 Overall = mean of VSI, SPAR, BLINK-Spatial, CV.
"""
import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402

PAPER = {  # SpatialStack-5B (Qwen3.5) as reported in the paper
    "vsi": {"Avg": 67.5, "ObjCount": 71.0, "AbsDist": 55.6, "ObjSize": 69.1, "RoomSize": 68.2,
            "RelDist": 67.3, "RelDir": 84.1, "RoutePlan": 41.2, "ApprOrder": 83.5},
    "cv": {"2D": 78.9, "3D": 92.2, "Avg": 85.5},
    "t2": {"VSI": 67.52, "SPAR": 71.39, "BLINK-Spatial": 52.12, "CV": 85.53, "Overall": 69.14},
}
VSI_CATS = [("object_counting", "ObjCount"), ("object_abs_distance", "AbsDist"),
            ("object_size_estimation", "ObjSize"), ("room_size_estimation", "RoomSize"),
            ("object_rel_distance", "RelDist"), ("object_rel_direction", "RelDir"),
            ("route_planning", "RoutePlan"), ("obj_appearance_order", "ApprOrder")]
BLINK_TASKS = ["blink_multi_view_reasoning", "blink_relative_depth", "blink_spatial_relation"]


# ---------------------------------------------------------------- loading
def samples(root, task):
    fs = sorted(glob.glob(os.path.join(root, "**", f"*_samples_{task}.jsonl"), recursive=True))
    return [json.loads(l) for l in open(fs[-1])] if fs else None


def mean(xs):
    xs = list(xs)
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def score_of(d):
    """accuracy for multiple choice, MRA for numeric answers, vci_metric for SPAR view-change questions."""
    for k in ("accuracy", "MRA:.5:.95:.05", "vci_metric"):
        if d.get(k) is not None:
            return d[k]
    return None


# ---------------------------------------------------------------- benchmarks
def vsi(root, scannetpp_only=False):
    s = None
    if scannetpp_only:
        s = samples(root, "vsibench_scannetpp")
    if s is None:
        s = samples(root, "vsibench")
    if s is None:
        return None
    rows = [(x["vsibench_score"]["question_type"], score_of(x["vsibench_score"]), x["doc"]["dataset"]) for x in s]
    if scannetpp_only:
        rows = [r for r in rows if r[2] == "scannetpp"]
    by = collections.defaultdict(list)
    for qt, sc, _ in rows:
        by[qt].append(sc)
    out = {}
    for qt, name in VSI_CATS:
        if qt == "object_rel_direction":  # mean of easy/medium/hard sub-accuracies
            subs = [mean(v) for k, v in by.items() if k.startswith(qt)]
            out[name] = sum(subs) / len(subs) if subs else float("nan")
        else:
            out[name] = mean(by.get(qt, []))
    valid = [v for v in out.values() if v == v]
    out = {"Avg": sum(valid) / len(valid), **out}
    return out, len(rows)


def cv(root):
    s = samples(root, "cvbench")
    if s is None:
        return None
    by = collections.defaultdict(list)
    for x in s:
        by[x["cvbench_score"]["source"]].append(x["cvbench_score"]["result"])
    d2 = (mean(by["ADE20K"]) + mean(by["COCO"])) / 2
    d3 = mean(by["Omni3D"])
    return {"2D": d2, "3D": d3, "Avg": (d2 + d3) / 2}, len(s)


def blink(root):
    out, n = {}, 0
    for t in BLINK_TASKS:
        s = samples(root, t)
        if s is None:
            return None
        acc = [float(v["is_correct"]) if isinstance(v, dict) else float(v) for v in (x["blink_acc"] for x in s)]
        out[t.replace("blink_", "")] = mean(acc)
        n += len(s)
    out = {"Avg": sum(out.values()) / 3, **out}
    return out, n


def spar(root):
    s = samples(root, "sparbench")
    if s is None:
        return None
    by = collections.defaultdict(list)
    for x in s:
        by[x["sparbench_score"]["task"]].append(score_of(x["sparbench_score"]))
    out = {t: mean(v) for t, v in sorted(by.items())}
    avg = None  # official aggregate from the results json
    for f in sorted(glob.glob(os.path.join(root, "**", "*_results.json"), recursive=True), reverse=True):
        r = json.load(open(f))["results"].get("sparbench")
        if r and "sparbench_score,none" in r:
            avg = r["sparbench_score,none"]
            break
    return {"Avg": avg if avg is not None else mean(sum(by.values(), [])), **out}, len(s)


def mmsi(root):
    s = samples(root, "mmsibench")
    if s is None:
        return None
    by = collections.defaultdict(list)
    for x in s:
        by[x["average"]["l2_category"]].append(x["average"]["score"])
    allv = [v for vs in by.values() for v in vs]
    return {"Avg": mean(allv), **{k: mean(v) for k, v in sorted(by.items())}}, len(s)


# ---------------------------------------------------------------- tables
def fmt(v):
    return "—" if v is None or v != v else f"{v:.2f}"


def table(title, cols, rows, base_name):
    """rows: list of (name, dict or None). Adds a Δ row (vs base_name) after every non-base, non-paper row."""
    base = dict(rows).get(base_name)
    lines = [f"### {title}", "", "| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, vals in rows:
        vals = vals or {}
        lines.append(f"| {name} | " + " | ".join(fmt(vals.get(c)) for c in cols) + " |")
        if base and name not in (base_name,) and not name.startswith("Paper") and vals:
            ds = []
            for c in cols:
                a, b = vals.get(c), base.get(c)
                ds.append("—" if a is None or b is None or a != a or b != b else f"{a - b:+.2f}")
            lines.append(f"| *Δ vs {base_name}* | " + " | ".join(ds) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=C.BASELINE_RESULTS)
    ap.add_argument("--run", action="append", default=[], help="name=results_dir (repeatable)")
    ap.add_argument("--scannetpp", action="store_true", help="VSI-Bench ScanNet++ subset only")
    ap.add_argument("--out")
    a = ap.parse_args()
    runs = [("Reproduced (baseline)", a.base)] + [tuple(r.split("=", 1)) for r in a.run]
    base_name = runs[0][0]
    parts = []

    if a.scannetpp:
        res = [(n, vsi(d, scannetpp_only=True)) for n, d in runs]
        cols = ["Avg"] + [c for _, c in VSI_CATS]
        parts.append(table("VSI-Bench, ScanNet++ scenes only (Table 3 layout)", cols,
                           [(n, r[0] if r else None) for n, r in res], base_name))
        parts.append("N: " + ", ".join(f"{n}={r[1] if r else 0}" for n, r in res) + "\n")
    else:
        R = {n: {"vsi": vsi(d), "cv": cv(d), "blink": blink(d), "spar": spar(d), "mmsi": mmsi(d)} for n, d in runs}
        get = lambda n, b: R[n][b][0] if R[n][b] else None  # noqa: E731

        cols = ["Avg"] + [c for _, c in VSI_CATS]
        parts.append(table("Table 3: VSI-Bench", cols,
                           [("Paper SpatialStack-5B", PAPER["vsi"])] + [(n, get(n, "vsi")) for n, _ in runs], base_name))
        parts.append(table("Table 4: CV-Bench", ["2D", "3D", "Avg"],
                           [("Paper SpatialStack-5B", PAPER["cv"])] + [(n, get(n, "cv")) for n, _ in runs], base_name))

        def t2(n):
            v = {"VSI": (get(n, "vsi") or {}).get("Avg"), "SPAR": (get(n, "spar") or {}).get("Avg"),
                 "BLINK-Spatial": (get(n, "blink") or {}).get("Avg"), "CV": (get(n, "cv") or {}).get("Avg")}
            if all(x is not None for x in v.values()):
                v["Overall"] = sum(v.values()) / 4
            return v
        parts.append(table("Table 2/6: spatial benchmarks overall", ["VSI", "SPAR", "BLINK-Spatial", "CV", "Overall"],
                           [("Paper SpatialStack-5B", PAPER["t2"])] + [(n, t2(n)) for n, _ in runs], base_name))

        for key, title in [("blink", "BLINK-Spatial subtasks"), ("spar", "SPAR-Bench per task"),
                           ("mmsi", "MMSI-Bench per category (not reported in the paper)")]:
            cols = []
            for n, _ in runs:
                for c in (get(n, key) or {}):
                    if c not in cols:
                        cols.append(c)
            if cols:
                # transpose for readability: one row per subtask
                lines = [f"### {title}", "", "| Subtask | " + " | ".join(n for n, _ in runs) +
                         "".join(f" | Δ {n}" for n, _ in runs[1:]) + " |",
                         "|---|" + "---|" * (2 * len(runs) - 1)]
                for c in cols:
                    vals = [(get(n, key) or {}).get(c) for n, _ in runs]
                    deltas = [("—" if v is None or vals[0] is None else f"{v - vals[0]:+.2f}") for v in vals[1:]]
                    lines.append(f"| {c} | " + " | ".join(fmt(v) for v in vals) + "".join(f" | {d}" for d in deltas) + " |")
                parts.append("\n".join(lines) + "\n")

        parts.append("N per benchmark: " + "; ".join(
            f"{n}: " + ", ".join(f"{b}={R[n][b][1]}" for b in R[n] if R[n][b]) for n, _ in runs) + "\n")

    txt = "\n".join(parts)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()
