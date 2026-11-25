"""Recompute benchmark scores from lmms-eval per-sample logs and compare to the paper.

Usage: python compare.py <results_dir> [<results_dir> ...]   (e.g. results/spatialstack)
Writes <results_dir>/summary.md and prints it.
"""
import glob, json, os, sys
import pandas as pd

PAPER = {  # SpatialStack (Qwen3.5) row from the paper / project page
    "VSI-Bench": {"Avg": 67.5, "ObjCount": 71.0, "AbsDist": 55.6, "ObjSize": 69.1, "RoomSize": 68.2,
                  "RelDist": 67.3, "RelDir": 84.1, "RoutePlan": 41.2, "ApprOrder": 83.5},
    "CV-Bench": {"2D": 78.9, "3D": 92.2, "Avg": 85.5},
    "BLINK-Spatial": {"Avg": 52.12},
    "SPAR-Bench": {"Avg": 71.39},
    "MMSI-Bench": {"Avg": None},
}
PAPER_QWEN25 = {  # SpatialStack-4B (Qwen2.5-VL-3B base) row; SPAR/BLINK not reported for this variant
    "VSI-Bench": {"Avg": 60.9, "ObjCount": 69.2, "AbsDist": 45.4, "ObjSize": 63.0, "RoomSize": 63.2,
                  "RelDist": 57.9, "RelDir": 68.4, "RoutePlan": 40.2, "ApprOrder": 79.6},
    "CV-Bench": {"2D": 75.4, "3D": 87.0, "Avg": 81.2},
}
VSI_CATS = {"object_counting": "ObjCount", "object_abs_distance": "AbsDist", "object_size_estimation": "ObjSize",
            "room_size_estimation": "RoomSize", "object_rel_distance": "RelDist", "object_rel_direction": "RelDir",
            "route_planning": "RoutePlan", "obj_appearance_order": "ApprOrder"}


def samples(root, task):
    fs = sorted(glob.glob(os.path.join(root, "**", f"*_samples_{task}.jsonl"), recursive=True))
    if not fs:
        return None
    return [json.loads(l) for l in open(fs[-1])]


def vsi(root):
    s = samples(root, "vsibench")
    if not s: return None
    df = pd.DataFrame([x["vsibench_score"] for x in s])
    df["score"] = df["accuracy"].where(df["accuracy"].notna(), df.get("MRA:.5:.95:.05"))
    df["cat"] = df["question_type"].str.replace(r"_(easy|medium|hard)$", "", regex=True)
    out = {}
    for qt, cat in VSI_CATS.items():
        if qt == "object_rel_direction":  # paper: mean of easy/medium/hard sub-accuracies
            subs = df[df["cat"] == qt].groupby("question_type")["score"].mean()
            out[cat] = subs.mean() * 100
        else:
            out[cat] = df[df["cat"] == qt]["score"].mean() * 100
    out["Avg"] = sum(out[c] for c in VSI_CATS.values()) / len(VSI_CATS)
    return out, len(df)


def cvbench(root):
    s = samples(root, "cvbench")
    if not s: return None
    df = pd.DataFrame([x["cvbench_score"] for x in s])
    acc = lambda src: df[df["source"] == src]["result"].mean()
    d2 = (acc("ADE20K") + acc("COCO")) / 2 * 100; d3 = acc("Omni3D") * 100
    return {"2D": d2, "3D": d3, "Avg": (d2 + d3) / 2}, len(df)


def blink(root):
    out, n = {}, 0
    for t in ["blink_multi_view_reasoning", "blink_relative_depth", "blink_spatial_relation"]:
        s = samples(root, t)
        if not s: return None
        accs = []
        for x in s:
            v = x["blink_acc"]
            accs.append(v["is_correct"] if isinstance(v, dict) and "is_correct" in v else v)
        out[t.replace("blink_", "")] = 100 * sum(float(a) for a in accs) / len(accs); n += len(s)
    out["Avg"] = sum(out.values()) / 3  # group config: weight_by_size False
    return out, n


def from_results_json(root, task, key):
    fs = sorted(glob.glob(os.path.join(root, "**", "*_results.json"), recursive=True))
    for f in reversed(fs):
        r = json.load(open(f))["results"].get(task)
        if r:
            v = next((v for k, v in r.items() if k.startswith(key)), None)
            n = json.load(open(f)).get("n-samples", {}).get(task, {}).get("effective")
            if v is not None:
                return {"Avg": v * (100 if v <= 1 else 1)}, n
    return None


def main():
    for root in sys.argv[1:]:
        paper = PAPER_QWEN25 if "qwen25" in root else PAPER
        res = {"VSI-Bench": vsi(root), "CV-Bench": cvbench(root), "BLINK-Spatial": blink(root),
               "SPAR-Bench": from_results_json(root, "sparbench", "sparbench_score"),
               "MMSI-Bench": from_results_json(root, "mmsibench", "average")}
        lines = [f"# Results: {root}\n", "| Benchmark | Metric | Reproduced | Paper | Δ | N |", "|---|---|---|---|---|---|"]
        for b, r in res.items():
            if r is None:
                lines.append(f"| {b} | — | not run | | | |"); continue
            vals, n = r
            for m, v in vals.items():
                p = paper.get(b, {}).get(m)
                d = f"{v - p:+.1f}" if p is not None and v == v else ""
                lines.append(f"| {b} | {m} | {v:.2f} | {p if p is not None else '—'} | {d} | {n} |")
        txt = "\n".join(lines)
        open(os.path.join(root, "summary.md"), "w").write(txt + "\n")
        print(txt)


if __name__ == "__main__":
    main()
