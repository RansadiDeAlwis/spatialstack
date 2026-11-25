"""Build the 5k stratified ScanNet++ training subset for the LoRA fine-tune.

Usage: python repo/src/lora/make_subset.py
Reads config.FULL_QA, writes config.SUBSET_QA, and checks counts, video files and test-scene overlap.
"""
import collections
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402


def vsibench_test_scenes():
    """Scene names used by VSI-Bench test, from the baseline per-sample log (all 5,130 questions)."""
    fs = sorted(glob.glob(os.path.join(C.BASELINE_RESULTS, "vsibench", "*", "*_samples_vsibench.jsonl")))
    if not fs:
        sys.exit("No baseline VSI-Bench samples found; cannot check test overlap.")
    return {json.loads(l)["doc"]["scene_name"] for l in open(fs[-1])}


def main():
    data = json.load(open(C.FULL_QA))
    by_type = collections.defaultdict(list)
    for x in data:
        by_type[x["question_type"]].append(x)

    rng = random.Random(C.SEED)
    for qtype, n in C.SUBSET_COUNTS.items():
        if len(by_type[qtype]) < n:
            sys.exit(f"Not enough '{qtype}' samples: need {n}, have {len(by_type[qtype])}")

    # Pass 1 - scene coverage: one question from every scene. For each scene pick the question type that is
    # furthest from its quota (largest fraction remaining), so the per-type counts stay exactly as configured.
    remaining = dict(C.SUBSET_COUNTS)
    by_scene = collections.defaultdict(list)
    for x in data:
        if x["question_type"] in remaining:
            by_scene[x["scene_name"]].append(x)
    scenes_all = sorted(by_scene)
    rng.shuffle(scenes_all)
    subset, used = [], set()
    for sc in scenes_all:
        types = sorted({x["question_type"] for x in by_scene[sc] if remaining[x["question_type"]] > 0})
        if not types:
            continue
        best = max(remaining[t] / C.SUBSET_COUNTS[t] for t in types)
        qtype = rng.choice([t for t in types if remaining[t] / C.SUBSET_COUNTS[t] == best])
        pick = rng.choice([x for x in by_scene[sc] if x["question_type"] == qtype])
        subset.append(pick)
        used.add(pick["id"])
        remaining[qtype] -= 1

    # Pass 2 - fill each type's remaining quota at random from the unused questions.
    for qtype, n in remaining.items():
        pool = [x for x in by_type[qtype] if x["id"] not in used]
        subset += rng.sample(pool, n)
    rng.shuffle(subset)

    n_scenes_total = len({x["scene_name"] for x in data})

    # Checks
    counts = collections.Counter(x["question_type"] for x in subset)
    scenes = {x["scene_name"] for x in subset}
    missing = sorted({x["video"] for x in subset if not os.path.isfile(os.path.join(C.MEDIA_ROOT, x["video"]))})
    overlap = scenes & vsibench_test_scenes()

    print(f"Subset: {len(subset)} samples, {len(scenes)} / {n_scenes_total} scenes")
    for k, v in sorted(counts.items()):
        print(f"  {v:5d}  {k}")
    print(f"Missing videos: {len(missing)}  | Scenes overlapping VSI-Bench test: {len(overlap)}")
    dup_ids = len(subset) - len({x["id"] for x in subset})
    print(f"Duplicate questions: {dup_ids}")
    if (missing or overlap or dup_ids or len(scenes) != n_scenes_total
            or counts != collections.Counter(C.SUBSET_COUNTS)):
        sys.exit("Check failed; subset NOT written.")

    os.makedirs(os.path.dirname(C.SUBSET_QA), exist_ok=True)
    json.dump(subset, open(C.SUBSET_QA, "w"))
    print("Wrote", C.SUBSET_QA)


if __name__ == "__main__":
    main()
