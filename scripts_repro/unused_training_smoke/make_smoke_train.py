"""PIPELINE SMOKE TEST DATA ONLY. Builds a tiny training set in the paper's VLM-3R format from
VSI-Bench ScanNet QAs (the real training videos, Journey9ni/aweb, are gated). A model trained on
this is contaminated and must never be evaluated on VSI-Bench."""
import json, os, random, datasets
ROOT = "/home/anujaya/ransadi/SpatialStack/repo"
ds = datasets.load_dataset("nyu-visionx/VSI-Bench", split="test")
ds = [x for x in ds if x["dataset"] == "scannet"]
random.seed(0); random.shuffle(ds)
out = []
for x in ds[:256]:
    q = "These are frames of a video.\n" + x["question"]
    if x["options"]:
        q += "\nOptions:\n" + "\n".join(x["options"]) + "\nAnswer with the option's letter from the given choices directly."
    else:
        q += "\nPlease answer the question using a single word or phrase."
    out.append({"id": str(x["id"]), "data_source": "scannet", "video": f"scannet/videos/{x['scene_name']}.mp4",
                "conversations": [{"from": "human", "value": q}, {"from": "gpt", "value": str(x["ground_truth"])}],
                "scene_name": x["scene_name"]})
os.makedirs(f"{ROOT}/data/smoke", exist_ok=True)
json.dump(out, open(f"{ROOT}/data/smoke/vsibench_scannet_SMOKE_ONLY.json", "w"), indent=1)
print(len(out), "samples,", len({o["scene_name"] for o in out}), "scenes"); print(json.dumps(out[0])[:400])
