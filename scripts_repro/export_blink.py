"""Export BLINK spatial subsets to the local load_from_disk layout the repo's task yamls expect."""
import datasets, os
out = "cache/datasets/BLINK-Benchmark__BLINK"
for sub in ["Multi-view_Reasoning", "Relative_Depth", "Spatial_Relation"]:
    ds = datasets.load_dataset("BLINK-Benchmark/BLINK", sub)
    ds.save_to_disk(os.path.join(out, sub))
    print(sub, {k: len(v) for k, v in ds.items()})
