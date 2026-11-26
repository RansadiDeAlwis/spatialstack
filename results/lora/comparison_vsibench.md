### Table 3: VSI-Bench

| Model | Avg | ObjCount | AbsDist | ObjSize | RoomSize | RelDist | RelDir | RoutePlan | ApprOrder |
|---|---|---|---|---|---|---|---|---|---|
| Paper SpatialStack-5B | 67.50 | 71.00 | 55.60 | 69.10 | 68.20 | 67.30 | 84.10 | 41.20 | 83.50 |
| Reproduced (baseline) | 67.54 | 70.53 | 55.59 | 68.95 | 68.33 | 67.18 | 84.30 | 42.27 | 83.17 |
| lora | 66.81 | 71.38 | 55.83 | 68.61 | 65.97 | 65.77 | 83.16 | 41.24 | 82.52 |
| *Δ vs Reproduced (baseline)* | -0.73 | +0.85 | +0.24 | -0.34 | -2.36 | -1.41 | -1.14 | -1.03 | -0.65 |

### Table 4: CV-Bench

| Model | 2D | 3D | Avg |
|---|---|---|---|
| Paper SpatialStack-5B | 78.90 | 92.20 | 85.50 |
| Reproduced (baseline) | 79.06 | 92.33 | 85.69 |
| lora | — | — | — |

### Table 2/6: spatial benchmarks overall

| Model | VSI | SPAR | BLINK-Spatial | CV | Overall |
|---|---|---|---|---|---|
| Paper SpatialStack-5B | 67.52 | 71.39 | 52.12 | 85.53 | 69.14 |
| Reproduced (baseline) | 67.54 | 68.40 | 51.30 | 85.69 | 68.23 |
| lora | 66.81 | — | — | — | — |
| *Δ vs Reproduced (baseline)* | -0.73 | — | — | — | — |

### BLINK-Spatial subtasks

| Subtask | Reproduced (baseline) | lora | Δ lora |
|---|---|---|---|
| Avg | 51.30 | — | — |
| multi_view_reasoning | 18.80 | — | — |
| relative_depth | 69.35 | — | — |
| spatial_relation | 65.73 | — | — |

### SPAR-Bench per task

| Subtask | Reproduced (baseline) | lora | Δ lora |
|---|---|---|---|
| Avg | 68.40 | — | — |
| camera_motion_infer | 67.25 | — | — |
| depth_prediction_oc | 81.56 | — | — |
| depth_prediction_oc_mv | 74.91 | — | — |
| depth_prediction_oo | 24.73 | — | — |
| depth_prediction_oo_mv | 18.09 | — | — |
| distance_infer_center_oo | 89.41 | — | — |
| distance_infer_center_oo_mv | 86.61 | — | — |
| distance_prediction_oc | 75.45 | — | — |
| distance_prediction_oc_mv | 76.08 | — | — |
| distance_prediction_oo | 64.27 | — | — |
| distance_prediction_oo_mv | 47.98 | — | — |
| obj_spatial_relation_oc_mv | 89.25 | — | — |
| obj_spatial_relation_oo | 83.24 | — | — |
| obj_spatial_relation_oo_mv | 81.99 | — | — |
| position_matching | 86.01 | — | — |
| spatial_imagination_oc | 74.73 | — | — |
| spatial_imagination_oc_mv | 73.84 | — | — |
| spatial_imagination_oo | 65.56 | — | — |
| spatial_imagination_oo_mv | 74.23 | — | — |
| view_change_infer | 32.91 | — | — |

### MMSI-Bench per category (not reported in the paper)

| Subtask | Reproduced (baseline) | lora | Δ lora |
|---|---|---|---|
| Avg | 28.20 | — | — |
| Attribute (Appr.) | 27.27 | — | — |
| Attribute (Meas.) | 26.56 | — | — |
| MSR | 25.25 | — | — |
| Motion (Cam.) | 20.27 | — | — |
| Motion (Obj.) | 23.68 | — | — |
| Positional Relationship (Cam.–Cam.) | 23.66 | — | — |
| Positional Relationship (Cam.–Obj.) | 47.67 | — | — |
| Positional Relationship (Cam.–Reg.) | 38.55 | — | — |
| Positional Relationship (Obj.–Obj.) | 23.40 | — | — |
| Positional Relationship (Obj.–Reg.) | 23.53 | — | — |
| Positional Relationship (Reg.–Reg.) | 33.33 | — | — |

N per benchmark: Reproduced (baseline): vsi=5130, cv=2638, blink=400, spar=7211, mmsi=1000; lora: vsi=5130
