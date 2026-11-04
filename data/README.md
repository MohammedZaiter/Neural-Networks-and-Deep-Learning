# Dataset

**Construction Site Safety, version 27** — Roboflow Universe Projects, licensed **CC BY 4.0**.
https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27

The images are not committed to this repository (about 150 MB). Only this README is tracked.

## Getting the data

Pick one:

1. **Automatic download.** Set the `ROBOFLOW_API_KEY` environment variable (or a Colab secret with the same name) and run the notebook. It downloads the YOLOv8 export into `data/Construction-Site-Safety-27/`.
2. **Manual download.** Export version 27 in **YOLOv8** format from the link above and extract it here, so this path exists:

   ```
   data/Construction-Site-Safety-27/data.yaml
   ```

   The notebook uses this folder automatically when it exists.
3. **Another location.** Set `LOCAL_DATASET_DIR` to any folder that contains `data.yaml`.

## Expected layout

```
data/Construction-Site-Safety-27/
├── data.yaml
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

## Facts used in the notebook

| Item | Value |
|---|---|
| Exported images | 2,799 (2,603 train / 114 valid / 82 test; Roboflow's page lists 2,801), augmented export of a 717-image source project |
| Annotation format | YOLO boxes: `class_id cx cy width height` (normalized) |
| Source classes | 10: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle |
| Classes used | 4: Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest |
| Crops after cleaning | 11,415 train / 255 valid / 269 test |
