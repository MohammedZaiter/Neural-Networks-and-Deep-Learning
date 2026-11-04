# SafeSight — PPE Compliance Classification with Deep Learning

SafeSight classifies cropped regions from construction-site images into PPE classes (e.g. `Hardhat` vs `NO-Hardhat`, `Safety Vest` vs `NO-Safety Vest`). It compares models of increasing capacity, from an MLP baseline through CNNs and a random hyperparameter search to transfer learning.

> **Scope:** this is a **crop / region classifier**. It relies on the dataset's annotated bounding boxes. It does not do full-scene detection or associate people with their gear, so mAP is not the main metric here.

## Repository layout

| Path | Contents |
|------|----------|
| `SafeSight_PPE_Deep_Learning.ipynb` | End-to-end experiment notebook (data loading, cropping, training, evaluation) |
| `requirements.txt` | Python dependencies |
| `artifacts/` | Run outputs: trained `.keras` models, metrics (CSV/JSON), plots |
| `docs/` | Project report (DOCX), presentation (PPTX), speaker notes |

## Results

Validation set (model selection by macro-F1):

| Model | Accuracy | Macro-F1 | Balanced acc. | Unsafe recall |
|-------|---------:|---------:|--------------:|--------------:|
| 01 MLP baseline | 0.573 | 0.568 | 0.592 | 0.574 |
| 02 CNN (ReLU) | 0.702 | 0.696 | 0.710 | 0.778 |
| 03 CNN (ELU + dropout) | 0.694 | 0.689 | 0.709 | 0.692 |
| 04 Best random-search CNN | 0.631 | 0.629 | 0.643 | 0.628 |
| 05 Transfer learning (frozen) | 0.855 | 0.849 | 0.859 | 0.846 |
| **06 Transfer learning (fine-tuned)** | **0.898** | **0.896** | **0.900** | **0.907** |

Held-out test set, chosen model `06_transfer_finetuned` (269 crops):

| Accuracy | Macro-F1 | Balanced acc. | Unsafe recall |
|---------:|---------:|--------------:|--------------:|
| 0.914 | 0.909 | 0.920 | 0.924 |

The test set was not used for tuning. See `artifacts/` for the confusion matrix, learning curves, and an error gallery.

## Dataset

[Construction Site Safety v27](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27) (YOLOv8 export, 2,801 augmented images, 10 classes), by Roboflow Universe Projects, licensed **CC BY 4.0**.

The dataset is **not committed** to this repository. The notebook downloads it automatically.

## Reproducing

1. Install the dependencies: `pip install -r requirements.txt` (or open the notebook in Google Colab with a GPU).
2. Provide the dataset in one of two ways:
   - Set the `ROBOFLOW_API_KEY` environment variable (or a Colab secret of the same name), **or**
   - Set `LOCAL_DATASET_DIR` to an extracted YOLOv8 export of the dataset.
3. Run all cells. Outputs are written to `artifacts/`.

Never commit API keys or put them in notebooks, slides, or screenshots.

Key configuration (`artifacts/configuration.json`): image size 160, batch size 32, 15 scratch epochs, 9 search epochs, 15 transfer epochs and 8 fine-tune epochs.
