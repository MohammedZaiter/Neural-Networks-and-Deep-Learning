# SafeSight — PPE Compliance Classification with Deep Learning

SafeSight classifies cropped regions from construction-site images into PPE classes (e.g. `Hardhat` vs `NO-Hardhat`, `Safety Vest` vs `NO-Safety Vest`). It compares models of increasing capacity, from an MLP baseline through CNNs and a random hyperparameter search to transfer learning.

> **Scope:** this is a **crop / region classifier**. It relies on the dataset's annotated bounding boxes. It does not do full-scene detection or associate people with their gear, so mAP is not the main metric here.

## Repository layout

```
├── SafeSight_PPE_Deep_Learning.ipynb   main deliverable: the full, documented experiment
├── requirements.txt
├── data/                               dataset location (not committed, see data/README.md)
├── models/
│   ├── final_ppe_classifier.keras      chosen model (fine-tuned MobileNetV2)
│   └── checkpoints/                    best checkpoint of every experiment (01_mlp … 06_transfer_finetuned)
├── results/
│   ├── figures/                        confusion matrix, learning curves, error gallery, dataset plots
│   └── tables/                         validation results, hyperparameter search, test metrics, manifests
└── docs/
    ├── report/                         project report (DOCX)
    └── presentation/                   slides (PPTX) and speaker notes
```

## Notebook structure

The notebook follows the project specification, section by section. Every stage has a rationale, the code, and observations on the output.

| Section | Contents |
|---|---|
| 1. Dataset selection | Source, licence, size, features, target |
| 2. Data preprocessing | Annotation audit (missing values), outlier rule, exploration, leakage check, normalization, augmentation, class weights |
| 3. Step-by-step model creation | MLP → CNN (ReLU) → CNN (ELU + dropout + L2) |
| 4. Hyperparameter tuning | Seeded random search over 7 hyperparameters |
| 5. Transfer learning | MobileNetV2: frozen backbone, then partial fine-tuning |
| 6. Integration of course topics | Map of course concepts to where they are used |
| 7. Evaluation and interpretation | Model comparison, held-out test, learning curves, error analysis, challenges |
| 8. Conclusion | Answer to the research question, gains per stage, limitations |

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

The test set was not used for tuning. The random search did not beat the hand-designed CNN within its 9-epoch, 6-trial budget; section 4 of the notebook explains why. See `results/figures/` for the confusion matrix, learning curves and error gallery.

## Where the data, weights and models come from

| Item | Source | Where it ends up |
|---|---|---|
| **Dataset** | Downloaded from Roboflow Universe with the `roboflow` Python package (YOLOv8 export, about 150 MB). Requires a free `ROBOFLOW_API_KEY`. | `data/Construction-Site-Safety-27/` (gitignored) |
| **Pretrained weights** | ImageNet weights for MobileNetV2 (`mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5`, 9.4 MB), downloaded automatically by `keras.applications` from Google's Keras storage on first use | Keras cache: `~/.keras/models/` |
| **Trained models** | Trained by the notebook itself; nothing else is downloaded | `models/checkpoints/*.keras` (best epoch of each experiment), `models/final_ppe_classifier.keras` (chosen model) |
| **Results** | Written by the notebook | `results/figures/`, `results/tables/` |

### Training environment of the saved run

The results in this repository come from **one run on a local machine, not Google Colab**:

| | |
|---|---|
| Machine | Windows, **CPU only** (no GPU detected) |
| Software | Python 3.12.10, TensorFlow / Keras 2.19.1, run as a Jupyter notebook in VS Code |
| Training time | **6.8 h** in total, as logged in `results/tables/validation_results.csv` |

| Stage | Experiments | Training time |
|---|---|---:|
| Step-by-step models | 01 MLP, 02 CNN (ReLU), 03 CNN (ELU) | 2.1 h |
| Hyperparameter search | 04_search_01 … 06 | 3.8 h |
| Transfer learning | 05 frozen, 06 fine-tuned | 0.8 h |

### Running in Google Colab (recommended for re-runs)

Colab provides a free GPU, which should cut the training time from hours to minutes.

1. Open the notebook in [Google Colab](https://colab.research.google.com/) (**File → Upload notebook**, or **File → Open notebook → GitHub** with this repository's URL).
2. **Runtime → Change runtime type → T4 GPU**.
3. Add a secret named `ROBOFLOW_API_KEY` (key icon in the left sidebar) and allow the notebook to access it.
4. **Runtime → Run all.** The first cell installs the dependencies; the dataset and MobileNetV2 weights download automatically.
5. Download the outputs before the session ends (Colab storage is temporary), e.g. by running `!zip -r outputs.zip models results` and downloading `outputs.zip` from the Files panel.

## Dataset

[Construction Site Safety v27](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27) (YOLOv8 export, 2,801 augmented images, 10 classes), by Roboflow Universe Projects, licensed **CC BY 4.0**.

The dataset is **not committed** to this repository. See [data/README.md](data/README.md) for how to obtain it.

## Reproducing

1. Install the dependencies: `pip install -r requirements.txt` (or open the notebook in Google Colab with a GPU).
2. Provide the dataset in one of three ways:
   - Set the `ROBOFLOW_API_KEY` environment variable (or a Colab secret of the same name); the notebook downloads into `data/`.
   - Extract a YOLOv8 export to `data/Construction-Site-Safety-27/`.
   - Set `LOCAL_DATASET_DIR` to any extracted YOLOv8 export.
3. Run all cells. Models are written to `models/`, plots and tables to `results/`.

Never commit API keys or put them in notebooks, slides, or screenshots.

Key configuration (`results/tables/configuration.json`): image size 160, batch size 32, 15 scratch epochs, 9 search epochs, 15 transfer epochs and 8 fine-tune epochs.
