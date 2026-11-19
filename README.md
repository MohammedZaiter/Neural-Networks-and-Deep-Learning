# SafeSight — PPE Compliance Classification with Deep Learning

SafeSight classifies cropped regions from construction-site images into 4 PPE classes: `Hardhat`, `NO-Hardhat`, `Safety Vest` and `NO-Safety Vest`. It compares models of increasing capacity: an MLP baseline, CNNs trained from scratch, controlled one-change-at-a-time experiments, a grid search and a combined best CNN, and transfer learning with MobileNetV2 (frozen, partial and full fine-tuning, on 10% and 100% of the data).

> **Scope:** this is a **crop / region classifier**. It relies on the dataset's annotated bounding boxes. It does not do full-scene detection or associate people with their gear, so mAP is not the main metric here.

## Repository layout

```
├── Checkpoints/                   best checkpoint of every experiment (.keras) + final_ppe_classifier.keras
├── Notebooks/
│   ├── SafeSight_PPE_Deep_Learning.ipynb   the project: sections 1-8 of the specification + demo appendix
│   └── utils.py                            code used by the notebook: data, models, training, metrics, plots
├── Presentation/                  slides (PPTX) and speaker notes
├── Report/                        project report (DOCX)
├── data/                          dataset location (not committed, see data/README.md)
├── results/                       CSV / JSON results of every experiment
│   └── figures/                   plots
├── requirements.txt
└── README.md
```

## Notebook structure

The notebook follows the project specification section by section. Every code cell has an explanation, and each stage ends with its observations.

| Section | Contents |
|---|---|
| 0. Google Colab setup | Drive, clone, working folder (skipped locally) |
| 1. Dataset selection | Source, licence, size, features, target |
| 2. Data preprocessing | Annotation audit (missing values), outlier rule, exploration, leakage check, normalization, augmentation, class weights |
| 3. Step-by-step model creation | MLP → CNN (ReLU) → CNN (ELU + dropout + L2) |
| 4. Hyperparameter tuning | One change at a time from the baseline CNN (architecture, activation, optimization, regularization, class weights, loss function, number of epochs) with a 3-seed noise floor; grid search over learning rate x batch size; combined best CNN |
| 5. Transfer learning | MobileNetV2: frozen, fine-tune last 30 layers, fine-tune all layers; setups compared on convergence, data efficiency (10% vs 100%) and cost |
| 6. Integration of course topics | Map of course concepts to where they are used |
| 7. Evaluation and interpretation | Model comparison, held-out test, learning curves, error analysis, challenges |
| 8. Conclusion | Answer to the research question, gains per stage, limitations, references |
| Appendix: Demo | Load any saved checkpoint and test it (for the defense) |

## Results

Validation set (model selection by macro-F1):

| Model | Accuracy | Macro-F1 | Balanced acc. | Unsafe recall |
|-------|---------:|---------:|--------------:|--------------:|
| 01 MLP baseline | 0.573 | 0.568 | 0.592 | 0.574 |
| 02 CNN (ReLU) | 0.702 | 0.696 | 0.710 | 0.778 |
| 03 CNN (ELU + dropout) | 0.694 | 0.689 | 0.709 | 0.692 |
| 05 Transfer learning (frozen) | 0.855 | 0.849 | 0.859 | 0.846 |
| **06 Transfer learning (fine-tuned)** | **0.898** | **0.896** | **0.900** | **0.907** |

Held-out test set, chosen model `06_transfer_finetuned` (269 crops):

| Accuracy | Macro-F1 | Balanced acc. | Unsafe recall |
|---------:|---------:|--------------:|--------------:|
| 0.914 | 0.909 | 0.920 | 0.924 |

The test set was not used for tuning. These results will be updated with the controlled experiments (section 4) and the transfer comparison (section 5) after the GPU run.

## Where the data, weights and models come from

| Item | Source | Where it ends up |
|---|---|---|
| **Dataset** | Downloaded from Roboflow Universe with the `roboflow` Python package (YOLOv8 export, about 150 MB). Uses the Roboflow key in `Notebooks/utils.py`, or a `ROBOFLOW_API_KEY` environment variable / Colab secret. | `data/Construction-Site-Safety-27/` (gitignored); `/content/data/` on Colab |
| **Pretrained weights** | ImageNet weights for MobileNetV2 (`mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5`, 9.4 MB), downloaded automatically by `keras.applications` on first use | Keras cache: `~/.keras/models/` |
| **Trained models** | Trained by the notebook; nothing else is downloaded | `Checkpoints/<name>.keras` (best epoch of each experiment), `Checkpoints/final_ppe_classifier.keras` (chosen model) |
| **Results** | Written by the notebook | `results/`, `results/figures/` |

### Training environment

Experiments 01–06 come from **a first run on a local machine** (Windows, CPU only, Python 3.12.10, TensorFlow / Keras 2.19.1). The experiments added later (`07_*`, `08_*`, `abl_*`, `tl_*`) are trained on a Google Colab T4 GPU; the `device` column of `results/experiment_benchmark.csv` records which hardware trained each one.

| Stage | Experiments | Training time (CPU) |
|---|---|---:|
| Step-by-step models | 01 MLP, 02 CNN (ReLU), 03 CNN (ELU) | 2.1 h |
| Transfer learning | 05 frozen, 06 fine-tuned | 0.8 h |

The first run also included a 6-trial random search (3.8 h), later replaced by the controlled experiments of section 4.

### Running in Google Colab (recommended for training)

1. Open the notebook in [Google Colab](https://colab.research.google.com/): **File → Open notebook → GitHub**, paste this repository's URL, choose the branch `ai-style-experiments` and `Notebooks/SafeSight_PPE_Deep_Learning.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. The dataset is downloaded with the Roboflow key set in `Notebooks/utils.py` (`ROBOFLOW_API_KEY`). To use another key, add a Colab secret named `ROBOFLOW_API_KEY`; it takes priority.
4. **Runtime → Run all.** Section 0 mounts Google Drive and clones the repository into `My Drive/Neural-Networks-and-Deep-Learning` on the first run, then moves into its `Notebooks/` folder. Saved checkpoints are loaded instead of retrained, and every new checkpoint and result is written to Drive. The dataset is downloaded to the local Colab disk (`/content/data`).
5. If a session disconnects, run all again: with `RETRAIN = False`, experiments that already have a checkpoint are loaded and only the missing ones are trained.
6. The results stay in `My Drive/Neural-Networks-and-Deep-Learning` (`Checkpoints/` and `results/`). Copy those two folders back into the local repository.

## Dataset

[Construction Site Safety v27](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27) (YOLOv8 export, 2,799 augmented images, 10 classes), by Roboflow Universe Projects, licensed **CC BY 4.0**.

The dataset is **not committed** to this repository. See [data/README.md](data/README.md) for how to obtain it.

## Reproducing locally

1. Install the dependencies (Python 3.12; TensorFlow 2.16–2.19 does not install on Python 3.13 or newer):

   ```powershell
   py -V:3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt ipykernel
   ```

   Then select the `.venv` kernel in VS Code / Jupyter.
2. Provide the dataset in one of three ways:
   - Set the `ROBOFLOW_API_KEY` environment variable; the notebook downloads into `data/`.
   - Extract a YOLOv8 export to `data/Construction-Site-Safety-27/`.
   - Set `LOCAL_DATASET_DIR` to any extracted YOLOv8 export.
3. Run `Notebooks/SafeSight_PPE_Deep_Learning.ipynb`. By default (`RETRAIN = False`) every experiment with a checkpoint in `Checkpoints/` is **loaded** instead of trained, and only the missing ones are trained. Set `RETRAIN = True` to train everything from scratch (use a GPU).

Never commit API keys or put them in notebooks, slides, or screenshots.

Key configuration (`results/configuration.json`): image size 160, batch size 32, 15 scratch epochs, 15 transfer epochs, 8 fine-tune epochs, 15 epochs per controlled experiment (30 without early stopping for the number-of-epochs run), 30 for the combined CNN and 10 per setup in the transfer comparison.

Result files written by the notebook (`results/`): `validation_results.csv` (every experiment), `experiment_benchmark.csv` (parameters, receptive field, latency, epochs, seconds per epoch, train–validation gap), `ablation_results.csv`, `best_cnn_config.json`, `transfer_comparison.csv`, `main_models_benchmark.csv`, `training_histories.json`, `final_metrics.json`, `classification_report.json`, `crop_manifest.csv`.

## References

Dataset: Roboflow Universe Projects, *Construction Site Safety* v27 (CC BY 4.0). Pre-trained model: MobileNetV2 (Sandler et al., 2018), ImageNet weights from `keras.applications`. The full reference list is at the end of the notebook (section 8).
