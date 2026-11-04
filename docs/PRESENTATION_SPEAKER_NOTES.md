# SafeSight — Presentation speaker notes (suggested 12–15 minutes)

1. **Title (20s)** — Introduce practical problem and hypothesis; say this is a crop-classification research prototype.
2. **Motivation (35s)** — PPE is an important visible workplace safety cue, but errors are costly.
3. **Question (35s)** — Hypothesis vs evidence: transfer might help, but results determine conclusion.
4. **Dataset (45s)** — Roboflow version 27; 2,801 is augmented export count, not 2,801 independent scenes. Show real class counts.
5. **Scope (45s)** — Explain supplied annotated boxes; not real-time detection or person-to-gear association.
6. **Preprocessing (50s)** — Validate annotation, crop, resize, scale, train-only augmentation.
7. **Leakage (55s)** — Crops must remain inside their parent image partition. Mention augmentation risks.
8. **Lecture coverage (40s)** — Perceptron, MLP, cross entropy, CNN and transfer; no RNN for unordered photos.
9. **Model progression (50s)** — Rationale for MLP baseline then CNN and regularized CNN.
10. **CNN internals (45s)** — Weight sharing, receptive fields, max-pooling, BatchNorm, dropout.
11. **Hyperparameters (65s)** — Six random trials, defined search space, validation macro-F1 selection, limited compute budget.
12. **Transfer (65s)** — Frozen features followed by last 30 layers fine-tune; why learning rate decreases.
13. **Metrics (65s)** — Macro-F1 and violation-class recall; per-class false negatives matter.
14. **Comparison (60s)** — Substitute measured table, explain changes without claiming every difference significant.
15. **Final test (60s)** — Show actual confusion matrix and test scores; test not used for tuning.
16. **Failures (50s)** — Explain bad crops, occlusion, illumination and confusing labels.
17. **Conclusions (45s)** — Specific observed gain/loss, what has not yet been proven.
18. **Defense (40s)** — Be ready for all six Q&A prompts.
19. **Q&A** — Credit Roboflow CC BY 4.0 and instructor lecture concepts.

## Essential last-minute checklist
- Fill every placeholder; insert actual `artifacts/*.png` in PowerPoint.
- Update results in report; report must match notebook values.
- If a class is absent in any split, fix dataset/version instead of suppressing the error.
- Save final executed `.ipynb` showing charts, learning curves and metrics.
- Prepare one screenshot of hardware/runtime and note total training time.
- Answer clearly: **This is region classification; mAP is a detection metric and not the main endpoint here.**
