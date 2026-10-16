# Model evaluation plots

## Current CNN retraining workflow

For duplicate-aware, leakage-safe CNN experiments, follow [CNN_TRAINING_WORKFLOW.md](../CNN_TRAINING_WORKFLOW.md). Use the manifest builder, candidate trainer, and held-out test evaluator described there.

The legacy `03_train_cnn_model.py` and `04_evaluate_models.py` use Keras' directory `validation_split=0.2`; do not use their results as the final independent test score for the duplicate-audited dataset.

Plots and metric JSON files must be generated from actual model runs, not fabricated example metrics. Keep dataset provenance and split details alongside results. Validation metrics must not be presented as independent test accuracy.

Optional labeled gas and temperature datasets can be evaluated with `04_evaluate_models.py --gas-csv` and `--temperature-csv`. The CNN candidate workflow does not replace real-world validation on unseen camera footage.
