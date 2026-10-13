# Model evaluation plots

Plots in this folder must be generated from actual model runs, not fabricated example metrics.

From the repository root:

    python notebooks/04_evaluate_models.py

This writes cnn_confusion_matrix.png, cnn_evaluation.json and evaluation_summary.json when the CNN model and dataset are present. The CNN training script also writes cnn_training_accuracy.png and its confusion matrix after training.

Optional labeled gas and temperature datasets can be evaluated with --gas-csv and --temperature-csv. Keep dataset provenance and split details alongside results. Validation metrics must not be presented as independent test accuracy.
