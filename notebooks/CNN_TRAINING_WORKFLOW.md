# Leakage-safe CNN training workflow

The CNN training script originally used Keras `validation_split=0.2`, which did not honor the audited duplicate-aware split manifest. Use the scripts below for reproducible model experiments.

## 1. Build a split manifest (if one is not already available)

Run from the repository root after the local image dataset is present:

```powershell
python notebooks/06_build_cnn_split_manifest.py
```

The script scans `data/fire_dataset/fire` and `data/fire_dataset/no_fire`, groups exact byte-identical images and perceptual-hash candidates (default Hamming distance <= 4), and assigns entire groups to one split. It uses deterministic group-stratified splitting with approximate 70/15/15 proportions. Actual counts can vary because duplicate groups cannot be split across subsets.

It writes `notebooks/plots/cnn_split_manifest_rebuilt.csv`. It refuses to overwrite an existing output file. It does **not** delete or modify source images. If a duplicate candidate group spans both labels, it stops for manual label review.

If the reviewed `notebooks/plots/cnn_split_manifest_corrected.csv` already exists, the training/evaluation scripts prefer that file by default.

## 2. Train a candidate model

```powershell
python notebooks/07_train_cnn_manifest.py
```

The script validates file paths, labels, split names, and duplicate-group boundaries before training. Only `train` and `validation` rows are loaded. The `test` split is not passed to training or model selection.

Outputs are saved under `models/cnn_fire_model_candidate.h5` and `models/cnn_fire_model_candidate_best.h5`, with metrics/plots under `notebooks/plots/cnn_manifest_run`. Existing model files are not overwritten. Use alternate `--output-model` and `--checkpoint` paths for another experiment.

Select a candidate using validation metrics only. A 92%+ validation score is a target, not a guaranteed result.

## 3. Evaluate the selected candidate on the held-out test split

Only after the model is selected using validation results:

```powershell
python notebooks/08_evaluate_cnn_manifest.py --model models/cnn_fire_model_candidate.h5
```

The default split is `test`, with threshold 0.5. Do not tune the threshold or choose between models based on test results. Run the held-out test evaluation once for the final reported result. Validation can be explicitly evaluated with `--split validation` during model selection.

## Important limitations

- The image dataset is local and is not committed to this repository; a clean clone needs the dataset before building a manifest or training.
- Perceptual-hash matches are candidate duplicates, not proof of identical semantic content. Grouping candidates prevents likely leakage but can reduce effective sample size.
- This workflow does not itself certify real-world fire detection or replace testing on unseen camera footage.
- Do not replace `models/cnn_fire_model.h5` with a candidate until its held-out results and application compatibility have been reviewed.
