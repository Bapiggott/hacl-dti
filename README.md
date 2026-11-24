# HACL-DTI

**History-Aware Connectome Learning for Diagnostic Progression**

Research code for longitudinal diagnostic-progression modeling with diffusion tensor imaging (DTI), structural connectivity, and clinical information. HACL combines visit-level brain representations with explicit changes between visits and a temporal model of the observed history.

This repository contains the model implementation and the single-model ablation campaign. It does not distribute ADNI data, trained checkpoints, participant-level predictions, or experiment results.

## Architecture

The model has four main components:

1. **State graph encoder.** Weighted GraphSAGE encodes regional FA, MD, RD, and axial diffusivity (AxD), together with tract-based connectivity.
2. **Change graph encoder.** A separate graph encoder represents adjacent-visit changes in regional measurements and tract summaries. A learned residual gate combines change and state representations.
3. **Temporal encoder.** An attention-based sequence model aggregates the available visit history, with configurable diagnosis and time-interval inputs.
4. **Clinical conditioning.** Structured clinical features condition the imaging representation before classification.

The prediction task is binary diagnostic progression at the next visit. The main temporal implementation is `DTISequenceClassifierV2.py`. The recurrent classifier supports the temporal-attention ablation and provides shared clinical-encoding utilities. `DTISequenceClassifierV13.py` is retained because the trainer imports it; its history branch is not selected in the included campaign.

## Repository layout

```text
hacl-dti/
├── campaign.py                         # Ablation scheduling and split audits
├── export_figures.py                   # Plot completed ablation groups
├── source/
│   ├── dti_main_stable_change_v13_history.py
│   ├── DTI_ADNIDataset.py
│   ├── DTIWindowDataLoader_stable_change.py
│   ├── model.py
│   ├── advanced_gnn.py
│   ├── longitudinal_innovation.py
│   ├── DTISequenceClassifier.py
│   ├── DTISequenceClassifierV2.py
│   ├── DTISequenceClassifierV13.py
│   ├── clinical_only_classifier.py
│   ├── clinical_fold_preprocessing_v2.py
│   ├── expanded_clinical_loader_bridge.py
│   ├── metadata_imputer.py
│   ├── temporal_gap_processor.py
│   ├── FocalLoss.py
│   └── v9_extras.py
└── README.md
```

## Environment

The campaign runs under Linux or WSL and uses CUDA-enabled PyTorch for model training. The recorded training environment included:

```text
torch              2.11.0
torch-geometric    2.7.0
numpy              1.26.4
pandas             2.3.3
scipy              1.15.3
scikit-learn       1.7.2
matplotlib         3.10.8
tqdm               4.67.3
```

Install PyTorch and PyTorch Geometric builds compatible with your CUDA environment. Additional dependencies vary with the enabled options; experiment tracking uses `wandb` when enabled. The campaign scheduler imports `fcntl` and therefore requires Linux/WSL rather than native Windows Python.

## Data preparation

Training expects externally prepared inputs:

- DTI visit directories containing regional FA, MD, RD, and AxD measurements.
- Tract-connectivity matrices and atlas/support arrays.
- Selected-participant and visit-manifest CSV files.
- Clinical feature tables and their feature schema.
- The clinical metadata and cohort/split references required by the selected configuration.

The primary imaging representation uses 116 brain regions. Raw image preprocessing and clinical-dataset construction are outside the scope of this source release.

ADNI data must be obtained separately through the applicable ADNI access process. Dataset files, identifiers, window lists, and generated outputs should remain outside version control.

## Training interface

From an environment with the dependencies installed, inspect the available arguments:

```bash
cd source
python dti_main_stable_change_v13_history.py --help
```

The trainer requires `--dti_root`, `--selected_csv`, `--manifest_csv`, and `--aal_adjacency`. Clinical inputs, tract support, model options, training settings, and an output directory must be specified for the intended experiment. The defaults alone do not define the paper configuration.

The source supports dataset checks through `--check_only`. Validate data paths, eligible windows, and participant splits before launching training.

## Ablation studies

The campaign evaluates architectural branches, clinical feature sets, temporal inputs, and connectivity choices. Additional groups cover graph processing, fusion, sampling, and training settings.

- Participant splits use five folds.
- The campaign uses seed 77.
- Scheduling supports three workers with two numerical threads per worker.
- Later runs use a maximum of 250 epochs with patience 75. Earlier retained runs use 400 epochs with patience 150. Comparisons must retain their recorded budgets and appropriate controls.

`campaign.py` expects a `manifest.json` defining experiments and a `reference_command.json` defining the base trainer arguments. It also audits windows against external reference runs. These configuration and reference files are not included in this code-only release. The script retains study-specific path constants; adapt them and provide the required inputs before using its `run` or `status` commands. Its `prepare` command additionally depends on the original study setup and is not a standalone setup utility.

`export_figures.py` reads completed job outputs and a manifest. It requires those external results and appropriate path configuration; the repository does not include precomputed figures.

## Scope and reproducibility

This release preserves the model and ablation source. It is not a self-contained benchmark package: data, exact experiment configuration files, split references, and fitted weights must be supplied separately. The conditional multi-atlas branch is not included and should remain disabled.

The repository focuses on single-model ablations. Four-member ensemble aggregation and separate tabular-baseline experiments are not included here.

