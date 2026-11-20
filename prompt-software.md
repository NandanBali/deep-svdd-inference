Implement a complete, reproducible software baseline for my Deep SVDD network anomaly detection project. Work directly in this repository and run the implementation. I need working code, trained artifacts, evaluation results, and a clear handoff for FPGA inference.

## Project context

- Target hardware: Pynq-Z2, Zynq-7020.
- FPGA tools: Vivado/Vitis 2023.2.
- Dataset: a NetFlow version of UNSW-NB15. Inspect the actual files to determine the exact variant and schema; do not assume NF-UNSW-NB15 and NF-UNSW-NB15-v2 have identical features.
- Available GPU: RTX 3050 Laptop GPU with 4 GB VRAM. Support CPU execution too.
- Overall project deadline: approximately two weeks.
- My immediate hardware goal is a first working Deep SVDD inference core. Keep the model small and the inference pipeline easy to reproduce in HLS.
- Training happens on the computer. FPGA training, online adaptation, packet parsing, and feature extraction hardware are outside this software baseline.

Prioritize a correct, compact end-to-end baseline over extensive hyperparameter tuning.

## 1. Inspect the repository and dataset

Read applicable repository instructions and inspect existing code before making changes. Reuse useful existing work.

Find the dataset locally. If it is missing, implement a configurable loader and document the expected path and schema. Do not silently substitute another dataset.

Inspect and record:

- Exact dataset filename and variant.
- Number of rows and features.
- Feature names and data types.
- Binary label and attack-category columns.
- Class distribution and attack distribution.
- Missing values, infinities, constant columns, and duplicate records.
- Whether timestamps or predefined splits are available.

Exclude labels, attack categories, row identifiers, and other target-derived information from model inputs.

Default to excluding raw IP addresses and flow identifiers. Explicitly justify whether ports and protocol fields are retained and how they are encoded.

Produce a concise dataset audit.

## 2. Define a leakage-safe experimental protocol

Use normal-only training for the main Deep SVDD model.

Create reproducible train, validation, and test partitions before fitting preprocessing or selecting features.

- Prefer a trustworthy official split or a meaningful chronological split when available.
- Otherwise, use a seeded split with approximately 60/20/20 proportions.
- Keep exact duplicate flow records together across partitions, or remove duplicates before splitting; document the policy.
- Use reliable session/group information when available to reduce related-flow leakage. Do not claim complete session isolation when the dataset lacks the information to establish it.
- Retain both normal and attack samples in validation and test data.
- Remove attack samples from the model’s training partition.
- Fit imputation, scaling, encoding, variance filters, and correlation filters using normal training data only.
- Keep the test partition untouched until model and threshold choices are frozen.

Save split indices or a deterministic split manifest, the random seed, and a dataset fingerprint.

Explicitly distinguish normal-only training from any use of attack labels for validation, feature selection, or model selection.

## 3. Implement preprocessing and compact feature selection

Build a saved preprocessing pipeline that handles non-finite values, missing data, categorical fields, and scaling consistently.

Use simple operations that can later be implemented on the processing system or in hardware. Begin with standardization unless the observed data justifies another transform. Make any log transforms or clipping explicit, deterministic, and configurable.

Prepare two feature configurations:

1. A baseline using all eligible features with a practical encoding.
2. A compact configuration targeting roughly 8–16 model inputs.

For compact feature selection:

- Remove constant or near-constant inputs.
- Reduce strongly correlated redundancy using training data.
- Favor useful features with manageable extraction requirements.
- Distinguish inexpensive counters/basic flow statistics from features requiring substantial flow state, timing histories, divisions, or complex parsing.
- Consider both predictive usefulness and eventual hardware cost.

Do not compute mutual information with the attack label on an all-normal training set: that label is constant. If label-based mutual information or another supervised selector is explored, use a separate development subset, keep it disjoint from the final test set, and report it as a supervised-assisted experiment.

Keep the main feature-selection procedure simple enough to finish quickly.

Save a feature table containing:

- Original feature name and meaning.
- Inclusion/exclusion decision and reason.
- Encoding and transformation.
- Final model input index.
- Approximate extraction requirements and hardware concerns.

## 4. Implement Deep SVDD correctly

Use PyTorch and implement one-class Deep SVDD as the primary model:

    z = phi(x; W)
    score(x) = sum_j (z_j - c_j)^2

Higher scores indicate more anomalous flows.

Train by minimizing mean squared distance to a fixed center, with clearly specified L2 regularization. Avoid accidentally applying the same regularization twice.

Start with this small architecture:

    input_dim → 32 → 16 → 8

Requirements:

- Bias-free linear layers.
- ReLU between hidden layers.
- Linear final embedding.
- No batch normalization or dropout in the initial baseline.
- No sigmoid, softmax, or reconstruction decoder.
- Fixed center initialized from mean embeddings of normal training data before optimization.
- Apply and document the usual small-epsilon treatment for center components near zero.
- Do not update the center during training.
- Record the exact loss, optimizer, regularization, initialization, and stopping rules.

Use a modest training budget and early stopping based on held-out normal validation loss. Track embedding variance and score distributions so representation collapse can be detected.

Do not make autoencoder pretraining a dependency for the first result. It can be an optional follow-up if the initial model collapses or performs poorly.

Implement one-class Deep SVDD first. Treat soft-boundary Deep SVDD as optional and only add it after the primary pipeline works.

## 5. Calibrate a deployable threshold

Use a threshold determined from normal validation scores.

Default target: approximately 1% false-positive rate on validation normal flows.

    threshold = chosen quantile of normal validation scores
    anomaly = score > threshold

Make the target false-positive rate configurable. Document the quantile method, comparison operator, and effects of tied scores.

Never optimize the threshold on test labels. Report the actual false-positive rate achieved on the test set.

## 6. Evaluate the baseline

Evaluate both feature configurations using the same experimental protocol and comparable training settings.

Report:

- AUROC.
- Average precision, clearly identified as average precision.
- Precision, recall, F1, and false-positive rate at the frozen threshold.
- Confusion matrix with class definitions and counts.
- Attack-category recall where attack labels exist.
- Test-set attack prevalence.
- Training duration.
- Model parameter count and approximate dense-layer MACs per flow.
- CPU batch-size-one inference timing and batched timing, including measurement methodology.

Separate network inference timing from preprocessing and data-loading time. Report measured software performance only; do not infer FPGA throughput from it.

Include score-distribution plots for normal and attack flows and concise training curves.

Run one seed initially. Once the complete pipeline works, run approximately three seeds for the selected compact configuration if practical. Keep seed-specific model centers and thresholds paired with their models.

If attack labels influence architecture or feature-set selection through validation metrics, state that explicitly.

## 7. Produce the FPGA handoff

For the selected compact model, export:

- PyTorch checkpoint and model configuration.
- Ordered input-feature list.
- Complete preprocessing parameters.
- Weight matrices with documented dimensions and orientation.
- Fixed center vector.
- Decision threshold.
- Machine-readable metadata describing all operations and numerical types.
- Plain NumPy arrays or similarly simple files that can be inspected independently of PyTorch.
- A generated C/C++ header containing floating-point weights, center, and threshold.

Define the inference contract precisely:

    ordered raw features
        → documented preprocessing
        → normalized input vector
        → dense/ReLU network
        → squared distance to center
        → threshold comparison

For the first FPGA core, define the PL input as the already-preprocessed feature vector. Document preprocessing as a separate stage that can initially run on the PS.

Export at least 100 representative golden test vectors, including:

- Raw selected inputs where available.
- Preprocessed network inputs.
- Expected embedding.
- Expected anomaly score.
- Expected decision.

Include normal samples, attacks, and samples close to the decision threshold.

Implement an independent NumPy inference function and a small portable C++ floating-point reference. Verify both against PyTorch using the exported parameters. Check intermediate activations as well as final scores so matrix orientation and activation errors are easy to diagnose.

Do not require Vivado/Vitis or FPGA access to complete this software handoff.

## 8. Add a lightweight numerical-range study

After the floating-point baseline works:

- Record observed ranges of normalized inputs, weights, layer outputs, distance terms, and final scores.
- Suggest initial fixed-point formats with explicit headroom.
- Explain accumulator-width requirements.
- If feasible, simulate one or two candidate fixed-point configurations and measure score error, classification disagreement, saturation, and detection metrics.

Specify rounding, overflow, saturation, and intermediate-cast behavior. Do not describe an approximate Python quantization experiment as bit-accurate HLS verification.

Evaluate quantized inference with the original threshold first. If recalibrating a quantized threshold, use validation data only and report that result separately.

## 9. Repository quality and reproducibility

Use a small, understandable project structure with modules for:

- Dataset loading and audit.
- Split generation.
- Preprocessing and feature selection.
- Model definition.
- Training.
- Evaluation and threshold calibration.
- Model export.
- Independent reference inference.
- Focused correctness tests.

Provide configuration files and copy-paste commands for the entire workflow.

Include meaningful checks for:

- Disjoint partitions and duplicate handling.
- Normal-only training.
- Preprocessing fitted only on training data.
- Consistent feature order.
- Threshold calibration without test-label access.
- PyTorch/NumPy/C++ agreement.
- Exported parameter dimensions and model identity.

Record package versions and seeds. Avoid unnecessary dependencies and external experiment-tracking services.

Keep raw data and large generated artifacts out of git as appropriate.

## 10. Execution order and completion criteria

Work in this order:

1. Dataset audit and fixed experimental protocol.
2. Complete floating-point pipeline for eligible features.
3. Compact-feature model.
4. Frozen-threshold evaluation.
5. Verified exports and golden vectors.
6. Numerical-range and optional fixed-point study.

Do not stop after scaffolding, writing a plan, or producing a notebook. Run the code and correct failures.

If the actual dataset is unavailable, complete and test the pipeline with clearly labeled synthetic smoke-test data, then state the exact command needed to run the real experiment. Never present synthetic results as dataset performance.

At completion, provide:

- Exact commands to reproduce the results.
- Dataset and split summary.
- Feature list and selected architecture.
- Actual measured metrics and limitations.
- Paths to checkpoints, preprocessing, reports, exports, and golden vectors.
- Any remaining blockers.
- A concise explanation of what the FPGA developer can implement immediately.

Make routine implementation decisions autonomously, document assumptions, and ask only when missing information genuinely blocks progress.
