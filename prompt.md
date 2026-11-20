Implement a reproducible feature-selection workflow for our Deep SVDD anomaly-detection project.

**Project context**

- Dataset: a NetFlow version of UNSW-NB15. Inspect the actual files and identify the exact release/schema; do not assume original NF-UNSW-NB15, v2, or v3.
- Target hardware: Pynq-Z2 / Zynq-7020.
- Deadline: approximately two weeks for an FPGA inference-core prototype.
- Another person is responsible for the software baseline. Your scope is feature selection and the preprocessing/feature contract needed to integrate with that baseline.
- Initial encoder will be a small bias-free MLP, approximately `d → 16 → 8`.
- V1 preprocessing can run in software. Eventually, we want feature extraction on the FPGA.

Build and run the workflow, inspect the results, and recommend a practical feature subset. Do not stop at a plan.

**1. Inspect the repository and dataset**

Reuse existing loaders, splits, preprocessing, and configuration where appropriate. Read applicable repository instructions before changing files.

Identify:

- Dataset release, columns, types, units where documented, and label mapping.
- Missing/infinite values, constant columns, duplicate rows, and suspicious identifiers.
- Class distribution and attack-family distribution.
- Which columns represent continuous measurements, categorical codes, counters, flags, addresses, or ports.

Exclude labels, attack categories, row identifiers, and split/source markers from inputs. Exclude raw IP addresses from the initial candidates. Treat protocol identifiers and ports according to their semantics; do not assume numeric codes have meaningful Euclidean distances.

If the dataset cannot be located, ask for its path. Do not invent dataset contents or results.

**2. Establish leakage-resistant splits**

Reuse valid existing splits. Otherwise, create reproducible train/validation/test splits before fitting preprocessing or feature selection.

Use capture/time/group separation where the available metadata supports it. Otherwise use a documented fallback and explain its limitations. Keep duplicate feature vectors from appearing across partitions.

Use training data to fit imputers, encoders, scalers, clipping limits, and feature rankings. Keep the test set untouched until the final configuration is frozen.

**3. Compare two clearly distinguished selection tracks**

Track A: selection without attack labels.

- Use normal training samples for variance and redundancy analysis.
- Remove constant and near-constant inputs.
- Inspect covariance after an appropriate training-fitted scaling.
- Use Pearson and, where useful, Spearman correlation to find redundant numeric measurements.
- Start with an absolute-correlation threshold of 0.95 as a configurable heuristic.
- When two features are redundant, prefer the one with lower extraction cost, better data quality, and clearer semantics.
- Do not present covariance or variance alone as evidence of anomaly-detection usefulness.

Track B: label-assisted feature selection.

- Estimate mutual information between candidate features and the binary normal/attack label using training data only.
- Handle discrete versus continuous inputs explicitly and set reproducible random seeds.
- Produce an MI ranking and, if practical, a small greedy mRMR-style subset that balances label relevance against feature redundancy.
- State the exact redundancy estimator and selection formula. Do not label simple MI sorting as mRMR.
- If sampling or class balancing is used for estimation, document it because it changes the distribution being measured.
- Clearly label this track as using attack labels for feature selection, even though Deep SVDD itself is trained only on normal samples.

Keep these tracks separate in the report. Also disclose any use of attack validation labels to choose between subsets.

**4. Assess FPGA suitability**

Create a feature audit covering:

- Meaning, units, direction convention, and availability time.
- Source: packet header, per-flow counter, timestamp, derived statistic, or application parsing.
- Update operation and required per-flow state.
- Arithmetic: increments/addition, comparison, multiplication, division, logarithms, etc.
- Whether it requires a completed flow, cross-flow history, or payload inspection.
- Encoding and preprocessing requirements.
- An ordinal extraction-cost assessment with a concrete explanation.
- Unknown definitions or assumptions requiring verification.

Assess extraction cost separately from preprocessing and neural-network input cost.

Do not invent LUT/DSP/BRAM estimates without synthesis. Do not assume a feature is cheap merely because it is numeric or belongs to a NetFlow dataset. Treat ratios, rates, and complex statistics individually rather than automatically excluding all of them.

Consult the dataset’s official feature documentation where necessary and record sources. Do not silently replace a feature with a hardware approximation.

**5. Produce a small number of candidate subsets**

Keep the search bounded. Produce no more than three serious candidates, such as:

- Compact, hardware-friendly subset.
- Balanced subset.
- Richer reference subset.

Aim for approximately 8–16 encoded inputs when the actual schema permits it. Report both original feature count and final encoded input dimension. Do not pad a smaller useful subset just to reach a target count.

Prefer interpretable original features for this milestone. Do not introduce PCA or a large automated search unless existing results establish a need.

For every retained feature, explain its contribution and implementation cost. For important rejected features, explain the reason.

**6. Validate usefulness with the baseline**

If a working Deep SVDD evaluator exists, compare candidates using the same architecture policy, training budget, splits, and seeds. Keep centers and thresholds specific to each trained model.

Report:

- ROC-AUC and PR-AUC.
- Recall and measured false-positive rate at a threshold calibrated on normal validation scores.
- Per-attack recall.
- Encoded input dimension and preprocessing complexity.
- Score distributions and signs of embedding collapse.

Choose a configurable target false-positive rate; use 1% as an initial evaluation point, not a claimed project requirement. Prefer multiple seeds if the runtime budget permits.

Use validation results for selection. Evaluate the frozen final choice on test data once.

If the baseline is not yet available, implement the integration interface and mark the recommendation as provisional. Statistical rankings alone must not be presented as demonstrated Deep SVDD performance. Do not build a separate large training framework.

**7. Deliver executable code and a precise handoff**

Adapt filenames to the repository, but provide:

- A command-line feature-selection workflow.
- A machine-readable feature audit and selection results.
- A concise report with actual findings, limitations, and a recommended subset.
- A saved preprocessing configuration/artifact.
- An ordered feature contract defining raw columns, encoding, scaling, missing-value behavior, out-of-range behavior, and final input dimension.
- Representative preprocessed normal/attack vectors for integration testing.
- A single documented command to reproduce the workflow.

Use chunked loading or bounded sampling where needed for large CSVs. Record sample sizes and seeds. Avoid expensive all-pairs MI over millions of rows.

Verify that preprocessing survives save/reload, feature ordering is stable, labels never enter the input matrix, and exported vectors match the preprocessing pipeline.

End with the recommended subset, encoded dimension, evidence supporting it, remaining uncertainty, and the exact files the baseline and FPGA developers should consume.
