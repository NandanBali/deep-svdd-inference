from __future__ import annotations

import time
import numpy as np
import torch
from sklearn.metrics import precision_score, f1_score, confusion_matrix

from svdd_features.evaluation import metrics, calibrate


def evaluate(scores, labels, attacks, threshold):
    result = metrics(scores, labels, attacks, threshold)
    pred = scores > threshold
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    result.update(precision=float(precision_score(labels, pred, zero_division=0)),
        f1=float(f1_score(labels, pred, zero_division=0)),
        confusion_matrix=dict(class_0="benign", class_1="attack", rows="true", columns="predicted",
                              matrix=[[int(tn), int(fp)], [int(fn), int(tp)]]),
        attack_prevalence=float(np.mean(labels)), rows=len(labels))
    return result


def benchmark(model, x, config):
    model = model.to("cpu").eval()
    torch.set_num_threads(1)
    results = {}
    for name, batch, warmup, count in [
        ("batch_one", 1, config["single_warmup"], config["single_calls"]),
        ("batched", min(config["batch_size"], len(x)), config["batch_warmup"], config["batch_calls"])]:
        tensor = torch.from_numpy(np.ascontiguousarray(x[:batch]))
        timings = []
        with torch.inference_mode():
            for _ in range(warmup):
                model.trace(tensor)[-1]
            for _ in range(config["trials"]):
                start = time.perf_counter_ns()
                for _ in range(count):
                    model.trace(tensor)[-1]
                timings.append((time.perf_counter_ns()-start)/count)
        results[name] = dict(batch_size=batch, calls_per_trial=count, warmup_calls=warmup,
            trial_mean_ns=timings, median_call_ns=float(np.median(timings)),
            p95_trial_mean_ns=float(np.quantile(timings, .95)),
            median_ns_per_flow=float(np.median(timings)/batch))
    results["methodology"] = "PyTorch eager float32 CPU, one thread, inference_mode, perf_counter_ns, preloaded CPU tensors, network + distance only; Python dispatch included. Quantiles summarize trial means, not individual-call tail latency. No preprocessing, transfer or file loading."
    return results
