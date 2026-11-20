from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .common import save_json, sha, ranges
from .model import DeepSVDD, objective
from svdd_features.evaluation import embedding_diagnostics


def configure(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def trace_numpy(model, x, device="cpu", batch_size=4096):
    model.eval()
    blocks = [[] for _ in range(5)]
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(np.ascontiguousarray(x[start:start+batch_size]), device=device)
            for result, array in zip(blocks, model.trace(batch)):
                result.append(array.cpu().numpy())
    return dict(zip(["h1", "h2", "embedding", "distance_terms", "scores"], [np.concatenate(a) for a in blocks]))


def load_model(folder, device="cpu"):
    folder = Path(folder)
    metadata = json.loads((folder/"training.json").read_text())
    if sha(folder/"model.pt") != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint identity mismatch")
    model = DeepSVDD(metadata["input_dimension"]).to(device)
    model.load_state_dict(torch.load(folder/"model.pt", map_location=device, weights_only=True)["state_dict"])
    model.eval()
    return model, metadata


def train(folder, x_train, x_normal_validation, seed, policy, device, experiment_id):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/"training.json").exists():
        model, metadata = load_model(folder, device)
        if metadata["experiment_id"] != experiment_id or metadata["seed"] != seed or metadata["device"] != device:
            raise ValueError("Cached model belongs to another experiment")
        return model, metadata
    configure(seed, policy["cpu_threads"])
    model = DeepSVDD(x_train.shape[1]).to(device)
    train_x = torch.from_numpy(x_train).to(device)
    val_x = torch.from_numpy(x_normal_validation).to(device)
    original_center = model.initialize_center(train_x, policy["center_epsilon"])
    frozen_center = model.center.clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=policy["learning_rate"],
                                 betas=(.9, .999), weight_decay=0., foreach=False, fused=False)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def normal_validation():
        return sum(float((model(b)-model.center).square().sum()) for b in val_x.split(4096))/len(val_x)

    initial = trace_numpy(model, x_normal_validation, device)
    initial_variance = float(np.var(initial["embedding"].astype(float), axis=0).sum())
    history, best, best_epoch, stale, best_state = [], float("inf"), 0, 0, None
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(1, policy["max_epochs"]+1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator).to(device)
        distance_total, penalty_total = 0., 0.
        for indices in order.split(policy["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss, distance, penalty = objective(model, train_x[indices], policy["regularization"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            distance_total += float(distance.detach())*len(indices)
            penalty_total += float(penalty.detach())*len(indices)
        model.eval()
        val = normal_validation()
        trace = trace_numpy(model, x_normal_validation[:8192], device)
        variance = float(np.var(trace["embedding"].astype(float), axis=0).sum())
        history.append(dict(epoch=epoch, training_distance=distance_total/len(train_x),
            l2_penalty=penalty_total/len(train_x), normal_validation_distance=val,
            embedding_total_variance=variance, score_std=float(trace["scores"].std()),
            hidden_zero_fractions=[float((trace[k] == 0).mean()) for k in ["h1", "h2"]]))
        # Save the literal minimum; patience requires a meaningful relative gain.
        significant = val < best*(1-policy["relative_min_delta"])
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        stale = 0 if significant else stale+1
        assert torch.equal(model.center, frozen_center)
        print(f"{folder.name} epoch {epoch:02d}: distance={history[-1]['training_distance']:.6g} normal_val={val:.6g} variance={variance:.3g}", flush=True)
        if epoch >= policy["min_epochs"] and stale >= policy["patience"]:
            break
    if device == "cuda":
        torch.cuda.synchronize()
    duration = time.perf_counter()-started
    model.load_state_dict(best_state)
    model.eval()
    model_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(dict(state_dict=model_cpu, input_dimension=x_train.shape[1], seed=seed,
                    experiment_id=experiment_id), folder/"model.pt")
    final = trace_numpy(model, x_normal_validation, device)
    diagnostics = embedding_diagnostics(final["embedding"], model.center.detach().cpu().numpy())
    diagnostics["initial_total_variance"] = initial_variance
    diagnostics["variance_ratio_to_initial"] = diagnostics["total_variance"]/max(initial_variance, 1e-30)
    diagnostics["relative_collapse_warning"] = diagnostics["variance_ratio_to_initial"] < 1e-4
    metadata = dict(experiment_id=experiment_id, seed=seed, input_dimension=x_train.shape[1],
        architecture=[x_train.shape[1], 32, 16, 8], device=device,
        training_rows=len(x_train), normal_validation_rows=len(x_normal_validation),
        center_before_epsilon=original_center, center=model.center.detach().cpu().tolist(),
        center_epsilon=policy["center_epsilon"], center_fixed=True,
        loss="mean(sum((embedding-center)^2)) + lambda/2 * sum(all weight squares)",
        optimizer="Adam; betas=(0.9,0.999); weight_decay=0; no scheduler; no AMP/TF32",
        initialization="Kaiming uniform hidden/ReLU; Xavier uniform final/linear",
        policy=policy, best_epoch=best_epoch, epochs_run=len(history), best_normal_validation_loss=best,
        training_seconds=duration, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        parameter_count=sum(p.numel() for p in model.parameters()),
        dense_macs_per_flow=sum(p.numel() for p in model.parameters()),
        embedding_diagnostics=diagnostics, checkpoint_sha256=sha(folder/"model.pt"))
    save_json(folder/"history.json", history)
    save_json(folder/"training.json", metadata)
    return model, metadata
