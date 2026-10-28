"""Exploratory matched PyTorch and DP-SGD federated-learning benchmark.

This uses public scikit-learn digits, the Stage 1 Opacus mechanism and OpenDP
accounting. It is not an official OpenDP or formally verified DP experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

from src import fl
from src import accounting as acc


@dataclass(frozen=True)
class Protocol:
    scenarios: tuple[str, ...] = ("iid", "label_dirichlet")
    seeds: tuple[int, ...] = (11, 23, 37)
    clients: int = 3
    rounds: int = 6
    local_steps: int = 3
    sample_rate: float = 0.1
    noise_multiplier: float = 1.2
    clip: float = 1.0
    public_normalizer: int = 40
    lr: float = 0.4
    dirichlet_alpha: float = 0.3
    delta: float = 1e-5
    epsilon_limit: float = 15.

    def dp_config(self):
        return fl.DPConfig(
            sample_rate=self.sample_rate, noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.clip, expected_batch_size=self.public_normalizer,
            local_steps=self.local_steps, rounds=self.rounds, lr=self.lr,
            target_epsilon=self.epsilon_limit, delta=self.delta,
            sampling_rng="csprng", secure_noise=True, reproducible_noise=False,
        )


def load_public_digits(seed):
    data = load_digits()
    X = (data.data / 16.).astype(np.float32)
    y = (data.target >= 5).astype(np.float32)
    train, test = train_test_split(
        np.arange(len(X)), random_state=seed, test_size=0.2, stratify=y
    )
    assert not set(train).intersection(test)
    return X[train], y[train], train, X[test], y[test], test


def client_indices(y, clients, scenario, seed, alpha=0.3, min_size=20):
    """Public benchmark IID or label-Dirichlet partition, frozen per seed.

    Deriving private client memberships from sensitive labels would require
    its own privacy analysis; that is NOT done here.
    """
    if clients < 2 or len(y) < clients * min_size:
        raise ValueError("insufficient records for requested client count")
    rng = np.random.default_rng(seed)
    if scenario == "iid":
        return [np.sort(p) for p in np.array_split(rng.permutation(len(y)), clients)]
    if scenario != "label_dirichlet":
        raise ValueError("unknown scenario")
    for _ in range(300):
        groups = [[] for _ in range(clients)]
        for label in sorted(np.unique(y)):
            ix = rng.permutation(np.flatnonzero(y == label))
            proportions = rng.dirichlet(np.full(clients, alpha))
            sizes = rng.multinomial(len(ix), proportions)
            at = 0
            for k, n in enumerate(sizes):
                groups[k].extend(ix[at:at + n].tolist())
                at += n
        if min(map(len, groups)) >= min_size:
            return [np.asarray(sorted(g), dtype=np.int64) for g in groups]
    raise ValueError("could not generate suitable label-Dirichlet groups")


def datasets_for_clients(X, y, ids, groups):
    indices = np.concatenate(groups)
    if not np.array_equal(np.sort(indices), np.arange(len(X))):
        raise ValueError("clients must cover every training row exactly once")
    return [fl.ClientDataset(X[g], y[g], k, record_ids=ids[g])
            for k, g in enumerate(groups)]


def plain_client_round(state, data, cfg, dimension, seed):
    """Unclipped, no-noise SGD; matched model, q, steps, LR and public B."""
    model = fl.make_model(dimension)
    model.load_state_dict(state)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    sampler_rng = torch.Generator().manual_seed(seed)
    loader = fl.make_poisson_loader(data, cfg, sampler_rng, use_csprng=False)
    steps = 0
    for xb, yb in loader:
        opt.zero_grad(set_to_none=True)
        if len(xb):
            # Sum / public B, not a data-dependent mean or dataset size.
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(xb).squeeze(-1), yb, reduction="sum"
            ) / cfg.expected_batch_size
            loss.backward()
            opt.step()
        steps += 1
    if steps != cfg.local_steps:
        raise RuntimeError("sampled step mismatch")
    return fl.get_state(model)


def run_pair(protocol, scenario, seed):
    acc.enable_opendp_features()  # Required by Stage 1's custom OpenDP domain.
    cfg = protocol.dp_config()
    X, y, ids, X_test, y_test, test_ids = load_public_digits(seed)
    groups = client_indices(
        y, protocol.clients, scenario, seed + 1009,
        alpha=protocol.dirichlet_alpha,
    )
    data = datasets_for_clients(X, y, ids, groups)
    dp_clients = fl.make_clients(data, cfg)
    init = fl.get_state(fl.make_model(X.shape[1], seed=seed))
    non_dp_state = dp_state = init
    histories = {"non_dp": [], "dp": []}
    elapsed = {"non_dp": 0., "dp": 0.}
    for i in range(protocol.rounds):
        tic = time.perf_counter()
        updates = [plain_client_round(
            non_dp_state, ds, cfg, X.shape[1], seed * 100000 + i * 1000 + k
        ) for k, ds in enumerate(data)]
        non_dp_state = fl.fedavg(updates)  # Uniform public aggregation weights.
        elapsed["non_dp"] += time.perf_counter() - tic
        row = fl.evaluate(non_dp_state, X_test, y_test, X.shape[1])
        histories["non_dp"].append({"round": i + 1, **row})
        tic = time.perf_counter()
        one = fl.run_federated(
            dp_clients, dp_state, cfg, X.shape[1],
            X_test=X_test, y_test=y_test, rounds=1,
        )
        dp_state = one["state"]
        elapsed["dp"] += time.perf_counter() - tic
        row = one["history"][0]
        histories["dp"].append({
            "round": i + 1, "accuracy": float(row["accuracy"]),
            "loss": float(row["loss"]),
            "epsilon": float(max(c.ledger[-1]["epsilon"] for c in dp_clients)),
        })
    target = histories["dp"][-1]["accuracy"]
    non_dp_round = next((r["round"] for r in histories["non_dp"]
                         if r["accuracy"] >= target), None)
    dp_round = next((r["round"] for r in histories["dp"]
                     if r["accuracy"] >= target), None)
    return {
        "scenario": scenario, "seed": seed,
        "client_sizes": [len(g) for g in groups],
        "client_positive_fractions": [float(y[g].mean()) for g in groups],
        "train_test_overlap": len(set(ids) & set(test_ids)),
        "final_non_dp_accuracy": histories["non_dp"][-1]["accuracy"],
        "final_dp_accuracy": target,
        "dp_max_client_epsilon": histories["dp"][-1]["epsilon"],
        "target_accuracy_from_dp_final": target,
        "non_dp_first_round_at_target": non_dp_round,
        "dp_first_round_at_target": dp_round,
        "elapsed_non_dp_seconds": elapsed["non_dp"],
        "elapsed_dp_seconds": elapsed["dp"],
        "history": histories,
    }


CAUTIONS = [
    "This is a small PUBLIC digits benchmark, not full MNIST/FEMNIST.",
    "Label-Dirichlet client splits depend on public benchmark labels.",
    "Dataset and model initialization are seeded; DP noise is intentionally unseeded.",
    "Stage 1's PyTorch Gaussian noise is not CSPRNG-backed and its RDP map assumes ideal noise.",
    "The DP final accuracy sets a post-hoc exploratory target; no confirmatory claim.",
    "Wall time includes local training AND accounting, not isolated RDP speed.",
    "The privacy epsilon is conditional on Stage 1 assumptions, not a formal proof.",
    "This experiment is not reviewed, accepted or integrated by OpenDP.",
]


def save_results(outdir, protocol, runs):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for scenario in protocol.scenarios:
        rs = [r for r in runs if r["scenario"] == scenario]
        if not rs:
            continue
        groups[scenario] = {
            "median_final_non_dp_accuracy": median(r["final_non_dp_accuracy"] for r in rs),
            "median_final_dp_accuracy": median(r["final_dp_accuracy"] for r in rs),
            "median_dp_max_client_epsilon": median(r["dp_max_client_epsilon"] for r in rs),
            "median_non_dp_seconds": median(r["elapsed_non_dp_seconds"] for r in rs),
            "median_dp_seconds": median(r["elapsed_dp_seconds"] for r in rs),
            "baseline_reached_dp_final_target_count": sum(
                r["non_dp_first_round_at_target"] is not None for r in rs
            ),
            "runs": len(rs),
        }
    with (outdir / "results.json").open("w") as f:
        json.dump({"protocol": asdict(protocol), "cautions": CAUTIONS,
                   "summary": groups, "runs": runs}, f, indent=2, allow_nan=False)
    with (outdir / "rounds.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["scenario", "seed", "arm", "round", "accuracy", "loss", "epsilon"]
        )
        w.writeheader()
        for r in runs:
            for arm in ("non_dp", "dp"):
                for point in r["history"][arm]:
                    w.writerow({
                        "scenario": r["scenario"], "seed": r["seed"], "arm": arm,
                        "round": point["round"], "accuracy": point["accuracy"],
                        "loss": point["loss"], "epsilon": point.get("epsilon", ""),
                    })
    lines = [
        "# Exploratory PyTorch vs Opacus federated-learning results", "",
        "Public-data research pilot; not an OpenDP-endorsed experiment or DP proof.", "",
        "## Protocol", "",
        f"Scenarios: {protocol.scenarios}; seeds: {protocol.seeds}; "
        f"clients: {protocol.clients}; rounds: {protocol.rounds}; "
        f"steps/round: {protocol.local_steps}; q: {protocol.sample_rate}; "
        f"sigma: {protocol.noise_multiplier}; C: {protocol.clip}; "
        f"public B: {protocol.public_normalizer}; LR: {protocol.lr}.",
        "", "## Summary across seeds", "",
        "| Scenario | Non-DP accuracy | DP accuracy | DP ε | Non-DP seconds | DP seconds | Baseline reached target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, d in groups.items():
        lines.append(
            f"| {name} | {d['median_final_non_dp_accuracy']:.4f} | "
            f"{d['median_final_dp_accuracy']:.4f} | "
            f"{d['median_dp_max_client_epsilon']:.4f} | "
            f"{d['median_non_dp_seconds']:.2f} | {d['median_dp_seconds']:.2f} | "
            f"{d['baseline_reached_dp_final_target_count']}/{d['runs']} |"
        )
    lines += [
        "", "## Target-reaching rounds", "",
        "| Scenario | Seed | DP final accuracy target | Baseline round | DP round |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in runs:
        b = r["non_dp_first_round_at_target"]
        lines.append(
            f"| {r['scenario']} | {r['seed']} | {r['target_accuracy_from_dp_final']:.4f} | "
            f"{b if b is not None else 'not reached'} | {r['dp_first_round_at_target']} |"
        )
    lines += ["", "## Caveats", ""] + ["- " + x for x in CAUTIONS]
    lines += ["", "Exact metrics, per-round histories, client fractions: results.json.", ""]
    (outdir / "results.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", default="iid,label_dirichlet")
    p.add_argument("--seeds", default="11,23,37")
    p.add_argument("--clients", type=int, default=3)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--local-steps", type=int, default=3)
    p.add_argument("--out", default="benchmark-output")
    a = p.parse_args()
    if a.clients < 2 or a.rounds < 1 or a.local_steps < 1:
        p.error("clients >= 2 and rounds/steps >= 1 required")
    protocol = Protocol(
        scenarios=tuple(a.scenarios.split(",")),
        seeds=tuple(int(x) for x in a.seeds.split(",")),
        clients=a.clients, rounds=a.rounds, local_steps=a.local_steps,
    )
    torch.set_num_threads(1)
    runs = []
    for scenario in protocol.scenarios:
        for seed in protocol.seeds:
            print(f"Running scenario={scenario} seed={seed}", flush=True)
            runs.append(run_pair(protocol, scenario, seed))
    save_results(a.out, protocol, runs)
    for scenario in protocol.scenarios:
        rs = [r for r in runs if r["scenario"] == scenario]
        print(
            f"{scenario}: median non-DP accuracy="
            f"{median(r['final_non_dp_accuracy'] for r in rs):.4f}; DP accuracy="
            f"{median(r['final_dp_accuracy'] for r in rs):.4f}", flush=True
        )
    print(f"Results written under {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
