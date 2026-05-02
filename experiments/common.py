"""Shared utilities for reproducible K-Prototypes experiments."""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GAMMA_RANGE = (1e-3, 1e3)
DEFAULT_SEED = 42
PRIMARY_REAL_DATASETS = ["Heart Disease", "Credit Approval", "Adult", "Bank Marketing"]
EXTENDED_REAL_DATASETS = ["Abalone", "Adult", "Australian Credit", "Automobile", "Bank Marketing", "Census KDD", "Credit Approval", "German Credit", "Heart Disease", "Horse Colic"]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def setup_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def preprocess_mixed_data(X: np.ndarray, categorical_indices: Sequence[int]) -> np.ndarray:
    categorical_indices = list(categorical_indices)
    continuous_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]
    out = np.empty(X.shape, dtype=object)
    if continuous_indices:
        cont = StandardScaler().fit_transform(X[:, continuous_indices].astype(float))
        for j, i in enumerate(continuous_indices):
            out[:, i] = cont[:, j]
    for i in categorical_indices:
        out[:, i] = X[:, i].astype(str)
    return out


def default_gamma(X: np.ndarray, categorical_indices: Sequence[int]) -> float:
    categorical_indices = set(categorical_indices)
    continuous_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]
    if not continuous_indices:
        return 1.0
    return max(0.5 * float(np.mean(np.std(X[:, continuous_indices].astype(float), axis=0))), 1e-12)


def run_kprototypes(X: np.ndarray, categorical_indices: Sequence[int], n_clusters: int, gamma: float, n_init: int, max_iter: int, seed: int) -> Tuple[np.ndarray, float]:
    model = KPrototypes(n_clusters=n_clusters, gamma=float(gamma), n_init=n_init, max_iter=max_iter, random_state=seed, verbose=0)
    labels = model.fit_predict(X, categorical=list(categorical_indices))
    return labels, float(model.cost_)


class BudgetedObjective:
    def __init__(self, X: np.ndarray, y_true: Optional[np.ndarray], categorical_indices: Sequence[int], n_clusters: int, n_init: int, max_iter: int, seed: int, budget: int, use_ari_objective: bool = False):
        self.X = X
        self.y_true = y_true
        self.categorical_indices = list(categorical_indices)
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.seed = seed
        self.budget = budget
        self.use_ari_objective = use_ari_objective
        self.n_eval = 0
        self.cache: Dict[float, float] = {}
        self.best_score = np.inf
        self.best_gamma = None
        self.best_labels = None

    def evaluate(self, gamma: float) -> float:
        if self.n_eval >= self.budget:
            return float(self.best_score)
        key = round(float(np.log10(max(gamma, 1e-12))), 6)
        if key in self.cache:
            return self.cache[key]
        self.n_eval += 1
        labels, cost = run_kprototypes(self.X, self.categorical_indices, self.n_clusters, gamma, self.n_init, self.max_iter, self.seed + self.n_eval)
        if self.use_ari_objective and self.y_true is not None and len(np.unique(labels)) > 1:
            score = 1.0 - adjusted_rand_score(self.y_true, labels)
        else:
            score = cost
        self.cache[key] = float(score)
        if score < self.best_score:
            self.best_score = float(score)
            self.best_gamma = float(gamma)
            self.best_labels = labels
        return float(score)


def dmkpo_search(obj: BudgetedObjective, gamma_range: Tuple[float, float], budget: int, log_space: bool, n_initial: int, alpha: float = 0.3) -> None:
    if log_space:
        current = list(np.logspace(np.log10(gamma_range[0]), np.log10(gamma_range[1]), min(n_initial, budget)))
        radius = 0.75
    else:
        current = list(np.linspace(gamma_range[0], gamma_range[1], min(n_initial, budget)))
        radius = (gamma_range[1] - gamma_range[0]) / 4.0
    gammas: List[float] = []
    scores: List[float] = []
    while obj.n_eval < budget and current:
        for gamma in current:
            if obj.n_eval >= budget:
                break
            gammas.append(float(gamma))
            scores.append(obj.evaluate(float(gamma)))
        if len(gammas) < 2 or obj.n_eval >= budget:
            break
        order = np.argsort(gammas)
        raw_gs = np.array(gammas, dtype=float)[order]
        raw_es = np.array(scores, dtype=float)[order]
        unique_scores: Dict[float, float] = {}
        for gamma_value, score_value in zip(raw_gs, raw_es):
            unique_scores[round(float(gamma_value), 12)] = float(score_value)
        gs = np.array(list(unique_scores.keys()), dtype=float)
        es = np.array(list(unique_scores.values()), dtype=float)
        if len(gs) < 2:
            break
        x = np.log10(np.maximum(gs, 1e-12)) if log_space else gs
        e_norm = (es - es.min()) / (es.max() - es.min() + 1e-12)
        if len(gs) >= 3:
            grad = np.gradient(es, x, edge_order=1)
        else:
            grad = np.array([0.0, 0.0])
        g_norm = (np.abs(grad) - np.abs(grad).min()) / (np.abs(grad).max() - np.abs(grad).min() + 1e-12)
        mark_score = (1 - alpha) * (1 - e_norm) + alpha * g_norm
        ranked = list(np.argsort(-mark_score))
        selected = []
        acc = 0.0
        total = float(mark_score.sum())
        for idx in ranked:
            selected.append(int(idx))
            acc += float(mark_score[idx])
            if total <= 0 or acc >= 0.5 * total:
                break
        new_points: List[float] = []
        for idx in selected:
            center = float(gs[idx])
            if log_space:
                c = np.log10(max(center, 1e-12))
                vals = np.linspace(max(np.log10(gamma_range[0]), c - radius), min(np.log10(gamma_range[1]), c + radius), 3)
                new_points.extend([10 ** v for v in vals])
            else:
                vals = np.linspace(max(gamma_range[0], center - radius), min(gamma_range[1], center + radius), 3)
                new_points.extend([float(v) for v in vals])
        current = new_points[: max(0, budget - obj.n_eval)]
        radius *= 0.7


def method_rows(X: np.ndarray, y: np.ndarray, cat: Sequence[int], n_clusters: int, run_id: int, cfg: Dict, include_optuna: bool = True, use_ari_objective: bool = False) -> List[Dict]:
    seed = int(cfg["seed"]) + run_id * 997
    budget = int(cfg["budget"])
    rows: List[Dict] = []

    def add(method: str, gamma, labels, n_eval: int, elapsed: float):
        if labels is None or gamma is None:
            rows.append({"method": method, "ari": np.nan, "nmi": np.nan, "gamma": np.nan, "n_eval": n_eval, "time_s": elapsed})
            return
        rows.append({"method": method, "ari": adjusted_rand_score(y, labels) if len(np.unique(labels)) > 1 else 0.0, "nmi": normalized_mutual_info_score(y, labels) if len(np.unique(labels)) > 1 else 0.0, "gamma": float(gamma), "n_eval": n_eval, "time_s": elapsed})

    t = time.time(); g = default_gamma(X, cat); labels, _ = run_kprototypes(X, cat, n_clusters, g, cfg["n_init"], cfg["max_iter"], seed); add("Default-KP", g, labels, 1, time.time() - t)
    t = time.time(); obj = BudgetedObjective(X, y, cat, n_clusters, cfg["n_init"], cfg["max_iter"], seed + 1000, budget, use_ari_objective); [obj.evaluate(g) for g in np.logspace(np.log10(cfg["gamma_range"][0]), np.log10(cfg["gamma_range"][1]), budget)]; add("Grid-Log", obj.best_gamma, obj.best_labels, obj.n_eval, time.time() - t)
    t = time.time(); obj = BudgetedObjective(X, y, cat, n_clusters, cfg["n_init"], cfg["max_iter"], seed + 2000, budget, use_ari_objective); rng = np.random.default_rng(seed + 2000); [obj.evaluate(g) for g in 10 ** rng.uniform(np.log10(cfg["gamma_range"][0]), np.log10(cfg["gamma_range"][1]), budget)]; add("Random", obj.best_gamma, obj.best_labels, obj.n_eval, time.time() - t)
    t = time.time()
    if include_optuna:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            obj = BudgetedObjective(X, y, cat, n_clusters, cfg["n_init"], cfg["max_iter"], seed + 3000, budget, use_ari_objective)
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed + 3000))
            study.optimize(lambda trial: obj.evaluate(trial.suggest_float("gamma", cfg["gamma_range"][0], cfg["gamma_range"][1], log=True)), n_trials=budget, show_progress_bar=False)
            add("Optuna-TPE", obj.best_gamma, obj.best_labels, obj.n_eval, time.time() - t)
        except Exception:
            add("Optuna-TPE", None, None, 0, time.time() - t)
    else:
        add("Optuna-TPE", None, None, 0, 0.0)
    for method, log_space, offset in [("DMKPO-Lin", False, 4000), ("DMKPO-Log", True, 5000)]:
        t = time.time(); obj = BudgetedObjective(X, y, cat, n_clusters, cfg["n_init"], cfg["max_iter"], seed + offset, budget, use_ari_objective); dmkpo_search(obj, cfg["gamma_range"], budget, log_space, cfg["n_initial"], cfg.get("alpha", 0.3)); add(method, obj.best_gamma, obj.best_labels, obj.n_eval, time.time() - t)
    return rows


def summarize(df: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    return df.groupby(list(groups) + ["method"], dropna=False).agg(mean_ari=("ari", "mean"), std_ari=("ari", "std"), mean_nmi=("nmi", "mean"), std_nmi=("nmi", "std"), mean_gamma=("gamma", "mean"), mean_n_eval=("n_eval", "mean"), mean_time_s=("time_s", "mean")).reset_index()


def tests(df: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    out = []
    for key, sub in df.groupby(list(groups), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        meta = dict(zip(groups, key))
        target = sub[sub.method == "DMKPO-Log"].sort_values("run_id").ari.values
        for baseline in [m for m in sub.method.unique() if m != "DMKPO-Log"]:
            base = sub[sub.method == baseline].sort_values("run_id").ari.values
            n = min(len(target), len(base)); a = target[:n]; b = base[:n]
            valid = np.isfinite(a) & np.isfinite(b)
            a = a[valid]; b = b[valid]; n = len(a)
            if n == 0:
                continue
            if n < 2 or np.all(a - b == 0):
                stat, p, used = np.nan, 1.0, "all_ties_or_small_n"
            elif np.mean((a - b) == 0) > 0.3:
                n_pos = int(np.sum((a - b) > 0)); n_eff = int(np.sum((a - b) != 0)); stat, p, used = np.nan, float(stats.binomtest(n_pos, n_eff, 0.5, alternative="greater").pvalue) if n_eff else 1.0, "sign_test_greater"
            else:
                stat, p = stats.wilcoxon(a, b, alternative="greater", zero_method="pratt"); used = "wilcoxon_pratt_greater"
            out.append({**meta, "baseline": baseline, "n_pairs": n, "mean_ari_dmkpo_log": float(np.nanmean(a)) if n else np.nan, "mean_ari_baseline": float(np.nanmean(b)) if n else np.nan, "test_statistic": stat, "p_value": p, "test_used": used, "significant_0_05": bool(p < 0.05)})
    return pd.DataFrame(out)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--no-optuna", action="store_true")
    return parser
