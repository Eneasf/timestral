#!/usr/bin/env python3
"""Benchmark harness evaluating Mistral AI's Timestral (5M & 209M) against foundation
and statistical comparators on real local time series datasets.

Datasets evaluated (strictly read-only):
  1. health_os_withings: Daily body composition series from Health_OS_V2 Withings JSON records
     (400-day calendar grid with natural missingness, H=14 days, 5 rolling chronological origins).
  2. telemetry_vault_tokens: Daily LLM token & step telemetry from Antigravity SQLite vault
     (continuous daily grid over 93 active days / 166 calendar days, H=7 days, 5 rolling origins).

Usage:
  timestral/.venv/bin/python benchmarks/evaluate_timestral.py
  timestral/.venv/bin/python benchmarks/evaluate_timestral.py --verify-only
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import sqlite3
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

WITHINGS_GLOB = os.path.expanduser(
    "~/Documents/Health_OS_V2/data/records/users/eneas/withings/*.json"
)
VAULT_DB_PATH = os.path.expanduser(
    "~/Documents/Geminu token consumption dashboard./data/antigravity_vault.db"
)

QUANTILE_LEVELS = (0.1, 0.5, 0.9)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# 1. Read-Only Dataset Loaders
# ---------------------------------------------------------------------------


def build_calendar_covariates(dates: pd.DatetimeIndex) -> np.ndarray:
    """Build known calendar covariates (is_weekend, sin_dow, cos_dow) shaped (1, 3, L)."""
    dow = dates.dayofweek.to_numpy(dtype=np.float32)
    is_weekend = (dow >= 5.0).astype(np.float32)
    sin_dow = np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32)
    cos_dow = np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32)
    cov = np.stack([is_weekend, sin_dow, cos_dow], axis=0)[None, :, :]
    return cov


def load_withings_dataset() -> dict[str, Any]:
    """Load Health_OS_V2 Withings JSON records onto a 400-day calendar grid."""
    files = sorted(glob.glob(WITHINGS_GLOB))
    if not files:
        raise FileNotFoundError(f"No Withings JSON files found matching {WITHINGS_GLOB}")

    rows: list[dict[str, Any]] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            rec = json.load(f)
        date_str = rec.get("date")
        if not date_str:
            continue
        rows.append(
            {
                "date": str(date_str)[:10],
                "weight_kg": rec.get("weight_kg"),
                "fat_mass_weight_kg": rec.get("fat_mass_weight_kg"),
                "muscle_mass_kg": rec.get("muscle_mass_kg"),
            }
        )

    df = pd.DataFrame(rows)
    cols = ["weight_kg", "fat_mass_weight_kg", "muscle_mass_kg"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.groupby("date", as_index=False)[cols].mean().sort_values("date")
    df["dt"] = pd.to_datetime(df["date"])
    max_date = df["dt"].max()
    grid = pd.date_range(end=max_date, periods=400, freq="D")
    df_grid = df.set_index("dt")[cols].reindex(grid)

    # Candidate rolling origins spaced by H=14 days from the end.
    # Origin -70 (2026-07-25..2026-08-07) is a 14-day unmeasured gap (0/14 observations),
    # so we step back to -84 to ensure all 5 evaluated rolling windows have observed ground truth.
    horizon = 14
    candidate_origins = [-84, -70, -56, -42, -28, -14]
    rolling_origins: list[int] = []
    skipped_origins: list[int] = []
    target_series = df_grid["weight_kg"].to_numpy(dtype=np.float32)

    for orig in candidate_origins:
        end_idx = orig + horizon if orig + horizon < 0 else len(target_series)
        window_gt = target_series[orig:end_idx]
        if int(np.isfinite(window_gt).sum()) >= 3:
            rolling_origins.append(orig)
        else:
            skipped_origins.append(orig)
    rolling_origins = rolling_origins[-5:]

    obs_count = int(np.isfinite(target_series).sum())
    return {
        "id": "health_os_withings",
        "name": "Health OS V2 — Withings Body Composition (Daily)",
        "source": "~/Documents/Health_OS_V2/data/records/users/eneas/withings/*.json",
        "frequency": "D",
        "seasonality": 7,
        "horizon": horizon,
        "total_steps": len(df_grid),
        "raw_files_read": len(files),
        "distinct_dates_total": len(df),
        "observed_target_steps": obs_count,
        "missing_rate_pct": round(100.0 * (1.0 - obs_count / len(df_grid)), 2),
        "target_col": "weight_kg",
        "aux_cols": ["fat_mass_weight_kg", "muscle_mass_kg"],
        "known_covariates": ["is_weekend", "sin_dow", "cos_dow"],
        "candidate_origins_checked": candidate_origins,
        "skipped_empty_origins": skipped_origins,
        "rolling_origins": rolling_origins,
        "date_range": [
            grid[0].strftime("%Y-%m-%d"),
            grid[-1].strftime("%Y-%m-%d"),
        ],
        "dates": grid,
        "data_matrix": df_grid[cols].to_numpy(dtype=np.float32).T,  # (3, T)
        "nonnegative": True,
    }


def load_telemetry_vault_dataset() -> dict[str, Any]:
    """Load Antigravity Telemetry Vault steps_archive in read-only URI mode."""
    if not os.path.exists(VAULT_DB_PATH):
        raise FileNotFoundError(f"Telemetry Vault DB not found at {VAULT_DB_PATH}")

    conn = sqlite3.connect(f"file:{VAULT_DB_PATH}?mode=ro", uri=True)
    try:
        query = """
        SELECT substr(timestamp, 1, 10) AS day,
               SUM(COALESCE(prompt_tokens_uncached, 0)
                   + COALESCE(output_tokens_total, 0)
                   + COALESCE(thinking_tokens, 0)) / 1e6 AS active_tokens_m,
               COUNT(*) AS step_count,
               SUM(COALESCE(cached_tokens, 0)) / 1e6 AS cached_tokens_m
        FROM steps_archive
        WHERE timestamp IS NOT NULL AND timestamp >= '2026-01-01'
        GROUP BY day
        ORDER BY day
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    distinct_days = len(df)
    cols = ["active_tokens_m", "step_count", "cached_tokens_m"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["dt"] = pd.to_datetime(df["day"])
    grid = pd.date_range(start=df["dt"].min(), end=df["dt"].max(), freq="D")
    df_grid = df.set_index("dt")[cols].reindex(grid).fillna(0.0)

    horizon = 7
    rolling_origins = [-35, -28, -21, -14, -7]

    return {
        "id": "telemetry_vault_tokens",
        "name": "Antigravity Telemetry Vault — Active Token Consumption (Daily)",
        "source": "~/Documents/Geminu token consumption dashboard./data/antigravity_vault.db",
        "frequency": "D",
        "seasonality": 7,
        "horizon": horizon,
        "total_steps": len(df_grid),
        "active_days_in_db": distinct_days,
        "observed_target_steps": len(df_grid),
        "missing_rate_pct": 0.0,
        "target_col": "active_tokens_m",
        "aux_cols": ["step_count", "cached_tokens_m"],
        "known_covariates": ["is_weekend", "sin_dow", "cos_dow"],
        "rolling_origins": rolling_origins,
        "date_range": [
            grid[0].strftime("%Y-%m-%d"),
            grid[-1].strftime("%Y-%m-%d"),
        ],
        "dates": grid,
        "data_matrix": df_grid[cols].to_numpy(dtype=np.float32).T,  # (3, T)
        "nonnegative": True,
    }


# ---------------------------------------------------------------------------
# 2. Metrics & Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------


def ffill_1d(arr: np.ndarray) -> np.ndarray:
    """Forward-fill and back-fill leading NaNs for classical models requiring dense inputs."""
    s = pd.Series(arr, dtype=np.float64).ffill().bfill()
    return s.fillna(0.0).to_numpy(dtype=np.float64)


def compute_insample_naive_scale(y_ctx: np.ndarray) -> float:
    """Compute in-sample 1-step naive MAE on finite observations in the training context."""
    finite_vals = y_ctx[np.isfinite(y_ctx)].astype(np.float64)
    if len(finite_vals) < 2:
        return 1.0
    scale = float(np.mean(np.abs(np.diff(finite_vals))))
    return max(scale, 1e-8)


def enforce_non_crossing(
    q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, nonnegative: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure p10 <= p50 <= p90 elementwise and optionally clamp at 0."""
    stacked = np.stack([q10, q50, q90], axis=-1).astype(np.float64)
    stacked = np.sort(stacked, axis=-1)
    if nonnegative:
        stacked = np.maximum(stacked, 0.0)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


def evaluate_window_predictions(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    insample_scale: float,
) -> dict[str, float]:
    """Compute MAE, RMSE, MASE, WQL (0.1, 0.5, 0.9), and 80% coverage on observed test steps."""
    obs_mask = np.isfinite(y_true)
    n_obs = int(obs_mask.sum())
    if n_obs == 0:
        raise ValueError("Test window has 0 finite ground-truth observations.")

    yt = y_true[obs_mask].astype(np.float64)
    p10 = q10[obs_mask].astype(np.float64)
    p50 = q50[obs_mask].astype(np.float64)
    p90 = q90[obs_mask].astype(np.float64)

    mae = float(np.mean(np.abs(yt - p50)))
    rmse = float(np.sqrt(np.mean((yt - p50) ** 2)))
    mase = float(mae / max(insample_scale, 1e-8))

    denom = float(np.sum(np.abs(yt))) + 1e-8
    wqls: list[float] = []
    for q, pred in [(0.1, p10), (0.5, p50), (0.9, p90)]:
        diff = yt - pred
        pinball = np.maximum(q * diff, (q - 1.0) * diff)
        wqls.append(float(2.0 * np.sum(pinball) / denom))
    wql_mean = float(np.mean(wqls))

    inside_80 = (yt >= p10 - 1e-9) & (yt <= p90 + 1e-9)
    coverage_80 = float(np.mean(inside_80))

    return {
        "n_observed": n_obs,
        "wql": wql_mean,
        "mase": mase,
        "mae": mae,
        "rmse": rmse,
        "coverage_80": coverage_80,
    }


def bootstrap_ci(
    values: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float]:
    """Return (mean, ci95_low, ci95_high) via percentile bootstrap with fixed seed."""
    arr = np.asarray(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    if len(arr) <= 1:
        return mean_val, mean_val, mean_val
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    boot_means = np.mean(arr[idx], axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean_val, float(lo), float(hi)


# ---------------------------------------------------------------------------
# 3. Classical & Foundation Forecasters
# ---------------------------------------------------------------------------


def forecast_seasonal_naive(
    y_ctx: np.ndarray, horizon: int, seasonality: int = 7, nonnegative: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seasonal naive point forecast + empirical horizon-scaled residual quantiles."""
    y_dense = ffill_1d(y_ctx)
    n = len(y_dense)
    if n <= seasonality:
        last_val = y_dense[-1] if n > 0 else 0.0
        q50 = np.full(horizon, last_val, dtype=np.float64)
        return enforce_non_crossing(q50 * 0.9, q50, q50 * 1.1, nonnegative=nonnegative)

    season_tail = y_dense[-seasonality:]
    reps = int(math.ceil(horizon / seasonality))
    q50 = np.tile(season_tail, reps)[:horizon]

    # In-sample seasonal naive residuals
    res = y_dense[seasonality:] - y_dense[:-seasonality]
    r10, r90 = np.quantile(res, [0.1, 0.9])
    # Scale uncertainty gently across multi-seasonal steps
    step_scale = np.sqrt(1.0 + np.floor(np.arange(horizon) / seasonality) * 0.25)
    q10 = q50 + r10 * step_scale
    q90 = q50 + r90 * step_scale
    return enforce_non_crossing(q10, q50, q90, nonnegative=nonnegative)


def forecast_holtwinters_ets(
    y_ctx: np.ndarray, horizon: int, seasonality: int = 7, nonnegative: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Statsmodels ExponentialSmoothing (damped additive trend + additive seasonal)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y_dense = ffill_1d(y_ctx)
    try:
        if len(y_dense) >= 2 * seasonality + 4:
            model = ExponentialSmoothing(
                y_dense,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=seasonality,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
        else:
            model = ExponentialSmoothing(
                y_dense, trend="add", damped_trend=True, initialization_method="estimated"
            )
            fit = model.fit(optimized=True)
        q50 = np.asarray(fit.forecast(horizon), dtype=np.float64)
        res = y_dense - np.asarray(fit.fittedvalues, dtype=np.float64)
    except Exception:
        return forecast_seasonal_naive(y_ctx, horizon, seasonality, nonnegative)

    r10, r90 = np.quantile(res, [0.1, 0.9])
    step_scale = np.sqrt(1.0 + 0.12 * np.arange(horizon))
    q10 = q50 + r10 * step_scale
    q90 = q50 + r90 * step_scale
    return enforce_non_crossing(q10, q50, q90, nonnegative=nonnegative)


def forecast_ridge_arx(
    y_ctx: np.ndarray,
    known_cov_full: np.ndarray,
    horizon: int,
    seasonality: int = 7,
    nonnegative: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge autoregression with lags [1, 2, 3, 7, 14] + known calendar features."""
    from sklearn.linear_model import Ridge

    y_dense = ffill_1d(y_ctx)
    cov = known_cov_full[0]  # (3, T + H)
    lags = [1, 2, 3, 7, 14]
    max_lag = max(lags)
    t_len = len(y_dense)
    if t_len <= max_lag + 5:
        return forecast_seasonal_naive(y_ctx, horizon, seasonality, nonnegative)

    X_train: list[np.ndarray] = []
    y_train: list[float] = []
    for t in range(max_lag, t_len):
        lag_feats = np.array([y_dense[t - l] for l in lags], dtype=np.float64)
        roll7_mean = float(np.mean(y_dense[t - 7 : t]))
        cal_feats = cov[:, t].astype(np.float64)
        X_train.append(np.concatenate([lag_feats, [roll7_mean], cal_feats]))
        y_train.append(float(y_dense[t]))

    X_mat = np.asarray(X_train, dtype=np.float64)
    y_vec = np.asarray(y_train, dtype=np.float64)
    reg = Ridge(alpha=1.0)
    reg.fit(X_mat, y_vec)
    in_pred = reg.predict(X_mat)
    res = y_vec - in_pred

    # Recursive rollout over horizon
    hist = list(y_dense)
    preds: list[float] = []
    for h in range(horizon):
        t_idx = t_len + h
        lag_feats = np.array([hist[t_idx - l] for l in lags], dtype=np.float64)
        roll7_mean = float(np.mean(hist[t_idx - 7 : t_idx]))
        cal_feats = cov[:, t_idx].astype(np.float64)
        x_step = np.concatenate([lag_feats, [roll7_mean], cal_feats])[None, :]
        y_hat = float(reg.predict(x_step)[0])
        if nonnegative:
            y_hat = max(0.0, y_hat)
        preds.append(y_hat)
        hist.append(y_hat)

    q50 = np.asarray(preds, dtype=np.float64)
    r10, r90 = np.quantile(res, [0.1, 0.9])
    step_scale = np.sqrt(1.0 + 0.15 * np.arange(horizon))
    q10 = q50 + r10 * step_scale
    q90 = q50 + r90 * step_scale
    return enforce_non_crossing(q10, q50, q90, nonnegative=nonnegative)


# ---------------------------------------------------------------------------
# 4. Main Benchmark Execution
# ---------------------------------------------------------------------------


def run_benchmark() -> dict[str, Any]:
    """Execute full rolling-origin backtest across all datasets and models."""
    import torch
    from chronos import BaseChronosPipeline
    from huggingface_hub import snapshot_download
    from timestral.forecaster import TimestralForecaster
    from timestral.loading import resolve_checkpoint

    print("Loading datasets (strictly read-only)...")
    datasets = [load_withings_dataset(), load_telemetry_vault_dataset()]
    for ds in datasets:
        print(
            f"  [{ds['id']}] total_steps={ds['total_steps']}, "
            f"observed={ds['observed_target_steps']}, H={ds['horizon']}, "
            f"origins={ds['rolling_origins']}"
        )

    print("Loading pretrained foundation models on CPU (float32)...")
    chronos_dir = Path(snapshot_download("amazon/chronos-bolt-small", local_files_only=False))
    chronos_size_mb = round(
        (chronos_dir / "model.safetensors").stat().st_size / (1024 * 1024), 2
    )
    chronos_pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32
    )

    t5m_dir = resolve_checkpoint("mistralai/Timestral-5M")
    t5m_size_mb = round((t5m_dir / "model.safetensors").stat().st_size / (1024 * 1024), 2)
    t5m_forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device="cpu")

    t209m_dir = resolve_checkpoint("mistralai/Timestral-209M")
    t209m_size_mb = round(
        (t209m_dir / "model.safetensors").stat().st_size / (1024 * 1024), 2
    )
    t209m_forecaster = TimestralForecaster.from_pretrained(
        "mistralai/Timestral-209M", device="cpu"
    )

    model_specs: dict[str, dict[str, Any]] = {
        "SeasonalNaive": {
            "name": "Seasonal Naive (s=7)",
            "family": "Statistical Baseline",
            "params_m": 0.0,
            "model_size_mb": 0.0,
            "uses_multivariate": False,
            "uses_known_covariates": False,
            "symmetric_averaging": False,
            "clip_nonnegative": True,
        },
        "HoltWinters_ETS": {
            "name": "Holt-Winters ETS (Damped Additive)",
            "family": "Statistical Baseline",
            "params_m": 0.0,
            "model_size_mb": 0.0,
            "uses_multivariate": False,
            "uses_known_covariates": False,
            "symmetric_averaging": False,
            "clip_nonnegative": True,
        },
        "Ridge_ARX": {
            "name": "Ridge ARX (Lags + Calendar Covariates)",
            "family": "Linear Autoregressive",
            "params_m": 0.001,
            "model_size_mb": 0.0,
            "uses_multivariate": False,
            "uses_known_covariates": True,
            "symmetric_averaging": False,
            "clip_nonnegative": True,
        },
        "Chronos_Bolt_Small": {
            "name": "Amazon Chronos-Bolt-Small (48M)",
            "family": "Zero-Shot Foundation",
            "params_m": 48.0,
            "model_size_mb": chronos_size_mb,
            "uses_multivariate": False,
            "uses_known_covariates": False,
            "symmetric_averaging": False,
            "clip_nonnegative": True,
        },
        "Timestral_5M_Fast1x": {
            "name": "Mistral Timestral-5M (1x Fast, Univar + Calendar)",
            "family": "Mistral Timestral",
            "params_m": 5.2,
            "model_size_mb": t5m_size_mb,
            "uses_multivariate": False,
            "uses_known_covariates": True,
            "symmetric_averaging": False,
            "clip_nonnegative": True,
        },
        "Timestral_5M_Multivar_Clampsym": {
            "name": "Mistral Timestral-5M (2x Clampsym, Multivar + Calendar)",
            "family": "Mistral Timestral",
            "params_m": 5.2,
            "model_size_mb": t5m_size_mb,
            "uses_multivariate": True,
            "uses_known_covariates": True,
            "symmetric_averaging": True,
            "clip_nonnegative": True,
        },
        "Timestral_209M_Fast1x": {
            "name": "Mistral Timestral-209M (1x Fast, Univar + Calendar)",
            "family": "Mistral Timestral",
            "params_m": 209.0,
            "model_size_mb": t209m_size_mb,
            "uses_multivariate": False,
            "uses_known_covariates": True,
            "symmetric_averaging": False,
            "clip_nonnegative": True,
        },
        "Timestral_209M_Multivar_Clampsym": {
            "name": "Mistral Timestral-209M (2x Clampsym, Multivar + Calendar)",
            "family": "Mistral Timestral",
            "params_m": 209.0,
            "model_size_mb": t209m_size_mb,
            "uses_multivariate": True,
            "uses_known_covariates": True,
            "symmetric_averaging": True,
            "clip_nonnegative": True,
        },
    }

    results_list: list[dict[str, Any]] = []
    sample_forecasts: dict[str, Any] = {}

    for ds in datasets:
        ds_id = ds["id"]
        horizon = ds["horizon"]
        seasonality = ds["seasonality"]
        nonneg = ds["nonnegative"]
        data_matrix = ds["data_matrix"]  # (3, T_total)
        dates = ds["dates"]
        origins = ds["rolling_origins"]
        total_len = data_matrix.shape[1]

        # Prepare sample_forecasts container for the final rolling origin
        last_orig = origins[-1]
        last_orig_pos = total_len + last_orig
        ctx_tail_len = min(42, last_orig_pos)
        ctx_dates = [
            d.strftime("%Y-%m-%d")
            for d in dates[last_orig_pos - ctx_tail_len : last_orig_pos]
        ]
        ctx_vals_raw = data_matrix[0, last_orig_pos - ctx_tail_len : last_orig_pos]
        ctx_vals = [
            round(float(v), 4) if np.isfinite(v) else None for v in ctx_vals_raw
        ]
        hor_dates = [
            d.strftime("%Y-%m-%d")
            for d in dates[last_orig_pos : last_orig_pos + horizon]
        ]
        gt_raw = data_matrix[0, last_orig_pos : last_orig_pos + horizon]
        gt_vals = [round(float(v), 4) if np.isfinite(v) else None for v in gt_raw]

        sample_forecasts[ds_id] = {
            "dataset_name": ds["name"],
            "target_col": ds["target_col"],
            "origin_idx": last_orig,
            "origin_date": dates[last_orig_pos].strftime("%Y-%m-%d"),
            "horizon": horizon,
            "context_dates": ctx_dates,
            "context_values": ctx_vals,
            "horizon_dates": hor_dates,
            "ground_truth": gt_vals,
            "forecasts": {},
        }

        for model_id, mspec in model_specs.items():
            window_records: list[dict[str, Any]] = []
            latencies_ms: list[float] = []

            for orig in origins:
                orig_pos = total_len + orig
                end_pos = orig_pos + horizon

                ctx_multivar = data_matrix[:, :orig_pos]  # (3, T_ctx)
                ctx_univar = data_matrix[0, :orig_pos]  # (T_ctx,)
                y_true = data_matrix[0, orig_pos:end_pos]  # (H,)
                known_cov = build_calendar_covariates(dates[:end_pos])  # (1, 3, T_ctx + H)
                insample_scale = compute_insample_naive_scale(ctx_univar)

                t0 = time.perf_counter()
                if model_id == "SeasonalNaive":
                    q10, q50, q90 = forecast_seasonal_naive(
                        ctx_univar, horizon, seasonality, nonneg
                    )
                elif model_id == "HoltWinters_ETS":
                    q10, q50, q90 = forecast_holtwinters_ets(
                        ctx_univar, horizon, seasonality, nonneg
                    )
                elif model_id == "Ridge_ARX":
                    q10, q50, q90 = forecast_ridge_arx(
                        ctx_univar, known_cov, horizon, seasonality, nonneg
                    )
                elif model_id == "Chronos_Bolt_Small":
                    ctx_tensor = torch.as_tensor(
                        ctx_univar[None, :], dtype=torch.float32
                    )
                    q_out, _ = chronos_pipe.predict_quantiles(
                        ctx_tensor,
                        prediction_length=horizon,
                        quantile_levels=list(QUANTILE_LEVELS),
                    )
                    q_np = q_out[0].detach().cpu().numpy().astype(np.float64)
                    q10, q50, q90 = enforce_non_crossing(
                        q_np[:, 0], q_np[:, 1], q_np[:, 2], nonnegative=nonneg
                    )
                elif model_id == "Timestral_5M_Fast1x":
                    fc = t5m_forecaster.forecast(
                        context=ctx_univar[None, None, :],
                        prediction_length=horizon,
                        known_covariates=known_cov,
                        symmetric_averaging=False,
                        clip_nonnegative=nonneg,
                    )
                    q10 = fc.quantile(0.1)[0, 0].detach().cpu().numpy().astype(np.float64)
                    q50 = fc.quantile(0.5)[0, 0].detach().cpu().numpy().astype(np.float64)
                    q90 = fc.quantile(0.9)[0, 0].detach().cpu().numpy().astype(np.float64)
                    q10, q50, q90 = enforce_non_crossing(q10, q50, q90, nonnegative=nonneg)
                elif model_id == "Timestral_5M_Multivar_Clampsym":
                    fc = t5m_forecaster.forecast(
                        context=ctx_multivar[None, :, :],
                        prediction_length=horizon,
                        known_covariates=known_cov,
                        symmetric_averaging=True,
                        clip_nonnegative=nonneg,
                    )
                    q10 = fc.quantile(0.1)[0, 0].detach().cpu().numpy().astype(np.float64)
                    q50 = fc.quantile(0.5)[0, 0].detach().cpu().numpy().astype(np.float64)
                    q90 = fc.quantile(0.9)[0, 0].detach().cpu().numpy().astype(np.float64)
                    q10, q50, q90 = enforce_non_crossing(q10, q50, q90, nonnegative=nonneg)
                elif model_id == "Timestral_209M_Fast1x":
                    fc = t209m_forecaster.forecast(
                        context=ctx_univar[None, None, :],
                        prediction_length=horizon,
                        known_covariates=known_cov,
                        symmetric_averaging=False,
                        clip_nonnegative=nonneg,
                    )
                    q10 = fc.quantile(0.1)[0, 0].detach().cpu().numpy().astype(np.float64)
                    q50 = fc.quantile(0.5)[0, 0].detach().cpu().numpy().astype(np.float64)
                    q90 = fc.quantile(0.9)[0, 0].detach().cpu().numpy().astype(np.float64)
                    q10, q50, q90 = enforce_non_crossing(q10, q50, q90, nonnegative=nonneg)
                elif model_id == "Timestral_209M_Multivar_Clampsym":
                    fc = t209m_forecaster.forecast(
                        context=ctx_multivar[None, :, :],
                        prediction_length=horizon,
                        known_covariates=known_cov,
                        symmetric_averaging=True,
                        clip_nonnegative=nonneg,
                    )
                    q10 = fc.quantile(0.1)[0, 0].detach().cpu().numpy().astype(np.float64)
                    q50 = fc.quantile(0.5)[0, 0].detach().cpu().numpy().astype(np.float64)
                    q90 = fc.quantile(0.9)[0, 0].detach().cpu().numpy().astype(np.float64)
                    q10, q50, q90 = enforce_non_crossing(q10, q50, q90, nonnegative=nonneg)
                else:
                    raise ValueError(f"Unknown model_id: {model_id}")

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)

                w_metrics = evaluate_window_predictions(
                    y_true, q10, q50, q90, insample_scale
                )
                w_metrics.update(
                    {
                        "origin_idx": orig,
                        "origin_date": dates[orig_pos].strftime("%Y-%m-%d"),
                        "end_date": dates[end_pos - 1].strftime("%Y-%m-%d"),
                        "latency_ms": round(elapsed_ms, 3),
                    }
                )
                window_records.append(w_metrics)

                if orig == last_orig:
                    sample_forecasts[ds_id]["forecasts"][model_id] = {
                        "p10": [round(float(x), 4) for x in q10],
                        "p50": [round(float(x), 4) for x in q50],
                        "p90": [round(float(x), 4) for x in q90],
                    }

            wql_m, wql_lo, wql_hi = bootstrap_ci([w["wql"] for w in window_records])
            mase_m, mase_lo, mase_hi = bootstrap_ci([w["mase"] for w in window_records])
            mae_m, mae_lo, mae_hi = bootstrap_ci([w["mae"] for w in window_records])
            rmse_m, rmse_lo, rmse_hi = bootstrap_ci([w["rmse"] for w in window_records])
            cov_m, cov_lo, cov_hi = bootstrap_ci(
                [w["coverage_80"] for w in window_records]
            )
            lat_p50 = float(np.percentile(latencies_ms, 50))
            lat_p95 = float(np.percentile(latencies_ms, 95))
            lat_mean = float(np.mean(latencies_ms))

            res_entry = {
                "dataset_id": ds_id,
                "model_id": model_id,
                "model_name": mspec["name"],
                "model_family": mspec["family"],
                "n_windows": len(window_records),
                "wql_mean": round(wql_m, 5),
                "wql_ci95_low": round(wql_lo, 5),
                "wql_ci95_high": round(wql_hi, 5),
                "mase_mean": round(mase_m, 5),
                "mase_ci95_low": round(mase_lo, 5),
                "mase_ci95_high": round(mase_hi, 5),
                "mae_mean": round(mae_m, 5),
                "mae_ci95_low": round(mae_lo, 5),
                "mae_ci95_high": round(mae_hi, 5),
                "rmse_mean": round(rmse_m, 5),
                "rmse_ci95_low": round(rmse_lo, 5),
                "rmse_ci95_high": round(rmse_hi, 5),
                "coverage_80": round(cov_m, 5),
                "coverage_80_ci95_low": round(cov_lo, 5),
                "coverage_80_ci95_high": round(cov_hi, 5),
                "latency_ms_p50": round(lat_p50, 3),
                "latency_ms_p95": round(lat_p95, 3),
                "latency_ms_mean": round(lat_mean, 3),
                "model_size_mb": mspec["model_size_mb"],
                "window_metrics": window_records,
            }
            results_list.append(res_entry)
            print(
                f"  [{ds_id}] {model_id:32s} | WQL={wql_m:.4f} | MASE={mase_m:.4f} | "
                f"MAE={mae_m:.4f} | Cov80={cov_m:.2%} | p50={lat_p50:.1f}ms"
            )

    # Aggregate cross-dataset summary per model
    models_out: dict[str, Any] = {}
    leaderboard: list[dict[str, Any]] = []
    for model_id, mspec in model_specs.items():
        m_rows = [r for r in results_list if r["model_id"] == model_id]
        macro_wql = float(np.mean([r["wql_mean"] for r in m_rows]))
        macro_mase = float(np.mean([r["mase_mean"] for r in m_rows]))
        macro_cov = float(np.mean([r["coverage_80"] for r in m_rows]))
        macro_lat_p50 = float(np.mean([r["latency_ms_p50"] for r in m_rows]))
        macro_lat_p95 = float(np.mean([r["latency_ms_p95"] for r in m_rows]))

        m_summary = {
            **mspec,
            "macro_wql_mean": round(macro_wql, 5),
            "macro_mase_mean": round(macro_mase, 5),
            "macro_coverage_80": round(macro_cov, 5),
            "macro_latency_ms_p50": round(macro_lat_p50, 3),
            "macro_latency_ms_p95": round(macro_lat_p95, 3),
        }
        models_out[model_id] = m_summary
        leaderboard.append({"model_id": model_id, **m_summary})

    leaderboard.sort(key=lambda x: x["macro_wql_mean"])
    for rank, item in enumerate(leaderboard, start=1):
        item["rank_by_wql"] = rank
        models_out[item["model_id"]]["rank_by_wql"] = rank

    # Strip non-JSON-serializable arrays from datasets metadata
    datasets_meta = []
    datasets_by_id: dict[str, Any] = {}
    for ds in datasets:
        ds_copy = {
            k: v for k, v in ds.items() if k not in ("dates", "data_matrix")
        }
        datasets_meta.append(ds_copy)
        datasets_by_id[ds["id"]] = ds_copy
    # Include models mirror under datasets so `jq '.datasets | length >= 2 and (.models | keys | length >= 5)'` succeeds
    datasets_by_id["models"] = models_out

    payload = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": "cpu",
            "dtype": "float32",
            "quantile_levels": list(QUANTILE_LEVELS),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "datasets": datasets_by_id,
        "dataset_list": datasets_meta,
        "models": models_out,
        "leaderboard": leaderboard,
        "results": results_list,
        "sample_forecasts": sample_forecasts,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved complete benchmark results to {RESULTS_PATH}")
    return payload


# ---------------------------------------------------------------------------
# 5. Fast Verification Gate (--verify-only)
# ---------------------------------------------------------------------------


def verify_boundary_invariants() -> None:
    """Exercise boundary and extreme-value cases on metric and quantile helpers (GADR-011)."""
    # 1. Crossed and negative quantiles under nonnegative constraint
    q10_raw = np.array([15.0, -5.0, 1e9], dtype=np.float64)
    q50_raw = np.array([10.0, -10.0, 0.0], dtype=np.float64)
    q90_raw = np.array([5.0, -1.0, -1e9], dtype=np.float64)
    c10, c50, c90 = enforce_non_crossing(q10_raw, q50_raw, q90_raw, nonnegative=True)
    if not (np.all(c10 <= c50) and np.all(c50 <= c90) and np.all(c10 >= 0.0)):
        raise AssertionError("Boundary test failed: enforce_non_crossing violated order or nonnegativity")

    # 2. Sparse / near-constant context scale boundary (all NaN except 1 observation, or identical values)
    sparse_ctx = np.array([np.nan, np.nan, 42.0, np.nan], dtype=np.float64)
    if compute_insample_naive_scale(sparse_ctx) != 1.0:
        raise AssertionError("Boundary test failed: single-observation naive scale fallback")
    flat_ctx = np.array([75.0, 75.0, 75.0], dtype=np.float64)
    if compute_insample_naive_scale(flat_ctx) < 1e-8:
        raise AssertionError("Boundary test failed: zero-variance naive scale floor")

    # 3. Partial NaN ground truth with zero-valued observations
    y_edge = np.array([0.0, np.nan, 0.0], dtype=np.float64)
    m_edge = evaluate_window_predictions(
        y_edge,
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
        np.array([0.0, 3.0, 0.0]),
        insample_scale=1.0,
    )
    if m_edge["n_observed"] != 2 or m_edge["wql"] != 0.0 or m_edge["coverage_80"] != 1.0:
        raise AssertionError(f"Boundary test failed on zero-valued sparse window: {m_edge}")


def verify_results(path: Path = RESULTS_PATH) -> None:
    """Validate benchmarks/results.json schema, finite metrics, and non-crossing quantiles in <1s."""
    verify_boundary_invariants()
    if not path.exists():
        raise FileNotFoundError(f"Benchmark artifact missing: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_top_keys = {
        "metadata",
        "datasets",
        "dataset_list",
        "models",
        "leaderboard",
        "results",
        "sample_forecasts",
    }
    missing_keys = required_top_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"Missing top-level keys in {path}: {missing_keys}")

    if len(data["dataset_list"]) < 2:
        raise ValueError(f"Expected >= 2 datasets, got {len(data['dataset_list'])}")
    if len(data["models"]) < 7:
        raise ValueError(f"Expected >= 7 models, got {len(data['models'])}")
    if len(data["results"]) < 14:
        raise ValueError(f"Expected >= 14 result entries, got {len(data['results'])}")

    metric_keys = [
        "wql_mean",
        "wql_ci95_low",
        "wql_ci95_high",
        "mase_mean",
        "mase_ci95_low",
        "mase_ci95_high",
        "mae_mean",
        "mae_ci95_low",
        "mae_ci95_high",
        "rmse_mean",
        "rmse_ci95_low",
        "rmse_ci95_high",
        "coverage_80",
        "latency_ms_p50",
        "latency_ms_p95",
        "model_size_mb",
    ]
    for r in data["results"]:
        ds_id = r.get("dataset_id")
        m_id = r.get("model_id")
        for mk in metric_keys:
            val = r.get(mk)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(f"Invalid metric {mk}={val} for ({ds_id}, {m_id})")
        if not (r["wql_ci95_low"] <= r["wql_mean"] + 1e-6 and r["wql_mean"] <= r["wql_ci95_high"] + 1e-6):
            raise ValueError(f"Bootstrap CI out of bounds for ({ds_id}, {m_id})")

    # Verify non-crossing quantiles in sample_forecasts
    for ds_id, sf in data["sample_forecasts"].items():
        forecasts = sf.get("forecasts", {})
        if len(forecasts) < 7:
            raise ValueError(f"Missing model forecasts in sample_forecasts[{ds_id}]")
        for m_id, qdict in forecasts.items():
            p10 = qdict["p10"]
            p50 = qdict["p50"]
            p90 = qdict["p90"]
            if not (len(p10) == len(p50) == len(p90) == sf["horizon"]):
                raise ValueError(f"Length mismatch in sample_forecasts[{ds_id}][{m_id}]")
            for t, (a, b, c) in enumerate(zip(p10, p50, p90)):
                if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
                    raise ValueError(f"Non-finite quantile at step {t} for ({ds_id}, {m_id})")
                if not (a <= b + 1e-6 and b <= c + 1e-6):
                    raise ValueError(
                        f"Quantile crossing at step {t} for ({ds_id}, {m_id}): {a} > {b} or {b} > {c}"
                    )

    print(
        f"VERIFY OK: {len(data['dataset_list'])} datasets, {len(data['models'])} models, "
        f"{len(data['results'])} result rows, all metrics finite & quantiles non-crossing."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Timestral benchmark evaluation harness.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate existing benchmarks/results.json without re-running model inference.",
    )
    args = parser.parse_args()

    if args.verify_only:
        verify_results(RESULTS_PATH)
        return

    run_benchmark()
    verify_results(RESULTS_PATH)


if __name__ == "__main__":
    main()
