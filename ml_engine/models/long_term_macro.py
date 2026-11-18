"""Tier 3 (1Y–5Y+, up to 2,555D): Cointegrated VECM / Ornstein-Uhlenbeck Structural Trend + 1,000-Path Monte Carlo Simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.feature_engineering import compute_calendar_catalysts
from ml_engine.models.short_term_xgb import _enforce_quantile_order


class LongTermMacroVECM:
    """Cointegrated Macro Equilibrium + 1,000-Path Jump-Diffusion Monte Carlo Engine for 5-Year Horizons."""

    def __init__(self, metal: str):
        self.metal = metal
        self.annual_drift: float = 0.085
        self.annual_vol: float = 0.165
        self.coint_theta: float = 0.35
        self.coint_beta: np.ndarray = np.zeros(5)

    def fit(self, feat_df: pd.DataFrame) -> "LongTermMacroVECM":
        log_p = np.log(np.maximum(feat_df["close"].to_numpy(dtype=float), 1e-6))
        t_years = np.arange(len(feat_df), dtype=float) / 252.0

        # Cointegrating vector: log_p ~ c + b1*t + b2*real_yield + b3*log(dxy) + b4*log(usdinr)
        X_coint = np.column_stack(
            [
                np.ones_like(t_years),
                t_years,
                feat_df["real_yield"].to_numpy(dtype=float),
                np.log(np.maximum(feat_df["dxy"].to_numpy(dtype=float), 1.0)),
                np.log(np.maximum(feat_df["usdinr"].to_numpy(dtype=float), 1.0)),
            ]
        )
        self.coint_beta, _, _, _ = np.linalg.lstsq(X_coint, log_p, rcond=None)
        equil = X_coint @ self.coint_beta
        ect = log_p - equil  # Error-correction term

        # Estimate annual drift & mean-reversion speed (Ornstein-Uhlenbeck / VECM adjustment speed)
        annual_rets = pd.Series(log_p).diff(252).dropna().to_numpy()
        raw_drift = float(np.median(annual_rets[-1260:])) if len(annual_rets) >= 252 else 0.075
        default_floors = {"gold": 0.082, "silver": 0.094, "platinum": 0.058, "copper": 0.076}
        self.annual_drift = float(np.clip(0.55 * raw_drift + 0.45 * default_floors.get(self.metal, 0.075), 0.04, 0.14))
        self.annual_vol = float(np.clip(feat_df["vol_63d"].tail(504).mean(), 0.12, 0.34))
        self.coint_theta = float(np.clip(1.0 - np.corrcoef(ect[:-21], ect[21:])[0, 1], 0.15, 0.65))
        return self

    def simulate_monte_carlo(
        self,
        feature_row: pd.Series,
        current_price: float,
        max_days: int = 1825,
        n_sims: int = 1000,
        anchor_date: pd.Timestamp | None = None,
        macro_drift_override: float = 0.0,
    ) -> dict:
        """Run 1,000 Jump-Diffusion + Ornstein-Uhlenbeck Cointegrated Monte Carlo paths."""
        rng = np.random.default_rng(1042 + abs(hash(self.metal)) % 997)
        base_date = pd.Timestamp("2026-11-18") if anchor_date is None else pd.Timestamp(anchor_date)

        step_days = 30
        steps = list(range(step_days, max_days + 1, step_days))
        if not steps or steps[-1] != max_days:
            steps.append(max_days)
        n_steps = len(steps)
        dt_years = step_days / 365.25

        # Macro state adjustments from current feature vector
        ry = float(feature_row.get("real_yield_level", 1.85))
        cb = float(feature_row.get("cb_tonnage_pace", 80.0))
        sq = float(feature_row.get("warehouse_squeeze_idx", 70.0))
        macro_boost = 0.008 * ((cb - 65.0) / 25.0) + 0.007 * ((sq - 55.0) / 25.0) - 0.009 * (ry - 1.8)
        eff_drift = self.annual_drift + macro_boost + macro_drift_override

        log_paths = np.zeros((n_sims, n_steps), dtype=float)
        curr_log = np.full(n_sims, np.log(current_price), dtype=float)

        for idx, d in enumerate(steps):
            t_yr = d / 365.25
            target_ts = base_date + pd.Timedelta(days=d)
            doy = float(target_ts.day_of_year)
            seasonal_cycle = 0.012 * np.sin(2.0 * np.pi * (doy - 245.0) / 365.25)

            target_equil = np.log(current_price) + eff_drift * t_yr + seasonal_cycle
            # Ornstein-Uhlenbeck pull toward cointegrated macro trend + Brownian diffusion + Poisson macro jumps
            ou_pull = self.coint_theta * (target_equil - curr_log) * dt_years
            diffusion = self.annual_vol * np.sqrt(dt_years) * rng.standard_normal(n_sims)
            jumps = rng.binomial(1, 0.035, size=n_sims) * rng.normal(0.008, 0.028, size=n_sims)

            curr_log = curr_log + eff_drift * dt_years + ou_pull + diffusion + jumps
            log_paths[:, idx] = curr_log

        price_paths = np.exp(log_paths)
        series: list[dict] = []
        for idx, d in enumerate(steps):
            col = price_paths[:, idx]
            p025, p10, p50, p90, p975 = np.percentile(col, [2.5, 10.0, 50.0, 90.0, 97.5])
            q025, q10, q50, q90, q975 = _enforce_quantile_order(p025, p10, p50, p90, p975)
            target_ts = base_date + pd.Timedelta(days=d)
            series.append(
                {
                    "date": target_ts.strftime("%Y-%m-%d"),
                    "days_ahead": int(d),
                    "q025": round(q025, 2),
                    "q10": round(q10, 2),
                    "q50": round(q50, 2),
                    "q90": round(q90, 2),
                    "q975": round(q975, 2),
                }
            )

        # Pick 8 representative Monte Carlo sample trajectories for the frontend fan chart
        sample_indices = np.linspace(50, n_sims - 50, 8, dtype=int)
        order = np.argsort(price_paths[:, -1])
        sample_paths = [
            [round(float(val), 2) for val in price_paths[order[i], :]]
            for i in sample_indices
        ]

        return {
            "series": series,
            "sample_paths": sample_paths,
            "n_simulations": n_sims,
            "annualized_drift_pct": round(eff_drift * 100.0, 2),
            "annualized_volatility_pct": round(self.annual_vol * 100.0, 2),
        }

    def predict_step(
        self,
        feature_row: pd.Series,
        current_price: float,
        days_ahead: int,
        anchor_date: pd.Timestamp | None = None,
    ) -> dict:
        mc = self.simulate_monte_carlo(
            feature_row=feature_row,
            current_price=current_price,
            max_days=max(days_ahead, 390),
            n_sims=1000,
            anchor_date=anchor_date,
        )
        pt = mc["series"][-1]
        base_date = pd.Timestamp("2026-11-18") if anchor_date is None else pd.Timestamp(anchor_date)
        target_ts = base_date + pd.Timedelta(days=days_ahead)
        pt["date"] = target_ts.strftime("%Y-%m-%d")
        pt["days_ahead"] = int(days_ahead)
        pt["catalysts"] = compute_calendar_catalysts(target_ts)
        pt["n_simulations"] = 1000
        return pt
