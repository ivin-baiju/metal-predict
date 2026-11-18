"""Tier 1 (1D–90D, 1M, 3M): Direct Multi-Horizon XGBoost Quantile Regressors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from data_pipeline.feature_engineering import FEATURE_COLUMNS, compute_calendar_catalysts

ANCHOR_CALENDAR_DAYS = [5, 15, 30, 60, 90]


def _enforce_quantile_order(
    q025: float,
    q10: float,
    q50: float,
    q90: float,
    q975: float,
    min_rel_gap: float = 0.0025,
) -> tuple[float, float, float, float, float]:
    """Ensure strict monotonicity q025 < q10 < q50 < q90 < q975."""
    vals = np.sort([q025, q10, q50, q90, q975]).astype(float)
    mid = vals[2]
    gap = max(abs(mid) * min_rel_gap, 1e-4)
    vals[1] = min(vals[1], mid - gap)
    vals[0] = min(vals[0], vals[1] - gap)
    vals[3] = max(vals[3], mid + gap)
    vals[4] = max(vals[4], vals[3] + gap)
    return float(vals[0]), float(vals[1]), float(mid), float(vals[3]), float(vals[4])


class ShortTermXGBEnsemble:
    """Direct multi-horizon XGBoost Quantile Ensemble for 1D to 90D forecasts."""

    def __init__(self, metal: str):
        self.metal = metal
        self.median_models: dict[int, XGBRegressor] = {}
        self.q10_models: dict[int, XGBRegressor] = {}
        self.q90_models: dict[int, XGBRegressor] = {}
        self.residual_std: dict[int, float] = {}
        self.annual_vol: float = 0.16

    def fit(self, feat_df: pd.DataFrame) -> "ShortTermXGBEnsemble":
        X_all = feat_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        log_p = np.log(np.maximum(feat_df["close"].to_numpy(dtype=float), 1e-6))
        self.annual_vol = float(np.clip(feat_df["vol_21d"].tail(252).mean(), 0.08, 0.45))

        for cal_d in ANCHOR_CALENDAR_DAYS:
            b_steps = max(1, int(round(cal_d * 5.0 / 7.0)))
            y_target = np.roll(log_p, -b_steps) - log_p
            valid_n = len(feat_df) - b_steps
            start_idx = max(252, valid_n - 1800)
            X_tr = X_all[start_idx:valid_n:2]
            y_tr = y_target[start_idx:valid_n:2]

            med_model = XGBRegressor(
                n_estimators=55,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=42 + cal_d,
                n_jobs=2,
            )
            med_model.fit(X_tr, y_tr)
            preds = med_model.predict(X_tr)
            res_std = float(np.std(y_tr - preds))
            self.median_models[cal_d] = med_model
            self.residual_std[cal_d] = max(res_std, self.annual_vol * np.sqrt(cal_d / 365.25) * 0.55)

            if cal_d in (30, 90):
                q10_m = XGBRegressor(
                    n_estimators=45,
                    max_depth=3,
                    learning_rate=0.07,
                    objective="reg:quantileerror",
                    quantile_alpha=0.10,
                    random_state=100 + cal_d,
                    n_jobs=2,
                )
                q90_m = XGBRegressor(
                    n_estimators=45,
                    max_depth=3,
                    learning_rate=0.07,
                    objective="reg:quantileerror",
                    quantile_alpha=0.90,
                    random_state=200 + cal_d,
                    n_jobs=2,
                )
                q10_m.fit(X_tr, y_tr)
                q90_m.fit(X_tr, y_tr)
                self.q10_models[cal_d] = q10_m
                self.q90_models[cal_d] = q90_m

        return self

    def _predict_anchor_log_returns(self, x_vec: np.ndarray) -> dict[int, tuple[float, float, float]]:
        x_2d = np.asarray(x_vec, dtype=np.float32).reshape(1, -1)
        out: dict[int, tuple[float, float, float]] = {0: (0.0, 0.0, 0.0)}
        for cal_d in ANCHOR_CALENDAR_DAYS:
            med = float(self.median_models[cal_d].predict(x_2d)[0])
            sigma = self.residual_std[cal_d]
            if cal_d in self.q10_models:
                q10_raw = float(self.q10_models[cal_d].predict(x_2d)[0])
                q90_raw = float(self.q90_models[cal_d].predict(x_2d)[0])
                low_spread = max(abs(med - q10_raw), 1.2816 * sigma * 0.75)
                high_spread = max(abs(q90_raw - med), 1.2816 * sigma * 0.75)
            else:
                low_spread = 1.2816 * sigma
                high_spread = 1.2816 * sigma
            out[cal_d] = (med - low_spread, med, med + high_spread)
        return out

    def predict_step(
        self,
        feature_row: pd.Series | np.ndarray,
        current_price: float,
        days_ahead: int,
        anchor_date: pd.Timestamp | None = None,
    ) -> dict:
        """Predict exact calendar day (1 <= days_ahead <= 90) using interpolated XGBoost quantile matrix."""
        d = int(np.clip(days_ahead, 1, 90))
        x_vec = (
            feature_row[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            if isinstance(feature_row, pd.Series)
            else np.asarray(feature_row, dtype=np.float32)
        )
        anchors = self._predict_anchor_log_returns(x_vec)
        xs = np.array(sorted(anchors.keys()), dtype=float)
        q10_arr = np.array([anchors[int(k)][0] for k in xs])
        q50_arr = np.array([anchors[int(k)][1] for k in xs])
        q90_arr = np.array([anchors[int(k)][2] for k in xs])

        r10 = float(np.interp(d, xs, q10_arr))
        r50 = float(np.interp(d, xs, q50_arr))
        r90 = float(np.interp(d, xs, q90_arr))

        base_date = pd.Timestamp("2026-11-18") if anchor_date is None else pd.Timestamp(anchor_date)
        target_ts = base_date + pd.Timedelta(days=d)
        cat = compute_calendar_catalysts(target_ts)

        doy = float(target_ts.day_of_year)
        seasonal_nudge = 0.0035 * np.sin(2.0 * np.pi * (doy - 245.0) / 365.25) * (d / 30.0)
        r50_adj = r50 + seasonal_nudge + (cat["wedding_season_multiplier"] - 1.0) * 0.35

        half_80_low = max(abs(r50 - r10), 0.008 * np.sqrt(d / 7.0)) * cat["volatility_cone_multiplier"]
        half_80_high = max(abs(r90 - r50), 0.008 * np.sqrt(d / 7.0)) * cat["volatility_cone_multiplier"]
        half_95_low = half_80_low * 1.53
        half_95_high = half_80_high * 1.53

        p50 = current_price * np.exp(r50_adj)
        p10 = current_price * np.exp(r50_adj - half_80_low)
        p90 = current_price * np.exp(r50_adj + half_80_high)
        p025 = current_price * np.exp(r50_adj - half_95_low)
        p975 = current_price * np.exp(r50_adj + half_95_high)

        q025, q10, q50, q90, q975 = _enforce_quantile_order(p025, p10, p50, p90, p975)
        return {
            "date": target_ts.strftime("%Y-%m-%d"),
            "days_ahead": d,
            "q025": round(q025, 2),
            "q10": round(q10, 2),
            "q50": round(q50, 2),
            "q90": round(q90, 2),
            "q975": round(q975, 2),
            "catalysts": cat,
        }

    def predict_trajectory(
        self,
        feature_row: pd.Series | np.ndarray,
        current_price: float,
        max_days: int = 90,
        anchor_date: pd.Timestamp | None = None,
    ) -> list[dict]:
        step = 1 if max_days <= 35 else 2
        days_list = list(range(1, max_days + 1, step))
        if days_list[-1] != max_days:
            days_list.append(max_days)
        return [
            self.predict_step(feature_row, current_price, d, anchor_date=anchor_date)
            for d in days_list
        ]
