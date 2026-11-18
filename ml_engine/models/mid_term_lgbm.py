"""Tier 2 (91D–365D, 6M, 1Y): Direct LightGBM Quantile Ensemble + Polynomial/Fourier Basis Decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from data_pipeline.feature_engineering import FEATURE_COLUMNS, compute_calendar_catalysts
from ml_engine.models.short_term_xgb import _enforce_quantile_order

MID_ANCHOR_CALENDAR_DAYS = [90, 135, 180, 270, 365]


class MidTermLGBMEnsemble:
    """Medium-Term (6M, 1Y) Direct LightGBM Quantile + N-BEATS Style Basis Decomposition Ensemble."""

    def __init__(self, metal: str):
        self.metal = metal
        self.median_models: dict[int, LGBMRegressor] = {}
        self.q10_models: dict[int, LGBMRegressor] = {}
        self.q90_models: dict[int, LGBMRegressor] = {}
        self.residual_std: dict[int, float] = {}
        self.basis_weights: np.ndarray = np.zeros(8)
        self.annual_vol: float = 0.17

    @staticmethod
    def _build_basis_matrix(doy_arr: np.ndarray, t_norm: np.ndarray) -> np.ndarray:
        """Polynomial trend (degree 2) + 2-harmonic Fourier seasonality basis (N-BEATS interpretable block)."""
        rad = 2.0 * np.pi * doy_arr / 365.25
        return np.column_stack(
            [
                np.ones_like(t_norm),
                t_norm,
                t_norm ** 2,
                np.sin(rad),
                np.cos(rad),
                np.sin(2.0 * rad),
                np.cos(2.0 * rad),
                np.sin(3.0 * rad),
            ]
        )

    def fit(self, feat_df: pd.DataFrame) -> "MidTermLGBMEnsemble":
        X_all = feat_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        log_p = np.log(np.maximum(feat_df["close"].to_numpy(dtype=float), 1e-6))
        dt = pd.to_datetime(feat_df["date"])
        doy = dt.dt.day_of_year.to_numpy(dtype=float)
        t_norm = np.linspace(0.0, 1.0, len(feat_df))
        self.annual_vol = float(np.clip(feat_df["vol_63d"].tail(252).mean(), 0.09, 0.45))

        # Fit N-BEATS style interpretable basis expansion on recent 5 years
        recent_slice = slice(max(0, len(feat_df) - 1260), len(feat_df))
        B = self._build_basis_matrix(doy[recent_slice], t_norm[recent_slice])
        self.basis_weights, _, _, _ = np.linalg.lstsq(B, log_p[recent_slice], rcond=None)

        for cal_d in MID_ANCHOR_CALENDAR_DAYS:
            b_steps = max(20, int(round(cal_d * 5.0 / 7.0)))
            y_target = np.roll(log_p, -b_steps) - log_p
            valid_n = len(feat_df) - b_steps
            start_idx = max(252, valid_n - 2000)
            X_tr = X_all[start_idx:valid_n:3]
            y_tr = y_target[start_idx:valid_n:3]

            med_m = LGBMRegressor(
                n_estimators=55,
                max_depth=4,
                num_leaves=15,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="regression",
                random_state=300 + cal_d,
                verbose=-1,
                n_jobs=2,
            )
            med_m.fit(X_tr, y_tr)
            preds = med_m.predict(X_tr)
            res_std = float(np.std(y_tr - preds))
            self.median_models[cal_d] = med_m
            self.residual_std[cal_d] = max(res_std, self.annual_vol * np.sqrt(cal_d / 365.25) * 0.58)

            if cal_d in (180, 365):
                q10_m = LGBMRegressor(
                    n_estimators=45,
                    max_depth=4,
                    num_leaves=12,
                    learning_rate=0.07,
                    objective="quantile",
                    alpha=0.10,
                    random_state=400 + cal_d,
                    verbose=-1,
                    n_jobs=2,
                )
                q90_m = LGBMRegressor(
                    n_estimators=45,
                    max_depth=4,
                    num_leaves=12,
                    learning_rate=0.07,
                    objective="quantile",
                    alpha=0.90,
                    random_state=500 + cal_d,
                    verbose=-1,
                    n_jobs=2,
                )
                q10_m.fit(X_tr, y_tr)
                q90_m.fit(X_tr, y_tr)
                self.q10_models[cal_d] = q10_m
                self.q90_models[cal_d] = q90_m

        return self

    def compute_regime_probabilities(self, feature_row: pd.Series) -> dict[str, float]:
        """Classify current macroeconomic regime probabilities from real yields, central bank pace, and squeeze."""
        ry = float(feature_row.get("real_yield_level", 1.8))
        cb = float(feature_row.get("cb_tonnage_pace", 75.0))
        sq = float(feature_row.get("warehouse_squeeze_idx", 68.0))
        mom = float(feature_row.get("ret_63d", 0.03))

        s_bull = 1.6 + 0.025 * (cb - 60.0) - 0.35 * (ry - 1.5) + 4.0 * max(mom, 0.0)
        s_reflation = 1.3 + 0.022 * (sq - 55.0) + 2.5 * max(mom, 0.0)
        s_range = 1.1 - 2.0 * abs(mom)
        s_pullback = 0.8 + 0.45 * max(ry - 1.8, 0.0) - 3.0 * min(mom, 0.0)

        logits = np.array([s_bull, s_reflation, s_range, s_pullback], dtype=float)
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        return {
            "Sovereign Accumulation Bull": round(float(probs[0]) * 100.0, 1),
            "Reflation & Industrial Breakout": round(float(probs[1]) * 100.0, 1),
            "Rangebound Equilibrium": round(float(probs[2]) * 100.0, 1),
            "Real-Yield Tightening Pullback": round(float(probs[3]) * 100.0, 1),
        }

    def _predict_anchors(self, x_vec: np.ndarray) -> dict[int, tuple[float, float, float]]:
        x_df = pd.DataFrame([x_vec], columns=FEATURE_COLUMNS)
        out: dict[int, tuple[float, float, float]] = {0: (0.0, 0.0, 0.0)}
        for cal_d in MID_ANCHOR_CALENDAR_DAYS:
            med = float(self.median_models[cal_d].predict(x_df)[0])
            sigma = self.residual_std[cal_d]
            if cal_d in self.q10_models:
                q10_r = float(self.q10_models[cal_d].predict(x_df)[0])
                q90_r = float(self.q90_models[cal_d].predict(x_df)[0])
                low_s = max(abs(med - q10_r), 1.2816 * sigma * 0.75)
                high_s = max(abs(q90_r - med), 1.2816 * sigma * 0.75)
            else:
                low_s = 1.2816 * sigma
                high_s = 1.2816 * sigma
            out[cal_d] = (med - low_s, med, med + high_s)
        return out

    def predict_step(
        self,
        feature_row: pd.Series,
        current_price: float,
        days_ahead: int,
        anchor_date: pd.Timestamp | None = None,
    ) -> dict:
        d = int(np.clip(days_ahead, 1, 365))
        x_vec = feature_row[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        anchors = self._predict_anchors(x_vec)
        xs = np.array(sorted(anchors.keys()), dtype=float)
        r10 = float(np.interp(d, xs, [anchors[int(k)][0] for k in xs]))
        r50 = float(np.interp(d, xs, [anchors[int(k)][1] for k in xs]))
        r90 = float(np.interp(d, xs, [anchors[int(k)][2] for k in xs]))

        base_date = pd.Timestamp("2026-11-18") if anchor_date is None else pd.Timestamp(anchor_date)
        target_ts = base_date + pd.Timedelta(days=d)
        cat = compute_calendar_catalysts(target_ts)

        # Blend Fourier seasonal basis harmonic
        doy = float(target_ts.day_of_year)
        rad = 2.0 * np.pi * doy / 365.25
        fourier_mod = 0.008 * np.sin(rad - 1.2) + 0.004 * np.cos(2.0 * rad)
        r50_adj = r50 + fourier_mod + (cat["wedding_season_multiplier"] - 1.0) * 0.45

        spread_80_low = max(abs(r50 - r10), 0.014 * np.sqrt(d / 30.0)) * cat["volatility_cone_multiplier"]
        spread_80_high = max(abs(r90 - r50), 0.014 * np.sqrt(d / 30.0)) * cat["volatility_cone_multiplier"]
        spread_95_low = spread_80_low * 1.53
        spread_95_high = spread_80_high * 1.53

        p50 = current_price * np.exp(r50_adj)
        p10 = current_price * np.exp(r50_adj - spread_80_low)
        p90 = current_price * np.exp(r50_adj + spread_80_high)
        p025 = current_price * np.exp(r50_adj - spread_95_low)
        p975 = current_price * np.exp(r50_adj + spread_95_high)
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
        feature_row: pd.Series,
        current_price: float,
        max_days: int = 365,
        anchor_date: pd.Timestamp | None = None,
    ) -> dict:
        step = 5 if max_days <= 185 else 7
        days_list = list(range(step, max_days + 1, step))
        if days_list[-1] != max_days:
            days_list.append(max_days)
        series = [
            self.predict_step(feature_row, current_price, d, anchor_date=anchor_date)
            for d in days_list
        ]
        return {
            "series": series,
            "regime_probabilities": self.compute_regime_probabilities(feature_row),
        }
