"""Core forecasting model for GTA VI PS5 Exophase trophy earn rates."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Student-t (df=8) quantiles scaled so IQR matches normal (0.6745 / 0.7064 = 0.9548)
T8_SCALED = [-1.7754, -0.6745, 0.0, 0.6745, 1.7754]
NORMAL_Z = [-1.64485, -0.67449, 0.0, 0.67449, 1.64485]

LAUNCH_ISO = "2026-11-19T00:00:00Z"
LOG_OFFSET_HOURS = 6.0


def parse_iso_utc(ts: str) -> float:
    """Parse an ISO-8601 UTC timestamp string into Unix epoch seconds."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc).timestamp()


def clamp_pct(x: float) -> float:
    """Clamp percentage rate into the valid logit scoring range [0.01, 99.99]."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    """Compute natural log-odds of a percentage in [0.01, 99.99]."""
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg: float) -> float:
    """Convert natural log-odds back to percentage in [0.0, 100.0]."""
    if lg >= 30.0:
        return 100.0
    if lg <= -30.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float) -> float | int:
    """Round to 4 decimal places half-up, returning int when exact integer."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def pinball_loss(actual_pct: float, quantiles: list[float]) -> float:
    """Compute mean logit-scale pinball loss for one trophy across LEVELS."""
    y_lg = logit(actual_pct)
    total = 0.0
    for tau, q in zip(LEVELS, quantiles):
        d = y_lg - logit(q)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(LEVELS)


def extract_refresh_snapshots(
    history: list[dict[str, Any]],
    t0_sec: float,
) -> list[dict[str, Any]]:
    """Collapse consecutive unchanged Exophase reads into distinct refresh snapshots."""
    snapshots: list[dict[str, Any]] = []
    prev_sig: tuple[Any, ...] | None = None
    prev_last_sec: float | None = None

    for row in history:
        ts_sec = parse_iso_utc(row.get("slot") or row["at"])
        rates = row["rates"]
        players = int(row["players_tracked"])
        earned = int(row["trophies_earned"])
        sig = (players, earned, tuple(rates))
        if sig != prev_sig:
            if snapshots and prev_last_sec is not None:
                # Refresh happened between prev_last_sec and ts_sec; midpoint approximates refresh time.
                eff_sec = 0.5 * (prev_last_sec + ts_sec)
            else:
                eff_sec = ts_sec
            snapshots.append(
                {
                    "first_sec": ts_sec,
                    "last_sec": ts_sec,
                    "eff_sec": eff_sec,
                    "t_hours": max(0.0, (eff_sec - t0_sec) / 3600.0),
                    "players": players,
                    "earned": earned,
                    "rates": list(rates),
                }
            )
            prev_sig = sig
        else:
            snapshots[-1]["last_sec"] = ts_sec
        prev_last_sec = ts_sec

    return snapshots


def time_basis(t_hours: float) -> float:
    """Concave time basis capturing launch-week progression deceleration."""
    return math.log(LOG_OFFSET_HOURS + max(0.0, t_hours))


def weighted_linear_fit(
    xs: list[float],
    ys: list[float],
    ws: list[float],
    prior_slope: float = 0.0,
    slope_prior_weight: float = 0.0,
) -> tuple[float, float, float, float]:
    """Fit y = a + b * (x - x_last) with ridge shrinkage on b toward prior_slope.

    Returns (intercept_at_last, slope, resid_rms, slope_se).
    """
    n = len(xs)
    if n == 0:
        return 0.0, prior_slope, 0.3, 0.5
    if n == 1:
        return ys[0], prior_slope, 0.3, 0.5

    x_last = xs[-1]
    dx = [x - x_last for x in xs]
    w_sum = sum(ws)
    if w_sum <= 0.0:
        return ys[-1], prior_slope, 0.3, 0.5

    x_bar = sum(w * d for w, d in zip(ws, dx)) / w_sum
    y_bar = sum(w * y for w, y in zip(ws, ys)) / w_sum

    sxx = sum(w * (d - x_bar) ** 2 for w, d in zip(ws, dx))
    sxy = sum(w * (d - x_bar) * (y - y_bar) for w, d, y in zip(ws, dx, ys))

    denom = sxx + slope_prior_weight
    if denom <= 1e-9:
        b = prior_slope
    else:
        b = (sxy + slope_prior_weight * prior_slope) / denom
    a = y_bar - b * x_bar

    resid_sq = sum(w * (y - (a + b * d)) ** 2 for w, d, y in zip(ws, dx, ys))
    dof = max(1.0, w_sum * max(1, n - 2) / n)
    resid_var = resid_sq / dof
    resid_rms = math.sqrt(max(0.0, resid_var))
    slope_se = math.sqrt(max(0.0, resid_var / max(denom, 1e-6)))
    return a, b, resid_rms, slope_se


def _pre_refresh_forecast(
    trophies: list[dict[str, Any]],
    snap: dict[str, Any],
    horizon_days: float,
    t_at_hours: float,
) -> list[dict[str, Any]]:
    """Forecast when only a single flat snapshot exists (no Exophase refresh observed yet).

    On launch night with a small early cohort (players < 2500), an imminent refresh
    usually dilutes rare/mid trophies substantially while near-100% starter trophies
    remain high. Spread must honestly cover both refresh and no-refresh outcomes.
    """
    players = snap["players"]
    early_cohort = players < 2500 and t_at_hours < 18.0
    rw_scale = 0.5 * math.sqrt(max(horizon_days, 1e-4))
    rows: list[dict[str, Any]] = []

    for i, trophy in enumerate(trophies):
        rate = snap["rates"][i]
        r = float(rate) if rate is not None else 0.01
        lg = logit(r)

        if early_cohort:
            # Probability/magnitude of dilution scales down smoothly as r approaches 100%.
            p_r = clamp_pct(r) / 100.0
            if p_r >= 0.92:
                shift_med = -0.15
                s_low = max(0.35, rw_scale * 1.4)
                s_high = max(0.25, rw_scale * 1.1)
            else:
                # Rare trophies (e.g. 0.6%-2%) typically drop ~4x-6x on first launch refresh,
                # with a bimodal risk of staying unchanged if no refresh happens by target.
                dilution_factor = 1.0 - p_r
                expected_log_mult = -1.25 * dilution_factor
                shift_med = expected_log_mult * 0.72
                s_low = max(0.65, 0.75 * dilution_factor + rw_scale)
                s_high = max(0.45, 0.55 * dilution_factor + rw_scale)

            qs_lg = [
                lg + shift_med + T8_SCALED[0] * s_low * 0.65,
                lg + shift_med + T8_SCALED[1] * s_low * 0.65,
                lg + shift_med,
                lg + shift_med + T8_SCALED[3] * s_high * 0.85,
                lg + max(0.25, shift_med + T8_SCALED[4] * s_high * 0.85),
            ]
        else:
            spread = max(0.25, rw_scale)
            qs_lg = [lg + z * spread for z in NORMAL_Z]

        qs = [round4(inv_logit_pct(val)) for val in qs_lg]
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return rows


def _extract_chapter_order(trophies: list[dict[str, Any]]) -> list[int]:
    """Identify sequential story chapter trophy indices [Ch1, Ch2, ..., Final, 100%]."""
    chapters: list[tuple[int, int]] = []
    final_idx: int | None = None
    pct100_idx: int | None = None
    for idx, t in enumerate(trophies):
        desc = (t.get("description") or "").lower()
        m = re.search(r"complete chapter\s+(\d+)", desc)
        if m:
            chapters.append((int(m.group(1)), idx))
        elif "complete the final mission" in desc:
            final_idx = idx
        elif "100% completion" in desc:
            pct100_idx = idx
    chapters.sort()
    chain = [idx for _, idx in chapters]
    if final_idx is not None:
        chain.append(final_idx)
    if pct100_idx is not None:
        chain.append(pct100_idx)
    return chain


def _fit_trophy_predictions(
    trophies: list[dict[str, Any]],
    used_snaps: list[dict[str, Any]],
    t_target_hours: float,
    staleness_days: float,
    horizon_days: float,
) -> list[tuple[float, float]]:
    """Estimate (median_logit, raw_sigma) for each trophy at t_target_hours."""
    xs = [time_basis(s["t_hours"]) for s in used_snaps]
    x_last = xs[-1]
    dx_target = max(0.0, time_basis(t_target_hours) - x_last)
    t_last_hours = used_snaps[-1]["t_hours"]
    n_last = used_snaps[-1]["players"]

    decay_hours = max(12.0, 0.45 * max(1.0, t_last_hours - used_snaps[0]["t_hours"]))
    base_ws = [
        math.exp(-(t_last_hours - s["t_hours"]) / decay_hours) * math.sqrt(max(1.0, s["players"] / 1000.0))
        for s in used_snaps
    ]

    raw_slopes: list[float] = []
    tier_slopes: dict[str, list[float]] = {}
    for i, trophy in enumerate(trophies):
        xs_i: list[float] = []
        ys_i: list[float] = []
        ws_i: list[float] = []
        for s, x, bw in zip(used_snaps, xs, base_ws):
            r = s["rates"][i]
            if r is None:
                continue
            k_est = (float(r) / 100.0) * s["players"]
            r_smooth = 100.0 * (k_est + 0.5) / (s["players"] + 1.0)
            xs_i.append(x)
            ys_i.append(logit(r_smooth))
            ws_i.append(bw * max(0.15, k_est / (k_est + 2.0)))

        if len(xs_i) >= 2:
            _, b_raw, _, _ = weighted_linear_fit(xs_i, ys_i, ws_i, prior_slope=0.0, slope_prior_weight=0.15)
            b_clamped = max(-1.5, min(3.5, b_raw))
        else:
            b_clamped = 0.0
        raw_slopes.append(b_clamped)
        tier_slopes.setdefault(trophy.get("tier", "bronze"), []).append(b_clamped)

    tier_prior: dict[str, float] = {}
    all_med = sorted(raw_slopes)[len(raw_slopes) // 2] if raw_slopes else 0.0
    for tier, vals in tier_slopes.items():
        svals = sorted(vals)
        tier_prior[tier] = 0.7 * svals[len(svals) // 2] + 0.3 * all_med

    eff_horizon_days = max(1e-4, horizon_days + 0.5 * staleness_days)
    rw_sigma = 0.34 * math.sqrt(eff_horizon_days)

    results: list[tuple[float, float]] = []
    for i, trophy in enumerate(trophies):
        xs_i = []
        ys_i = []
        ws_i = []
        last_raw_r = used_snaps[-1]["rates"][i]
        r_now = float(last_raw_r) if last_raw_r is not None else 0.01

        for s, x, bw in zip(used_snaps, xs, base_ws):
            r = s["rates"][i]
            if r is None:
                continue
            k_est = (float(r) / 100.0) * s["players"]
            r_smooth = 100.0 * (k_est + 0.35) / (s["players"] + 0.7)
            xs_i.append(x)
            ys_i.append(logit(r_smooth))
            ws_i.append(bw * max(0.15, k_est / (k_est + 1.5)))

        prior_b = tier_prior.get(trophy.get("tier", "bronze"), 0.0)
        if r_now >= 92.0 and prior_b > 0.0:
            prior_b *= max(0.0, (100.0 - r_now) / 8.0)

        k_now = (r_now / 100.0) * n_last
        prior_wt = 0.25 + 1.5 / (1.0 + k_now / 8.0)
        a_fit, b_fit, resid_rms, slope_se = weighted_linear_fit(
            xs_i, ys_i, ws_i, prior_slope=prior_b, slope_prior_weight=prior_wt
        )

        lg_now = logit(r_now)
        lg_base = 0.65 * lg_now + 0.35 * a_fit
        # Negative slopes are mostly transient launch dilution and decay faster than positive growth.
        damping = 0.90 if b_fit >= 0.0 else 0.55
        lg_med = lg_base + b_fit * dx_target * damping

        slope_unc = min(0.75, slope_se * dx_target)
        count_unc = 0.28 / math.sqrt(max(1.0, k_now + 1.0))
        raw_sigma = math.sqrt(rw_sigma ** 2 + slope_unc ** 2 + (0.65 * resid_rms) ** 2 + count_unc ** 2)
        raw_sigma = max(0.12, min(1.8, raw_sigma))
        results.append((lg_med, raw_sigma))

    return results


def _calibrate_spread(
    trophies: list[dict[str, Any]],
    used_snaps: list[dict[str, Any]],
) -> float:
    """Use walk-forward splits inside history to calibrate spread scale toward empirical errors."""
    if len(used_snaps) < 6:
        return 1.0
    span_hours = used_snaps[-1]["t_hours"] - used_snaps[0]["t_hours"]
    if span_hours < 18.0:
        return 1.0

    t_end = used_snaps[-1]["t_hours"]
    target_rates = used_snaps[-1]["rates"]
    ratios: list[float] = []

    for frac in (0.55, 0.72):
        cutoff_h = used_snaps[0]["t_hours"] + frac * span_hours
        prefix = [s for s in used_snaps if s["t_hours"] <= cutoff_h]
        if len(prefix) < 3 or len(prefix) >= len(used_snaps):
            continue
        cv_horizon_days = max(1e-3, (t_end - prefix[-1]["t_hours"]) / 24.0)
        preds = _fit_trophy_predictions(trophies, prefix, t_end, 0.0, cv_horizon_days)
        for i, (lg_med, raw_sig) in enumerate(preds):
            y_r = target_rates[i]
            if y_r is None:
                continue
            ratios.append(abs(logit(float(y_r)) - lg_med) / max(0.05, raw_sig))

    if len(ratios) < 20:
        return 1.0
    # Under standard normal errors, E[|Z|] = sqrt(2 / pi) ~= 0.7979.
    emp_mult = (sum(ratios) / len(ratios)) / 0.7979
    # Shrink empirical multiplier toward 1.0 to avoid under-coverage on longer final horizon.
    blended = 0.55 * emp_mult + 0.45 * 1.0
    return max(0.60, min(1.50, blended))


def forecast_case(case: dict[str, Any], name: str = "case") -> dict[str, Any]:
    """Generate calibrated 5-quantile forecasts for all trophies in a case."""
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    release_date = case.get("game", {}).get("release_date", "2026-11-19")
    t0_sec = parse_iso_utc(f"{release_date}T00:00:00Z")
    t_at_sec = parse_iso_utc(case["at"])
    t_target_sec = parse_iso_utc(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    t_at_hours = max(0.0, (t_at_sec - t0_sec) / 3600.0)
    t_target_hours = max(t_at_hours, (t_target_sec - t0_sec) / 3600.0)

    snapshots = extract_refresh_snapshots(case["history"], t0_sec)
    trophies = case["trophies"]

    if len(snapshots) <= 1:
        return {"trophies": _pre_refresh_forecast(trophies, snapshots[0], horizon_days, t_at_hours)}

    n_last = snapshots[-1]["players"]
    post_transient = [
        s for s in snapshots if s["players"] >= max(1500, int(0.12 * n_last)) and s["t_hours"] >= 6.0
    ]
    if len(post_transient) >= 3:
        used_snaps = post_transient
    elif len(snapshots) >= 3 and snapshots[0]["players"] < 0.25 * snapshots[1]["players"]:
        used_snaps = snapshots[1:]
    else:
        used_snaps = snapshots

    staleness_days = max(0.0, (t_at_sec - used_snaps[-1]["last_sec"]) / 86400.0)
    preds = _fit_trophy_predictions(trophies, used_snaps, t_target_hours, staleness_days, horizon_days)
    spread_mult = _calibrate_spread(trophies, used_snaps)

    pred_qs_lg: list[list[float]] = []
    for lg_med, raw_sigma in preds:
        scale = raw_sigma * spread_mult
        pred_qs_lg.append([lg_med + q_t * scale for q_t in T8_SCALED])

    # Structural PlayStation rule: Platinum requires every other trophy -> rate(platinum) <= min(others).
    plat_indices = [i for i, t in enumerate(trophies) if t.get("tier") == "platinum"]
    if plat_indices and len(trophies) > 1:
        for p_idx in plat_indices:
            other_indices = [j for j in range(len(trophies)) if j != p_idx]
            for q_k in range(len(LEVELS)):
                min_other = min(pred_qs_lg[j][q_k] for j in other_indices)
                if pred_qs_lg[p_idx][q_k] > min_other:
                    pred_qs_lg[p_idx][q_k] = min_other

    # Soft monotonicity along sequential story chapters (Ch1 >= Ch2 >= Ch3 >= Ch4 >= Final >= 100%).
    chain = _extract_chapter_order(trophies)
    for pos in range(1, len(chain)):
        prev_idx = chain[pos - 1]
        curr_idx = chain[pos]
        if pred_qs_lg[curr_idx][2] > pred_qs_lg[prev_idx][2]:
            mid = 0.5 * (pred_qs_lg[curr_idx][2] + pred_qs_lg[prev_idx][2])
            shift = mid - pred_qs_lg[curr_idx][2]
            pred_qs_lg[curr_idx] = [v + shift for v in pred_qs_lg[curr_idx]]

    rows: list[dict[str, Any]] = []
    for i, trophy in enumerate(trophies):
        qs = [round4(inv_logit_pct(v)) for v in pred_qs_lg[i]]
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}
