"""Fast, calibrated dual-engine forecaster for GTA VI PS5 Exophase trophy earn rates."""

import json
import math
import os

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Student-t (df=8) quantiles scaled so IQR matches standard normal (0.6745 / 0.7064).
T8_SCALED = [-1.7754, -0.6745, 0.0, 0.6745, 1.7754]
NORMAL_Z = [-1.64485, -0.67449, 0.0, 0.67449, 1.64485]


def _iso_to_sec(ts):
    """Parse ISO-8601 UTC string 'YYYY-MM-DDTHH:MM:SS(.fff)Z' to Unix seconds without datetime."""
    y = int(ts[0:4])
    m = int(ts[5:7])
    d = int(ts[8:10])
    hh = int(ts[11:13])
    mm = int(ts[14:16])
    ss = int(ts[17:19])
    # Howard Hinnant civil-to-days algorithm relative to 1970-01-01.
    if m <= 2:
        y -= 1
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (m + (9 if m <= 2 else -3)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return days * 86400.0 + hh * 3600.0 + mm * 60.0 + ss


def _clamp_pct(x):
    if x < 0.01:
        return 0.01
    if x > 99.99:
        return 99.99
    return x


def _logit(pct):
    p = _clamp_pct(pct) * 0.01
    return math.log(p / (1.0 - p))


def _inv_logit_pct(lg):
    if lg >= 30.0:
        return 100.0
    if lg <= -30.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-lg))


def _round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    val = whole / 10000.0
    return int(val) if val.is_integer() else val


def _extract_snapshots(history, t0_sec):
    """Collapse unchanged periodic Exophase reads into unique refresh snapshots."""
    snaps = []
    prev_sig = None
    prev_last = None
    for row in history:
        ts = _iso_to_sec(row.get("slot") or row["at"])
        players = int(row["players_tracked"])
        earned = int(row["trophies_earned"])
        rates = row["rates"]
        sig = (players, earned, tuple(rates))
        if sig != prev_sig:
            eff = 0.5 * (prev_last + ts) if (snaps and prev_last is not None) else ts
            t_hr = (eff - t0_sec) / 3600.0
            if t_hr < 0.0:
                t_hr = 0.0
            snaps.append((ts, ts, eff, t_hr, players, earned, rates))
            prev_sig = sig
        else:
            last = snaps[-1]
            snaps[-1] = (last[0], ts, last[2], last[3], last[4], last[5], last[6])
        prev_last = ts
    return snaps


def _ridge_fit(xs, ys, ws, prior_b=0.0, prior_w=0.0):
    """Weighted linear regression y = a + b*(x - x_last) with ridge shrinkage toward prior_b."""
    n = len(xs)
    if n == 0:
        return 0.0, prior_b, 0.25, 0.4
    if n == 1:
        return ys[0], prior_b, 0.25, 0.4

    x_last = xs[-1]
    w_sum = 0.0
    wx = 0.0
    wy = 0.0
    for i in range(n):
        w = ws[i]
        d = xs[i] - x_last
        w_sum += w
        wx += w * d
        wy += w * ys[i]
    if w_sum <= 1e-12:
        return ys[-1], prior_b, 0.25, 0.4

    x_bar = wx / w_sum
    y_bar = wy / w_sum
    sxx = 0.0
    sxy = 0.0
    for i in range(n):
        w = ws[i]
        dx = (xs[i] - x_last) - x_bar
        sxx += w * dx * dx
        sxy += w * dx * (ys[i] - y_bar)

    denom = sxx + prior_w
    b = (sxy + prior_w * prior_b) / denom if denom > 1e-9 else prior_b
    a = y_bar - b * x_bar

    res_sq = 0.0
    for i in range(n):
        err = ys[i] - (a + b * (xs[i] - x_last))
        res_sq += ws[i] * err * err
    dof = max(1.0, w_sum * max(1, n - 2) / n)
    res_var = res_sq / dof
    resid_rms = math.sqrt(res_var) if res_var > 0.0 else 0.0
    slope_se = math.sqrt(res_var / max(denom, 1e-6)) if res_var > 0.0 else 0.0
    return a, b, resid_rms, slope_se


def _chapter_chain(trophies):
    """Identify sequential story chapter indices [Ch1, Ch2, ..., Final, 100%]."""
    chapters = []
    final_idx = None
    pct100_idx = None
    for idx, t in enumerate(trophies):
        desc = (t.get("description") or "").lower()
        pos = desc.find("complete chapter ")
        if pos != -1:
            rest = desc[pos + 17 :].strip()
            digits = ""
            for ch in rest:
                if "0" <= ch <= "9":
                    digits += ch
                else:
                    break
            if digits:
                chapters.append((int(digits), idx))
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


def _pre_refresh_forecast(trophies, snap, horizon_days, t_at_hours):
    """Count-conservation launch-night dilution forecast when no refresh has occurred yet."""
    players = snap[4]
    rates = snap[6]
    early_cohort = players < 2500 and t_at_hours < 18.0
    rw_scale = 0.45 * math.sqrt(max(horizon_days, 1e-4))
    rows = []

    # Expected first-refresh player wave on launch night (~6x from pre-launch cohort).
    n_pred = players * 6.0
    for i, trophy in enumerate(trophies):
        raw_r = rates[i]
        r = float(raw_r) if raw_r is not None else 0.01
        lg = _logit(r)

        if early_cohort:
            if r >= 98.0:
                # Intro/prologue trophy earned by virtually every synced player.
                qs_lg = [lg - 0.55, lg - 0.15, lg, lg, lg]
            else:
                # Exact implied pre-launch earner count diluted by launch-night player influx.
                k0 = max(0.5, round(r * 0.01 * players))
                tier = trophy.get("tier", "bronze")
                is_secret = bool(trophy.get("secret", False))
                # Slight new-earner increment during launch window (higher for accessible bronze).
                dk = 0.15 if tier == "platinum" else (0.35 if tier in ("gold", "silver") else (0.7 if is_secret else 1.1))
                r_med = _clamp_pct(100.0 * (k0 + dk) / n_pred)
                lg_med = _logit(r_med)
                s_low = 0.25 + rw_scale
                s_high = 0.38 + rw_scale
                qs_lg = [
                    lg_med + T8_SCALED[0] * s_low,
                    lg_med + T8_SCALED[1] * s_low,
                    lg_med,
                    lg_med + T8_SCALED[3] * s_high,
                    lg_med + T8_SCALED[4] * s_high,
                ]
        else:
            sp = max(0.22, rw_scale)
            qs_lg = [lg + z * sp for z in NORMAL_Z]

        qs = [_round4(_inv_logit_pct(v)) for v in qs_lg]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return rows


def _damped_horizon(dx, lam):
    """Integral of exp(-lam * u) from 0 to dx: exact Holt continuous-time damped horizon."""
    if dx <= 0.0:
        return 0.0
    if lam <= 1e-6:
        return dx
    return (1.0 - math.exp(-lam * dx)) / lam


def _fit_snapshot_stats(trophies, snaps):
    """Precompute per-trophy Empirical-Bayes regression stats once for a snapshot sequence."""
    m = len(snaps)
    n_trophies = len(trophies)
    t_last = snaps[-1][3]
    n_last = snaps[-1][4]
    ln_n_last = math.log(max(1.0, float(n_last)))

    xs = [math.log(1.0 + s[3]) for s in snaps]
    x_last = xs[-1]
    span_h = max(1.0, t_last - snaps[0][3])
    decay_h = max(9.0, 0.38 * span_h)

    base_ws = [
        math.exp(-(t_last - s[3]) / decay_h) * math.sqrt(max(1.0, s[4] * 0.001))
        for s in snaps
    ]

    ln_ns = [math.log(max(1.0, float(s[4]))) for s in snaps]
    _, b_n, _, _ = _ridge_fit(xs, ln_ns, base_ws, prior_b=0.5, prior_w=0.15)
    b_n = max(0.0, min(3.0, b_n))

    ln_avg_e = [math.log(max(0.05, s[5] / max(1.0, float(s[4])))) for s in snaps]
    a_me, b_me, _, _ = _ridge_fit(xs, ln_avg_e, base_ws, prior_b=0.2, prior_w=0.18)
    b_me = max(-0.4, min(2.5, b_me))
    me_base = 0.80 * ln_avg_e[-1] + 0.20 * a_me

    lg_nows = [0.0] * n_trophies
    r_nows = [0.01] * n_trophies
    k_nows = [0.5] * n_trophies
    raw_slopes = [0.0] * n_trophies
    tiers = [t.get("tier", "bronze") for t in trophies]
    secrets = [bool(t.get("secret", False)) for t in trophies]

    series_cache = []
    x_mid = 0.5 * (xs[0] + x_last)
    x_rec_cut = xs[0] + 0.42 * (x_last - xs[0])

    for i in range(n_trophies):
        last_r = snaps[-1][6][i]
        r_now = float(last_r) if last_r is not None else 0.01
        r_nows[i] = r_now
        lg_nows[i] = _logit(r_now)
        k_nows[i] = max(0.0, r_now * 0.01 * n_last)

        xs_i = []
        ys_i = []
        ws_i = []
        for idx_s in range(m):
            s = snaps[idx_s]
            r = s[6][i]
            if r is None:
                continue
            n_s = float(s[4])
            k_raw = float(r) * 0.01 * n_s
            r_smooth = 100.0 * (k_raw + 0.4) / (n_s + 0.8)
            xs_i.append(xs[idx_s])
            ys_i.append(_logit(r_smooth))
            cw = k_raw / (k_raw + 1.8)
            ws_i.append(base_ws[idx_s] * (0.18 if cw < 0.18 else cw))

        series_cache.append((xs_i, ys_i, ws_i))
        if len(xs_i) >= 2:
            _, b_raw, _, _ = _ridge_fit(xs_i, ys_i, ws_i, prior_b=0.0, prior_w=0.12)
            raw_slopes[i] = max(-1.5, min(4.0, b_raw))
        else:
            raw_slopes[i] = 0.0

    # Leave-one-out rarity-kernel Empirical Bayes slope prior.
    prior_slopes = [0.0] * n_trophies
    for i in range(n_trophies):
        lgi = lg_nows[i]
        ti = tiers[i]
        si = secrets[i]
        sw = 0.0
        sb = 0.0
        for j in range(n_trophies):
            if i == j:
                continue
            diff = (lgi - lg_nows[j]) * 0.72
            kw = math.exp(-diff * diff)
            if ti == tiers[j]:
                kw *= 1.5
            if si == secrets[j]:
                kw *= 1.2
            sw += kw
            sb += kw * raw_slopes[j]
        if sw > 1e-9:
            prior_slopes[i] = sb / sw

    trophy_stats = []
    for i in range(n_trophies):
        xs_i, ys_i, ws_i = series_cache[i]
        r_now = r_nows[i]
        lg_now = lg_nows[i]
        k_now = k_nows[i]

        p_b = prior_slopes[i]
        if r_now >= 92.0 and p_b > 0.0:
            p_b *= max(0.0, (100.0 - r_now) / 8.0)

        prior_wt = 0.15 + 1.05 / (1.0 + k_now * 0.1)
        a_fit, b_fit, resid_rms, slope_se = _ridge_fit(xs_i, ys_i, ws_i, prior_b=p_b, prior_w=prior_wt)

        b_rec = b_fit
        decel_adj = 0.0
        slope_shift = 0.0
        if len(xs_i) >= 5:
            rec_x, rec_y, rec_w = [], [], []
            ear_x, ear_y, ear_w = [], [], []
            for idx_p in range(len(xs_i)):
                xp = xs_i[idx_p]
                if xp >= x_rec_cut:
                    rec_x.append(xp)
                    rec_y.append(ys_i[idx_p])
                    rec_w.append(ws_i[idx_p])
                if xp <= x_mid:
                    ear_x.append(xp)
                    ear_y.append(ys_i[idx_p])
                    ear_w.append(ws_i[idx_p])
            if len(rec_x) >= 3:
                _, br, _, _ = _ridge_fit(rec_x, rec_y, rec_w, prior_b=b_fit, prior_w=0.22)
                b_rec = max(-1.5, min(4.2, br))
                slope_shift = abs(b_rec - b_fit)
                if len(ear_x) >= 2 and b_rec > 0.05:
                    _, b_ear, _, _ = _ridge_fit(ear_x, ear_y, ear_w, prior_b=b_fit, prior_w=0.25)
                    if b_ear > b_rec + 0.15:
                        decel_adj = min(0.45, 0.25 * (b_ear - b_rec) / max(0.3, b_ear))
                    elif b_rec > b_ear + 0.25:
                        decel_adj = -0.12

        lg_base = 0.84 * lg_now + 0.16 * a_fit
        trophy_stats.append((r_now, k_now, lg_base, b_fit, b_rec, decel_adj, slope_shift, resid_rms, slope_se))

    return x_last, ln_n_last, b_n, me_base, b_me, trophy_stats


def _project_from_stats(stats, t_target_hours, staleness_days, horizon_days, lam_global, rec_blend):
    """Project (meds, sigmas) at target horizon from precomputed snapshot regression stats."""
    x_last, ln_n_last, b_n, me_base, b_me, trophy_stats = stats
    n_trophies = len(trophy_stats)
    dx_target = max(0.0, math.log(1.0 + t_target_hours) - x_last)
    n_target = math.exp(ln_n_last + b_n * _damped_horizon(dx_target, 0.35))
    avg_e_target = math.exp(me_base + b_me * _damped_horizon(dx_target, lam_global))

    eff_horizon_days = max(1e-4, horizon_days + 0.5 * staleness_days)
    sqrt_h = math.sqrt(eff_horizon_days)

    meds = [0.0] * n_trophies
    sigmas = [0.2] * n_trophies
    sum_r = 0.0

    for i in range(n_trophies):
        r_now, k_now, lg_base, b_fit, b_rec, decel_adj, slope_shift, resid_rms, slope_se = trophy_stats[i]
        b_eff = (1.0 - rec_blend) * b_fit + rec_blend * b_rec
        if r_now >= 95.0 and b_eff > 0.0:
            b_eff *= max(0.0, (100.0 - r_now) / 5.0)

        lam_i = max(0.05, lam_global + decel_adj)
        if b_eff >= 0.0:
            eff_dx = _damped_horizon(dx_target, lam_i)
        else:
            eff_dx = _damped_horizon(dx_target, max(0.75, lam_i + 0.45))
        lg_med = lg_base + b_eff * eff_dx

        # Physical non-decreasing earner floor: k(T) >= k_now.
        if k_now >= 2.0:
            lg_floor = _logit(100.0 * k_now / max(1.0, n_target * 1.40))
            if lg_med < lg_floor:
                lg_med = 0.65 * lg_floor + 0.35 * lg_med

        activity_rw = (0.11 + 0.13 * min(2.0, abs(b_eff))) * sqrt_h
        curv_unc = min(0.45, 0.42 * slope_shift * dx_target)
        slope_unc = min(0.55, slope_se * dx_target)
        count_unc = 0.20 / math.sqrt(k_now + 1.0)
        raw_sig = math.sqrt(
            activity_rw * activity_rw
            + curv_unc * curv_unc
            + slope_unc * slope_unc
            + (0.50 * resid_rms) ** 2
            + count_unc * count_unc
        )
        if r_now >= 95.0:
            raw_sig = min(raw_sig, 0.28 + 0.12 * sqrt_h)
        raw_sig = max(0.07, min(1.5, raw_sig))

        meds[i] = lg_med
        sigmas[i] = raw_sig
        sum_r += _inv_logit_pct(lg_med) * 0.01

    # Macro conservation reconciliation: gently pull total rate mass toward forecasted E(T)/N(T).
    if sum_r > 0.15 and avg_e_target > 0.15:
        ratio_shift = 0.22 * math.log(avg_e_target / sum_r)
        if ratio_shift > 0.16:
            ratio_shift = 0.16
        elif ratio_shift < -0.16:
            ratio_shift = -0.16
        for i in range(n_trophies):
            if trophy_stats[i][0] < 85.0:
                meds[i] += ratio_shift

    return meds, sigmas


def _walk_forward_calibrate(trophies, used_snaps):
    """Run fast cached walk-forward pinball-loss CV on history to tune damping & spread scale."""
    n_snaps = len(used_snaps)
    if n_snaps < 6:
        return 0.32, 0.45, 0.88
    span_h = used_snaps[-1][3] - used_snaps[0][3]
    if span_h < 16.0:
        return 0.32, 0.45, 0.88

    t_end = used_snaps[-1][3]
    y_lgs = [_logit(float(r)) if r is not None else None for r in used_snaps[-1][6]]
    splits = []
    for frac in (0.52, 0.70):
        cut = used_snaps[0][3] + frac * span_h
        pref = [s for s in used_snaps if s[3] <= cut]
        if 3 <= len(pref) < n_snaps:
            h_days = max(1e-3, (t_end - pref[-1][3]) / 24.0)
            splits.append((_fit_snapshot_stats(trophies, pref), h_days))
    if not splits:
        return 0.32, 0.45, 0.88

    best_lam = 0.32
    best_rec = 0.45
    best_mae = 1e9
    best_eval = None

    for lam in (0.15, 0.32, 0.52):
        for rec_b in (0.35, 0.55):
            tot_err = 0.0
            cnt = 0
            split_preds = []
            for stats, h_days in splits:
                meds, sigs = _project_from_stats(stats, t_end, 0.0, h_days, lam, rec_b)
                split_preds.append((meds, sigs))
                for i, y_lg in enumerate(y_lgs):
                    if y_lg is not None:
                        tot_err += abs(y_lg - meds[i])
                        cnt += 1
            reg = 0.004 * (abs(lam - 0.32) + abs(rec_b - 0.45))
            mae = (tot_err / max(1, cnt)) + reg
            if mae < best_mae:
                best_mae = mae
                best_lam = lam
                best_rec = rec_b
                best_eval = split_preds

    safe_lam = 0.65 * best_lam + 0.35 * 0.35

    best_mult = 0.88
    best_pin = 1e9
    for mult in (0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.20, 1.35):
        pin_sum = 0.0
        pin_cnt = 0
        for meds, sigs in best_eval:
            for i, y_lg in enumerate(y_lgs):
                if y_lg is None:
                    continue
                sc = sigs[i] * mult
                m_lg = meds[i]
                for k_q in range(5):
                    tau = LEVELS[k_q]
                    d = y_lg - (m_lg + T8_SCALED[k_q] * sc)
                    pin_sum += 2.0 * (tau * d if d >= 0.0 else (tau - 1.0) * d)
                pin_cnt += 5
        loss = (pin_sum / max(1, pin_cnt)) + 0.005 * (mult - 0.88) ** 2
        if loss < best_pin:
            best_pin = loss
            best_mult = mult

    safe_mult = max(0.55, min(1.35, 0.72 * best_mult + 0.28 * 0.88))
    return safe_lam, best_rec, safe_mult


def forecast_case(case, name="case"):
    """Generate calibrated 5-quantile forecasts for all trophies in a case."""
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    release_date = case.get("game", {}).get("release_date", "2026-11-19")
    t0_sec = _iso_to_sec(f"{release_date}T00:00:00Z")
    t_at_sec = _iso_to_sec(case["at"])
    t_target_sec = _iso_to_sec(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    t_at_hours = max(0.0, (t_at_sec - t0_sec) / 3600.0)
    t_target_hours = max(t_at_hours, (t_target_sec - t0_sec) / 3600.0)

    snapshots = _extract_snapshots(case["history"], t0_sec)
    trophies = case["trophies"]

    if len(snapshots) <= 1:
        return {"trophies": _pre_refresh_forecast(trophies, snapshots[0], horizon_days, t_at_hours)}

    n_last = snapshots[-1][4]
    min_n = max(1500, int(0.12 * n_last))
    post_transient = [s for s in snapshots if s[4] >= min_n and s[3] >= 6.0]
    if len(post_transient) >= 3:
        used_snaps = post_transient
    elif len(snapshots) >= 3 and snapshots[0][4] < 0.25 * snapshots[1][4]:
        used_snaps = snapshots[1:]
    else:
        used_snaps = snapshots

    lam_global, rec_blend, spread_mult = _walk_forward_calibrate(trophies, used_snaps)
    staleness_days = max(0.0, (t_at_sec - used_snaps[-1][1]) / 86400.0)
    full_stats = _fit_snapshot_stats(trophies, used_snaps)
    meds, sigmas = _project_from_stats(
        full_stats, t_target_hours, staleness_days, horizon_days, lam_global, rec_blend
    )

    n_trophies = len(trophies)
    pred_qs_lg = [
        [meds[i] + q_t * (sigmas[i] * spread_mult) for q_t in T8_SCALED]
        for i in range(n_trophies)
    ]

    # PlayStation system invariant: Platinum requires every other trophy -> rate(plat) <= min(others).
    plat_indices = [i for i, t in enumerate(trophies) if t.get("tier") == "platinum"]
    if plat_indices and n_trophies > 1:
        for p_idx in plat_indices:
            for q_k in range(5):
                min_other = min(pred_qs_lg[j][q_k] for j in range(n_trophies) if j != p_idx)
                if pred_qs_lg[p_idx][q_k] > min_other:
                    pred_qs_lg[p_idx][q_k] = min_other

    # Soft monotonicity along sequential story chapters (Ch1 >= Ch2 >= Ch3 >= Ch4 >= Final >= 100%).
    chain = _chapter_chain(trophies)
    for pos in range(1, len(chain)):
        prev_idx = chain[pos - 1]
        curr_idx = chain[pos]
        if pred_qs_lg[curr_idx][2] > pred_qs_lg[prev_idx][2]:
            mid = 0.5 * (pred_qs_lg[curr_idx][2] + pred_qs_lg[prev_idx][2])
            shift = mid - pred_qs_lg[curr_idx][2]
            pred_qs_lg[curr_idx] = [v + shift for v in pred_qs_lg[curr_idx]]

    rows = []
    for i, trophy in enumerate(trophies):
        qs = [_round4(_inv_logit_pct(v)) for v in pred_qs_lg[i]]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(input_dir) if f.endswith(".in"))
    for name in names:
        in_path = os.path.join(input_dir, name)
        with open(in_path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case, name)
        out_name = name[:-3] + ".out"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
