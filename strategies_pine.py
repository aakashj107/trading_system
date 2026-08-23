"""
Indian Market Trading System — Pine Script Strategies (Converted)
=================================================================
Converted from TradingView Pine Script v5/v6

Strategy 10: RTS Adaptive Trail        (from Reactive Trail System)
Strategy 11: Structure Entry           (from WAIT+ADX+RSI+CCI+REG SuperTrend)
Strategy 12: Combined (Struct + RTS)   (Structure entry + RTS exit)

Usage:
  from strategies_pine import get_pine_strategies
  strategies = get_pine_strategies()
"""

import pandas as pd
import numpy as np
from strategies import Strategy, Signal


# ═══════════════════════════════════════════════════════════════════════
#  MOVING AVERAGE FUNCTIONS (Pine ta.* → Python equivalents)
# ═══════════════════════════════════════════════════════════════════════

def hma(series, period):
    """Hull Moving Average — fast, low lag."""
    half = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    w = lambda x: np.average(x, weights=np.arange(1, len(x) + 1), axis=0)
    wma_h = series.rolling(half).apply(w, raw=True)
    wma_f = series.rolling(period).apply(w, raw=True)
    diff = 2 * wma_h - wma_f
    return diff.rolling(sqrt_p).apply(w, raw=True)


def alma(series, period, offset=0.85, sigma=6):
    """Arnaud Legoux Moving Average — smooth, balanced."""
    m = offset * (period - 1)
    s = period / sigma
    wts = np.array([np.exp(-((i - m) ** 2) / (2 * s ** 2)) for i in range(period)])
    wts /= wts.sum()
    return series.rolling(period).apply(lambda x: np.dot(x, wts), raw=True)


def kama(series, period):
    """Kaufman Adaptive Moving Average — adapts to market efficiency."""
    fast_sc, slow_sc = 2.0 / 3.0, 2.0 / 31.0
    v = series.values.astype(float)
    n = len(v)
    out = np.full(n, np.nan)
    if n < period:
        return pd.Series(out, index=series.index)
    out[period - 1] = v[period - 1]
    for i in range(period, n):
        direction = abs(v[i] - v[i - period])
        volatility = sum(abs(v[j] - v[j - 1]) for j in range(i - period + 1, i + 1))
        er = direction / volatility if volatility != 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[i] = out[i - 1] + sc * (v[i] - out[i - 1])
    return pd.Series(out, index=series.index)


def t3(series, period, vf=0.7):
    """Tillson T3 — triple-smoothed, very low noise."""
    e1 = series.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    e4 = e3.ewm(span=period, adjust=False).mean()
    e5 = e4.ewm(span=period, adjust=False).mean()
    e6 = e5.ewm(span=period, adjust=False).mean()
    c1, c2 = -(vf ** 3), 3 * vf ** 2 + 3 * vf ** 3
    c3 = -6 * vf ** 2 - 3 * vf - 3 * vf ** 3
    c4 = 1 + 3 * vf + vf ** 3 + 3 * vf ** 2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def vwma(series, volume, period):
    """Volume Weighted Moving Average."""
    vs = (series * volume).rolling(period).sum()
    vol_s = volume.rolling(period).sum().replace(0, np.nan)
    return vs / vol_s


# ═══════════════════════════════════════════════════════════════════════
#  INDICATOR FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def _sma(series, period):
    return series.rolling(period).mean()


def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    ag = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    al = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))


def _rma(series, period):
    """Wilder's RMA (Pine ta.rma)."""
    return series.ewm(alpha=1.0 / period, min_periods=period).mean()


def _atr(high, low, close, period=14):
    h_l = high - low
    h_c = (high - close.shift()).abs()
    l_c = (low - close.shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    return _rma(tr, period)


def _adx(high, low, close, period=14):
    """Wilder's ADX."""
    h_l = high - low
    h_c = (high - close.shift()).abs()
    l_c = (low - close.shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    up = high - high.shift(1)
    dn = low.shift(1) - low
    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)
    plus_dm[(up > dn) & (up > 0)] = up
    minus_dm[(dn > up) & (dn > 0)] = dn
    s_tr = _rma(tr, period)
    s_p = _rma(plus_dm, period)
    s_m = _rma(minus_dm, period)
    pdi = 100 * s_p / s_tr.replace(0, np.nan)
    mdi = 100 * s_m / s_tr.replace(0, np.nan)
    di_sum = pdi + mdi
    dx = 100 * (pdi - mdi).abs() / di_sum.replace(0, np.nan)
    return _rma(dx, period)


def _linreg(series, length, offset=0):
    """Rolling linear regression endpoint (Pine ta.linreg)."""
    x = np.arange(length, dtype=float)
    xm = x.mean()
    den = ((x - xm) ** 2).sum()

    def calc(y):
        ym = y.mean()
        slope = ((x - xm) * (y - ym)).sum() / den
        intercept = ym - slope * xm
        return intercept + slope * (length - 1 - offset)

    return series.rolling(length).apply(calc, raw=True)


def _pivot_high(hv, left, right):
    n = len(hv)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        pb = i - right
        if all(hv[pb - j] < hv[pb] for j in range(1, left + 1)) and \
           all(hv[pb + j] < hv[pb] for j in range(1, right + 1)):
            out[i] = hv[pb]
    return out


def _pivot_low(lv, left, right):
    n = len(lv)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        pb = i - right
        if all(lv[pb - j] > lv[pb] for j in range(1, left + 1)) and \
           all(lv[pb + j] > lv[pb] for j in range(1, right + 1)):
            out[i] = lv[pb]
    return out


# ═══════════════════════════════════════════════════════════════════════
#  RTS TRAIL ENGINE (shared by Strategy 10 & 12)
# ═══════════════════════════════════════════════════════════════════════

def _rts_trail(close, high, low, volume,
               ma_type="ALMA", ma_len=21,
               rsi_len=13, rsi_sm=3,
               vol_type="Hybrid", vol_len=13,
               trail_mult=2.0, adapt=1.0):
    """
    Core RTS trail engine.
    Returns: trend (1/-1 array), trail (price array), signal_score array.
    """
    n = len(close)
    cv = close.values

    # Baseline MA
    ma_fn = {
        "HMA": lambda: hma(close, ma_len),
        "ALMA": lambda: alma(close, ma_len),
        "KAMA": lambda: kama(close, ma_len),
        "T3": lambda: t3(close, ma_len),
        "VWMA": lambda: vwma(close, volume, ma_len),
        "EMA": lambda: _ema(close, ma_len),
    }
    base = ma_fn.get(ma_type, ma_fn["EMA"])().fillna(close)

    # Momentum
    rsi_raw = _rsi(close, rsi_len).fillna(50)
    rsi_s = _ema(rsi_raw, rsi_sm).fillna(50)
    mom_dist = ((rsi_s - 50).abs() / 50).clip(0, 1)
    eff_m = trail_mult * (1 - adapt * mom_dist * 0.4)

    # Volatility
    atr_v = _atr(high, low, close, vol_len).fillna(0)
    std_v = close.rolling(vol_len).std().fillna(0)
    vol = {
        "ATR": atr_v,
        "StDev": std_v,
    }.get(vol_type, (atr_v + std_v) / 2)

    offset = (vol * eff_m).values
    ma_v = base.values
    uc = ma_v - offset
    dc = ma_v + offset

    # Ratcheting state machine
    trend = np.zeros(n, dtype=int)
    trail = np.full(n, np.nan)

    start = 0
    for i in range(n):
        if not np.isnan(uc[i]) and not np.isnan(dc[i]) and uc[i] != dc[i]:
            start = i
            break

    trend[start] = 1 if cv[start] >= ma_v[start] else -1
    trail[start] = uc[start] if trend[start] == 1 else dc[start]

    for i in range(start + 1, n):
        if np.isnan(uc[i]) or np.isnan(dc[i]):
            trend[i], trail[i] = trend[i - 1], trail[i - 1]
            continue
        if trend[i - 1] == 1:
            t = max(trail[i - 1], uc[i])
            if cv[i] < t:
                trend[i], trail[i] = -1, dc[i]
            else:
                trend[i], trail[i] = 1, t
        else:
            t = min(trail[i - 1], dc[i])
            if cv[i] > t:
                trend[i], trail[i] = 1, uc[i]
            else:
                trend[i], trail[i] = -1, t

    # Signal score 0-100
    mom_score = mom_dist.clip(0, 0.6) / 0.6 * 40
    vol_ratio = volume / volume.rolling(20).mean().replace(0, np.nan)
    vol_score = ((vol_ratio - 0.5).clip(0, 2) / 2 * 30).fillna(15)
    ema50 = _ema(close, 50)
    ema50_slope = ema50 - ema50.shift(5)
    trend_s = pd.Series(trend, index=close.index)
    htf_ok = ((trend_s == 1) & (ema50_slope > 0)) | \
             ((trend_s == -1) & (ema50_slope < 0))
    htf_score = htf_ok.astype(float) * 20 + 10
    score = (mom_score + vol_score + htf_score).values

    return trend, trail, score


# ═══════════════════════════════════════════════════════════════════════
#  STRATEGY 10 — RTS ADAPTIVE TRAIL
# ═══════════════════════════════════════════════════════════════════════

class RTSAdaptiveTrail(Strategy):
    """
    Reactive Trail System — standalone.
    Entry on trail flip, exit on opposite flip.
    """

    def __init__(self, ma_type="ALMA", ma_length=21, rsi_length=13,
                 rsi_smooth=3, vol_type="Hybrid", vol_length=13,
                 trail_mult=2.0, adapt=1.0):
        super().__init__(
            f"RTS_Adaptive_{ma_type}_{ma_length}_T{trail_mult}"
        )
        self.ma_type = ma_type
        self.ma_length = ma_length
        self.rsi_length = rsi_length
        self.rsi_smooth = rsi_smooth
        self.vol_type = vol_type
        self.vol_length = vol_length
        self.trail_mult = trail_mult
        self.adapt = adapt

    def generate_signals(self, df):
        df = df.copy()
        n = len(df)

        trend, trail_arr, score = _rts_trail(
            df["Close"], df["High"], df["Low"], df["Volume"],
            self.ma_type, self.ma_length, self.rsi_length,
            self.rsi_smooth, self.vol_type, self.vol_length,
            self.trail_mult, self.adapt,
        )

        signals = np.zeros(n, dtype=int)
        for i in range(1, n):
            if trend[i] == 1 and trend[i - 1] == -1:
                signals[i] = Signal.BUY.value
            elif trend[i] == -1 and trend[i - 1] == 1:
                signals[i] = Signal.SELL.value

        df["Signal"] = signals
        df["_trend"] = trend
        df["_trail"] = trail_arr
        df["_score"] = score
        return df


# ═══════════════════════════════════════════════════════════════════════
#  STRATEGY 11 — STRUCTURE ENTRY
# ═══════════════════════════════════════════════════════════════════════

class StructureEntry(Strategy):
    """
    Structure Entry — converted from WAIT+ADX+RSI+CCI+REG SuperTrend.

    Entry engines:
      1. Regression SuperTrend flip + body-level retest + break (v2)
      2. CHOCH / BOS (swing structure + neckline + retest + 2nd cross)

    Filters:
      - Regime (TREND vs RANGE)
      - ADX minimum
      - Break Quality (body ratio + ATR)

    Exit: ATR trailing stop.
    """

    def __init__(self, reg_length=20, rmse_mult=2.0,
                 adx_length=14, adx_min=20.0, ema_filter_len=50,
                 regime_lookback=60, regime_flip_thresh=4,
                 regime_adx_max=18.0, regime_slope_len=8,
                 regime_slope_min_pct=0.03,
                 bq_mode="SOFT", bq_body_ratio=0.35, bq_body_atr=0.10,
                 choch_pivot_left=5, choch_pivot_right=5,
                 choch_max_wait=40, choch_only_trend=True,
                 atr_exit_mult=2.0,
                 use_entry_v2=True, use_choch=True):

        super().__init__(
            f"Structure_Reg{reg_length}_Choch_ADX{int(adx_min)}"
        )
        self.reg_length = reg_length
        self.rmse_mult = rmse_mult
        self.adx_length = adx_length
        self.adx_min = adx_min
        self.ema_filter_len = ema_filter_len
        self.regime_lookback = regime_lookback
        self.regime_flip_thresh = regime_flip_thresh
        self.regime_adx_max = regime_adx_max
        self.regime_slope_len = regime_slope_len
        self.regime_slope_min_pct = regime_slope_min_pct
        self.bq_mode = bq_mode
        self.bq_body_ratio = bq_body_ratio
        self.bq_body_atr = bq_body_atr
        self.choch_pivot_left = choch_pivot_left
        self.choch_pivot_right = choch_pivot_right
        self.choch_max_wait = choch_max_wait
        self.choch_only_trend = choch_only_trend
        self.atr_exit_mult = atr_exit_mult
        self.use_entry_v2 = use_entry_v2
        self.use_choch = use_choch

    def generate_signals(self, df):
        df = df.copy()
        close, high, low = df["Close"], df["High"], df["Low"]
        open_ = df["Open"]
        n = len(df)
        cv, hv, lv, ov = close.values, high.values, low.values, open_.values

        # ── 1. Core indicators ────────────────────────────────────
        ema50 = _ema(close, self.ema_filter_len)
        adx_v = _adx(high, low, close, self.adx_length).fillna(0).values
        ema50_v = ema50.fillna(close).values

        state = np.full(n, "WAIT", dtype=object)
        for i in range(n):
            if cv[i] > ema50_v[i] and adx_v[i] >= self.adx_min:
                state[i] = "WAIT(L)"
            elif cv[i] < ema50_v[i] and adx_v[i] >= self.adx_min:
                state[i] = "WAIT(S)"

        # ── 2. Regression SuperTrend ──────────────────────────────
        reg_line = _linreg(close, self.reg_length).bfill()
        rl_v = reg_line.values

        rmse_v = np.full(n, np.nan)
        for i in range(self.reg_length, n):
            sq = 0.0
            for j in range(self.reg_length):
                sq += (cv[i - j] - rl_v[i - j]) ** 2
            rmse_v[i] = np.sqrt(sq / max(self.reg_length - 1, 1))

        upper_raw = rl_v + rmse_v * self.rmse_mult
        lower_raw = rl_v - rmse_v * self.rmse_mult

        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        rt = np.ones(n, dtype=int)

        for i in range(1, n):
            if np.isnan(upper_raw[i]) or np.isnan(lower_raw[i]):
                upper[i], lower[i], rt[i] = upper[i-1], lower[i-1], rt[i-1]
                continue
            u_prev = upper[i-1] if not np.isnan(upper[i-1]) else upper_raw[i]
            l_prev = lower[i-1] if not np.isnan(lower[i-1]) else lower_raw[i]
            upper[i] = min(upper_raw[i], u_prev) if rt[i-1] == -1 else upper_raw[i]
            lower[i] = max(lower_raw[i], l_prev) if rt[i-1] == 1 else lower_raw[i]
            if rt[i-1] == 1 and cv[i] < lower[i]:
                rt[i] = -1
            elif rt[i-1] == -1 and cv[i] > upper[i]:
                rt[i] = 1
            else:
                rt[i] = rt[i-1]

        flip = np.zeros(n, dtype=bool)
        for i in range(1, n):
            flip[i] = rt[i] != rt[i-1]

        # ── 3. Regime classification ──────────────────────────────
        eslope = np.full(n, np.nan)
        for i in range(self.regime_slope_len, n):
            eslope[i] = ema50_v[i] - ema50_v[i - self.regime_slope_len]
        eslope_pct = np.where(ema50_v != 0, np.abs(eslope) / ema50_v * 100, 0)
        flat = eslope_pct < self.regime_slope_min_pct
        low_adx = adx_v <= self.regime_adx_max

        fc = np.zeros(n, dtype=int)
        for i in range(self.regime_lookback, n):
            fc[i] = np.sum(flip[i - self.regime_lookback + 1: i + 1])

        is_range = ((fc >= self.regime_flip_thresh) & low_adx) | \
                   (flat & low_adx)
        allow_flip = ~is_range

        # ── 4. Break Quality ──────────────────────────────────────
        bq_atr = _atr(high, low, close, 14).fillna(0).values
        bq_rng = np.maximum(hv - lv, 0.0001)
        bq_body = np.abs(cv - ov)
        bq_ratio = bq_body / bq_rng
        if self.bq_mode == "OFF":
            bq_ok = np.ones(n, dtype=bool)
        else:
            bq_ok = (bq_ratio >= self.bq_body_ratio) & \
                    (bq_body >= bq_atr * self.bq_body_atr)

        # ── 5. Entry v2 (flip body → retest → break) ─────────────
        v2_buy = np.zeros(n, dtype=bool)
        v2_sell = np.zeros(n, dtype=bool)

        if self.use_entry_v2:
            e_dir, e_level = 0, np.nan
            e_retest, e_active, e_bar = False, False, 0
            retr_req = bq_atr * 0.25

            for i in range(1, n):
                if flip[i] and allow_flip[i] and \
                   (self.bq_mode != "HARD" or bq_ok[i]):
                    e_dir = rt[i]
                    e_level = cv[i]
                    e_retest, e_active, e_bar = False, True, i

                if e_active and (i - e_bar) > 30:
                    e_active, e_dir = False, 0

                if not e_active or np.isnan(e_level):
                    continue

                if not e_retest:
                    if e_dir == 1 and lv[i] <= e_level:
                        e_retest = True
                    elif e_dir == -1 and hv[i] >= e_level:
                        e_retest = True

                if e_retest:
                    if e_dir == 1 and cv[i] > e_level and \
                       cv[i-1] <= e_level:
                        v2_buy[i] = True
                        e_active = False
                    elif e_dir == -1 and cv[i] < e_level and \
                         cv[i-1] >= e_level:
                        v2_sell[i] = True
                        e_active = False

        # ── 6. CHOCH / BOS ───────────────────────────────────────
        cho_buy = np.zeros(n, dtype=bool)
        cho_sell = np.zeros(n, dtype=bool)

        if self.use_choch:
            ph = _pivot_high(hv, self.choch_pivot_left,
                             self.choch_pivot_right)
            pl = _pivot_low(lv, self.choch_pivot_left,
                            self.choch_pivot_right)
            lsh = np.full(n, np.nan)
            lsl = np.full(n, np.nan)
            for i in range(n):
                if not np.isnan(ph[i]):
                    lsh[i] = ph[i]
                elif i > 0:
                    lsh[i] = lsh[i-1]
                if not np.isnan(pl[i]):
                    lsl[i] = pl[i]
                elif i > 0:
                    lsl[i] = lsl[i-1]

            s_dir = 0
            c_dir, c_level = 0, np.nan
            c_retest, c_fired, c_active, c_bar = False, False, False, 0

            for i in range(1, n):
                if s_dir == 0:
                    if not np.isnan(lsh[i]) and cv[i] > lsh[i]:
                        s_dir = 1
                    if not np.isnan(lsl[i]) and cv[i] < lsl[i]:
                        s_dir = -1

                allowed = True
                if self.choch_only_trend and is_range[i]:
                    allowed = False
                if adx_v[i] < self.adx_min:
                    allowed = False

                if not allowed:
                    continue

                bos_up = s_dir == 1 and not np.isnan(lsh[i]) and \
                         cv[i] > lsh[i] and cv[i-1] <= lsh[i]
                bos_dn = s_dir == -1 and not np.isnan(lsl[i]) and \
                         cv[i] < lsl[i] and cv[i-1] >= lsl[i]
                cho_up = s_dir == -1 and not np.isnan(lsh[i]) and \
                         cv[i] > lsh[i] and cv[i-1] <= lsh[i]
                cho_dn = s_dir == 1 and not np.isnan(lsl[i]) and \
                         cv[i] < lsl[i] and cv[i-1] >= lsl[i]

                new_brk = bos_up or bos_dn or cho_up or cho_dn

                if new_brk and (self.bq_mode != "HARD" or bq_ok[i]):
                    if bos_up or cho_up:
                        c_dir = 1
                        s_dir = 1
                    else:
                        c_dir = -1
                        s_dir = -1
                    c_level = max(ov[i], cv[i]) if c_dir == 1 \
                        else min(ov[i], cv[i])
                    c_retest, c_fired = False, False
                    c_active, c_bar = True, i

                if c_active and (i - c_bar) > self.choch_max_wait:
                    c_active, c_retest, c_fired = False, False, False

                if c_active and not c_retest:
                    if c_dir == 1 and min(ov[i], cv[i]) <= c_level:
                        c_retest = True
                    elif c_dir == -1 and max(ov[i], cv[i]) >= c_level:
                        c_retest = True

                if c_active and c_retest and not c_fired:
                    if c_dir == 1 and cv[i] > c_level and \
                       cv[i-1] <= c_level:
                        cho_buy[i] = True
                        c_fired, c_active = True, False
                    elif c_dir == -1 and cv[i] < c_level and \
                         cv[i-1] >= c_level:
                        cho_sell[i] = True
                        c_fired, c_active = True, False

        # ── 7. Combine entries + ATR trailing-stop exit ───────────
        buy_sig = v2_buy | cho_buy
        sell_sig = v2_sell | cho_sell
        atr14 = _atr(high, low, close, 14).fillna(0).values

        signals = np.zeros(n, dtype=int)
        in_pos = 0
        peak = 0.0
        trough = 0.0
        sl = 0.0

        for i in range(n):
            if in_pos == 0:
                if buy_sig[i]:
                    signals[i] = Signal.BUY.value
                    in_pos, peak, sl = 1, cv[i], \
                        cv[i] - self.atr_exit_mult * atr14[i]
                elif sell_sig[i]:
                    signals[i] = Signal.SELL.value
                    in_pos, trough, sl = -1, cv[i], \
                        cv[i] + self.atr_exit_mult * atr14[i]
            elif in_pos == 1:
                peak = max(peak, cv[i])
                sl = max(sl, peak - self.atr_exit_mult * atr14[i])
                if cv[i] < sl or sell_sig[i]:
                    signals[i] = Signal.SELL.value
                    in_pos = 0
            elif in_pos == -1:
                trough = min(trough, cv[i])
                sl = min(sl, trough + self.atr_exit_mult * atr14[i])
                if cv[i] > sl or buy_sig[i]:
                    signals[i] = Signal.BUY.value
                    in_pos = 0

        df["Signal"] = signals
        df["_regime"] = np.where(is_range, "RANGE", "TREND")
        df["_adx"] = adx_v
        df["_state"] = state
        return df


# ═══════════════════════════════════════════════════════════════════════
#  STRATEGY 12 — COMBINED (Structure Entry + RTS Exit)
# ═══════════════════════════════════════════════════════════════════════

class CombinedRTSStructure(Strategy):
    """
    Best of both worlds:
      Entry  →  CHOCH/BOS + Break Quality + Regime  (from Script 2)
      Exit   →  RTS adaptive trail flip              (from Script 1)

    BUY when Structure gives long entry AND RTS trail is bullish.
    SELL when RTS trail flips from bullish to bearish.
    """

    def __init__(self, **kwargs):
        rts_kw = {k: kwargs.pop(k) for k in list(kwargs.keys())
                  if k.startswith("rts_")}
        str_kw = {k: kwargs.pop(k) for k in list(kwargs.keys())
                  if k.startswith("str_")}

        self._rts = RTSAdaptiveTrail(
            ma_type=rts_kw.get("rts_ma_type", "ALMA"),
            ma_length=rts_kw.get("rts_ma_length", 21),
            rsi_length=rts_kw.get("rts_rsi_length", 13),
            rsi_smooth=rts_kw.get("rts_rsi_smooth", 3),
            vol_type=rts_kw.get("rts_vol_type", "Hybrid"),
            vol_length=rts_kw.get("rts_vol_length", 13),
            trail_mult=rts_kw.get("rts_trail_mult", 2.0),
            adapt=rts_kw.get("rts_adapt", 1.0),
        )
        self._strat = StructureEntry(
            reg_length=str_kw.get("str_reg_length", 20),
            rmse_mult=str_kw.get("str_rmse_mult", 2.0),
            adx_length=str_kw.get("str_adx_length", 14),
            adx_min=str_kw.get("str_adx_min", 20.0),
            ema_filter_len=str_kw.get("str_ema_filter", 50),
            bq_mode=str_kw.get("str_bq_mode", "SOFT"),
            choch_pivot_left=str_kw.get("str_choch_left", 5),
            choch_pivot_right=str_kw.get("str_choch_right", 5),
            choch_max_wait=str_kw.get("str_choch_wait", 40),
            choch_only_trend=str_kw.get("str_choch_trend", True),
            atr_exit_mult=str_kw.get("str_atr_exit", 2.0),
            use_entry_v2=str_kw.get("str_use_v2", False),
            use_choch=str_kw.get("str_use_choch", True),
        )
        super().__init__("Combined_Structure_RTS")

    def generate_signals(self, df):
        df = df.copy()
        n = len(df)
        cv = df["Close"].values

        # RTS trail → trend direction
        trend, trail_arr, score = _rts_trail(
            df["Close"], df["High"], df["Low"], df["Volume"],
            self._rts.ma_type, self._rts.ma_length,
            self._rts.rsi_length, self._rts.rsi_smooth,
            self._rts.vol_type, self._rts.vol_length,
            self._rts.trail_mult, self._rts.adapt,
        )

        # Structure entry signals (raw)
        cho_buy, cho_sell = self._get_structure_entries(df)

        # Combined logic
        signals = np.zeros(n, dtype=int)
        in_long = False

        for i in range(1, n):
            if not in_long:
                if cho_buy[i] and trend[i] == 1:
                    signals[i] = Signal.BUY.value
                    in_long = True
            else:
                if trend[i] == -1 and trend[i - 1] == 1:
                    signals[i] = Signal.SELL.value
                    in_long = False

        df["Signal"] = signals
        df["_trend"] = trend
        df["_trail"] = trail_arr
        df["_score"] = score
        return df

    def _get_structure_entries(self, df):
        """Extract raw CHOCH/BOS entry signals without exit loop."""
        close = df["Close"]
        high, low = df["High"], df["Low"]
        open_ = df["Open"]
        n = len(df)
        cv, hv, lv, ov = close.values, high.values, low.values, open_.values

        s = self._strat
        adx_v = _adx(high, low, close, s.adx_length).fillna(0).values
        ema50_v = _ema(close, s.ema_filter_len).fillna(close).values

        # Regime
        eslope = np.full(n, np.nan)
        for i in range(s.regime_slope_len, n):
            eslope[i] = ema50_v[i] - ema50_v[i - s.regime_slope_len]
        esp = np.where(ema50_v != 0, np.abs(eslope) / ema50_v * 100, 0)
        flat = esp < s.regime_slope_min_pct
        low_adx = adx_v <= s.regime_adx_max

        reg_line = _linreg(close, s.reg_length).bfill()
        rl_v = reg_line.values
        rmse_v = np.full(n, np.nan)
        for i in range(s.reg_length, n):
            sq = sum((cv[i - j] - rl_v[i - j]) ** 2
                     for j in range(s.reg_length))
            rmse_v[i] = np.sqrt(sq / max(s.reg_length - 1, 1))

        upper_raw = rl_v + rmse_v * s.rmse_mult
        lower_raw = rl_v - rmse_v * s.rmse_mult
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        rt = np.ones(n, dtype=int)

        for i in range(1, n):
            if np.isnan(upper_raw[i]) or np.isnan(lower_raw[i]):
                upper[i] = upper[i-1]
                lower[i] = lower[i-1]
                rt[i] = rt[i-1]
                continue
            u_prev = upper[i-1] if not np.isnan(upper[i-1]) \
                else upper_raw[i]
            l_prev = lower[i-1] if not np.isnan(lower[i-1]) \
                else lower_raw[i]
            upper[i] = min(upper_raw[i], u_prev) \
                if rt[i-1] == -1 else upper_raw[i]
            lower[i] = max(lower_raw[i], l_prev) \
                if rt[i-1] == 1 else lower_raw[i]
            if rt[i-1] == 1 and cv[i] < lower[i]:
                rt[i] = -1
            elif rt[i-1] == -1 and cv[i] > upper[i]:
                rt[i] = 1
            else:
                rt[i] = rt[i-1]

        flip = np.zeros(n, dtype=bool)
        for i in range(1, n):
            flip[i] = rt[i] != rt[i-1]

        fc = np.zeros(n, dtype=int)
        for i in range(s.regime_lookback, n):
            fc[i] = np.sum(flip[i - s.regime_lookback + 1: i + 1])
        is_range = ((fc >= s.regime_flip_thresh) & low_adx) | \
                   (flat & low_adx)

        bq_atr = _atr(high, low, close, 14).fillna(0).values
        bq_rng = np.maximum(hv - lv, 0.0001)
        bq_body = np.abs(cv - ov)
        bq_ratio = bq_body / bq_rng
        if s.bq_mode == "OFF":
            bq_ok = np.ones(n, dtype=bool)
        else:
            bq_ok = (bq_ratio >= s.bq_body_ratio) & \
                    (bq_body >= bq_atr * s.bq_body_atr)

        # CHOCH/BOS
        cho_buy = np.zeros(n, dtype=bool)
        cho_sell = np.zeros(n, dtype=bool)

        ph = _pivot_high(hv, s.choch_pivot_left, s.choch_pivot_right)
        pl = _pivot_low(lv, s.choch_pivot_left, s.choch_pivot_right)
        lsh = np.full(n, np.nan)
        lsl = np.full(n, np.nan)
        for i in range(n):
            if not np.isnan(ph[i]):
                lsh[i] = ph[i]
            elif i > 0:
                lsh[i] = lsh[i-1]
            if not np.isnan(pl[i]):
                lsl[i] = pl[i]
            elif i > 0:
                lsl[i] = lsl[i-1]

        s_dir = 0
        c_dir, c_level = 0, np.nan
        c_retest, c_fired, c_active, c_bar = False, False, False, 0

        for i in range(1, n):
            if s_dir == 0:
                if not np.isnan(lsh[i]) and cv[i] > lsh[i]:
                    s_dir = 1
                if not np.isnan(lsl[i]) and cv[i] < lsl[i]:
                    s_dir = -1

            allowed = (not s.choch_only_trend or not is_range[i]) \
                      and adx_v[i] >= s.adx_min
            if not allowed:
                continue

            bos_up = s_dir == 1 and not np.isnan(lsh[i]) and \
                     cv[i] > lsh[i] and cv[i-1] <= lsh[i]
            bos_dn = s_dir == -1 and not np.isnan(lsl[i]) and \
                     cv[i] < lsl[i] and cv[i-1] >= lsl[i]
            cho_up = s_dir == -1 and not np.isnan(lsh[i]) and \
                     cv[i] > lsh[i] and cv[i-1] <= lsh[i]
            cho_dn = s_dir == 1 and not np.isnan(lsl[i]) and \
                     cv[i] < lsl[i] and cv[i-1] >= lsl[i]

            if (bos_up or bos_dn or cho_up or cho_dn) and \
               (s.bq_mode != "HARD" or bq_ok[i]):
                c_dir = 1 if (bos_up or cho_up) else -1
                s_dir = c_dir
                c_level = max(ov[i], cv[i]) if c_dir == 1 \
                    else min(ov[i], cv[i])
                c_retest, c_fired = False, False
                c_active, c_bar = True, i

            if c_active and (i - c_bar) > s.choch_max_wait:
                c_active, c_retest, c_fired = False, False, False

            if c_active and not c_retest:
                if c_dir == 1 and min(ov[i], cv[i]) <= c_level:
                    c_retest = True
                elif c_dir == -1 and max(ov[i], cv[i]) >= c_level:
                    c_retest = True

            if c_active and c_retest and not c_fired:
                if c_dir == 1 and cv[i] > c_level and \
                   cv[i-1] <= c_level:
                    cho_buy[i] = True
                    c_fired, c_active = True, False
                elif c_dir == -1 and cv[i] < c_level and \
                     cv[i-1] >= c_level:
                    cho_sell[i] = True
                    c_fired, c_active = True, False

        return cho_buy, cho_sell


# ═══════════════════════════════════════════════════════════════════════
#  STRATEGY 13: ORDER BLOCK (SMC) — ported from "Order Block Engine [JOAT]"
#  Pine v6 indicator. Detects institutional order blocks: the last
#  opposing-close candle before a displacement leg that BREAKS a confirmed
#  swing by > dispMult x ATR (with volume confirmation), grades the zone
#  0-10, then fires BUY on a bullish-OB retest and SHORT on a bearish-OB
#  retest. Pure Smart-Money-Concepts logic — structurally new vs the
#  trend/mean-reversion families already in the system.
# ═══════════════════════════════════════════════════════════════════════


def _grade(disp_atr, vol_ratio, body_ratio):
    """Faithful port of f_grade() from the Pine source (max 10)."""
    s = 0.0
    s += 4.0 if disp_atr >= 2.0 else 3.2 if disp_atr >= 1.4 else 2.5 if disp_atr >= 1.0 else 1.7 if disp_atr >= 0.6 else 1.0
    s += 3.5 if vol_ratio >= 2.2 else 2.7 if vol_ratio >= 1.6 else 1.8 if vol_ratio >= 1.2 else 1.0 if vol_ratio >= 0.9 else 0.4
    s += 2.5 if body_ratio >= 0.8 else 1.8 if body_ratio >= 0.6 else 1.0 if body_ratio >= 0.4 else 0.4
    return max(0.0, min(10.0, s))


class OrderBlockStrategy(Strategy):
    def __init__(self, atr_len=14, pivot_len=8, disp_mult=0.5,
                 ob_search=10, min_grade=5.0, vol_len=20, vol_mult=1.3,
                 use_vol=True):
        super().__init__(
            f"OrderBlock_A{atr_len}_P{pivot_len}_D{disp_mult}_G{min_grade}",
            "smc")
        self.atr_len = atr_len
        self.pivot_len = pivot_len
        self.disp_mult = disp_mult
        self.ob_search = ob_search
        self.min_grade = min_grade
        self.vol_len = vol_len
        self.vol_mult = vol_mult
        self.use_vol = use_vol

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Signal"] = Signal.HOLD.value
        min_bars = max(self.atr_len, self.pivot_len * 2, self.vol_len) + self.ob_search + 5
        if len(df) < min_bars:
            return df

        high, low, close, open_ = df["High"], df["Low"], df["Close"], df["Open"]
        # ATR (Wilder) — matches Pine ta.atr
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / self.atr_len, min_periods=self.atr_len,
                     adjust=False).mean()
        # Volume baseline + ratio
        vol = df["Volume"].fillna(0) if "Volume" in df else pd.Series(0, index=df.index)
        vol_sma = vol.rolling(self.vol_len).mean()
        vol_ratio = (vol / vol_sma).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        vol_ok = (~self.use_vol) | (vol_ratio >= self.vol_mult) | (vol_sma <= 0)
        body_ratio = ((close - open_).abs() / (high - low).replace(0, np.nan)).fillna(0.0)

        # Confirmed swing high/low: extremes over a centered pivot_len window
        pl = self.pivot_len
        roll_h = high.rolling(2 * pl + 1, center=True).max()
        roll_l = low.rolling(2 * pl + 1, center=True).min()
        swing_high = roll_h.where(high == roll_h)
        swing_low = roll_l.where(low == roll_l)

        last_sh = swing_high.ffill()
        last_sl = swing_low.ffill()
        sh_broken = pd.Series(False, index=df.index)
        sl_broken = pd.Series(False, index=df.index)

        closes = close.values
        highs = high.values
        lows = low.values
        opens = open_.values
        atrs = atr.values
        volratio = vol_ratio.values
        bodyrat = body_ratio.values
        lsh = last_sh.values
        lsl = last_sl.values
        vok = vol_ok.values
        sig = np.array(df["Signal"].values, copy=True)

        active_bull = []  # list of (top, bot, grade)
        active_bear = []

        for i in range(pl + 1, len(df)):
            if not np.isnan(swing_high.iloc[i]):
                sh_broken.iloc[:i] = True
            if not np.isnan(swing_low.iloc[i]):
                sl_broken.iloc[:i] = True

            a = atrs[i]
            if a <= 0 or np.isnan(a):
                continue
            up = closes[i] > opens[i]
            dn = closes[i] < opens[i]
            # Bullish displacement break
            if (up and not np.isnan(lsh[i]) and not sh_broken.iloc[i] and
                    (closes[i] - lsh[i]) > self.disp_mult * a and vok[i]):
                sh_broken.iloc[i] = True
                for j in range(max(0, i - self.ob_search), i):
                    if opens[j] > closes[j]:
                        disp_atr = (closes[i] - lsh[i]) / a
                        gr = _grade(disp_atr, volratio[i], bodyrat[i])
                        if gr >= self.min_grade:
                            active_bull.append((highs[j], lows[j], gr))
                        break
            # Bearish displacement break
            if (dn and not np.isnan(lsl[i]) and not sl_broken.iloc[i] and
                    (lsl[i] - closes[i]) > self.disp_mult * a and vok[i]):
                sl_broken.iloc[i] = True
                for j in range(max(0, i - self.ob_search), i):
                    if opens[j] < closes[j]:
                        disp_atr = (lsl[i] - closes[i]) / a
                        gr = _grade(disp_atr, volratio[i], bodyrat[i])
                        if gr >= self.min_grade:
                            active_bear.append((highs[j], lows[j], gr))
                        break

            # Retest entry: price returns into an active zone
            c = closes[i]
            if active_bull:
                hit = [z for z in active_bull if z[1] <= c <= z[0]]
                if hit:
                    sig[i] = Signal.BUY.value
                    active_bull = [z for z in active_bull if not (z[1] <= c <= z[0])]
            if active_bear:
                hit = [z for z in active_bear if z[1] <= c <= z[0]]
                if hit:
                    sig[i] = Signal.SHORT.value
                    active_bear = [z for z in active_bear if not (z[1] <= c <= z[0])]

        df["Signal"] = sig
        return df


# ═══════════════════════════════════════════════════════════════════════
#  REGISTRY
# ═══════════════════════════════════════════════════════════════════════

def get_pine_strategies():
    """Return all converted Pine Script strategies."""
    return [
        RTSAdaptiveTrail(),          # Strategy 10
        StructureEntry(),             # Strategy 11
        CombinedRTSStructure(),       # Strategy 12
        OrderBlockStrategy(),         # Strategy 13 (SMC order blocks)
    ]


def get_all_strategies_with_pines():
    """Return original 9 strategies + 3 Pine strategies."""
    from strategies import get_all_strategies
    return get_all_strategies() + get_pine_strategies()