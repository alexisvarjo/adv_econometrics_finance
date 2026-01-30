#!/usr/bin/env python3
"""
table4.py — Transaction Costs of Volatility Timing (Table IV, Moreira & Muir 2017)

Market-only. Uses:
- Fama-French daily Mkt-RF (excess return) from 1926+
- FRED VIXCLS daily from 1990+
- Monthly volatility timing weights based on realized variance, plus variants
- Transaction costs proportional to turnover |Δw| (monthly)

Outputs:
  ../tables/table4.tex

Notes on samples:
- Gross, 1bp, 10bp columns use full available sample (1926+).
- VIX-adjusted column uses overlap where VIX exists (1990+). This is economically correct and avoids
  implicitly extrapolating VIX back before it existed.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# =============================
# Paths
# =============================
DATA_DIR = "../data"
OUT_DIR = "../tables"

FF_DAILY = os.path.join(DATA_DIR, "F-F_Research_Data_Factors_daily.csv")
VIX_PATH = os.path.join(DATA_DIR, "VIXCLS.csv")

# =============================
# Parameters
# =============================
ANNUALIZE = 12

COST_1BP = 0.0001
COST_10BP = 0.0010

# VIX-dependent add-on: calibrated so moving VIX from 20 -> 40 adds 4bps
# i.e. slope = 0.0002 per VIX point above 20
VIX_BASE = 20.0
VIX_SLOPE = 0.0002

# =============================
# Loaders
# =============================
def load_ff_market_daily() -> pd.Series:
    """Return daily Mkt-RF excess return in decimals, indexed by date."""
    raw = pd.read_csv(FF_DAILY, header=None)
    c0 = raw.iloc[:, 0].astype(str).str.strip()
    mask = c0.str.fullmatch(r"\d{8}")
    df = raw.loc[mask, [0, 1]]
    df.columns = ["date", "MktRF"]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["MktRF"] = pd.to_numeric(df["MktRF"], errors="coerce") / 100.0
    df = df.dropna(subset=["date", "MktRF"]).set_index("date").sort_index()
    return df["MktRF"]


def load_vix_daily() -> pd.Series:
    """Return daily VIX level (not returns), indexed by date."""
    vix = pd.read_csv(VIX_PATH)
    vix["date"] = pd.to_datetime(vix["observation_date"], errors="coerce")
    vix["VIX"] = pd.to_numeric(vix["VIXCLS"], errors="coerce")
    vix = vix.dropna(subset=["date", "VIX"]).set_index("date").sort_index()
    return vix["VIX"]


# =============================
# Time aggregation helpers
# =============================
def month_end(s: pd.Series) -> pd.Series:
    """Force month-end index (Timestamp at month end)."""
    x = s.dropna().sort_index()
    if isinstance(x.index, pd.PeriodIndex):
        x.index = x.index.to_timestamp("ME")
    return x


def monthly_compound(daily_ret: pd.Series) -> pd.Series:
    """Compound daily returns into month returns, month-end indexed."""
    x = daily_ret.dropna().sort_index()
    return x.resample("ME").apply(lambda z: (1.0 + z).prod() - 1.0)


def realized_variance_monthly(daily_ret: pd.Series) -> pd.Series:
    """
    Monthly realized variance proxy (decimal^2) using within-month de-meaned RV:
      RV_t^2 = mean_d (r_{t,d} - mean_t)^2 * N_t
    month-end indexed.
    """
    x = daily_ret.dropna().sort_index()

    def _rv(z: pd.Series) -> float:
        n = z.shape[0]
        if n < 2:
            return np.nan
        m = z.mean()
        return np.mean((z - m) ** 2) * n

    return x.resample("ME").apply(_rv)


def expected_variance_ar1(log_var: pd.Series) -> pd.Series:
    """
    Fit AR(1) on log variance and produce fitted (in-sample) expected variance.
      log(var_t) = a + b log(var_{t-1}) + u_t
    Returns exp(fitted), aligned to var_t timestamps.
    """
    y = log_var.dropna()
    y_lag = y.shift(1)
    df = pd.concat([y, y_lag], axis=1).dropna()
    df.columns = ["y", "y_lag"]
    X = sm.add_constant(df["y_lag"])
    res = sm.OLS(df["y"], X).fit()
    fitted = res.predict(X)
    out = pd.Series(np.exp(fitted), index=df.index, name="E_var")
    return out


# =============================
# Portfolio construction & evaluation
# =============================
def normalize_weights_to_match_vol(w: pd.Series, r: pd.Series) -> pd.Series:
    """
    Choose scalar c so that std(c*w*r) == std(r) on overlapping sample.
    Return c*w.
    """
    df = pd.concat([w, r], axis=1).dropna()
    ww = df.iloc[:, 0]
    rr = df.iloc[:, 1]
    strat = ww * rr
    s_rr = rr.std(ddof=1)
    s_strat = strat.std(ddof=1)
    if not np.isfinite(s_strat) or s_strat == 0:
        c = 1.0
    else:
        c = s_rr / s_strat
    return c * w


def alpha_hc1(y: pd.Series, x: pd.Series):
    """
    Regress y on constant + x, HC1 SE.
    Returns alpha, se(alpha), nobs, r2, rmse
    """
    df = pd.concat([y, x], axis=1).dropna()
    yy = df.iloc[:, 0]
    xx = sm.add_constant(df.iloc[:, 1], has_constant="add")
    res = sm.OLS(yy.values, xx.values).fit(cov_type="HC1")
    alpha = float(res.params[0])
    se = float(res.bse[0])
    n = int(res.nobs)
    r2 = float(res.rsquared)
    rmse = float(np.sqrt(np.mean(res.resid ** 2)))
    return alpha, se, n, r2, rmse


def sharpe_monthly(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty or r.std(ddof=1) == 0:
        return np.nan
    return np.sqrt(12) * r.mean() / r.std(ddof=1)


def evaluate_strategy(
    w_raw: pd.Series,
    r_m: pd.Series,
    vix_m: pd.Series | None,
):
    """
    Evaluate:
    - Gross returns: w_{t-1} * r_t
    - Turnover: mean |Δw|
    - Alpha from regressing managed returns on buy-and-hold returns
    - Costs: 1bp, 10bp on full sample; VIX-adjusted on overlap with VIX
    - Break-even bps: alpha / mean(|Δw|) converted to bps
    """
    w_raw = month_end(w_raw)
    r_m = month_end(r_m)

    # Use lagged weight for return in month t (decided at end of t-1)
    w_lag = w_raw.shift(1)

    # Normalize exposure so managed portfolio has same unconditional volatility as buy-and-hold
    w = normalize_weights_to_match_vol(w_lag, r_m)

    # Align and compute gross
    df = pd.concat([w, r_m], axis=1).dropna()
    df.columns = ["w", "r"]
    gross = df["w"] * df["r"]

    # Turnover uses contemporaneous change in w (month to month)
    dw = df["w"].diff().abs()
    avg_turnover = float(dw.mean(skipna=True))

    # Expected return (annualized, percent)
    ER = float(gross.mean() * 12 * 100)

    # Alpha (annualized percent) from regressing managed on buy&hold
    a, se, n, r2, rmse = alpha_hc1(gross * 12 * 100, df["r"] * 12 * 100)

    # Net under constant costs (full sample where dw exists)
    net_1bp = gross - COST_1BP * dw
    net_10bp = gross - COST_10BP * dw

    a1, _, _, _, _ = alpha_hc1(net_1bp.dropna() * 12 * 100, df.loc[net_1bp.dropna().index, "r"] * 12 * 100)
    a10, _, _, _, _ = alpha_hc1(net_10bp.dropna() * 12 * 100, df.loc[net_10bp.dropna().index, "r"] * 12 * 100)

    # VIX-adjusted costs on overlap sample only
    a_vix = np.nan
    if vix_m is not None:
        vix_m = month_end(vix_m)
        d2 = df.join(vix_m.rename("VIX"), how="inner")
        dw2 = d2["w"].diff().abs()
        kappa = COST_10BP + VIX_SLOPE * np.maximum(d2["VIX"] - VIX_BASE, 0.0)
        net_vix = (d2["w"] * d2["r"]) - kappa * dw2
        net_vix = net_vix.dropna()
        if not net_vix.empty:
            a_vix, _, _, _, _ = alpha_hc1(net_vix * 12 * 100, d2.loc[net_vix.index, "r"] * 12 * 100)

    # Break-even bps (cost per unit turnover needed to drive alpha to zero)
    # approx: alpha_net = alpha_gross - kappa*E|dw|  => kappa_be = alpha_gross / E|dw|
    # alpha_gross is in percent/year; convert to decimal/year before dividing:
    be_bps = np.nan
    if avg_turnover and np.isfinite(avg_turnover) and avg_turnover > 0:
        alpha_dec_per_year = (a / 100.0)
        kappa_be = alpha_dec_per_year / avg_turnover
        be_bps = kappa_be * 10000.0  # decimal -> bps

    return {
        "turnover": avg_turnover,
        "ER": ER,
        "alpha": a,
        "alpha_1bp": a1,
        "alpha_10bp": a10,
        "alpha_vix": a_vix,
        "be_bps": be_bps,
        "n": n,
        "r2": r2,
        "rmse": rmse,
    }


# =============================
# Strategies
# =============================
def build_strategies(rv: pd.Series, exp_rv: pd.Series):
    """
    Return dict of strategy label -> raw weight series (month-end indexed)
    Raw weights get normalized later to match volatility.
    """
    rv = month_end(rv)
    exp_rv = month_end(exp_rv)

    out = {}

    # 1) Realized variance
    out["Realized variance"] = 1.0 / rv

    # 2) Realized vol
    out["Realized vol"] = 1.0 / np.sqrt(rv)

    # 3) Expected variance (AR(1) log variance forecast)
    out["Expected variance"] = 1.0 / exp_rv

    # 4) No leverage cap at 1
    out["No leverage"] = np.minimum(1.0 / rv, 1.0)

    # 5) 50% leverage (cap at 1.5)
    out["50% leverage"] = np.minimum(1.0 / rv, 1.5)

    return out


# =============================
# Table formatting
# =============================
def fmt(x, digits=2, pct=False, suffix=""):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    if pct:
        return f"{x:.{digits}f}%{suffix}"
    return f"{x:.{digits}f}{suffix}"


def to_latex_table(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame(
        rows,
        columns=[
            "Description",
            r"$|\Delta w|$",
            r"$E[R]$",
            r"$\alpha$",
            r"$\alpha$ (1bp)",
            r"$\alpha$ (10bps)",
            r"$\alpha$ (VIX adj)",
            "Break Even (bps)",
        ],
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by table4.py\n\n")
        f.write(df.to_latex(index=False, escape=False))
    print(f"Saved: {out_path}")


# =============================
# Main
# =============================
def main():
    r_d = load_ff_market_daily()
    vix_d = load_vix_daily()

    # Monthly market return and realized variance
    r_m = monthly_compound(r_d)
    rv = realized_variance_monthly(r_d)

    # Expected variance via AR(1) on log RV
    exp_rv = expected_variance_ar1(np.log(rv))

    # Monthly VIX (end-of-month). Will start 1990+.
    vix_m = vix_d.resample("ME").last()

    strategies = build_strategies(rv, exp_rv)

    rows = []
    for desc, w_raw in strategies.items():
        res = evaluate_strategy(w_raw, r_m, vix_m)
        rows.append([
            desc,
            fmt(res["turnover"], 2),
            fmt(res["ER"], 2, pct=True),
            fmt(res["alpha"], 2, pct=True),
            fmt(res["alpha_1bp"], 2, pct=True),
            fmt(res["alpha_10bp"], 2, pct=True),
            fmt(res["alpha_vix"], 2, pct=True),
            fmt(res["be_bps"], 0),
        ])

    out_path = os.path.join(OUT_DIR, "table4.tex")
    to_latex_table(rows, out_path)


if __name__ == "__main__":
    main()
