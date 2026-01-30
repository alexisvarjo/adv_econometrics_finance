#!/usr/bin/env python3
"""
fig3.py — Replicate Figure 3 (Moreira & Muir, 2017): Volatility-managed market portfolio.

Uses Ken French daily factors to construct:
  - Buy-and-hold market excess return (Mkt-RF)
  - Volatility-managed market excess return: f^{σ}_{t+1} = (c / \hat{σ}^2_t) f_{t+1}

Key design choice:
  Volatility/variance estimation is isolated behind a single function call, so you can
  easily swap estimators later.

Inputs:
  - ../data/F-F_Research_Data_Factors_daily.csv  (daily; 1926+; columns Mkt-RF, SMB, HML, RF)

Output:
  - ../figures/figure3.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Robust loader for Ken French CSV
# -----------------------------
def parse_french_factors_daily(path: str) -> pd.DataFrame:
    """
    Robustly loads 'F-F_Research_Data_Factors_daily.csv' exported from Ken French library.
    Keeps only rows where first column is an 8-digit date (YYYYMMDD), discards header/footer.

    Returns DataFrame indexed by datetime with columns in DECIMAL units:
      ['Mkt-RF','SMB','HML','RF']
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    raw = pd.read_csv(path, header=None)
    c0 = raw.iloc[:, 0].astype(str).str.strip()
    mask = c0.str.fullmatch(r"\d{8}")
    df = raw.loc[mask].iloc[:, :5].copy()
    df.columns = ["date", "Mkt-RF", "SMB", "HML", "RF"]

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    for c in ["Mkt-RF", "SMB", "HML", "RF"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0  # percent -> decimal

    return df.dropna(subset=["date"]).set_index("date").sort_index()


# -----------------------------
# Returns aggregation utilities
# -----------------------------
def compound_monthly_return(daily_ret: pd.Series) -> pd.Series:
    """Daily (decimal) -> monthly (decimal) via compounding: Π(1+r_d)-1"""
    daily_ret = daily_ret.dropna().sort_index()
    return daily_ret.resample("M").apply(lambda x: (1.0 + x).prod() - 1.0)


# -----------------------------
# Volatility estimator interface (swap this easily)
# -----------------------------
def estimate_variance(daily_excess_ret: pd.Series, method: str = "rolling22") -> pd.Series:
    """
    Return a MONTHLY variance proxy \hat{σ}^2_t indexed at month-end.

    The managed return at month t+1 uses \hat{σ}^2_t (i.e., this series will be shifted by +1 month
    when forming the strategy).

    Parameters
    ----------
    daily_excess_ret : pd.Series
        Daily excess returns in decimals.
    method : str
        Currently supported:
          - "rolling22": Eq (2)-style variance using last 22 trading days ending each day,
                         sampled at month-end.

    Returns
    -------
    pd.Series
        Monthly variance proxy at month-end (decimal^2).
    """
    x = daily_excess_ret.dropna().sort_index()

    if method == "rolling22":
        # Eq (2): 1/22 * sum_{d=1..22} (f_{t+d} - mean)^2 over last 22 trading days
        def _rv2(arr: np.ndarray) -> float:
            m = arr.mean()
            return np.mean((arr - m) ** 2)

        # daily rolling variance proxy (demeaned within window)
        rv2_daily = x.rolling(window=22, min_periods=22).apply(_rv2, raw=True)

        # take month-end value (variance estimate for that month)
        rv2_monthly = rv2_daily.resample("M").last()
        return rv2_monthly

    raise ValueError(f"Unknown variance estimator method: {method}")


# -----------------------------
# Strategy construction
# -----------------------------
def build_vol_managed_strategy(
    daily_excess_ret: pd.Series,
    var_method: str = "rolling22",
) -> pd.DataFrame:
    """
    Builds buy-and-hold and volatility-managed MONTHLY excess return series.

    Managed portfolio:
      f^{σ}_{t+1} = (c / \hat{σ}^2_t) * f_{t+1}
    where c is chosen so std(managed) == std(buy_and_hold), using full sample.

    Returns a DataFrame indexed by month-end with columns:
      ['bh', 'vm', 'weight', 'varhat']
    """
    # Monthly excess returns f_{t} (buy-and-hold)
    bh = compound_monthly_return(daily_excess_ret)

    # Monthly variance proxy \hat{σ}^2_t at month-end
    varhat = estimate_variance(daily_excess_ret, method=var_method)

    # Align: managed return at month t uses varhat_{t-1}
    df = pd.DataFrame({"bh": bh, "varhat": varhat}).dropna()

    # weight_{t-1} applied to bh_t
    weight = 1.0 / df["varhat"].shift(1)
    vm_raw = weight * df["bh"]

    # Drop first month (no lagged var)
    out = pd.DataFrame({"bh": df["bh"], "vm_raw": vm_raw, "weight": weight, "varhat": df["varhat"]}).dropna()

    # Scale constant c so that unconditional monthly std matches
    std_bh = out["bh"].std(ddof=1)
    std_vm_raw = out["vm_raw"].std(ddof=1)
    c = std_bh / std_vm_raw if std_vm_raw and np.isfinite(std_vm_raw) else 1.0

    out["vm"] = c * out["vm_raw"]
    out = out.drop(columns=["vm_raw"])

    # Keep c around for debugging if desired
    out.attrs["scale_c"] = c
    out.attrs["std_bh"] = std_bh
    out.attrs["std_vm_raw"] = std_vm_raw
    return out


# -----------------------------
# Plot utilities
# -----------------------------
def cumulative_wealth(monthly_ret: pd.Series, start: float = 1.0) -> pd.Series:
    """Wealth index from monthly returns."""
    r = monthly_ret.dropna().sort_index()
    w = (1.0 + r).cumprod()
    return start * w


def rolling_1y_return(monthly_ret: pd.Series, window: int = 12) -> pd.Series:
    """Rolling 12-month compounded return: Π(1+r)-1."""
    r = monthly_ret.dropna().sort_index()
    return (1.0 + r).rolling(window).apply(np.prod, raw=True) - 1.0


def drawdown(wealth: pd.Series) -> pd.Series:
    """Drawdown level as wealth / running_peak (1 = no drawdown)."""
    w = wealth.dropna().sort_index()
    peak = w.cummax()
    return w / peak


# -----------------------------
# Main
# -----------------------------
ff = parse_french_factors_daily("../data/F-F_Research_Data_Factors_daily.csv")
mktrf_d = ff["Mkt-RF"].dropna()  # daily excess market return in decimals

# Build monthly strategies (easy to swap variance estimator here)
STRATEGY_VAR_METHOD = "rolling22"
res = build_vol_managed_strategy(mktrf_d, var_method=STRATEGY_VAR_METHOD)

bh_m = res["bh"]
vm_m = res["vm"]

# Performance series
w_bh = cumulative_wealth(bh_m)
w_vm = cumulative_wealth(vm_m)

roll_bh = rolling_1y_return(bh_m, 12)
roll_vm = rolling_1y_return(vm_m, 12)

dd_bh = drawdown(w_bh)
dd_vm = drawdown(w_vm)

# -----------------------------
# Plot layout to match Figure 3
# -----------------------------
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.3], hspace=0.32, wspace=0.25)

ax_top = fig.add_subplot(gs[0, :])
ax_bl = fig.add_subplot(gs[1, 0])
ax_br = fig.add_subplot(gs[1, 1])

# Top: cumulative performance (log scale)
ax_top.plot(w_bh.index, w_bh.values, label="Buy and hold", linewidth=1.5)
ax_top.plot(w_vm.index, w_vm.values, label="Volatility managed", linewidth=1.5, linestyle="--")
ax_top.set_title("Cumulative performance")
ax_top.set_yscale("log")
ax_top.set_xlabel("")
ax_top.legend(loc="upper left", frameon=True)

# Bottom-left: rolling one-year returns
ax_bl.plot(roll_bh.index, roll_bh.values, label="Buy and hold", linewidth=1.2)
ax_bl.plot(roll_vm.index, roll_vm.values, label="Volatility managed", linewidth=1.2, linestyle="--")
ax_bl.set_title("One-Year Rolling Returns")
ax_bl.legend(loc="upper left", frameon=True)

# Bottom-right: drawdowns (wealth/peak)
ax_br.plot(dd_bh.index, dd_bh.values, label="Buy and hold", linewidth=1.2)
ax_br.plot(dd_vm.index, dd_vm.values, label="Volatility managed", linewidth=1.2, linestyle="--")
ax_br.set_title("Drawdowns")
ax_br.set_ylim(0.1, 1.0)
ax_br.legend(loc="lower right", frameon=True)

# Aesthetics
for ax in [ax_top, ax_bl, ax_br]:
    ax.grid(False)

plt.tight_layout()

# -----------------------------
# Save as PDF for LaTeX
# -----------------------------
out_dir = "../figures"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "figure3.pdf")
fig.savefig(out_path, format="pdf", bbox_inches="tight")
print(f"Saved Figure 3 to: {out_path}")

# Diagnostics
print("\nDiagnostics:")
print(f"Variance estimator: {STRATEGY_VAR_METHOD}")
print(f"Scale constant c: {res.attrs.get('scale_c')}")
print(f"Std(buy&hold) monthly: {res.attrs.get('std_bh')}")
print(f"Std(managed, raw) monthly: {res.attrs.get('std_vm_raw')}")
print(f"Sample (monthly): {bh_m.index.min().date()} to {bh_m.index.max().date()}")
