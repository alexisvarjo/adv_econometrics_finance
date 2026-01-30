#!/usr/bin/env python3
"""
fig4.py — Replicate Figure 4 (Moreira & Muir, 2017): Utility benefits and leverage constraints

Key fix vs. the "negative constant" issue:
- Unrestricted ΔU% is computed using the Sharpe-ratio identity:
    ΔU% = 100 * (SR_new^2 - SR_old^2) / SR_old^2
  because the unrestricted volatility-managed allocation is MV-optimal and the gain is γ-invariant.

- Leverage-capped lines are computed from the MV utility ratio:
    100 * ( U(min(w_t, wbar)) / U(min(w, wbar)) - 1 )
  where w = mu/(gamma*sigma^2) is the target buy-and-hold weight, and
        w_t = w * sigma^2 / sigma_t^2 with sigma_t^2 proxied by realized variance from the prior month.

Data:
  - ../data/F-F_Research_Data_Factors_daily.csv (Ken French export; daily Mkt-RF in percent)

Output:
  - ../figures/figure4.pdf

Volatility estimator is modular: edit estimate_monthly_sigma2(...) to explore alternatives.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================
# Paths
# =============================
DATA_DIR = "../data"
FIG_DIR = "../figures"
FF_DAILY = os.path.join(DATA_DIR, "F-F_Research_Data_Factors_daily.csv")


# =============================
# Loading / preprocessing
# =============================
def parse_french_daily_mktrf(path: str) -> pd.Series:
    """Load Ken French daily CSV export; return Mkt-RF in decimals indexed by date."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    raw = pd.read_csv(path, header=None)
    c0 = raw.iloc[:, 0].astype(str).str.strip()
    mask = c0.str.fullmatch(r"\d{8}")

    df = raw.loc[mask, [0, 1]].copy()
    df.columns = ["date", "Mkt-RF"]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["Mkt-RF"] = pd.to_numeric(df["Mkt-RF"], errors="coerce") / 100.0

    df = df.dropna(subset=["date", "Mkt-RF"]).set_index("date").sort_index()
    return df["Mkt-RF"]


def compound_monthly_return(daily_ret: pd.Series) -> pd.Series:
    """Daily decimal -> monthly decimal via compounding (month-end index)."""
    x = daily_ret.dropna().sort_index()
    return x.resample("ME").apply(lambda z: (1.0 + z).prod() - 1.0)


# =============================
# Modular variance proxy (sigma_t^2)
# =============================
def estimate_monthly_sigma2(
    daily_ret: pd.Series,
    method: str = "eq2_22demeaned_to_monthly",
    min_obs: int = 15,
    n_target: int = 22,
) -> pd.Series:
    """
    Returns a MONTHLY variance proxy sigma_t^2 (decimal^2), indexed at month-end.

    Default method matches the paper's Eq.(2) spirit:
      RV_t^2 = (1/22) Σ_{d=1..22} ( f_{t,d} - mean_t )^2     [daily variance proxy]
    Then convert to a monthly variance proxy by multiplying by 22:
      sigma_t^2 ≈ 22 * RV_t^2

    Practical details:
    - We compute within each calendar month using available daily obs.
    - If a month has fewer than min_obs daily returns, we return NaN for that month.
    - We still scale by n_target=22 to keep the unit convention stable.

    Alternative methods can be plugged in later by adding branches.
    """
    x = daily_ret.dropna().sort_index()

    if method == "eq2_22demeaned_to_monthly":
        def _sigma2_month(z: pd.Series) -> float:
            n = z.shape[0]
            if n < min_obs:
                return np.nan
            m = z.mean()
            rv_daily = np.mean((z - m) ** 2)          # ~ Eq(2) daily variance proxy (no *n)
            sigma2 = n_target * rv_daily              # convert to monthly variance proxy
            return sigma2

        return x.resample("ME").apply(_sigma2_month)

    raise ValueError(f"Unknown variance method: {method}")


# =============================
# Mean-variance utility + ΔU% for capped cases
# =============================
def mv_utility(port_ret: pd.Series, gamma: float) -> float:
    """Sample MV utility: E[r] - (gamma/2) Var[r]."""
    r = port_ret.dropna()
    if r.empty:
        return np.nan
    mu = float(r.mean())
    v = float(r.var(ddof=1))
    return mu - 0.5 * gamma * v


def deltaU_capped_percent(
    r_m: pd.Series,
    sigma2_m: pd.Series,
    w_target: float,
    wbar: float,
    mu_uncond: float,
    sigma2_uncond: float,
) -> float:
    """
    Capped ΔU%:
      100 * ( U(min(w_t,wbar)) / U(min(w,wbar)) - 1 )

    Where:
      w = w_target = mu/(gamma*sigma^2)  => gamma = mu/(w*sigma^2)
      w_t = w * sigma^2 / sigma_{t-1}^2
    """
    if w_target <= 0 or not np.isfinite(w_target):
        return np.nan

    # implied gamma from target weight definition
    gamma = mu_uncond / (w_target * sigma2_uncond) if sigma2_uncond > 0 else np.nan
    if not np.isfinite(gamma) or gamma <= 0:
        return np.nan

    # use lagged conditional variance (sigma_{t-1}^2)
    df = pd.concat([r_m, sigma2_m.shift(1)], axis=1).dropna()
    df.columns = ["r", "sigma2_lag"]

    # avoid insane weights when sigma2 is (near) zero
    df = df[df["sigma2_lag"] > 0]

    if df.empty:
        return np.nan

    w_t = w_target * (sigma2_uncond / df["sigma2_lag"])
    w_t_cap = np.minimum(w_t, wbar)
    w_bh_cap = min(w_target, wbar)

    r_vm = w_t_cap * df["r"]
    r_bh = w_bh_cap * df["r"]

    U_vm = mv_utility(r_vm, gamma)
    U_bh = mv_utility(r_bh, gamma)

    if not np.isfinite(U_vm) or not np.isfinite(U_bh) or U_bh == 0:
        return np.nan

    return 100.0 * (U_vm / U_bh - 1.0)


# =============================
# Unrestricted ΔU% from Sharpe ratios
# =============================
def sharpe_ratio(x: pd.Series) -> float:
    r = x.dropna()
    if r.empty:
        return np.nan
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    return mu / sd if sd > 0 else np.nan


def deltaU_unrestricted_percent(r_m: pd.Series, sigma2_m: pd.Series) -> float:
    """
    Unrestricted ΔU% using Eq.(4):
      ΔU% = 100 * (SR_new^2 - SR_old^2) / SR_old^2

    We construct the volatility-managed *return* series (paper Eq.(1)):
      r^σ_t = (c / sigma^2_{t-1}) * r_t
    with c chosen so that std(r^σ) == std(r).
    """
    df = pd.concat([r_m, sigma2_m.shift(1)], axis=1).dropna()
    df.columns = ["r", "sigma2_lag"]
    df = df[df["sigma2_lag"] > 0]
    if df.empty:
        return np.nan

    # raw managed return
    r_sig_raw = (1.0 / df["sigma2_lag"]) * df["r"]

    # normalize by c to match unconditional volatility of r
    sd_r = float(df["r"].std(ddof=1))
    sd_raw = float(r_sig_raw.std(ddof=1))
    c = sd_r / sd_raw if sd_raw > 0 else 1.0

    r_sig = c * r_sig_raw

    sr_old = sharpe_ratio(df["r"])
    sr_new = sharpe_ratio(r_sig)

    if not np.isfinite(sr_old) or sr_old == 0 or not np.isfinite(sr_new):
        return np.nan

    return 100.0 * ((sr_new ** 2 - sr_old ** 2) / (sr_old ** 2))


# =============================
# Main
# =============================
def main():
    # Daily market excess returns
    mktrf_d = parse_french_daily_mktrf(FF_DAILY)

    # Monthly market excess returns
    r_m = compound_monthly_return(mktrf_d)

    # Monthly conditional variance proxy (sigma_t^2) from daily data
    sigma2_m = estimate_monthly_sigma2(
        mktrf_d,
        method="eq2_22demeaned_to_monthly",
        min_obs=15,
        n_target=22,
    )

    # Align sample for unconditional moments
    base = pd.concat([r_m, sigma2_m], axis=1).dropna()
    r_m = base.iloc[:, 0]
    sigma2_m = base.iloc[:, 1]

    mu_uncond = float(r_m.mean())
    sigma2_uncond = float(r_m.var(ddof=1))

    if not np.isfinite(mu_uncond) or not np.isfinite(sigma2_uncond) or sigma2_uncond <= 0:
        raise RuntimeError("Unconditional moments not well-defined; check return series.")

    # x-axis: target buy-and-hold weight w = mu/(gamma*sigma^2)
    w_grid = np.linspace(0.05, 1.5, 120)

    # Unrestricted line: constant from SR identity
    du_un = deltaU_unrestricted_percent(r_m, sigma2_m)
    y_un = np.full_like(w_grid, du_un, dtype=float)

    # Leverage constraints: compute capped utility ratio
    y_15 = np.array([deltaU_capped_percent(r_m, sigma2_m, w, 1.5, mu_uncond, sigma2_uncond) for w in w_grid], dtype=float)
    y_10 = np.array([deltaU_capped_percent(r_m, sigma2_m, w, 1.0, mu_uncond, sigma2_uncond) for w in w_grid], dtype=float)

    # Plot
    fig, ax = plt.subplots(figsize=(8.4, 6.3))

    ax.plot(w_grid, y_un, label="Unrestricted", linewidth=2.2)
    ax.plot(w_grid, y_15, label="Leverage<1.5", linestyle="--", linewidth=2.2)
    ax.plot(w_grid, y_10, label="Leverage<1", linestyle=":", marker="o", markersize=5, linewidth=1.6)

    ax.set_title("Percentage Utility Gain From Volatility Timing, Δ U %")
    ax.set_xlabel(r"Target buy-and-hold weight  $\frac{\mu}{\gamma\,\sigma^2}$")
    ax.set_ylabel("Δ U %")
    ax.set_xlim(float(w_grid.min()), float(w_grid.max()))

    # Paper figure shows up to ~100; keep same
    ax.set_ylim(0, 100)

    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "figure4.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")

    print(f"Saved Figure 4 to: {out_path}")
    print(f"Unrestricted ΔU% (SR-identity): {du_un:.2f}")


if __name__ == "__main__":
    main()
