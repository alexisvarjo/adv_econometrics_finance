#!/usr/bin/env python3
"""
fig1.py — Replicate Figure 1 (Moreira & Muir, 2017): Sorts on the previous month's volatility.

Data:
  - ../data/F-F_Research_Data_Factors_daily.csv  (Ken French daily 3 factors; uses Mkt-RF, SMB, HML, RF)
  - ../data/USREC.csv                           (monthly NBER recession dummy from FRED)

Output:
  - ../figures/figure1.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def parse_ff_daily_factors(path: str) -> pd.DataFrame:
    """
    Load Ken French 'F-F_Research_Data_Factors_daily.csv' robustly.

    The French library CSV often has a few header lines + a footer.
    This function reads it by coercing the first column to YYYYMMDD dates and
    discarding non-date rows.

    Returns a DataFrame indexed by datetime with columns:
      ['Mkt-RF','SMB','HML','RF'] in DECIMAL units.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing factors file: {path}")

    raw = pd.read_csv(path, header=None)
    # Find rows where first column looks like an 8-digit date
    c0 = raw.iloc[:, 0].astype(str).str.strip()
    mask = c0.str.fullmatch(r"\d{8}")
    df = raw.loc[mask].copy()

    # Name columns (French file is typically: date, Mkt-RF, SMB, HML, RF)
    df.columns = ["date", "Mkt-RF", "SMB", "HML", "RF"] + [f"extra_{i}" for i in range(df.shape[1] - 5)]
    df = df[["date", "Mkt-RF", "SMB", "HML", "RF"]]

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    for c in ["Mkt-RF", "SMB", "HML", "RF"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0  # percent -> decimal

    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df


def compound_monthly_return(daily_ret: pd.Series) -> pd.Series:
    """Daily (decimal) -> monthly (decimal) via compounding."""
    daily_ret = daily_ret.dropna().sort_index()
    return daily_ret.resample("M").apply(lambda x: (1.0 + x).prod() - 1.0)


def realized_monthly_vol_annualized(daily_ret: pd.Series) -> pd.Series:
    """
    Monthly realized volatility (annualized, decimal units) from daily returns.

    Within each month:
      rv2 = mean_d (r_d - mean_month)^2
      monthly_sd = sqrt(rv2 * N)
      annualized_vol = monthly_sd * sqrt(12)
    """
    daily_ret = daily_ret.dropna().sort_index()

    def _month_vol(x: pd.Series) -> float:
        n = x.shape[0]
        if n < 2:
            return np.nan
        m = x.mean()
        rv2 = ((x - m) ** 2).mean()
        monthly_sd = np.sqrt(rv2 * n)
        return monthly_sd * np.sqrt(12)

    return daily_ret.resample("M").apply(_month_vol)


# -----------------------------
# Load USREC (monthly recession dummy)
# -----------------------------
if not os.path.exists("../data/USREC.csv"):
    raise FileNotFoundError("Missing recession file: ../data/USREC.csv")

usrec = pd.read_csv("../data/USREC.csv")
if "observation_date" not in usrec.columns or "USREC" not in usrec.columns:
    raise ValueError("../data/USREC.csv must have columns ['observation_date','USREC'].")

usrec["date"] = pd.to_datetime(usrec["observation_date"], errors="coerce")
usrec["USREC"] = pd.to_numeric(usrec["USREC"], errors="coerce")
usrec["month"] = usrec["date"].dt.to_period("M")
usrec_m = (
    usrec.dropna(subset=["month", "USREC"])
         .drop_duplicates("month")
         .set_index("month")["USREC"]
         .astype(float)
)

# -----------------------------
# Load Ken French daily factors (3 factors)
# -----------------------------
ff = parse_ff_daily_factors("../data/F-F_Research_Data_Factors_daily.csv")

# Use Mkt-RF for the figure (as in the paper)
mktrf_d = ff["Mkt-RF"].dropna()

# Monthly realized vol (t) and monthly return (t)
mktrf_m = compound_monthly_return(mktrf_d)           # decimal
vol_m = realized_monthly_vol_annualized(mktrf_d)     # annualized decimal

# Sort on previous month's volatility (lagged)
vol_lag = vol_m.shift(1)

# Build main sort df: next-month returns matched to prior-month vol
df = pd.DataFrame({"ret_m": mktrf_m, "vol_lag": vol_lag}).dropna()
df["bucket"] = pd.qcut(df["vol_lag"], q=5, labels=[1, 2, 3, 4, 5])

# -----------------------------
# Return-side panels (next-month return properties by bucket)
# -----------------------------
bucket_stats = (
    df.groupby("bucket")
      .agg(
          mean_ret=("ret_m", "mean"),
          std_ret=("ret_m", "std"),
          var_ret=("ret_m", "var"),
          n=("ret_m", "count"),
      )
)

# Annualize mean and std for plotting scale
bucket_stats["mean_ret_ann_pct"] = bucket_stats["mean_ret"] * 12 * 100
bucket_stats["std_ret_ann_pct"] = bucket_stats["std_ret"] * np.sqrt(12) * 100

# E[R]/Var(R): monthly decimals (no annualization, no percent scaling)
bucket_stats["mean_over_var"] = bucket_stats["mean_ret"] / bucket_stats["var_ret"]

# -----------------------------
# Recession panel (contemporaneous with volatility month)
#   Bucket on vol_m in month t, compute mean(USREC_t) within buckets.
# -----------------------------
vol_m_month = vol_m.copy()
vol_m_month.index = vol_m_month.index.to_period("M")

df_rec = pd.DataFrame({"vol": vol_m_month}).join(usrec_m, how="inner").dropna()
df_rec["bucket"] = pd.qcut(df_rec["vol"], q=5, labels=[1, 2, 3, 4, 5])

p_rec = df_rec.groupby("bucket")["USREC"].mean()
p_rec_vals = p_rec.reindex([1, 2, 3, 4, 5]).values

# -----------------------------
# Prepare plot inputs
# -----------------------------
x_labels = ["Low Vol", "2", "3", "4", "High Vol"]
x = np.arange(5)

avg_ret = bucket_stats["mean_ret_ann_pct"].values
std_ret = bucket_stats["std_ret_ann_pct"].values
m_over_v = bucket_stats["mean_over_var"].values

# -----------------------------
# Plot Figure 1 (2x2)
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
(ax1, ax2), (ax3, ax4) = axes

bar_kws = dict(edgecolor="black", linewidth=0.6)

ax1.bar(x, avg_ret, **bar_kws)
ax1.set_title("Average Return")
ax1.set_xticks(x, x_labels)
ax1.set_ylim(0, max(12, np.nanmax(avg_ret) * 1.15))

ax2.bar(x, std_ret, **bar_kws)
ax2.set_title("Standard Deviation")
ax2.set_xticks(x, x_labels)
ax2.set_ylim(0, max(40, np.nanmax(std_ret) * 1.15))

ax3.bar(x, m_over_v, **bar_kws)
ax3.set_title("E[R]/Var(R)")
ax3.set_xticks(x, x_labels)
ax3.set_ylim(0, max(8, np.nanmax(m_over_v) * 1.15))

ax4.bar(x, p_rec_vals, **bar_kws)
ax4.set_title("Probability of Recession")
ax4.set_xticks(x, x_labels)
ax4.set_ylim(0, 0.5)

for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(False)

plt.tight_layout()

# -----------------------------
# Save PDF
# -----------------------------
out_dir = "../figures"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "figure1.pdf")
fig.savefig(out_path, format="pdf", bbox_inches="tight")
print(f"Saved Figure 1 to: {out_path}")

# -----------------------------
# Diagnostics
# -----------------------------
print("\nReturn-panel bucket counts (next-month returns sorted by lagged vol):")
print(bucket_stats["n"])

print("\nProbability of recession by contemporaneous vol bucket:")
print(p_rec)

print("\nSample start/end (daily Mkt-RF):", mktrf_d.index.min().date(), "to", mktrf_d.index.max().date())
