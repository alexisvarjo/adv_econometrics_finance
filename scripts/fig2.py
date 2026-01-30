import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def _to_datetime_yyyymmdd(s: pd.Series) -> pd.DatetimeIndex:
    s = s.astype(str).str.strip()
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def _to_datetime_mdy(s: pd.Series) -> pd.DatetimeIndex:
    # Handles '11/30/1983' style (month/day/year)
    return pd.to_datetime(s.astype(str).str.strip(), format="%m/%d/%Y", errors="coerce")


def _pct_str_to_decimal(s: pd.Series) -> pd.Series:
    # Converts '-1.30%' -> -0.013
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False), errors="coerce") / 100.0


def realized_monthly_vol_annualized(daily_ret: pd.Series) -> pd.Series:
    """
    Monthly realized volatility (annualized) computed from daily returns within each calendar month.

    RV^2_t = mean_d (r_{t,d} - mean_t)^2, within month t
    monthly_sd = sqrt(RV^2_t * N_t)
    annualized_vol = monthly_sd * sqrt(12)

    Returns a monthly series aligned to month-end.
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


def _parse_french_csv_by_dates(path: str, n_cols: int, colnames: list[str]) -> pd.DataFrame:
    """
    Robust loader for Ken French CSV exports that contain header/footer text.
    Keeps only rows where first column is an 8-digit date (YYYYMMDD).
    Assumes the next columns are numeric in percent units.

    Parameters:
      n_cols: number of columns to keep (including date)
      colnames: names for those columns (length n_cols)

    Returns: DataFrame indexed by datetime, numeric columns in DECIMAL units.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    raw = pd.read_csv(path, header=None)
    c0 = raw.iloc[:, 0].astype(str).str.strip()
    mask = c0.str.fullmatch(r"\d{8}")
    df = raw.loc[mask].iloc[:, :n_cols].copy()
    df.columns = colnames

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    for c in colnames[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0  # percent -> decimal

    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df


# -----------------------------
# Load data
# -----------------------------
# 1) Mkt-RF, SMB, HML, RF from 1926+
ff3 = _parse_french_csv_by_dates(
    "../data/F-F_Research_Data_Factors_daily.csv",
    n_cols=5,
    colnames=["date", "Mkt-RF", "SMB", "HML", "RF"],
)

# 2) RMW, CMA from 1967+ (2x3 daily)
ff5_2x3 = _parse_french_csv_by_dates(
    "../data/F-F_Research_Data_5_Factors_2x3_daily.csv",
    n_cols=7,
    colnames=["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
)

# 3) Momentum daily (already decimal)
mom = pd.read_csv("../data/usa_momentum_daily_vwcap.csv")
mom["date"] = pd.to_datetime(mom["date"], errors="coerce")
mom = mom.dropna(subset=["date"]).set_index("date").sort_index()
mom_ret = pd.to_numeric(mom["ret"], errors="coerce")

# 4) Carry monthly sheet (fallback only)
carry_m = pd.read_csv("../data/currency-portfolios/all-currencies.csv")


# -----------------------------
# Carry volatility
# Use Verdelhan portfolio changes file:
#   High portfolio = Port. 6
#   Low  portfolio = Port. 1
#   Carry (HML)    = Port. 6 - Port. 1
# If the file has many obs per month (daily), compute realized monthly vol.
# If it has ~1 obs per month (monthly), compute rolling 12m annualized vol.
# -----------------------------
carry_vol_m = None
carry_port_path_candidates = [
    "../data/data-augmented-uip/spotchge-dollar-port.csv",
    "../data/data-augmented-uip/spotchge-dollar-port.csv".replace("-", "_"),
    "../data/spotchge-dollar-port.csv",
]

for p in carry_port_path_candidates:
    if not os.path.exists(p):
        continue

    carry_port = pd.read_csv(p)

    # Your file has first column as dates (e.g. 11/30/1988) and columns "Port. 1"..."Port. 6"
    date_col = carry_port.columns[0]
    carry_port[date_col] = pd.to_datetime(carry_port[date_col], errors="coerce")
    carry_port = carry_port.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # Column names in your snippet are exactly "Port. 1" and "Port. 6"
    if "Port. 1" not in carry_port.columns or "Port. 6" not in carry_port.columns:
        # fallback: try variants
        cols = list(carry_port.columns)

        def _find_exact_or_contains(targets):
            for t in targets:
                for c in cols:
                    if str(c).strip().lower() == t.lower():
                        return c
            for t in targets:
                for c in cols:
                    if t.lower() in str(c).strip().lower():
                        return c
            return None

        c1 = _find_exact_or_contains(["port. 1", "port 1", "p1", "portfolio1"])
        c6 = _find_exact_or_contains(["port. 6", "port 6", "p6", "portfolio6"])
        if c1 is None or c6 is None:
            raise ValueError(f"Found {p} but cannot locate Port. 1 / Port. 6 columns. Columns: {cols}")
        p1 = pd.to_numeric(carry_port[c1], errors="coerce")
        p6 = pd.to_numeric(carry_port[c6], errors="coerce")
    else:
        p1 = pd.to_numeric(carry_port["Port. 1"], errors="coerce")
        p6 = pd.to_numeric(carry_port["Port. 6"], errors="coerce")

    carry_ret = (p6 - p1).dropna()

    # Detect frequency: median obs per month
    obs_per_month = carry_ret.groupby(carry_ret.index.to_period("M")).size().median()

    if obs_per_month >= 10:
        # likely daily: compute realized monthly vol
        carry_vol_m = realized_monthly_vol_annualized(carry_ret)
    else:
        # monthly: rolling 12-month annualized vol (proxy)
        carry_vol_m = carry_ret.rolling(12).std() * np.sqrt(12)

    print(f"Carry source: {p} | median obs/month: {obs_per_month} | method: {'realized(daily)' if obs_per_month>=10 else 'rolling12m(monthly)'}")
    break

# Final fallback: use the all-currencies sheet HML column if portfolio file not found
if carry_vol_m is None:
    date_col = "Dates" if "Dates" in carry_m.columns else carry_m.columns[0]
    carry_m["date"] = _to_datetime_mdy(carry_m[date_col])
    carry_m = carry_m.dropna(subset=["date"]).set_index("date").sort_index()

    hml_col = None
    for cand in ["HML = P6 - P1", "HML=P6-P1", "HML", "Carry", "FX"]:
        if cand in carry_m.columns:
            hml_col = cand
            break
    if hml_col is None:
        raise ValueError(f"Couldn't find carry/HML column in {date_col} file. Columns: {list(carry_m.columns)}")

    carry_monthly_ret = _pct_str_to_decimal(carry_m[hml_col])
    carry_vol_m = carry_monthly_ret.rolling(12).std() * np.sqrt(12)
    print("Carry source: all-currencies.csv | method: rolling12m(monthly) (fallback)")


# -----------------------------
# Build monthly vol series (each factor from its best-available source)
# -----------------------------
vol_series = {}

# Mkt-RF, SMB, HML from 1926+ file
for c in ["Mkt-RF", "SMB", "HML"]:
    vol_series[c] = realized_monthly_vol_annualized(ff3[c])

# RMW, CMA from 1967+ 2x3 file
for c in ["RMW", "CMA"]:
    vol_series[c] = realized_monthly_vol_annualized(ff5_2x3[c])

# Momentum
vol_series["Mom"] = realized_monthly_vol_annualized(mom_ret)

# Carry
vol_series["Carry"] = carry_vol_m

vol_df = pd.DataFrame(vol_series).sort_index()
vol_df_pct = vol_df * 100.0  # annualized vol in percent


# -----------------------------
# Plot (Figure 2 style) — improved legibility
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 6))

rename = {"Mkt-RF": "MktRF"}

plot_cols = ["Mkt-RF", "HML", "SMB", "Mom", "RMW", "CMA", "Carry"]
plot_cols = [c for c in plot_cols if c in vol_df_pct.columns]

tab10 = plt.get_cmap("tab10").colors
color_map = {
    "Mkt-RF": tab10[0],
    "HML":    tab10[2],
    "SMB":    tab10[1],
    "Mom":    tab10[5],
    "RMW":    tab10[3],
    "CMA":    tab10[4],
    "Carry":  tab10[6],
}

for c in plot_cols:
    ax.plot(
        vol_df_pct.index,
        vol_df_pct[c],
        label=rename.get(c, c),
        color=color_map.get(c, None),
        linewidth=1.4,
        alpha=0.95,
    )

ax.set_title("Time series of volatility by factor")
ax.set_xlabel("date")
ax.set_ylabel("Annualized volatility (%)")
ax.grid(True, linewidth=0.5, alpha=0.25)

leg = ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.01, 0.99),
    frameon=True,
    ncol=1,
    fontsize=10,
)
leg.get_frame().set_alpha(0.9)

plt.tight_layout()

# -----------------------------
# Save as PDF for LaTeX
# -----------------------------
out_dir = "../figures"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "figure2.pdf")
fig.savefig(out_path, format="pdf", bbox_inches="tight")
print(f"Saved Figure 2 to: {out_path}")
