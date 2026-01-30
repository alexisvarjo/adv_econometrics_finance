#!/usr/bin/env python3
"""
table1.py — Replicate Table I (Moreira & Muir, 2017): Volatility-Managed Factor Alphas
(extended: includes ROE, IA, BAB; produces monthly table + daily analog table)

Outputs:
  - ../tables/table1_monthly.tex
  - ../tables/table1_daily.tex

Robustness fixes:
  - Uses 'ME' (month-end) resampling (pandas deprecation of 'M')
  - Treats Ken French missing codes (-99.99) as NaN
  - Skips factors with insufficient aligned sample after lagged variance
  - Prevents statsmodels zero-size crash
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


# =============================
# Paths
# =============================
DATA_DIR = "../data"
TABLE_DIR = "../tables"

FF3_DAILY = os.path.join(DATA_DIR, "F-F_Research_Data_Factors_daily.csv")
FF5_2X3_DAILY = os.path.join(DATA_DIR, "F-F_Research_Data_5_Factors_2x3_daily.csv")

Q5_DAILY = os.path.join(DATA_DIR, "q5_factors_daily_2024.csv")
Q5_MONTHLY = os.path.join(DATA_DIR, "q5_factors_monthly_2024.csv")

MOM_DAILY = os.path.join(DATA_DIR, "usa_momentum_daily_vwcap.csv")
MOM_MONTHLY = os.path.join(DATA_DIR, "usa_momentum_monthly_vwcap.csv")

BAB_DAILY = os.path.join(DATA_DIR, "usa_betabab_daily_vwcap.csv")
BAB_MONTHLY = os.path.join(DATA_DIR, "usa_betabab_monthly_vwcap.csv")

ROE_DAILY = os.path.join(DATA_DIR, "usa_ni-be_daily_vwcap.csv")      # confirmed ROE
ROE_MONTHLY = os.path.join(DATA_DIR, "usa_ni-be_monthly_vwcap.csv")

IA_DAILY = os.path.join(DATA_DIR, "usa_gpat_daily_vwcap.csv")        # confirmed IA
IA_MONTHLY = os.path.join(DATA_DIR, "usa_gpat_monthly_vwcap.csv")

CARRY_PORT = os.path.join(DATA_DIR, "data-augmented-uip", "spotchge-dollar-port.csv")

# =============================
# Estimator knobs
# =============================
VAR_METHOD_MONTHLY_FROM_DAILY = "month_realized"   # "month_realized" or "rv22_monthend"
VAR_METHOD_MONTHLY_FROM_MONTHLY = "rolling12"      # "rolling12"
VAR_METHOD_DAILY = "rolling22"                     # "rolling22"

ANNUALIZE_MONTHLY = 12
ANNUALIZE_DAILY = 252

# Minimum sample sizes to run regressions safely
MIN_OBS_MONTHLY = 60     # paper samples are large; you can lower, but <24 is risky
MIN_OBS_DAILY = 252 * 3  # 3 years of dailies


# =============================
# Utilities
# =============================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _clean_missing_codes(x: pd.Series) -> pd.Series:
    """Replace common Ken French missing codes with NaN."""
    # Ken French sometimes uses -99.99 or -999 as missing sentinel
    return x.replace([-99.99, -999, -9999], np.nan)


def parse_french_csv_by_dates(path: str, n_cols: int, colnames: list[str]) -> pd.DataFrame:
    """Robust Ken French CSV loader (filters YYYYMMDD rows), percent->decimal, cleans missing codes."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    raw = pd.read_csv(path, header=None)
    c0 = raw.iloc[:, 0].astype(str).str.strip()
    mask = c0.str.fullmatch(r"\d{8}")
    df = raw.loc[mask].iloc[:, :n_cols].copy()
    df.columns = colnames

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    for c in colnames[1:]:
        s = pd.to_numeric(df[c], errors="coerce")
        s = _clean_missing_codes(s)
        df[c] = s / 100.0

    return df.dropna(subset=["date"]).set_index("date").sort_index()


def load_vwcap_factor_csv(path: str) -> pd.Series:
    """Loads usa_*_daily/monthly_vwcap.csv files, returns Series(date->ret decimal)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns or "ret" not in df.columns:
        raise ValueError(f"{path} must have columns ['date','ret']. Found {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    s = pd.to_numeric(df["ret"], errors="coerce")
    s.index = df["date"]
    return s.dropna().sort_index()


def compound_monthly_return(daily_ret: pd.Series) -> pd.Series:
    """Daily decimal -> monthly decimal via compounding (month-end index)."""
    x = daily_ret.dropna().sort_index()
    return x.resample("ME").apply(lambda r: (1.0 + r).prod() - 1.0)


def to_month_end_index(s: pd.Series) -> pd.Series:
    """Force month-end timestamp index."""
    x = s.dropna().sort_index()
    if isinstance(x.index, pd.PeriodIndex):
        x.index = x.index.to_timestamp("M")
    # If higher freq, collapse to month-end
    if len(x) > 2 and (pd.infer_freq(x.index) or "").startswith("D"):
        x = x.resample("ME").last()
    return x


# =============================
# Variance estimators (modular)
# =============================
def estimate_monthly_variance_from_daily(daily_ret: pd.Series, method: str) -> pd.Series:
    """Monthly variance proxy varhat_t (decimal^2), month-end indexed."""
    x = daily_ret.dropna().sort_index()

    if method == "month_realized":
        def _month_var(y: pd.Series) -> float:
            n = y.shape[0]
            if n < 2:
                return np.nan
            m = y.mean()
            rv2 = ((y - m) ** 2).mean()
            return rv2 * n
        return x.resample("ME").apply(_month_var)

    if method == "rv22_monthend":
        def _rv2(arr: np.ndarray) -> float:
            m = arr.mean()
            return np.mean((arr - m) ** 2)
        rv2_daily = x.rolling(22, min_periods=22).apply(_rv2, raw=True)
        return rv2_daily.resample("ME").last()

    raise ValueError(f"Unknown monthly-from-daily variance method: {method}")


def estimate_monthly_variance_from_monthly(monthly_ret: pd.Series, method: str) -> pd.Series:
    """Monthly variance proxy varhat_t (decimal^2), month-end indexed."""
    x = to_month_end_index(monthly_ret)

    if method == "rolling12":
        # Keep min_periods=12 to match typical 12m window; if too strict for your sample, lower it.
        return x.rolling(12, min_periods=12).var(ddof=1)

    raise ValueError(f"Unknown monthly-from-monthly variance method: {method}")


def estimate_daily_variance(daily_ret: pd.Series, method: str) -> pd.Series:
    """Daily variance proxy (decimal^2)."""
    x = daily_ret.dropna().sort_index()

    if method == "rolling22":
        def _rv2(arr: np.ndarray) -> float:
            m = arr.mean()
            return np.mean((arr - m) ** 2)
        return x.rolling(22, min_periods=22).apply(_rv2, raw=True)

    raise ValueError(f"Unknown daily variance method: {method}")


# =============================
# Vol-managed construction
# =============================
def build_vol_managed_monthly(monthly_ret: pd.Series, varhat_m: pd.Series) -> pd.DataFrame:
    """
    f^σ_t = (c / varhat_{t-1}) * f_t, with c matching std.
    Returns DataFrame with columns: f, f_sigma, weight, varhat.
    """
    f = to_month_end_index(monthly_ret)
    v = to_month_end_index(varhat_m)

    # Align strictly on intersection
    df = pd.DataFrame({"f": f}).join(pd.DataFrame({"varhat": v}), how="inner").dropna().sort_index()
    if df.empty:
        return pd.DataFrame()

    # Lag variance
    w_raw = 1.0 / df["varhat"].shift(1)
    # Avoid inf from zero variance
    w_raw = w_raw.replace([np.inf, -np.inf], np.nan)

    f_sig_raw = w_raw * df["f"]
    out = pd.DataFrame({"f": df["f"], "f_sig_raw": f_sig_raw, "w_raw": w_raw, "varhat": df["varhat"]}).dropna()

    if out.empty:
        return pd.DataFrame()

    std_f = out["f"].std(ddof=1)
    std_fs = out["f_sig_raw"].std(ddof=1)
    if (std_fs is None) or (not np.isfinite(std_fs)) or std_fs == 0:
        return pd.DataFrame()

    c = std_f / std_fs if np.isfinite(std_f) else 1.0

    out["f_sigma"] = c * out["f_sig_raw"]
    out["weight"] = c * out["w_raw"]
    out = out.drop(columns=["f_sig_raw", "w_raw"])
    out.attrs["c"] = c
    return out


def build_vol_managed_daily(daily_ret: pd.Series, varhat_d: pd.Series) -> pd.DataFrame:
    """Daily analog of vol-managed factor."""
    df = pd.DataFrame({"f": daily_ret, "varhat": varhat_d}).dropna().sort_index()
    if df.empty:
        return pd.DataFrame()

    w_raw = 1.0 / df["varhat"].shift(1)
    w_raw = w_raw.replace([np.inf, -np.inf], np.nan)
    f_sig_raw = w_raw * df["f"]
    out = pd.DataFrame({"f": df["f"], "f_sig_raw": f_sig_raw, "w_raw": w_raw, "varhat": df["varhat"]}).dropna()

    if out.empty:
        return pd.DataFrame()

    std_f = out["f"].std(ddof=1)
    std_fs = out["f_sig_raw"].std(ddof=1)
    if (std_fs is None) or (not np.isfinite(std_fs)) or std_fs == 0:
        return pd.DataFrame()

    c = std_f / std_fs if np.isfinite(std_f) else 1.0

    out["f_sigma"] = c * out["f_sig_raw"]
    out["weight"] = c * out["w_raw"]
    out = out.drop(columns=["f_sig_raw", "w_raw"])
    out.attrs["c"] = c
    return out


# =============================
# Regressions (HC1) with guards
# =============================
def ols_hc1(y: pd.Series, X: pd.DataFrame, min_obs: int) -> sm.regression.linear_model.RegressionResultsWrapper:
    df = pd.concat([y, X], axis=1).dropna()
    if df.shape[0] < min_obs:
        raise ValueError(f"Insufficient observations after alignment: {df.shape[0]} < {min_obs}")
    yv = df.iloc[:, 0]
    Xv = df.iloc[:, 1:]
    Xv = sm.add_constant(Xv, has_constant="add")
    return sm.OLS(yv.values, Xv.values).fit(cov_type="HC1")


def reg_panel_A(fsig: pd.Series, f: pd.Series, min_obs: int):
    res = ols_hc1(fsig, pd.DataFrame({"f": f}), min_obs=min_obs)
    alpha, beta = res.params[0], res.params[1]
    se_a, se_b = res.bse[0], res.bse[1]
    n = int(res.nobs)
    r2 = float(res.rsquared)
    rmse = float(np.sqrt(np.mean(res.resid ** 2)))
    return dict(alpha=alpha, se_alpha=se_a, beta=beta, se_beta=se_b, N=n, R2=r2, RMSE=rmse)


def reg_panel_B(fsig: pd.Series, f: pd.Series, ff3_controls: pd.DataFrame, min_obs: int):
    X = pd.DataFrame({"f": f}).join(ff3_controls, how="inner")
    res = ols_hc1(fsig, X, min_obs=min_obs)
    alpha = res.params[0]
    se_a = res.bse[0]
    n = int(res.nobs)
    return dict(alpha=alpha, se_alpha=se_a, N=n)


# =============================
# Factor loading
# =============================
def load_q5_daily_monthly():
    out = {"daily": None, "monthly": None}

    if os.path.exists(Q5_DAILY):
        df = pd.read_csv(Q5_DAILY)
        df["date"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d", errors="coerce") if "DATE" in df.columns else pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        for c in df.columns:
            if c in ["DATE", "date"]:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
        out["daily"] = df

    if os.path.exists(Q5_MONTHLY):
        df = pd.read_csv(Q5_MONTHLY)
        if "DATE" in df.columns:
            s = df["DATE"].astype(str).str.strip()
            dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
            if dt.isna().mean() > 0.5:
                dt = pd.to_datetime(s, format="%Y%m", errors="coerce")
            df["date"] = dt
        else:
            df["date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index().resample("ME").last()
        for c in df.columns:
            if c in ["DATE", "date"]:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
        out["monthly"] = df

    return out


def load_carry_monthly() -> pd.Series:
    if not os.path.exists(CARRY_PORT):
        raise FileNotFoundError(f"Missing FX carry file: {CARRY_PORT}")
    df = pd.read_csv(CARRY_PORT)
    date_col = df.columns[0]
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    if "Port. 1" not in df.columns or "Port. 6" not in df.columns:
        raise ValueError(f"Expected 'Port. 1' and 'Port. 6' in {CARRY_PORT}. Columns: {list(df.columns)}")

    p1 = pd.to_numeric(df["Port. 1"], errors="coerce")
    p6 = pd.to_numeric(df["Port. 6"], errors="coerce")
    carry = (p6 - p1).dropna().resample("ME").last()
    carry.name = "Carry"
    return carry


def load_factors():
    factors = {}

    # FF3 daily (Mkt-RF, SMB, HML)
    ff3 = parse_french_csv_by_dates(FF3_DAILY, 5, ["date", "Mkt-RF", "SMB", "HML", "RF"])
    for name in ["Mkt-RF", "SMB", "HML"]:
        d = ff3[name].dropna()
        factors[name] = {"daily": d, "monthly": compound_monthly_return(d)}

    # RMW/CMA daily (2x3 file)
    ff5 = parse_french_csv_by_dates(FF5_2X3_DAILY, 7, ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    for name in ["RMW", "CMA"]:
        d = ff5[name].dropna()
        factors[name] = {"daily": d, "monthly": compound_monthly_return(d)}

    # Momentum
    mom_d = load_vwcap_factor_csv(MOM_DAILY)
    mom_m = load_vwcap_factor_csv(MOM_MONTHLY) if os.path.exists(MOM_MONTHLY) else None
    factors["Mom"] = {"daily": mom_d, "monthly": (to_month_end_index(mom_m) if mom_m is not None else compound_monthly_return(mom_d))}

    # BAB
    bab_d = load_vwcap_factor_csv(BAB_DAILY)
    bab_m = load_vwcap_factor_csv(BAB_MONTHLY) if os.path.exists(BAB_MONTHLY) else None
    factors["BAB"] = {"daily": bab_d, "monthly": (to_month_end_index(bab_m) if bab_m is not None else compound_monthly_return(bab_d))}

    # IA and ROE — prefer q5 if present
    q5 = load_q5_daily_monthly()
    q5d, q5m = q5["daily"], q5["monthly"]

    if q5d is not None and "R_IA" in q5d.columns and "R_ROE" in q5d.columns:
        factors["IA"] = {
            "daily": q5d["R_IA"].dropna(),
            "monthly": (q5m["R_IA"].dropna() if (q5m is not None and "R_IA" in q5m.columns) else compound_monthly_return(q5d["R_IA"]))
        }
        factors["ROE"] = {
            "daily": q5d["R_ROE"].dropna(),
            "monthly": (q5m["R_ROE"].dropna() if (q5m is not None and "R_ROE" in q5m.columns) else compound_monthly_return(q5d["R_ROE"]))
        }
    else:
        # fallback: ROE=ni_be, IA=gpat
        roe_d = load_vwcap_factor_csv(ROE_DAILY)
        roe_m = load_vwcap_factor_csv(ROE_MONTHLY) if os.path.exists(ROE_MONTHLY) else None
        ia_d = load_vwcap_factor_csv(IA_DAILY)
        ia_m = load_vwcap_factor_csv(IA_MONTHLY) if os.path.exists(IA_MONTHLY) else None
        factors["ROE"] = {"daily": roe_d, "monthly": (to_month_end_index(roe_m) if roe_m is not None else compound_monthly_return(roe_d))}
        factors["IA"] = {"daily": ia_d, "monthly": (to_month_end_index(ia_m) if ia_m is not None else compound_monthly_return(ia_d))}

    # FX Carry (monthly only)
    factors["Carry"] = {"daily": None, "monthly": load_carry_monthly()}

    return factors


# =============================
# Table builder
# =============================
def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x:.{digits}f}"


def build_monthly_table(factors: dict):
    order = ["Mkt-RF", "SMB", "HML", "Mom", "RMW", "CMA", "Carry", "ROE", "IA", "BAB"]

    ff3_ctrl = pd.DataFrame({
        "MktRF": to_month_end_index(factors["Mkt-RF"]["monthly"]),
        "SMB": to_month_end_index(factors["SMB"]["monthly"]),
        "HML": to_month_end_index(factors["HML"]["monthly"]),
    }).dropna()
    ff3_ctrl_ann = ff3_ctrl * ANNUALIZE_MONTHLY * 100

    resultsA, resultsB, cols = {}, {}, []

    for name in order:
        if name not in factors or factors[name]["monthly"] is None:
            continue

        f_m = to_month_end_index(factors[name]["monthly"]).dropna()

        if factors[name]["daily"] is not None:
            var_m = estimate_monthly_variance_from_daily(factors[name]["daily"], VAR_METHOD_MONTHLY_FROM_DAILY)
        else:
            var_m = estimate_monthly_variance_from_monthly(f_m, VAR_METHOD_MONTHLY_FROM_MONTHLY)

        vm = build_vol_managed_monthly(f_m, var_m)
        if vm.empty or vm.shape[0] < MIN_OBS_MONTHLY:
            print(f"[MONTHLY] Skipping {name}: vm sample too small after lagged variance (n={vm.shape[0]})")
            continue

        f_ann = vm["f"] * ANNUALIZE_MONTHLY * 100
        fs_ann = vm["f_sigma"] * ANNUALIZE_MONTHLY * 100

        try:
            resultsA[name] = reg_panel_A(fs_ann, f_ann, min_obs=MIN_OBS_MONTHLY)
            resultsB[name] = reg_panel_B(fs_ann, f_ann, ff3_ctrl_ann, min_obs=MIN_OBS_MONTHLY)
            cols.append(name)
        except Exception as e:
            print(f"[MONTHLY] Skipping {name}: regression failed ({e})")
            continue

    panelA = pd.DataFrame(index=["Beta", "SE(Beta)", "Alpha", "SE(Alpha)", "N", "R2", "RMSE"], columns=cols, dtype=object)
    panelB = pd.DataFrame(index=["Alpha", "SE(Alpha)", "N"], columns=cols, dtype=object)

    for c in cols:
        ra, rb = resultsA[c], resultsB[c]
        panelA.loc["Beta", c] = fmt(ra["beta"], 2)
        panelA.loc["SE(Beta)", c] = f"({fmt(ra['se_beta'], 2)})"
        panelA.loc["Alpha", c] = fmt(ra["alpha"], 2)
        panelA.loc["SE(Alpha)", c] = f"({fmt(ra['se_alpha'], 2)})"
        panelA.loc["N", c] = str(int(ra["N"]))
        panelA.loc["R2", c] = fmt(ra["R2"], 2)
        panelA.loc["RMSE", c] = fmt(ra["RMSE"], 2)

        panelB.loc["Alpha", c] = fmt(rb["alpha"], 2)
        panelB.loc["SE(Alpha)", c] = f"({fmt(rb['se_alpha'], 2)})"
        panelB.loc["N", c] = str(int(rb["N"]))

    return panelA, panelB


def build_daily_table(factors: dict):
    order = ["Mkt-RF", "SMB", "HML", "Mom", "RMW", "CMA", "ROE", "IA", "BAB"]

    ff3_d = pd.DataFrame({
        "MktRF": factors["Mkt-RF"]["daily"],
        "SMB": factors["SMB"]["daily"],
        "HML": factors["HML"]["daily"],
    }).dropna()
    ff3_d_ann = ff3_d * ANNUALIZE_DAILY * 100

    resultsA, resultsB, cols = {}, {}, []

    for name in order:
        if name not in factors or factors[name]["daily"] is None:
            continue

        f_d = factors[name]["daily"].dropna()
        var_d = estimate_daily_variance(f_d, VAR_METHOD_DAILY)
        vm = build_vol_managed_daily(f_d, var_d)

        if vm.empty or vm.shape[0] < MIN_OBS_DAILY:
            print(f"[DAILY] Skipping {name}: vm sample too small after lagged variance (n={vm.shape[0]})")
            continue

        f_ann = vm["f"] * ANNUALIZE_DAILY * 100
        fs_ann = vm["f_sigma"] * ANNUALIZE_DAILY * 100

        try:
            resultsA[name] = reg_panel_A(fs_ann, f_ann, min_obs=MIN_OBS_DAILY)
            resultsB[name] = reg_panel_B(fs_ann, f_ann, ff3_d_ann, min_obs=MIN_OBS_DAILY)
            cols.append(name)
        except Exception as e:
            print(f"[DAILY] Skipping {name}: regression failed ({e})")
            continue

    panelA = pd.DataFrame(index=["Beta", "SE(Beta)", "Alpha", "SE(Alpha)", "N", "R2", "RMSE"], columns=cols, dtype=object)
    panelB = pd.DataFrame(index=["Alpha", "SE(Alpha)", "N"], columns=cols, dtype=object)

    for c in cols:
        ra, rb = resultsA[c], resultsB[c]
        panelA.loc["Beta", c] = fmt(ra["beta"], 2)
        panelA.loc["SE(Beta)", c] = f"({fmt(ra['se_beta'], 2)})"
        panelA.loc["Alpha", c] = fmt(ra["alpha"], 2)
        panelA.loc["SE(Alpha)", c] = f"({fmt(ra['se_alpha'], 2)})"
        panelA.loc["N", c] = str(int(ra["N"]))
        panelA.loc["R2", c] = fmt(ra["R2"], 2)
        panelA.loc["RMSE", c] = fmt(ra["RMSE"], 2)

        panelB.loc["Alpha", c] = fmt(rb["alpha"], 2)
        panelB.loc["SE(Alpha)", c] = f"({fmt(rb['se_alpha'], 2)})"
        panelB.loc["N", c] = str(int(rb["N"]))

    return panelA, panelB


def save_latex(panelA: pd.DataFrame, panelB: pd.DataFrame, out_path: str, title: str):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by table1.py\n\n")
        f.write("\\begin{table}[!ht]\n\\centering\n")
        f.write(f"\\caption{{{title}}}\n")
        f.write("\\label{tab:vm_factor_alphas}\n\n")

        f.write("\\textbf{Panel A: Univariate Regressions}\\\\\n")
        f.write(panelA.to_latex(escape=False))
        f.write("\n\\\\[6pt]\n")

        f.write("\\textbf{Panel B: Alphas Controlling for Fama-French Three Factors}\\\\\n")
        f.write(panelB.to_latex(escape=False))
        f.write("\n\\\\[4pt]\n")

        f.write("{\\footnotesize Notes: Returns are annualized to percent/year by multiplying monthly returns by $12\\times 100$ (monthly table) ")
        f.write("or daily returns by $252\\times 100$ (daily analog). Volatility-managed factors use lagged variance estimates and $c$ chosen so that managed and unmanaged have the same unconditional standard deviation in the corresponding frequency. ")
        f.write("Standard errors are HC1.}\n")
        f.write("\\end{table}\n")


def main():
    factors = load_factors()

    mA, mB = build_monthly_table(factors)
    out_monthly = os.path.join(TABLE_DIR, "table1_monthly.tex")
    save_latex(mA, mB, out_monthly, "Volatility-Managed Factor Alphas (Monthly; Replication)")

    dA, dB = build_daily_table(factors)
    out_daily = os.path.join(TABLE_DIR, "table1_daily.tex")
    save_latex(dA, dB, out_daily, "Volatility-Managed Factor Alphas (Daily Analog; Replication)")

    print("\nSaved:")
    print(f"  {out_monthly}")
    print(f"  {out_daily}")

    print("\nEstimator settings:")
    print(f"  Monthly varhat from daily:   {VAR_METHOD_MONTHLY_FROM_DAILY}")
    print(f"  Monthly varhat from monthly: {VAR_METHOD_MONTHLY_FROM_MONTHLY}")
    print(f"  Daily varhat:                {VAR_METHOD_DAILY}")


if __name__ == "__main__":
    main()
