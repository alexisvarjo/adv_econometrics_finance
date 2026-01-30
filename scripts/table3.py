#!/usr/bin/env python3
"""
table3.py — Replicate Table III (Moreira & Muir, 2017): Recession Betas by Factor

Regression (monthly):
    f^σ_t = α0 + α1 * 1_rec,t + β0 * f_t + β1 * (1_rec,t × f_t) + ε_t

We report:
  - β0 (unconditional beta)
  - β1 (incremental beta in recessions; i.e. interaction term)
  - robust (HC1) standard errors
  - N and R^2

Data (expected in ../data, consistent with your prior scripts):
  - F-F_Research_Data_Factors_daily.csv            (daily; Mkt-RF, SMB, HML, RF; 1926+)
  - F-F_Research_Data_5_Factors_2x3_daily.csv      (daily; includes RMW, CMA; 1967+)
  - usa_momentum_monthly_vwcap.csv (or daily fallback)
  - q5_factors_monthly_2024.csv (preferred for IA/ROE) OR:
      usa_ni-be_monthly_vwcap.csv  (ROE fallback)
      usa_gpat_monthly_vwcap.csv   (IA fallback)
  - data-augmented-uip/spotchge-dollar-port.csv    (FX carry portfolios; we form Carry = Port.6 - Port.1)
  - USREC.csv                                      (monthly recession dummy from FRED)

Outputs:
  - ../tables/table3.tex

Notes:
  - Volatility-managed factor construction matches your Table I script:
        f^σ_t = (c / varhat_{t-1}) * f_t
    where c normalizes std(f^σ) == std(f) over the available sample.
  - Variance estimator is modular; change VAR_METHOD_* below easily.
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

Q5_MONTHLY = os.path.join(DATA_DIR, "q5_factors_monthly_2024.csv")

MOM_MONTHLY = os.path.join(DATA_DIR, "usa_momentum_monthly_vwcap.csv")
MOM_DAILY = os.path.join(DATA_DIR, "usa_momentum_daily_vwcap.csv")

ROE_MONTHLY = os.path.join(DATA_DIR, "usa_ni-be_monthly_vwcap.csv")  # ROE fallback
IA_MONTHLY = os.path.join(DATA_DIR, "usa_gpat_monthly_vwcap.csv")     # IA fallback

CARRY_PORT = os.path.join(DATA_DIR, "data-augmented-uip", "spotchge-dollar-port.csv")

USREC_PATH = os.path.join(DATA_DIR, "USREC.csv")


# =============================
# Estimator knobs (easy to change)
# =============================
VAR_METHOD_MONTHLY_FROM_DAILY = "month_realized"   # "month_realized" or "rv22_monthend"
VAR_METHOD_MONTHLY_FROM_MONTHLY = "rolling12"      # "rolling12"


# =============================
# Utilities
# =============================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_french_csv_by_dates(path: str, n_cols: int, colnames: list[str]) -> pd.DataFrame:
    """Robust Ken French CSV loader (filters YYYYMMDD rows), percent->decimal."""
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

    return df.dropna(subset=["date"]).set_index("date").sort_index()


def load_vwcap_factor_csv(path: str) -> pd.Series:
    """Loads usa_*_daily/monthly_vwcap.csv files into Series(date->ret decimal)."""
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
    """Daily decimal -> monthly decimal via compounding."""
    x = daily_ret.dropna().sort_index()
    return x.resample("ME").apply(lambda r: (1.0 + r).prod() - 1.0)


def to_month_end_index(s: pd.Series) -> pd.Series:
    """Ensure month-end timestamp index."""
    x = s.dropna().sort_index()
    if isinstance(x.index, pd.PeriodIndex):
        x.index = x.index.to_timestamp("M")
    # If daily, sample month-end (this is not compounding; only for already-monthly series)
    if x.index.inferred_freq and x.index.inferred_freq.startswith("D"):
        x = x.resample("ME").last()
    # Force month-end alignment
    x = x.resample("ME").last()
    return x


def load_usrec_monthly(path: str = USREC_PATH) -> pd.Series:
    """
    Load FRED USREC:
        observation_date,USREC
        1854-12-01,1
    Returns month-end indexed Series of 0/1.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing recession file: {path}")

    df = pd.read_csv(path)
    if "observation_date" not in df.columns or "USREC" not in df.columns:
        raise ValueError(f"{path} must have columns ['observation_date','USREC'].")

    df["date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["USREC"] = pd.to_numeric(df["USREC"], errors="coerce")
    s = df.dropna(subset=["date", "USREC"]).set_index("date").sort_index()["USREC"]
    s = s.resample("ME").last().astype(float)
    s.name = "USREC"
    return s


# =============================
# Variance estimators (modular)
# =============================
def estimate_monthly_variance_from_daily(daily_ret: pd.Series, method: str) -> pd.Series:
    """
    Monthly variance proxy varhat_t (decimal^2), month-end indexed.

    - "month_realized": within-month realized variance proxy:
        var_month = mean_d (r - mean_month)^2 * N_month
    - "rv22_monthend": rolling 22 trading days de-meaned variance sampled at month-end
    """
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
        return x.rolling(12, min_periods=12).var(ddof=1)

    raise ValueError(f"Unknown monthly-from-monthly variance method: {method}")


# =============================
# Vol-managed construction
# =============================
def build_vol_managed_monthly(monthly_ret: pd.Series, varhat_m: pd.Series) -> pd.DataFrame:
    """
    f^σ_t = (c / varhat_{t-1}) * f_t, with c matching std(f^σ) == std(f).
    Returns DataFrame with columns: f, f_sigma, weight, varhat.
    """
    df = pd.DataFrame(
        {"f": to_month_end_index(monthly_ret), "varhat": to_month_end_index(varhat_m)}
    ).dropna().sort_index()

    w_raw = 1.0 / df["varhat"].shift(1)
    f_sig_raw = w_raw * df["f"]
    out = pd.DataFrame(
        {"f": df["f"], "f_sig_raw": f_sig_raw, "w_raw": w_raw, "varhat": df["varhat"]}
    ).dropna()

    if out.shape[0] < 30:
        # Too short to do anything meaningful; keep empty to avoid zero-size regressions
        out = out.iloc[0:0]
        out.attrs["c"] = np.nan
        return out

    std_f = out["f"].std(ddof=1)
    std_fs = out["f_sig_raw"].std(ddof=1)
    c = std_f / std_fs if (std_fs is not None and np.isfinite(std_fs) and std_fs > 0) else 1.0

    out["f_sigma"] = c * out["f_sig_raw"]
    out["weight"] = c * out["w_raw"]
    out = out.drop(columns=["f_sig_raw", "w_raw"])
    out.attrs["c"] = c
    return out


# =============================
# Regression for Table III
# =============================
def ols_hc1(y: pd.Series, X: pd.DataFrame):
    """OLS with HC1 SE; returns fitted result, aligned df."""
    df = pd.concat([y, X], axis=1).dropna()
    if df.shape[0] == 0:
        return None, df
    yv = df.iloc[:, 0]
    Xv = df.iloc[:, 1:]
    Xv = sm.add_constant(Xv, has_constant="add")
    res = sm.OLS(yv.values, Xv.values).fit(cov_type="HC1")
    return res, df


def recession_beta_regression(f_sigma: pd.Series, f: pd.Series, rec: pd.Series) -> dict:
    """
    f_sigma_t = α0 + α1 rec_t + β0 f_t + β1 (rec_t * f_t) + ε
    Return β0, β1 and their HC1 SEs + N and R2.
    """
    rec = rec.rename("rec")
    X = pd.DataFrame({"f": f, "rec": rec})
    X["f_x_rec"] = X["f"] * X["rec"]

    res, df = ols_hc1(f_sigma, X)
    if res is None:
        return dict(beta=np.nan, se_beta=np.nan, beta_rec=np.nan, se_beta_rec=np.nan, N=0, R2=np.nan)

    # params order: const, f, rec, f_x_rec (because we built X in that order)
    # statsmodels will keep column order after add_constant
    params = res.params
    bse = res.bse

    beta = float(params[1])
    se_beta = float(bse[1])
    beta_rec = float(params[3])
    se_beta_rec = float(bse[3])

    return dict(beta=beta, se_beta=se_beta, beta_rec=beta_rec, se_beta_rec=se_beta_rec,
                N=int(res.nobs), R2=float(res.rsquared))


# =============================
# Factor loaders needed for Table III
# =============================
def load_q5_monthly() -> pd.DataFrame | None:
    """
    q5_factors_monthly_2024.csv format (your example):
        year,month,R_F,R_MKT,R_ME,R_IA,R_ROE,R_EG
    Values appear in percent -> convert to decimals.
    """
    if not os.path.exists(Q5_MONTHLY):
        return None

    df = pd.read_csv(Q5_MONTHLY)
    needed = {"year", "month"}
    if not needed.issubset(set(df.columns)):
        return None

    # month-end date
    df["date"] = pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str).str.zfill(2) + "-01",
        errors="coerce",
    ) + pd.offsets.MonthEnd(0)

    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    for c in df.columns:
        if c in ["year", "month", "date"]:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0

    return df.resample("ME").last()


def load_carry_monthly() -> pd.Series:
    """
    FX carry monthly: Carry = Port.6 - Port.1 from spotchge-dollar-port.csv.
    """
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

    carry = (p6 - p1).dropna()
    carry = carry.resample("ME").last()
    carry.name = "Carry"
    return carry


def load_required_factors() -> dict:
    """
    Returns dict factor_name -> dict with:
        daily (Series or None),
        monthly (Series),
        var_source ("daily" or "monthly")
    Only the factors needed for Table III.
    """
    factors = {}

    # FF3 daily: Mkt-RF and HML
    ff3 = parse_french_csv_by_dates(FF3_DAILY, 5, ["date", "Mkt-RF", "SMB", "HML", "RF"])
    for name in ["Mkt-RF", "HML"]:
        d = ff3[name].dropna()
        factors[name] = {
            "daily": d,
            "monthly": compound_monthly_return(d),
            "var_source": "daily",
        }

    # RMW/CMA daily
    ff5 = parse_french_csv_by_dates(FF5_2X3_DAILY, 7, ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    for name in ["RMW", "CMA"]:
        d = ff5[name].dropna()
        factors[name] = {
            "daily": d,
            "monthly": compound_monthly_return(d),
            "var_source": "daily",
        }

    # Momentum monthly preferred
    if os.path.exists(MOM_MONTHLY):
        mom_m = load_vwcap_factor_csv(MOM_MONTHLY)
        factors["Mom"] = {"daily": None, "monthly": to_month_end_index(mom_m), "var_source": "monthly"}
    else:
        mom_d = load_vwcap_factor_csv(MOM_DAILY)
        factors["Mom"] = {"daily": mom_d, "monthly": compound_monthly_return(mom_d), "var_source": "daily"}

    # IA/ROE: prefer q5 monthly if present
    q5m = load_q5_monthly()
    if q5m is not None and "R_IA" in q5m.columns and "R_ROE" in q5m.columns:
        factors["IA"] = {"daily": None, "monthly": q5m["R_IA"].dropna(), "var_source": "monthly"}
        factors["ROE"] = {"daily": None, "monthly": q5m["R_ROE"].dropna(), "var_source": "monthly"}
    else:
        # fallbacks
        roe_m = load_vwcap_factor_csv(ROE_MONTHLY)
        ia_m = load_vwcap_factor_csv(IA_MONTHLY)
        factors["ROE"] = {"daily": None, "monthly": to_month_end_index(roe_m), "var_source": "monthly"}
        factors["IA"] = {"daily": None, "monthly": to_month_end_index(ia_m), "var_source": "monthly"}

    # FX Carry monthly only
    carry_m = load_carry_monthly()
    factors["Carry"] = {"daily": None, "monthly": carry_m, "var_source": "monthly"}

    return factors


# =============================
# Table construction + LaTeX
# =============================
def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x:.{digits}f}"


def build_table3() -> pd.DataFrame:
    """
    Build Table III in the layout shown: each column has only its own factor + interaction.
    """
    rec = load_usrec_monthly()

    factors = load_required_factors()

    # Table III columns/order in paper
    # (1) Mktσ (2) HMLσ (3) Momσ (4) RMWσ (5) CMAσ (6) FXσ (7) ROEσ (8) IAσ
    order = [
        ("MktRF", "Mkt-RF"),
        ("HML", "HML"),
        ("Mom", "Mom"),
        ("RMW", "RMW"),
        ("CMA", "CMA"),
        ("Carry", "Carry"),
        ("ROE", "ROE"),
        ("IA", "IA"),
    ]

    results = {}
    for col_label, fac_name in order:
        f_m = factors[fac_name]["monthly"].dropna()
        if f_m.empty:
            results[col_label] = dict(beta=np.nan, se_beta=np.nan, beta_rec=np.nan, se_beta_rec=np.nan, N=0, R2=np.nan)
            continue

        # variance proxy (month t), used as lag var_{t-1}
        if factors[fac_name]["var_source"] == "daily" and factors[fac_name]["daily"] is not None:
            var_m = estimate_monthly_variance_from_daily(
                factors[fac_name]["daily"], VAR_METHOD_MONTHLY_FROM_DAILY
            )
        else:
            var_m = estimate_monthly_variance_from_monthly(
                f_m, VAR_METHOD_MONTHLY_FROM_MONTHLY
            )

        vm = build_vol_managed_monthly(f_m, var_m)
        if vm.shape[0] == 0:
            results[col_label] = dict(beta=np.nan, se_beta=np.nan, beta_rec=np.nan, se_beta_rec=np.nan, N=0, R2=np.nan)
            continue

        # align recession dummy
        df = pd.DataFrame({"f": vm["f"], "f_sigma": vm["f_sigma"]}).join(rec, how="inner").dropna()
        if df.shape[0] == 0:
            results[col_label] = dict(beta=np.nan, se_beta=np.nan, beta_rec=np.nan, se_beta_rec=np.nan, N=0, R2=np.nan)
            continue

        results[col_label] = recession_beta_regression(df["f_sigma"], df["f"], df["USREC"])

    # Assemble in paper-like sparse matrix:
    cols = [c for c, _ in order]
    idx = [
        "MktRF", "MktRF × 1_rec",
        "HML", "HML × 1_rec",
        "Mom", "Mom × 1_rec",
        "RMW", "RMW × 1_rec",
        "CMA", "CMA × 1_rec",
        "Carry", "Carry × 1_rec",
        "ROE", "ROE × 1_rec",
        "IA", "IA × 1_rec",
        "Observations", "R^2",
    ]
    T = pd.DataFrame("", index=idx, columns=cols)

    # Fill each column only on its own rows
    for col_label, fac_name in order:
        r = results[col_label]
        # main beta rows
        T.loc[col_label if col_label != "MktRF" else "MktRF", col_label] = fmt(r["beta"], 2)
        T.loc[(col_label if col_label != "MktRF" else "MktRF") + " × 1_rec", col_label] = fmt(r["beta_rec"], 2)

        # SEs in parentheses on next line? The screenshot uses parentheses on the next line
        # We'll store SEs in the same cell line as parentheses by inserting extra rows would be messy.
        # Instead: mimic screenshot by writing SEs in the row below via overwriting pattern:
        # We'll append SEs in parentheses on the next index row directly by adding them as separate rows:
        # => easiest: replace cell contents with "coef\n(se)" using LaTeX line breaks.
        T.loc[col_label if col_label != "MktRF" else "MktRF", col_label] = f"{fmt(r['beta'],2)}\\\\({fmt(r['se_beta'],2)})"
        T.loc[(col_label if col_label != "MktRF" else "MktRF") + " × 1_rec", col_label] = f"{fmt(r['beta_rec'],2)}\\\\({fmt(r['se_beta_rec'],2)})"

        T.loc["Observations", col_label] = str(int(r["N"]))
        T.loc["R^2", col_label] = fmt(r["R2"], 2)

    return T


def save_table3_latex(T: pd.DataFrame, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by table3.py\n\n")
        f.write("\\begin{table}[!ht]\n\\centering\n")
        f.write("\\caption{Recession Betas by Factor}\n")
        f.write("\\label{tab:recession_betas}\n\n")
        f.write("{\\small\n")
        f.write(T.to_latex(escape=False, column_format="l" + "c"*T.shape[1]))
        f.write("}\n\n")
        f.write("{\\footnotesize Notes: Monthly regression $f^\\sigma_t = \\alpha_0 + \\alpha_1\\,1_{rec,t} + "
                "\\beta_0 f_t + \\beta_1(1_{rec,t}\\times f_t) + \\varepsilon_t$. "
                "Reported coefficients are $\\beta_0$ and $\\beta_1$ with HC1 standard errors in parentheses. "
                "Recession indicator is NBER (FRED USREC). Volatility-managed factors use lagged variance estimates "
                "and $c$ chosen so that $\\mathrm{std}(f^\\sigma)=\\mathrm{std}(f)$ over the available sample.}\n")
        f.write("\\end{table}\n")


def main():
    T = build_table3()
    out = os.path.join(TABLE_DIR, "table3.tex")
    save_table3_latex(T, out)
    print(f"Saved: {out}")
    print("\nEstimator settings:")
    print(f"  Monthly varhat from daily:   {VAR_METHOD_MONTHLY_FROM_DAILY}")
    print(f"  Monthly varhat from monthly: {VAR_METHOD_MONTHLY_FROM_MONTHLY}")


if __name__ == "__main__":
    main()
