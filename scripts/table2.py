#!/usr/bin/env python3
"""
table2.py — Replicate Table II (Moreira & Muir, 2017): Mean-Variance Efficient Factors
(Produces monthly table + daily analog table)

Outputs:
  - ../tables/table2_monthly.tex
  - ../tables/table2_daily.tex

Table II logic:
  1) Build unconditional MVE (tangency) portfolio using a given factor set:
       f^{MVE}_t = b' F_t,  b ∝ Σ^{-1} μ
  2) Volatility-manage that MVE portfolio:
       f^{MVE,σ}_t = (c / varhat_{t-1}) * f^{MVE}_t
     where c matches unconditional std of managed to original.
  3) Time-series regression:
       f^{MVE,σ}_t = α + β f^{MVE}_t + ε_t
     Report α (annualized, %/yr), SE(α), N, R^2, RMSE
  4) Report:
       - Original Sharpe (annualized)
       - Vol-Managed Sharpe (annualized)
       - Appraisal ratio (annualized): alpha / tracking error

Panel B: subsample α’s for:
  1926–1955, 1956–1985, 1986–2015

Data used (../data):
  - F-F_Research_Data_Factors_daily.csv (Mkt-RF, SMB, HML, RF), 1926+
  - F-F_Research_Data_5_Factors_2x3_daily.csv (RMW, CMA), 1963/1967+
  - usa_momentum_daily_vwcap.csv (+ monthly version if present)
  - q5_factors_daily_2024.csv (+ monthly version):
      daily: has DATE column (YYYYMMDD)
      monthly: has columns year, month, R_F, R_MKT, R_ME, R_IA, R_ROE, R_EG
    We use HXZ = (R_MKT, R_ME, R_IA, R_ROE) by default (EG optional)

Volatility estimator is modular and easy to swap.
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

MOM_DAILY = os.path.join(DATA_DIR, "usa_momentum_daily_vwcap.csv")
MOM_MONTHLY = os.path.join(DATA_DIR, "usa_momentum_monthly_vwcap.csv")

Q5_DAILY = os.path.join(DATA_DIR, "q5_factors_daily_2024.csv")
Q5_MONTHLY = os.path.join(DATA_DIR, "q5_factors_monthly_2024.csv")


# =============================
# Settings
# =============================
HXZ_INCLUDE_EG = False

ANN_M = 12
ANN_D = 252

MIN_OBS_MONTHLY = 120   # 10y
MIN_OBS_DAILY = 252 * 5 # 5y

SUBSAMPLES = [
    ("1926–1955", "1926-01-01", "1955-12-31"),
    ("1956–1985", "1956-01-01", "1985-12-31"),
    ("1986–2015", "1986-01-01", "2015-12-31"),
]


# =============================
# Helpers
# =============================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _clean_missing_codes(x: pd.Series) -> pd.Series:
    return x.replace([-99.99, -999, -9999], np.nan)


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
        s = pd.to_numeric(df[c], errors="coerce")
        s = _clean_missing_codes(s)
        df[c] = s / 100.0

    return df.dropna(subset=["date"]).set_index("date").sort_index()


def load_vwcap_factor_csv(path: str) -> pd.Series:
    """Loads usa_*_daily/monthly_vwcap.csv files (ret already decimal)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns or "ret" not in df.columns:
        raise ValueError(f"{path} must have columns ['date','ret']. Found {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    s = pd.to_numeric(df["ret"], errors="coerce")
    s.index = df["date"]
    return s.dropna().sort_index()


def compound_monthly_from_daily(daily_ret: pd.Series) -> pd.Series:
    x = daily_ret.dropna().sort_index()
    return x.resample("ME").apply(lambda r: (1.0 + r).prod() - 1.0)


def to_month_end(s: pd.Series) -> pd.Series:
    x = s.dropna().sort_index()
    if isinstance(x.index, pd.PeriodIndex):
        x.index = x.index.to_timestamp("M")
    # if daily/higher freq, collapse to month-end
    inf = pd.infer_freq(x.index)
    if inf is not None and inf.startswith("D"):
        x = x.resample("ME").last()
    return x


# =============================
# Variance estimators (modular)
# =============================
def estimate_monthly_var_from_daily(daily_ret: pd.Series, method: str = "month_realized") -> pd.Series:
    """
    Monthly variance proxy from daily returns (decimal^2), month-end index.
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

    raise ValueError(f"Unknown method: {method}")


def estimate_daily_var(daily_ret: pd.Series, method: str = "rolling22") -> pd.Series:
    """
    Daily variance proxy from daily returns (decimal^2).
    """
    x = daily_ret.dropna().sort_index()

    if method == "rolling22":
        def _rv2(arr: np.ndarray) -> float:
            m = arr.mean()
            return np.mean((arr - m) ** 2)
        return x.rolling(22, min_periods=22).apply(_rv2, raw=True)

    raise ValueError(f"Unknown method: {method}")


# =============================
# Core: MVE + Vol-managed
# =============================

def tangency_weights(F: pd.DataFrame) -> np.ndarray:
    """
    b ∝ Σ^{-1} μ (unconstrained tangency on excess returns).
    Robust to k=1 where np.cov returns a scalar.
    """
    X = F.dropna()
    k = X.shape[1]
    if k == 0:
        return np.array([])

    mu = X.mean().values.reshape(-1, 1)  # (k,1)

    # --- k=1 special case: tangency is just 100% in the single asset ---
    if k == 1:
        # Any positive scalar multiple is equivalent; return 1.0
        return np.array([1.0])

    Sigma = np.cov(X.values, rowvar=False, ddof=1)  # (k,k)

    # Guard against degenerate shapes
    Sigma = np.atleast_2d(Sigma)

    try:
        invS = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        invS = np.linalg.pinv(Sigma)

    b = (invS @ mu).flatten()

    # fallback if pathological
    if not np.all(np.isfinite(b)) or np.all(np.abs(b) < 1e-12):
        b = np.ones(k)

    return b



def mve_return(F: pd.DataFrame, b: np.ndarray) -> pd.Series:
    X = F.dropna()
    r = X.values @ b
    return pd.Series(r, index=X.index, name="MVE")


def normalize_to_unit_variance(r: pd.Series) -> pd.Series:
    s = r.dropna()
    v = s.var(ddof=1)
    if (v is None) or (not np.isfinite(v)) or v <= 0:
        return r
    return r / np.sqrt(v)


def vol_manage_by_lagged_var(r: pd.Series, varhat: pd.Series) -> pd.DataFrame:
    """
    f_sigma_t = (c / varhat_{t-1}) * f_t
    Choose c so std(f_sigma) == std(f).
    """
    df = pd.DataFrame({"f": r, "varhat": varhat}).dropna().sort_index()
    if df.empty:
        return pd.DataFrame()

    w_raw = 1.0 / df["varhat"].shift(1)
    w_raw = w_raw.replace([np.inf, -np.inf], np.nan)
    f_sig_raw = w_raw * df["f"]

    out = pd.DataFrame({"f": df["f"], "f_sig_raw": f_sig_raw, "w_raw": w_raw}).dropna()
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
# Regression + metrics
# =============================
def ols_hc1(y: pd.Series, x: pd.Series):
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if df.shape[0] < 10:
        raise ValueError("Too few obs.")
    X = sm.add_constant(df["x"].values, has_constant="add")
    res = sm.OLS(df["y"].values, X).fit(cov_type="HC1")
    return res, df


def annualized_sharpe(r: pd.Series, ann: int) -> float:
    x = r.dropna()
    if x.shape[0] < 2:
        return np.nan
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return (mu / sd) * np.sqrt(ann)


def appraisal_ratio(alpha_per_period: float, resid: np.ndarray, ann: int) -> float:
    te = np.std(resid, ddof=1)
    if te == 0 or not np.isfinite(te):
        return np.nan
    return (alpha_per_period / te) * np.sqrt(ann)


# =============================
# Q5 loaders (FIXED for your formats)
# =============================
def load_q5_daily(path: str) -> pd.DataFrame:
    """
    q5_factors_daily_2024.csv format:
      DATE,R_F,R_MKT,R_ME,R_IA,R_ROE,R_EG
    where DATE is YYYYMMDD and returns are in percent.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    # date col could be DATE or first col
    if "DATE" in df.columns:
        s = df["DATE"].astype(str).str.strip()
        dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    else:
        s = df.iloc[:, 0].astype(str).str.strip()
        # try yyyymmdd then generic
        dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        if dt.isna().mean() > 0.5:
            dt = pd.to_datetime(s, errors="coerce")

    df["date"] = dt
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    # percent -> decimal for numeric columns
    for c in df.columns:
        if c in ["DATE", "date", "year", "month"]:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0

    return df


def load_q5_monthly(path: str) -> pd.DataFrame:
    """
    q5_factors_monthly_2024.csv format (your example):
      year,month,R_F,R_MKT,R_ME,R_IA,R_ROE,R_EG
    values are in percent (monthly).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    if "year" in df.columns and "month" in df.columns:
        y = pd.to_numeric(df["year"], errors="coerce")
        m = pd.to_numeric(df["month"], errors="coerce")
        # Use first day then shift to month-end
        dt = pd.to_datetime(
            (y.astype("Int64").astype(str) + "-" + m.astype("Int64").astype(str) + "-01"),
            errors="coerce",
        )
        df["date"] = dt + pd.offsets.MonthEnd(0)
    else:
        # fallback: try DATE/date/first column
        if "DATE" in df.columns:
            s = df["DATE"].astype(str).str.strip()
        elif "date" in df.columns:
            s = df["date"].astype(str).str.strip()
        else:
            s = df.iloc[:, 0].astype(str).str.strip()

        dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        if dt.isna().mean() > 0.5:
            dt = pd.to_datetime(s, format="%Y%m", errors="coerce")
        if dt.isna().mean() > 0.5:
            dt = pd.to_datetime(s, errors="coerce")

        df["date"] = dt + pd.offsets.MonthEnd(0)

    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    for c in df.columns:
        if c in ["DATE", "date", "year", "month"]:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0

    # ensure month-end frequency
    df = df.resample("ME").last()
    return df


# =============================
# Load factor panels (daily + monthly)
# =============================
def load_all_factors():
    # FF3 daily
    ff3 = parse_french_csv_by_dates(FF3_DAILY, 5, ["date", "Mkt-RF", "SMB", "HML", "RF"])

    # FF5 2x3 daily (for RMW, CMA)
    ff5 = parse_french_csv_by_dates(FF5_2X3_DAILY, 7, ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])

    # Mom daily + optional monthly
    mom_d = load_vwcap_factor_csv(MOM_DAILY)
    if os.path.exists(MOM_MONTHLY):
        mom_m = load_vwcap_factor_csv(MOM_MONTHLY)
        mom_m = to_month_end(mom_m)
    else:
        mom_m = to_month_end(compound_monthly_from_daily(mom_d))

    # Q5 daily + monthly (HXZ)
    q5d = load_q5_daily(Q5_DAILY)
    q5m = load_q5_monthly(Q5_MONTHLY) if os.path.exists(Q5_MONTHLY) else q5d.resample("ME").apply(lambda z: (1.0 + z).prod() - 1.0)

    # Build monthly FF series by compounding daily
    ff3_m = pd.DataFrame({
        "Mkt-RF": compound_monthly_from_daily(ff3["Mkt-RF"]),
        "SMB": compound_monthly_from_daily(ff3["SMB"]),
        "HML": compound_monthly_from_daily(ff3["HML"]),
        "RF": compound_monthly_from_daily(ff3["RF"]),
    }).dropna()

    ff5_m = pd.DataFrame({
        "RMW": compound_monthly_from_daily(ff5["RMW"]),
        "CMA": compound_monthly_from_daily(ff5["CMA"]),
    }).dropna()

    return {
        "ff3_d": ff3,
        "ff5_d": ff5,
        "mom_d": mom_d,
        "mom_m": mom_m,
        "ff3_m": ff3_m,
        "ff5_m": ff5_m,
        "q5_d": q5d,
        "q5_m": q5m,
    }


# =============================
# Build Table II for a given frequency
# =============================
def compute_table_for_frequency(freq: str, data: dict):
    if freq == "monthly":
        ann = ANN_M

        mkt = pd.DataFrame({"Mkt": data["ff3_m"]["Mkt-RF"]})
        ff3 = pd.DataFrame({"Mkt": data["ff3_m"]["Mkt-RF"], "SMB": data["ff3_m"]["SMB"], "HML": data["ff3_m"]["HML"]})
        ff3_mom = ff3.join(data["mom_m"].rename("Mom"), how="inner")

        ff5 = ff3.join(data["ff5_m"], how="inner")
        ff5_mom = ff5.join(data["mom_m"].rename("Mom"), how="inner")

        qcols = ["R_MKT", "R_ME", "R_IA", "R_ROE"]
        if HXZ_INCLUDE_EG:
            qcols = qcols + ["R_EG"]
        hxz = data["q5_m"][qcols].rename(columns={"R_MKT": "MKT", "R_ME": "ME", "R_IA": "IA", "R_ROE": "ROE", "R_EG": "EG"})
        hxz_mom = hxz.join(data["mom_m"].rename("Mom"), how="inner")

        sets = [
            ("Mkt", mkt),
            ("FF3", ff3),
            ("FF3 Mom", ff3_mom),
            ("FF5", ff5),
            ("FF5 Mom", ff5_mom),
            ("HXZ", hxz),
            ("HXZ Mom", hxz_mom),
        ]

        # daily panels for realized monthly variance from daily MVE returns
        mom_d = data["mom_d"]
        q5_d = data["q5_d"][qcols].rename(columns={"R_MKT": "MKT", "R_ME": "ME", "R_IA": "IA", "R_ROE": "ROE", "R_EG": "EG"})

        daily_panels = {
            "Mkt": pd.DataFrame({"Mkt": data["ff3_d"]["Mkt-RF"]}),
            "FF3": pd.DataFrame({"Mkt": data["ff3_d"]["Mkt-RF"], "SMB": data["ff3_d"]["SMB"], "HML": data["ff3_d"]["HML"]}),
            "FF3 Mom": pd.DataFrame({"Mkt": data["ff3_d"]["Mkt-RF"], "SMB": data["ff3_d"]["SMB"], "HML": data["ff3_d"]["HML"]}).join(mom_d.rename("Mom"), how="inner"),
            "FF5": pd.DataFrame({"Mkt": data["ff3_d"]["Mkt-RF"], "SMB": data["ff3_d"]["SMB"], "HML": data["ff3_d"]["HML"]}).join(data["ff5_d"][["RMW", "CMA"]], how="inner"),
            "FF5 Mom": pd.DataFrame({"Mkt": data["ff3_d"]["Mkt-RF"], "SMB": data["ff3_d"]["SMB"], "HML": data["ff3_d"]["HML"]}).join(data["ff5_d"][["RMW", "CMA"]], how="inner").join(mom_d.rename("Mom"), how="inner"),
            "HXZ": q5_d,
            "HXZ Mom": q5_d.join(mom_d.rename("Mom"), how="inner"),
        }

        def varhat_builder(label: str, b: np.ndarray) -> pd.Series:
            Fd = daily_panels[label].dropna()
            if Fd.empty:
                return pd.Series(dtype=float)
            rd = pd.Series(Fd.values @ b, index=Fd.index)
            return estimate_monthly_var_from_daily(rd, method="month_realized")

    elif freq == "daily":
        ann = ANN_D

        mkt = pd.DataFrame({"Mkt": data["ff3_d"]["Mkt-RF"]})
        ff3 = pd.DataFrame({"Mkt": data["ff3_d"]["Mkt-RF"], "SMB": data["ff3_d"]["SMB"], "HML": data["ff3_d"]["HML"]})
        ff3_mom = ff3.join(data["mom_d"].rename("Mom"), how="inner")

        ff5 = ff3.join(data["ff5_d"][["RMW", "CMA"]], how="inner")
        ff5_mom = ff5.join(data["mom_d"].rename("Mom"), how="inner")

        qcols = ["R_MKT", "R_ME", "R_IA", "R_ROE"]
        if HXZ_INCLUDE_EG:
            qcols = qcols + ["R_EG"]
        hxz = data["q5_d"][qcols].rename(columns={"R_MKT": "MKT", "R_ME": "ME", "R_IA": "IA", "R_ROE": "ROE", "R_EG": "EG"})
        hxz_mom = hxz.join(data["mom_d"].rename("Mom"), how="inner")

        sets = [
            ("Mkt", mkt),
            ("FF3", ff3),
            ("FF3 Mom", ff3_mom),
            ("FF5", ff5),
            ("FF5 Mom", ff5_mom),
            ("HXZ", hxz),
            ("HXZ Mom", hxz_mom),
        ]

        def varhat_builder(label: str, b: np.ndarray) -> pd.Series:
            Fd = dict(sets)[label].dropna()
            if Fd.empty:
                return pd.Series(dtype=float)
            rd = pd.Series(Fd.values @ b, index=Fd.index)
            return estimate_daily_var(rd, method="rolling22")

    else:
        raise ValueError("freq must be 'monthly' or 'daily'")

    # -------------------------
    # Panel A
    # -------------------------
    panelA_rows = [
        "Alpha", "SE(Alpha)", "Observations", "R2", "RMSE",
        "Original Sharpe", "Vol-Managed Sharpe", "Appraisal Ratio"
    ]
    panelA = pd.DataFrame(index=panelA_rows)
    panelB = pd.DataFrame(index=[f"α: {nm}" for nm, _, _ in SUBSAMPLES])

    for label, F in sets:
        F = F.dropna()
        minobs = MIN_OBS_MONTHLY if freq == "monthly" else MIN_OBS_DAILY
        if F.shape[0] < minobs:
            panelA[label] = [""] * len(panelA_rows)
            for nm, _, _ in SUBSAMPLES:
                panelB.loc[f"α: {nm}", label] = ""
            continue

        b = tangency_weights(F)
        r_mve = normalize_to_unit_variance(mve_return(F, b))

        varhat = varhat_builder(label, b)
        if varhat.empty:
            panelA[label] = [""] * len(panelA_rows)
            for nm, _, _ in SUBSAMPLES:
                panelB.loc[f"α: {nm}", label] = ""
            continue

        if freq == "monthly":
            r_idx = to_month_end(r_mve)
            v_idx = to_month_end(varhat)
        else:
            r_idx = r_mve
            v_idx = varhat

        vm = vol_manage_by_lagged_var(r_idx, v_idx)
        if vm.empty or vm.shape[0] < minobs:
            panelA[label] = [""] * len(panelA_rows)
            for nm, _, _ in SUBSAMPLES:
                panelB.loc[f"α: {nm}", label] = ""
            continue

        try:
            res, _ = ols_hc1(vm["f_sigma"], vm["f"])
        except Exception:
            panelA[label] = [""] * len(panelA_rows)
            for nm, _, _ in SUBSAMPLES:
                panelB.loc[f"α: {nm}", label] = ""
            continue

        alpha = float(res.params[0])      # per-period
        se_alpha = float(res.bse[0])
        nobs = int(res.nobs)
        r2 = float(res.rsquared)
        rmse = float(np.sqrt(np.mean(res.resid ** 2)))

        sr_orig = annualized_sharpe(vm["f"], ann=ann)
        sr_vm = annualized_sharpe(vm["f_sigma"], ann=ann)
        ar = appraisal_ratio(alpha_per_period=alpha, resid=res.resid, ann=ann)

        alpha_ann = alpha * ann * 100
        se_alpha_ann = se_alpha * ann * 100
        rmse_ann_pct = rmse * ann * 100

        panelA[label] = [
            f"{alpha_ann:.2f}",
            f"({se_alpha_ann:.2f})",
            f"{nobs:d}",
            f"{r2:.2f}",
            f"{rmse_ann_pct:.2f}",
            f"{sr_orig:.2f}",
            f"{sr_vm:.2f}",
            f"{ar:.2f}",
        ]

        for nm, start, end in SUBSAMPLES:
            sub = vm.loc[(vm.index >= pd.to_datetime(start)) & (vm.index <= pd.to_datetime(end))].copy()
            if sub.shape[0] < minobs:
                panelB.loc[f"α: {nm}", label] = ""
                continue
            try:
                res_sub, _ = ols_hc1(sub["f_sigma"], sub["f"])
                a_sub = float(res_sub.params[0])
                se_sub = float(res_sub.bse[0])
                panelB.loc[f"α: {nm}", label] = f"{(a_sub*ann*100):.2f} ({(se_sub*ann*100):.2f})"
            except Exception:
                panelB.loc[f"α: {nm}", label] = ""

    return panelA, panelB


# =============================
# LaTeX writer
# =============================
def save_table2_latex(panelA: pd.DataFrame, panelB: pd.DataFrame, out_path: str, title: str):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by table2.py\n\n")
        f.write("\\begin{table}[!ht]\n\\centering\n")
        f.write(f"\\caption{{{title}}}\n")
        f.write("\\label{tab:mve_factors}\n\n")

        f.write("\\textbf{Panel A: Mean-Variance Efficient Portfolios (Full Sample)}\\\\\n")
        f.write(panelA.to_latex(escape=False))
        f.write("\n\\\\[6pt]\n")

        f.write("\\textbf{Panel B: Subsample Analysis}\\\\\n")
        f.write(panelB.to_latex(escape=False))
        f.write("\n\\\\[4pt]\n")

        f.write("{\\footnotesize Notes: MVE portfolios use tangency weights $b\\propto \\Sigma^{-1}\\mu$ computed in-sample on the corresponding factor set. ")
        f.write("Volatility management scales by inverse lagged variance of the MVE portfolio and normalizes to match the unconditional standard deviation of the unmanaged MVE portfolio. ")
        f.write("Alphas are from $f^{MVE,\\sigma}=\\alpha+\\beta f^{MVE}+\\varepsilon$ with HC1 standard errors. ")
        f.write("Sharpe ratios are annualized. Appraisal ratio is annualized alpha divided by tracking error.}\n")
        f.write("\\end{table}\n")


# =============================
# Main
# =============================
def main():
    data = load_all_factors()

    A_m, B_m = compute_table_for_frequency("monthly", data)
    out_m = os.path.join(TABLE_DIR, "table2_monthly.tex")
    save_table2_latex(A_m, B_m, out_m, "Mean-Variance Efficient Factors (Monthly; Replication)")

    A_d, B_d = compute_table_for_frequency("daily", data)
    out_d = os.path.join(TABLE_DIR, "table2_daily.tex")
    save_table2_latex(A_d, B_d, out_d, "Mean-Variance Efficient Factors (Daily Analog; Replication)")

    print("Saved:")
    print(f"  {out_m}")
    print(f"  {out_d}")

    print("\nHXZ definition:")
    print("  Uses q5:", "R_MKT, R_ME, R_IA, R_ROE" + (", R_EG" if HXZ_INCLUDE_EG else ""))


if __name__ == "__main__":
    main()
