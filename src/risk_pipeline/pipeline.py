import os, json, math
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import yaml
import requests

from .connectors.worldbank import fetch_indicator as wb_fetch
from .connectors.wgi import fetch_wgi
from .connectors.owid import fetch_grapher, fetch_percap
from .connectors.ndgain import fetch_nd_gain
from .connectors.acled import fetch_acled_fatalities
from .connectors.vendor_csv import fetch_vendor_csv

WB_COUNTRIES = "https://api.worldbank.org/v2/country?format=json&per_page=400"

def load_yaml(path="config/indicators.yaml")->Dict[str,Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fetch_wb_countries()->pd.DataFrame:
    try:
        r = requests.get(WB_COUNTRIES, timeout=30)
        r.raise_for_status()
        js = r.json()
        rows = js[1]
        out=[]
        for x in rows:
            out.append({
                "iso3": x.get("id"),
                "name": x.get("name"),
                "region": (x.get("region") or {}).get("value"),
                "incomeLevel": (x.get("incomeLevel") or {}).get("value"),
            })
        return pd.DataFrame(out)
    except Exception:
        # Safe fallback minimal set (no crash)
        return pd.DataFrame([
            {"iso3":"AUT","name":"Austria","region":"Europe & Central Asia","incomeLevel":"High income"},
            {"iso3":"USA","name":"United States","region":"North America","incomeLevel":"High income"},
            {"iso3":"BRA","name":"Brazil","region":"Latin America & Caribbean","incomeLevel":"Upper middle income"},
            {"iso3":"NGA","name":"Nigeria","region":"Sub-Saharan Africa","incomeLevel":"Lower middle income"},
            {"iso3":"IND","name":"India","region":"South Asia","incomeLevel":"Lower middle income"},
        ])

def source_fetch(ind: Dict[str,Any]) -> pd.DataFrame:
    s = ind["source"]
    t = s["type"]
    if t=="worldbank":
        return wb_fetch(s["code"])
    if t=="wgi":
        return fetch_wgi(s["code"])
    if t=="owid":
        return fetch_grapher(s["slug"])
    if t=="owid_percap":
        return fetch_percap(s["slug_value"], s["slug_population"], s.get("per",100000.0))
    if t=="nd_gain":
        return fetch_nd_gain()
    if t=="acled":
        return fetch_acled_fatalities()
    if t=="vendor_csv":
        return fetch_vendor_csv(s["filename"])
    if t=="vendor_csv_or_gpi":
        df = fetch_vendor_csv("gpi_scores.csv")
        if not df.empty: return df
        return pd.DataFrame(columns=["iso3","year","value"])
    if t=="worldbank_or_imf_weo":
        return wb_fetch(s["wb_code_fallback"])
    return pd.DataFrame(columns=["iso3","year","value"])

def pick_latest_with_lookback(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if df.empty: return df
    df = df.sort_values(["iso3","year"], ascending=[True, False])
    latest_year = df.groupby("iso3")["year"].max().to_dict()
    keep=[]
    for iso, y in latest_year.items():
        sub = df[df["iso3"]==iso]
        for k in range(0, lookback+1):
            yy = y - k
            row = sub[sub["year"]==yy]
            if not row.empty:
                keep.append(row.iloc[0])
                break
    return pd.DataFrame(keep)

def impute_missing(series: pd.Series, df_values: pd.DataFrame, countries: pd.DataFrame, order: List[str]) -> pd.Series:
    out = series.copy()
    meta = countries.set_index("iso3")
    for level in order:
        mask = out.isna()
        if not mask.any(): break
        if level=="region":
            grp = df_values.join(meta[["region"]], on="iso3").groupby("region")["value"].median()
            for iso in out[mask].index:
                r = meta.loc[iso]["region"] if iso in meta.index else None
                v = grp.get(r, np.nan)
                if pd.notna(v): out.loc[iso]=v
        elif level=="income":
            grp = df_values.join(meta[["incomeLevel"]], on="iso3").groupby("incomeLevel")["value"].median()
            for iso in out[mask].index:
                g = meta.loc[iso]["incomeLevel"] if iso in meta.index else None
                v = grp.get(g, np.nan)
                if pd.notna(v): out.loc[iso]=v
        elif level=="global":
            v = df_values["value"].median() if not df_values.empty else np.nan
            out[mask] = v
    return out

def normalize(values: pd.Series, spec: Dict[str,Any]) -> pd.Series:
    v = values.astype(float)
    method = spec.get("method","percentile_clamp")
    higher_is_riskier = spec.get("higher_is_riskier", True)
    if spec.get("invert_sign"): v = -v
    if spec.get("invert"): v = v.max() + v.min() - v
    if method=="invert_0_100":
        return (100 - v).clip(0,100)
    if method=="wgi":
        r = ((2.5 - v) / 5.0) * 100.0
        return np.clip(r, 0, 100)
    if method=="linear_minmax":
        mn, mx = float(spec.get("min", v.min())), float(spec.get("max", v.max()))
        r = ((v - mn) / (mx - mn)) * 100.0
        if not higher_is_riskier:
            r = 100.0 - r
        return np.clip(r, 0, 100)
    if method=="log_cap":
        cap = float(spec.get("cap", np.nanpercentile(v.dropna(), 95) if v.notna().any() else 1.0))
        r = np.log1p(v.clip(lower=0)) / np.log1p(cap) * 100.0
        return np.clip(r, 0, 100)
    if v.notna().sum() >= 4:
        lo, hi = np.nanpercentile(v, [spec.get("p_low",5), spec.get("p_high",95)])
    else:
        lo, hi = (v.min(), v.max())
    r = ((v - lo) / (hi - lo)) * 100.0 if hi!=lo else pd.Series(50.0, index=v.index)
    if not higher_is_riskier:
        r = 100.0 - r
    return np.clip(r, 0, 100)

def run_pipeline():
    cfg = load_yaml()
    os.makedirs("out", exist_ok=True)
    countries = fetch_wb_countries()
    iso_list = countries["iso3"].tolist()

    meta_status = {}
    raw_values = {}
    years_used  = {}

    for ind in cfg["indicators"]:
        iid = ind["id"]
        lookback = ind.get("lookback_years", cfg["defaults"]["lookback_years"])
        df = source_fetch(ind)

        if "transform" in ind and "scale" in ind["transform"] and not df.empty:
            df["value"] = df["value"] * float(ind["transform"]["scale"])

        used = pick_latest_with_lookback(df, lookback)
        used = used[used["iso3"].isin(iso_list)]

        ser = used.set_index("iso3")["value"] if not used.empty else pd.Series(dtype=float)
        yr  = used.set_index("iso3")["year"] if not used.empty else pd.Series(dtype=float)

        imputed = impute_missing(ser.reindex(iso_list), used, countries, cfg["defaults"]["imputation"]["order"])
        raw_values[iid] = imputed
        years_used[iid] = yr.reindex(iso_list)

        meta_status[iid] = {
            "name": ind["name"],
            "source_type": ind["source"]["type"],
            "optional": ind.get("optional", False),
            "coverage_share": float(imputed.notna().mean()) if len(imputed)>0 else 0.0,
        }

    norm_values = {}
    for ind in cfg["indicators"]:
        iid = ind["id"]
        spec = {**cfg["defaults"]["normalization"], **ind.get("normalization",{})}
        v = raw_values[iid]
        norm_values[iid] = normalize(v, spec)

    df_wide = pd.DataFrame({"iso3": iso_list}).set_index("iso3")
    for ind in cfg["indicators"]:
        iid = ind["id"]
        df_wide[f"raw__{iid}"] = raw_values[iid]
        df_wide[f"norm__{iid}"] = norm_values[iid]

    group_map = {}
    pillar_map = {}
    for ind in cfg["indicators"]:
        gid = f"{ind['pillar']}::{ind['group']}"
        group_map.setdefault(gid, []).append(ind["id"])
        pillar_map.setdefault(ind["pillar"], []).append(ind["id"])

    for gid, ids in group_map.items():
        cols = [f"norm__{i}" for i in ids]
        df_wide[f"group__{gid}"] = df_wide[cols].mean(axis=1, skipna=True)

    for pid, ids in pillar_map.items():
        cols = [f"norm__{i}" for i in ids]
        df_wide[f"pillar__{pid}"] = df_wide[cols].mean(axis=1, skipna=True)

    pillars = [p["id"] for p in cfg["pillars"]]
    df_wide["risk_total"] = df_wide[[f"pillar__{p}" for p in pillars]].mean(axis=1, skipna=True)
    df_wide = df_wide.reset_index()

    years_df = pd.DataFrame({"iso3": iso_list}).set_index("iso3")
    for iid, yr in years_used.items():
        years_df[f"year__{iid}"] = yr
    years_df = years_df.reset_index()
    df_wide = df_wide.merge(years_df, on="iso3", how="left")

    recs=[]
    for _, row in df_wide.iterrows():
        iso = row["iso3"]
        for ind in cfg["indicators"]:
            iid = ind["id"]
            recs.append({"iso3": iso, "kind":"indicator", "id": iid, "raw": row.get(f"raw__{iid}"), "norm": row.get(f"norm__{iid}"), "year": row.get(f"year__{iid}")})
        for gid in group_map.keys():
            recs.append({"iso3": iso, "kind":"group", "id": gid, "norm": row.get(f"group__{gid}")})
        for pid in pillar_map.keys():
            recs.append({"iso3": iso, "kind":"pillar", "id": pid, "norm": row.get(f"pillar__{pid}")})
        recs.append({"iso3": iso, "kind":"total", "id":"risk_total", "norm": row["risk_total"]})
    df_long = pd.DataFrame(recs)

    df_wide.to_csv("out/risk_wide.csv", index=False)
    df_long.to_csv("out/risk_long.csv", index=False)
    meta = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "notes": [
            "Sources fetched with robust fallbacks; optional sources (ACLED/GPI/Robbery) do not break the run.",
            "Imputation: 3y lookback, then region/income/global medians.",
            "Normalization: indicator-specific to 0..100 (higher=worse).",
        ]
    }
    with open("out/meta.json","w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("OK - wrote out/risk_wide.csv, out/risk_long.csv, out/meta.json")
