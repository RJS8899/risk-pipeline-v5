import os, json
from datetime import datetime
from typing import Dict, Any, List, Tuple
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

# -----------------------------
# Config & Country metadata
# -----------------------------
def load_yaml(path="config/indicators.yaml")->Dict[str,Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fetch_wb_countries()->pd.DataFrame:
    """Lädt Länder-Metadaten; fällt bei API-Fehler auf eine kleine Liste zurück (kein Crash)."""
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
        df = pd.DataFrame(out)
        df = df[df["iso3"].str.len()==3]
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame([
            {"iso3":"AUT","name":"Austria","region":"Europe & Central Asia","incomeLevel":"High income"},
            {"iso3":"USA","name":"United States","region":"North America","incomeLevel":"High income"},
            {"iso3":"BRA","name":"Brazil","region":"Latin America & Caribbean","incomeLevel":"Upper middle income"},
            {"iso3":"NGA","name":"Nigeria","region":"Sub-Saharan Africa","incomeLevel":"Lower middle income"},
            {"iso3":"IND","name":"India","region":"South Asia","incomeLevel":"Lower middle income"},
        ])

# -----------------------------
# Source fetch with fallbacks
# -----------------------------
def source_fetch(ind: Dict[str,Any]) -> pd.DataFrame:
    """Gibt DF mit Spalten iso3, year, value zurück (leeres DF bei Fehler)."""
    s = ind["source"]
    t = s["type"]
    try:
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
            return df if not df.empty else pd.DataFrame(columns=["iso3","year","value"])
        if t=="worldbank_or_imf_weo":
            # Robust: vorerst WB-Fallback (IMF WEO kann später ergänzt werden)
            return wb_fetch(s["wb_code_fallback"])
    except Exception:
        pass
    return pd.DataFrame(columns=["iso3","year","value"])

# -----------------------------
# Latest + lookback selection
# -----------------------------
def pick_latest_with_lookback_and_basis(df: pd.DataFrame, lookback: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """
    Nimmt pro iso3 den jüngsten Wert; wenn im letzten Jahr None, schaut bis lookback Jahre zurück.
    Liefert zusätzlich pro iso3 die Basis: 'raw' oder 'lookback'.
    """
    if df.empty:
        return df, {}
    df = df.sort_values(["iso3","year"], ascending=[True, False])
    latest_year = df.groupby("iso3")["year"].max().to_dict()
    chosen_rows = []
    basis_map: Dict[str,str] = {}
    for iso, y_latest in latest_year.items():
        sub = df[df["iso3"]==iso]
        used_row = None
        used_basis = None
        for k in range(0, lookback+1):
            yy = y_latest - k
            row = sub[sub["year"]==yy]
            if not row.empty:
                used_row = row.iloc[0]
                used_basis = "raw" if k==0 else "lookback"
                break
        if used_row is not None:
            chosen_rows.append(used_row)
            basis_map[iso] = used_basis
    used = pd.DataFrame(chosen_rows).reset_index(drop=True)
    return used, basis_map

# -----------------------------
# Imputation with basis tags
# -----------------------------
def impute_with_basis(series: pd.Series, df_values: pd.DataFrame, countries: pd.DataFrame,
                      order: List[str], basis_init: Dict[str,str]) -> Tuple[pd.Series, pd.Series]:
    """
    series: index=iso3, value=raw/used (nan möglich)
    df_values: 'used' (iso3,year,value) für die, die wir haben
    basis_init: iso3-> 'raw'|'lookback' für vorhandene Werte
    returns: (values, basis) mit basis in {'raw','lookback','region','income','global'}
    """
    values = series.copy()
    basis = pd.Series(index=series.index, dtype=object)
    for iso, b in basis_init.items():
        if iso in basis.index:
            basis.loc[iso] = b

    meta = countries.set_index("iso3")
    # Helper: Median pro Region/Income
    for level in order:
        mask = values.isna()
        if not mask.any():
            break
        if level=="region":
            grp = df_values.join(meta[["region"]], on="iso3").groupby("region")["value"].median()
            for iso in values[mask].index:
                r = meta.loc[iso]["region"] if iso in meta.index else None
                v = grp.get(r, np.nan)
                if pd.notna(v):
                    values.loc[iso] = v
                    basis.loc[iso] = "region"
        elif level=="income":
            grp = df_values.join(meta[["incomeLevel"]], on="iso3").groupby("incomeLevel")["value"].median()
            for iso in values[mask].index:
                g = meta.loc[iso]["incomeLevel"] if iso in meta.index else None
                v = grp.get(g, np.nan)
                if pd.notna(v):
                    values.loc[iso] = v
                    basis.loc[iso] = "income"
        elif level=="global":
            v = df_values["value"].median() if not df_values.empty else np.nan
            if pd.notna(v):
                values[mask] = v
                basis[mask] = "global"
    return values, basis

# -----------------------------
# Normalization 0..100
# -----------------------------
def normalize(values: pd.Series, spec: Dict[str,Any]) -> pd.Series:
    v = values.astype(float)
    method = spec.get("method","percentile_clamp")
    higher_is_riskier = spec.get("higher_is_riskier", True)

    if spec.get("invert_sign"): v = -v
    if spec.get("invert"): v = v.max() + v.min() - v

    if method=="invert_0_100":
        return (100 - v).clip(0,100)

    if method=="wgi":
        # WGI ca. -2.5..+2.5; höher = besser -> Risiko hoch bei niedrig
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

    # default: robuste Perzentil-Klammerung
    if v.notna().sum() >= 4:
        lo, hi = np.nanpercentile(v, [spec.get("p_low",5), spec.get("p_high",95)])
    else:
        lo, hi = (v.min(), v.max())
    r = ((v - lo) / (hi - lo)) * 100.0 if hi!=lo else pd.Series(50.0, index=v.index)
    if not higher_is_riskier:
        r = 100.0 - r
    return np.clip(r, 0, 100)

# -----------------------------
# Main run
# -----------------------------
def run_pipeline():
    cfg = load_yaml()
    os.makedirs("out", exist_ok=True)

    countries = fetch_wb_countries()
    iso_list = countries["iso3"].tolist()

    # Speicher
    raw_values: Dict[str,pd.Series] = {}
    norm_values: Dict[str,pd.Series] = {}
    basis_values: Dict[str,pd.Series] = {}
    years_used: Dict[str,pd.Series] = {}
    indicator_meta: Dict[str,Any] = {}

    # ---- fetch + lookback + impute + normalize per indicator ----
    for ind in cfg["indicators"]:
        iid = ind["id"]
        lookback = ind.get("lookback_years", cfg["defaults"]["lookback_years"])
        spec_norm = {**cfg["defaults"]["normalization"], **ind.get("normalization",{})}

        df = source_fetch(ind)  # iso3, year, value
        source_empty = df.empty

        # Optional Transform (z. B. per 100k → per 1k)
        if "transform" in ind and "scale" in ind["transform"] and not df.empty:
            df["value"] = df["value"] * float(ind["transform"]["scale"])

        # Auswahl: jüngstes / Lookback + Basis markieren
        used, basis0 = pick_latest_with_lookback_and_basis(df, lookback)

        # Nur Länder aus WB-Liste
        used = used[used["iso3"].isin(iso_list)]

        ser_used = used.set_index("iso3")["value"] if not used.empty else pd.Series(dtype=float)
        ser_year = used.set_index("iso3")["year"] if not used.empty else pd.Series(dtype=float)

        # Imputation (Region → Income → Global) + Basis-Tags
        impute_order = cfg["defaults"]["imputation"]["order"]
        values_full, basis_full = impute_with_basis(
            ser_used.reindex(iso_list),
            used,
            countries,
            impute_order,
            basis0
        )

        # Normalisierung 0..100
        norm_full = normalize(values_full, spec_norm)

        # Speichern
        raw_values[iid] = values_full
        norm_values[iid] = norm_full
        basis_values[iid] = basis_full
        years_used[iid] = ser_year.reindex(iso_list)

        # Meta pro Indikator
        bcounts = basis_full.value_counts(dropna=False).to_dict()
        indicator_meta[iid] = {
            "name": ind["name"],
            "pillar": ind["pillar"],
            "group": ind["group"],
            "source_type": ind["source"]["type"],
            "optional": bool(ind.get("optional", False)),
            "coverage_share": float(values_full.notna().mean()) if len(values_full)>0 else 0.0,
            "basis_counts": {str(k if pd.notna(k) else "missing"): int(v) for k,v in bcounts.items()},
            "year_min": int(pd.to_numeric(ser_year, errors="coerce").min()) if ser_year.notna().any() else None,
            "year_max": int(pd.to_numeric(ser_year, errors="coerce").max()) if ser_year.notna().any() else None,
            "source_empty": bool(source_empty),
        }

    # ---- Aggregation: Gruppen, Pillars, Total (skipna mean) ----
    df_wide = pd.DataFrame({"iso3": iso_list}).set_index("iso3")

    # Roh- / Norm- / Basis-Spalten je Indikator
    for ind in cfg["indicators"]:
        iid = ind["id"]
        df_wide[f"raw__{iid}"] = raw_values[iid]
        df_wide[f"norm__{iid}"] = norm_values[iid]
        df_wide[f"basis__{iid}"] = basis_values[iid]

    # Gruppen- & Pillar-Mapping
    group_map: Dict[str,List[str]] = {}
    pillar_map: Dict[str,List[str]] = {}
    for ind in cfg["indicators"]:
        gid = f"{ind['pillar']}::{ind['group']}"
        group_map.setdefault(gid, []).append(ind["id"])
        pillar_map.setdefault(ind["pillar"], []).append(ind["id"])

    # Gruppenwerte (gleiches Gewicht über Indikatoren; skipna)
    for gid, ids in group_map.items():
        cols = [f"norm__{i}" for i in ids]
        df_wide[f"group__{gid}"] = df_wide[cols].mean(axis=1, skipna=True)

    # Pillarwerte
    for pid, ids in pillar_map.items():
        cols = [f"norm__{i}" for i in ids]
        df_wide[f"pillar__{pid}"] = df_wide[cols].mean(axis=1, skipna=True)

    # Total (gleiches Gewicht über die definierten Pillars)
    pillars = [p["id"] for p in cfg["pillars"]]
    df_wide["risk_total"] = df_wide[[f"pillar__{p}" for p in pillars]].mean(axis=1, skipna=True)

    df_wide = df_wide.reset_index()

    # Years je Indikator für Transparenz
    years_df = pd.DataFrame({"iso3": iso_list}).set_index("iso3")
    for iid, yr in years_used.items():
        years_df[f"year__{iid}"] = yr
    years_df = years_df.reset_index()
    df_wide = df_wide.merge(years_df, on="iso3", how="left")

    # LONG-Format (API-freundlich)
    recs=[]
    for _, row in df_wide.iterrows():
        iso = row["iso3"]
        # Indicators
        for iid in raw_values.keys():
            recs.append({
                "iso3": iso,
                "kind": "indicator",
                "id": iid,
                "raw": row.get(f"raw__{iid}"),
                "norm": row.get(f"norm__{iid}"),
                "basis": row.get(f"basis__{iid}"),
                "year": row.get(f"year__{iid}"),
            })
        # Groups
        for gid in group_map.keys():
            recs.append({"iso3": iso, "kind":"group", "id": gid, "norm": row.get(f"group__{gid}")})
        # Pillars
        for pid in pillar_map.keys():
            recs.append({"iso3": iso, "kind":"pillar", "id": pid, "norm": row.get(f"pillar__{pid}")})
        # Total
        recs.append({"iso3": iso, "kind":"total", "id":"risk_total", "norm": row["risk_total"]})
    df_long = pd.DataFrame(recs)

    # OUTPUTS
    os.makedirs("out", exist_ok=True)
    df_wide.to_csv("out/risk_wide.csv", index=False)
    df_long.to_csv("out/risk_long.csv", index=False)

    # Coverage-Zusammenfassung
    coverage_rows = []
    for iid, meta in indicator_meta.items():
        coverage_rows.append({
            "indicator": iid,
            "coverage_share": meta["coverage_share"],
            "source_type": meta["source_type"],
            "optional": meta["optional"],
            "year_min": meta["year_min"],
            "year_max": meta["year_max"],
            "source_empty": meta["source_empty"],
            **{f"basis_{k}": v for k,v in meta["basis_counts"].items()}
        })
    pd.DataFrame(coverage_rows).to_csv("out/coverage.csv", index=False)

    # META
    meta = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "indicators": indicator_meta,
        "notes": [
            "Nicht erreichbare/optionale Quellen werden neutral ausgelassen (skipna), Aggregation bleibt robust.",
            "Imputation: Lookback, dann Region/Income/Global (Median).",
            "Normalization: indikator-spezifisch, 0 (gut) .. 100 (schlecht).",
        ]
    }
    with open("out/meta.json","w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("OK - wrote out/risk_wide.csv, out/risk_long.csv, out/coverage.csv, out/meta.json")
