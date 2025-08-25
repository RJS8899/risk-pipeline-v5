import io
import requests
import pandas as pd

OWID_CSV = "https://ourworldindata.org/grapher/{slug}.csv"

ISO_FIX = {
    "OWID_KOS": "XKX",
}

def _clean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if "Code" in df.columns:
        df["iso3"] = df["Code"].map(lambda x: ISO_FIX.get(x, x))
    else:
        df["iso3"] = None
    df = df[df["iso3"].str.len() == 3]
    out = df.rename(columns={"Year": "year", value_col: "value"})[["iso3", "year", "value"]]
    out = out.dropna(subset=["value"])
    out["year"] = out["year"].astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"])

def fetch_grapher(slug: str, value_col: str = None) -> pd.DataFrame:
    url = OWID_CSV.format(slug=slug)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        if value_col is None:
            cols = [c for c in df.columns if c not in ("Entity", "Code", "Year")]
            if not cols:
                return pd.DataFrame(columns=["iso3", "year", "value"])
            value_col = cols[0]
        return _clean(df, value_col)
    except Exception:
        return pd.DataFrame(columns=["iso3", "year", "value"])

def _asof_join_population(values: pd.DataFrame, pop: pd.DataFrame, max_lag_years: int = 5) -> pd.DataFrame:
    """
    Robust: mapt je (iso3, year_val) die zuletzt verfügbare Population (<= year_val, bis max_lag_years zurück).
    """
    if values.empty or pop.empty:
        return pd.DataFrame(columns=["iso3", "year", "value"])

    vals = values.sort_values(["iso3", "year"]).copy()
    p = pop.sort_values(["iso3", "year"]).rename(columns={"value": "pop"}).copy()

    out_rows = []
    for iso, grp in vals.groupby("iso3"):
        vg = grp.sort_values("year")
        pg = p[p["iso3"] == iso].sort_values("year")
        if pg.empty:
            continue
        # Für jedes value-Jahr die neueste Pop <= year nehmen
        pop_years = pg["year"].values
        pop_vals = pg["pop"].values
        for _, row in vg.iterrows():
            y = int(row["year"])
            # Index des letzten Pop-Jahres <= y
            idx = (pop_years <= y).nonzero()[0]
            if len(idx) == 0:
                continue
            j = idx[-1]
            py, pv = int(pop_years[j]), float(pop_vals[j])
            # nur verwenden, wenn nicht zu alt
            if y - py <= max_lag_years and pv > 0:
                out_rows.append({"iso3": iso, "year": y, "value": float(row["value"]) / pv})
    if not out_rows:
        return pd.DataFrame(columns=["iso3", "year", "value"])
    return pd.DataFrame(out_rows)

def fetch_percap(slug_value: str, slug_population: str, per: float) -> pd.DataFrame:
    v = fetch_grapher(slug_value)
    p = fetch_grapher(slug_population)
    if v.empty or p.empty:
        return pd.DataFrame(columns=["iso3", "year", "value"])
    df = _asof_join_population(v, p, max_lag_years=5)
    if df.empty:
        return df
    df["value"] = df["value"] * per
    return df.dropna(subset=["value"])[["iso3", "year", "value"]]

