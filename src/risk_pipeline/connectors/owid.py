import io, requests, pandas as pd

OWID_CSV = "https://ourworldindata.org/grapher/{slug}.csv"

ISO_FIX = {
    "OWID_KOS": "XKX",
}

def _clean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if "Code" in df.columns:
        df["iso3"] = df["Code"].map(lambda x: ISO_FIX.get(x, x))
    else:
        df["iso3"] = None
    df = df[df["iso3"].str.len()==3]
    return df.rename(columns={"Year":"year", value_col:"value"})[["iso3","year","value"]].dropna()

def fetch_grapher(slug: str, value_col: str=None) -> pd.DataFrame:
    url = OWID_CSV.format(slug=slug)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        if value_col is None:
            cols = [c for c in df.columns if c not in ("Entity","Code","Year")]
            if not cols:
                return pd.DataFrame(columns=["iso3","year","value"])
            value_col = cols[0]
        return _clean(df, value_col)
    except Exception:
        return pd.DataFrame(columns=["iso3","year","value"])

def fetch_percap(slug_value: str, slug_population: str, per: float) -> pd.DataFrame:
    v = fetch_grapher(slug_value)
    p = fetch_grapher(slug_population)
    if v.empty or p.empty:
        return pd.DataFrame(columns=["iso3","year","value"])
    p = p.rename(columns={"value":"pop"})
    df = v.merge(p, on=["iso3","year"], how="left")
    df["value"] = (df["value"] / df["pop"]) * per
    return df.dropna(subset=["value"])[["iso3","year","value"]]
