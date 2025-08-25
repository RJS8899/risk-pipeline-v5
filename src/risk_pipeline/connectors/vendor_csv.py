import os, pandas as pd

def fetch_vendor_csv(filename: str) -> pd.DataFrame:
    path = os.path.join("data","vendor",filename)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["iso3","year","value"])
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    iso = cols.get("iso3", "iso3")
    year = cols.get("year", "year")
    val = cols.get("value", "value")
    out = df.rename(columns={iso:"iso3", year:"year", val:"value"})[["iso3","year","value"]]
    out["year"] = out["year"].astype(int)
    return out.dropna()
