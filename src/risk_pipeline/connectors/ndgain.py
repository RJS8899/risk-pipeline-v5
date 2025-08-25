import io, re, zipfile, requests, pandas as pd

ND_GAIN_DL = "https://gain.nd.edu/our-work/country-index/download-data/"

def fetch_nd_gain() -> pd.DataFrame:
    try:
        html = requests.get(ND_GAIN_DL, timeout=60).text
        m = re.search(r'href="([^"]+\.zip)"', html)
        if not m:
            return pd.DataFrame(columns=["iso3","year","value"])
        zurl = m.group(1)
        zr = requests.get(zurl, timeout=120)
        zr.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(zr.content)) as zf:
            name = next((n for n in zf.namelist() if n.lower().endswith(".csv") and "index" in n.lower()), None)
            if not name:
                return pd.DataFrame(columns=["iso3","year","value"])
            df = pd.read_csv(zf.open(name))
        iso = [c for c in df.columns if c.upper()=="ISO3"]
        year = [c for c in df.columns if c.lower()=="year"]
        val = [c for c in df.columns if "index" in c.lower()]
        if not iso or not year or not val:
            return pd.DataFrame(columns=["iso3","year","value"])
        out = df.rename(columns={iso[0]:"iso3", year[0]:"year", val[0]:"value"})[["iso3","year","value"]]
        out["year"] = out["year"].astype(int)
        return out.dropna()
    except Exception:
        return pd.DataFrame(columns=["iso3","year","value"])
