import io
import re
import zipfile
import requests
import pandas as pd

# Zwei mögliche Seiten (alte & neue Struktur)
PAGES = [
    "https://gain.nd.edu/our-work/country-index/download-data/",
    "https://gain-new.crc.nd.edu/about/download",
]

def _find_zip_url(html: str) -> str:
    # 1) Direkter .zip-Link im HTML
    m = re.search(r'href=["\']([^"\']+\\.zip)["\']', html, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # 2) Fallback: data-href o. ä.
    m = re.search(r'data-href=["\']([^"\']+\\.zip)["\']', html, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return ""

def fetch_nd_gain() -> pd.DataFrame:
    """
    Holt ND-GAIN Index (Land-Jahr-Score) aus ZIP (CSV) – robust:
    - Probiert beide bekannten Download-Seiten
    - Folgt Redirects
    - Sucht die CSV mit 'index' im Namen
    """
    zip_url = ""
    for page in PAGES:
        try:
            html = requests.get(page, timeout=60, allow_redirects=True).text
            z = _find_zip_url(html)
            if z:
                zip_url = z if z.startswith("http") else requests.compat.urljoin(page, z)
                break
        except Exception:
            continue

    if not zip_url:
        return pd.DataFrame(columns=["iso3", "year", "value"])

    try:
        zr = requests.get(zip_url, timeout=120, allow_redirects=True)
        zr.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(zr.content)) as zf:
            # bevorzugt CSV mit "index" im Namen
            name = next((n for n in zf.namelist() if n.lower().endswith(".csv") and "index" in n.lower()), None)
            if not name:
                # sonst erste CSV
                name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
            if not name:
                return pd.DataFrame(columns=["iso3", "year", "value"])
            df = pd.read_csv(zf.open(name))
        # Spalten erkennen
        cols_lower = {c.lower(): c for c in df.columns}
        iso_col = cols_lower.get("iso3") or cols_lower.get("iso") or cols_lower.get("country code")
        year_col = cols_lower.get("year")
        # häufige Value-Bezeichner
        val_col = None
        for k in ["index", "score", "nd-gain", "nd_gain", "ndgain"]:
            if k in cols_lower:
                val_col = cols_lower[k]; break
        if not (iso_col and year_col and val_col):
            return pd.DataFrame(columns=["iso3", "year", "value"])
        out = df.rename(columns={iso_col: "iso3", year_col: "year", val_col: "value"})[["iso3", "year", "value"]]
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["iso3", "year", "value"])
        out["year"] = out["year"].astype(int)
        return out
    except Exception:
        return pd.DataFrame(columns=["iso3", "year", "value"])
