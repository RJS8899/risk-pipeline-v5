import io
import os
import re
import zipfile
import requests
import pandas as pd
from urllib.parse import urljoin

# Bekannte Download-Seiten (alt + neu)
PAGES = [
    "https://gain.nd.edu/our-work/country-index/download-data/",
    "https://gain-new.crc.nd.edu/about/download",
]

UA = {
    "User-Agent": "risk-pipeline/1.0 (+github-actions)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def _wb_country_map() -> pd.DataFrame:
    """
    Holt ISO3 + Ländername von der World Bank, um ggf. 'Country' -> ISO3 zu mappen.
    Rückgabe: DataFrame mit ['iso3','name','name_lc'].
    """
    try:
        r = requests.get("https://api.worldbank.org/v2/country?format=json&per_page=400",
                         timeout=30, headers=UA)
        r.raise_for_status()
        js = r.json()
        rows = js[1]
        out=[]
        for x in rows:
            iso3 = x.get("id")
            name = x.get("name")
            if iso3 and isinstance(iso3, str) and len(iso3) == 3:
                out.append({"iso3": iso3, "name": name})
        df = pd.DataFrame(out)
        df["name_lc"] = df["name"].astype(str).str.strip().str.lower()
        return df
    except Exception:
        # Minimaler Fallback
        df = pd.DataFrame([
            {"iso3":"USA","name":"United States","name_lc":"united states"},
            {"iso3":"GBR","name":"United Kingdom","name_lc":"united kingdom"},
            {"iso3":"DEU","name":"Germany","name_lc":"germany"},
            {"iso3":"FRA","name":"France","name_lc":"france"},
            {"iso3":"JPN","name":"Japan","name_lc":"japan"},
            {"iso3":"IND","name":"India","name_lc":"india"},
        ])
        return df

def _find_download_candidates(html: str, base: str) -> list[str]:
    """
    Sucht .zip/.csv/.xlsx Links in HTML (href= oder data-href=) und macht sie absolut.
    """
    urls = []
    for ext in (".zip", ".csv", ".xlsx", ".xls"):
        pat = rf'(?:href|data-href)=["\']([^"\']+{re.escape(ext)})["\']'
        urls += re.findall(pat, html, flags=re.IGNORECASE)
    abs_urls, seen = [], set()
    for u in urls:
        full = u if u.startswith("http") else urljoin(base, u)
        if full not in seen:
            seen.add(full)
            abs_urls.append(full)
    return abs_urls

def _read_ndgain_from_csv_bytes(raw: bytes) -> pd.DataFrame:
    """
    Versucht, ND-GAIN aus einer CSV zu lesen.
    Erkennt Spalten: iso3|iso|country code|country, year, (index|score|nd_gain*|...).
    Mappt ggf. Country -> ISO3 via World-Bank-Länderliste.
    """
    # Primärer Read (UTF-8), Fallback Latin-1
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    if df is None or df.empty:
        return pd.DataFrame(columns=["iso3","year","value"])

    cols_lower = {c.lower(): c for c in df.columns}
    iso_col = cols_lower.get("iso3") or cols_lower.get("iso") or cols_lower.get("country code")
    country_col = cols_lower.get("country") or cols_lower.get("country_name") or cols_lower.get("name")
    year_col = cols_lower.get("year")

    # Value-Spalte identifizieren
    val_col = None
    for k in ["index", "score", "nd-gain", "nd_gain", "ndgain", "nd_gain_score", "indexscore", "country_index", "gain_score"]:
        if k in cols_lower:
            val_col = cols_lower[k]; break

    if not year_col or not val_col:
        return pd.DataFrame(columns=["iso3","year","value"])

    out = df.copy()
    if iso_col:
        out = out.rename(columns={iso_col: "iso3", year_col: "year", val_col: "value"})
    elif country_col:
        out = out.rename(columns={country_col: "country", year_col: "year", val_col: "value"})
        cmap = _wb_country_map()
        out["country_lc"] = out["country"].astype(str).str.strip().str.lower()
        out = out.merge(cmap[["iso3","name_lc"]], left_on="country_lc", right_on="name_lc", how="left")
    else:
        return pd.DataFrame(columns=["iso3","year","value"])

    # Säubern
    if "iso3" not in out.columns:
        return pd.DataFrame(columns=["iso3","year","value"])

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["iso3","year","value"])
    out["year"] = out["year"].astype(int)
    out = out[["iso3","year","value"]].drop_duplicates()
    # ISO3-Filter
    out = out[out["iso3"].astype(str).str.len() == 3]
    return out

def _read_index_from_zip(raw_zip: bytes) -> pd.DataFrame:
    """
    Durchsucht ZIP nach plausiblen CSVs/XLSX mit Index/Score-Spalten.
    Bevorzugt Dateien mit 'index' im Namen.
    """
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        names = zf.namelist()
        prefer = [n for n in names if n.lower().endswith(".csv") and "index" in n.lower()]
        others_csv = [n for n in names if n.lower().endswith(".csv") and n not in prefer]
        xlsx = [n for n in names if n.lower().endswith((".xlsx",".xls"))]

        for name in prefer + others_csv:
            try:
                df = _read_ndgain_from_csv_bytes(zf.read(name))
            except Exception:
                continue
            if not df.empty:
                return df

        # XLSX-Unterstützung
        for name in xlsx:
            try:
                data = zf.read(name)
                xdf = pd.read_excel(io.BytesIO(data))
                # Heuristik wie bei CSV
                cols_lower = {c.lower(): c for c in xdf.columns}
                iso_col = cols_lower.get("iso3") or cols_lower.get("iso") or cols_lower.get("country code")
                country_col = cols_lower.get("country") or cols_lower.get("country_name") or cols_lower.get("name")
                year_col = cols_lower.get("year")
                val_col = None
                for k in ["index", "score", "nd-gain", "nd_gain", "ndgain", "nd_gain_score", "indexscore", "country_index", "gain_score"]:
                    if k in cols_lower:
                        val_col = cols_lower[k]; break
                if not year_col or not val_col:
                    continue
                out = xdf.copy()
                if iso_col:
                    out = out.rename(columns={iso_col: "iso3", year_col: "year", val_col: "value"})
                elif country_col:
                    out = out.rename(columns={country_col: "country", year_col: "year", val_col: "value"})
                    cmap = _wb_country_map()
                    out["country_lc"] = out["country"].astype(str).str.strip().str.lower()
                    out = out.merge(cmap[["iso3","name_lc"]], left_on="country_lc", right_on="name_lc", how="left")
                else:
                    continue
                out["year"] = pd.to_numeric(out["year"], errors="coerce")
                out["value"] = pd.to_numeric(out["value"], errors="coerce")
                out = out.dropna(subset=["iso3","year","value"])
                out["year"] = out["year"].astype(int)
                out = out[["iso3","year","value"]].drop_duplicates()
                out = out[out["iso3"].astype(str).str.len() == 3]
                if not out.empty:
                    return out
            except Exception:
                continue
    return pd.DataFrame(columns=["iso3","year","value"])

def _from_url(url: str) -> pd.DataFrame:
    """
    Lädt ND-GAIN von einer direkten URL (CSV/XLSX/ZIP) – mit Content-Type-Erkennung.
    """
    resp = requests.get(url, timeout=180, allow_redirects=True, headers=UA)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type","").lower()

    if url.lower().endswith(".zip") or "zip" in ctype:
        return _read_index_from_zip(resp.content)
    if url.lower().endswith((".csv",".txt")) or "csv" in ctype or "text/plain" in ctype:
        return _read_ndgain_from_csv_bytes(resp.content)
    if url.lower().endswith((".xlsx",".xls")) or "spreadsheet" in ctype or "excel" in ctype:
        xdf = pd.read_excel(io.BytesIO(resp.content))
        buf = io.BytesIO()
        xdf.to_csv(buf, index=False)
        return _read_ndgain_from_csv_bytes(buf.getvalue())
    # Unbekanntes Format -> leer
    return pd.DataFrame(columns=["iso3","year","value"])

def _from_vendor_csv() -> pd.DataFrame:
    """
    Vendor-Fallback: data/vendor/nd_gain.csv (Schema iso3,year,value), wenn vorhanden.
    """
    path = os.path.join("data","vendor","nd_gain.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["iso3","year","value"])
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding="latin-1")
    cols = {c.lower(): c for c in df.columns}
    iso = cols.get("iso3")
    year = cols.get("year")
    val = cols.get("value")
    if not (iso and year and val):
        return pd.DataFrame(columns=["iso3","year","value"])
    out = df.rename(columns={iso:"iso3", year:"year", val:"value"})[["iso3","year","value"]]
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["iso3","year","value"])
    out["year"] = out["year"].astype(int)
    out = out[out["iso3"].astype(str).str.len()==3]
    return out

def fetch_nd_gain() -> pd.DataFrame:
    """
    Holt ND-GAIN Country Index robust:
      1) ND_GAIN_URL (CSV/XLSX/ZIP) via ENV – wenn gesetzt, hat Vorrang.
      2) Scraper über bekannte Download-Seiten (alt + neu).
      3) Vendor-Fallback: data/vendor/nd_gain.csv
      -> Rückgabe: DataFrame [iso3, year, value] (value = 0..100, höher = besser).
    """
    # 1) Direkte URL per ENV
    direct = os.getenv("ND_GAIN_URL")
    if direct:
        try:
            df = _from_url(direct)
            if not df.empty:
                # Plausibilitätsfilter
                return df[(df["year"] >= 2000) & (df["year"] <= 2035)]
        except Exception:
            pass

    # 2) Download-Seiten parsen
    candidates = []
    for page in PAGES:
        try:
            html = requests.get(page, timeout=60, allow_redirects=True, headers=UA).text
            candidates.extend(_find_download_candidates(html, base=page))
        except Exception:
            continue

    for url in candidates:
        try:
            df = _from_url(url)
            if not df.empty:
                df = df[(df["year"] >= 2000) & (df["year"] <= 2035)]
                if not df.empty:
                    return df
        except Exception:
            continue

    # 3) Vendor-Fallback
    df = _from_vendor_csv()
    if not df.empty:
        return df[(df["year"] >= 2000) & (df["year"] <= 2035)]

    # 4) Leer zurück (kein Crash, Aggregation reweighted)
    return pd.DataFrame(columns=["iso3","year","value"])

