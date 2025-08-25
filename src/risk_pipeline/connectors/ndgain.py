import io
import re
import zipfile
import requests
import pandas as pd
from urllib.parse import urljoin

# Zwei mögliche Download-Seiten
PAGES = [
    "https://gain.nd.edu/our-work/country-index/download-data/",
    "https://gain-new.crc.nd.edu/about/download",
]

UA = {"User-Agent": "risk-pipeline/1.0 (+github-actions)"}

def _find_zip_urls(html: str) -> list[str]:
    # Alle .zip-Links einsammeln (relativ/absolut)
    urls = re.findall(r'href=["\']([^"\']+\\.zip)["\']', html, flags=re.IGNORECASE)
    # manche Seiten verwenden data-href
    urls += re.findall(r'data-href=["\']([^"\']+\\.zip)["\']', html, flags=re.IGNORECASE)
    # deduplizieren, Reihenfolge beibehalten
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _read_index_from_zip(zr: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zr)) as zf:
        # Kandidaten in sinnvoller Reihenfolge
        prefer = [n for n in zf.namelist() if n.lower().endswith(".csv") and "index" in n.lower()]
        anycsv = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        for name in prefer + anycsv:
            try:
                df = pd.read_csv(zf.open(name))
            except Exception:
                continue
            # Spalten-Heuristik
            cols_lower = {c.lower(): c for c in df.columns}
            iso_col = cols_lower.get("iso3") or cols_lower.get("iso") or cols_lower.get("country code")
            year_col = cols_lower.get("year")
            # Mögliche Werte-Spalten
            val_col = None
            for k in ["index", "score", "nd-gain", "nd_gain", "ndgain"]:
                if k in cols_lower:
                    val_col = cols_lower[k]; break
            # In manchen ZIPs liegt der Index als "index.csv" mit Spalte "nd_gain_score"
            if not val_col:
                for k in ["nd_gain_score", "indexscore", "country_index", "gain_score"]:
                    if k in cols_lower:
                        val_col = cols_lower[k]; break
            if not (iso_col and year_col and val_col):
                continue
            out = df.rename(columns={iso_col: "iso3", year_col: "year", val_col: "value"})[["iso3", "year", "value"]]
            out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
            out["value"] = pd.to_numeric(out["value"], errors="coerce")
            out = out.dropna(subset=["iso3", "year", "value"])
            if out.empty:
                continue
            out["year"] = out["year"].astype(int)
            return out
    return pd.DataFrame(columns=["iso3", "year", "value"])

def fetch_nd_gain() -> pd.DataFrame:
    """
    Holt ND-GAIN Index (Land/Jahr/Score) robust:
      - probiert alte und neue Download-Seite
      - findet ZIP-Links (href oder data-href)
      - folgt Redirects, setzt User-Agent
      - durchsucht ZIP nach einer plausiblen Index-CSV
    """
    # 1) Download-Seiten parsen
    zip_candidates = []
    for page in PAGES:
        try:
            html = requests.get(page, timeout=60, allow_redirects=True, headers=UA).text
            urls = _find_zip_urls(html)
            for u in urls:
                zip_candidates.append(u if u.startswith("http") else urljoin(page, u))
        except Exception:
            continue

    # Kein Link gefunden -> leer zurück
    if not zip_candidates:
        return pd.DataFrame(columns=["iso3", "year", "value"])

    # 2) Kandidaten durchprobieren
    for zurl in zip_candidates:
        try:
            zr = requests.get(zurl, timeout=180, allow_redirects=True, headers=UA)
            zr.raise_for_status()
            df = _read_index_from_zip(zr.content)
            if not df.empty:
                return df
        except Exception:
            continue

    # 3) Fallback: leer (kein Crash)
    return pd.DataFrame(columns=["iso3", "year", "value"])
