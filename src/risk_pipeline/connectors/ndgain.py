import io
import re
import zipfile
import requests
import pandas as pd
from urllib.parse import urljoin

# Download-Seiten (alt + neu)
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
    Rückgabe: DataFrame mit Spalten ['iso3','name'] + Name-Varianten in lowercase.
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
        df["name_lc"] = df["name"].str.strip().str.lower()
        return df
    except Exception:
        # Minimaler Fallback, falls API down – deckt die häufigsten Fälle rudimentär ab
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
    Sucht in HTML nach .zip, .csv, .xlsx Links und gibt absolute URLs zurück.
    """
    urls = []
    # href= oder data-href=
    for ext in (".zip", ".csv", ".xlsx", ".xls"):
        pat = rf'(?:href|data-href)=["\']([^"\']+{re.escape(ext)})["\']'
        urls += re.findall(pat, html, flags=re.IGNORECASE)
    # absolut machen + deduplizieren
    abs_urls, seen = [], set()
    for u in urls:
        full = u if u.startswith("http") else urljoin(base, u)
        if full not in seen:
            seen.add(full)
            abs_urls.append(full)
    return abs_urls

def _read_ndgain_from_csv_bytes(raw: bytes) -> pd.DataFrame:
    """
    Versucht, eine CSV mit ND-GAIN Index zu lesen.
    Erkennt Spalten: iso3|iso|country code|country, year, (index|score|nd_gain*|...).
    Mappt ggf. Country -> ISO3 per World-Bank-Namensliste.
    """
    # Primärer Read
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        # Encoding-Fallback
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    if df is None or df.empty:
        return pd.DataFrame(columns=["iso3","year","value"])

    cols_lower = {c.lower(): c for c in df.columns}
    iso_col = cols_lower.get("iso3") or cols_lower.get("iso") or cols_lower.get("country code")
    country_col = cols_lower.get("country") or cols_lower.get("country_name") or cols_lower.get("name")
    year_col = cols_lower.get("year")

    # Value-Kandidaten
    val_col = None
    for k in ["index", "score", "nd-gain", "nd_gain", "ndgain", "nd_gain_score", "indexscore", "country_index", "gain_score"]:
        if k in cols_lower:
            val_col = cols_lower[k]; break

    if not year_col or not val_col:
        return pd.DataFrame(columns=["iso3","year","value"])

    out = df.copy()
    # ISO3 ableiten / mappen
    if iso_col:
        out = out.rename(columns={iso_col: "iso3", year_col: "year", val_col: "value"})
    elif country_col:
        out = out.rename(columns={country_col: "country", year_col: "year", val_col: "value"})
        cmap = _wb_country_map()
        out["country_lc"] = out["country"].astype(str).str.strip().str.lower()
        out = out.merge(cmap[["iso3","name_lc"]], left_on="country_lc", right_on="name_lc", how="left")
    else:
        return pd.DataFrame(columns=["iso3","year","value"])

    # putzen
    if "iso3" not in out.columns:
        return pd.DataFrame(columns=["iso3","year","value"])

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["iso3","year","value"])
    out["year"] = out["year"].astype(int)
    out = out[["iso3","year","value"]].drop_duplicates()
    # Filter: ISO3 genau 3 Zeichen
    out = out[out["iso3"].astype(str).str.len() == 3]
    return out

def _read_index_from_zip(raw_zip: bytes) -> pd.DataFrame:
    """
    Durchsucht ZIP nach plausiblen CSVs/XLSX mit Index/Score-Spalten.
    Bevorzugt Dateien mit 'index' im Namen, fällt sonst auf andere CSVs zurück.
    """
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        names = zf.namelist()

        # Kandidatenreihenfolge
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

        # XLSX-Unterstützung (falls CSVs nicht passen)
        for name in xlsx:
            try:
                data = zf.read(name)
                xdf = pd.read_excel(io.BytesIO(data))
                # gleich wie CSV behandeln (Bytes aus Excel erneut über CSV-Parser geht nicht; wir nehmen DataFrame direkt)
                # Spaltenheuristik:
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

def fetch_nd_gain() -> pd.DataFrame:
    """
    Holt ND-GAIN Country Index robust:
      - parst beide Download-Seiten
      - sucht .zip/.csv/.xlsx Links
      - folgt Redirects, setzt User-Agent
      - liest unterschiedliche Layouts
      - mappt Country -> ISO3, wenn nötig
    Gibt DataFrame mit Spalten [iso3, year, value] zurück (value = 0..100, höher = besser).
    """
    # 1) Download-Seiten parsen und Link-Kandidaten einsammeln
    candidates = []
    for page in PAGES:
        try:
            html = requests.get(page, timeout=60, allow_redirects=True, headers=UA).text
            urls = _find_download_candidates(html, base=page)
            candidates.extend(urls)
        except Exception:
            continue

    # Nichts gefunden -> leer zurück
    if not candidates:
        return pd.DataFrame(columns=["iso3","year","value"])

    # 2) Kandidaten durchprobieren
    for url in candidates:
        try:
            resp = requests.get(url, timeout=180, allow_redirects=True, headers=UA)
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type","").lower()

            if url.lower().endswith(".zip") or "zip" in ctype:
                df = _read_index_from_zip(resp.content)
            elif url.lower().endswith((".csv",".txt")) or "csv" in ctype or "text/plain" in ctype:
                df = _read_ndgain_from_csv_bytes(resp.content)
            elif url.lower().endswith((".xlsx",".xls")) or "spreadsheet" in ctype or "excel" in ctype:
                xdf = pd.read_excel(io.BytesIO(resp.content))
                # Re-use CSV-Heuristik
                buf = io.BytesIO()
                xdf.to_csv(buf, index=False)
                df = _read_ndgain_from_csv_bytes(buf.getvalue())
            else:
                continue

            if not df.empty:
                # sanity-check: plausible Jahre (2000..2030)
                df = df[(df["year"] >= 2000) & (df["year"] <= 2030)]
                if not df.empty:
                    return df
        except Exception:
            continue

    # 3) Fallback: leer (kein Crash, Aggregation bleibt robust)
    return pd.DataFrame(columns=["iso3","year","value"])

