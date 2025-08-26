import os
import time
import io
import requests
import pandas as pd

# Hinweis:
# - Liefert Fatalities pro 100k Einwohner (value) im Schema [iso3, year, value]
# - Benötigt ACLED_EMAIL und ACLED_KEY als Umgebungsvariablen (GitHub Secrets)
# - Robust gegen Rate Limits, Pagination, fehlende Felder

ACLED_BASE = "https://api.acleddata.com/acled/read"
REQUEST_TIMEOUT = 120
PAGE_SIZE = 5000
MAX_PAGES = 200        # Schutz vor Endlosschleifen
BACKOFF_SEC = [2, 4, 8, 16]  # bei 429/5xx

def _fetch_owid_population() -> pd.DataFrame:
    """OWID Population (iso3, year, pop)."""
    # Lightweight eigener Fetch (vermeidet interne Imports)
    OWID_CSV = "https://ourworldindata.org/grapher/population.csv"
    try:
        r = requests.get(OWID_CSV, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        # Erwartete Spalten: Entity, Code, Year, population
        if not set(["Code", "Year"]).issubset(df.columns):
            return pd.DataFrame(columns=["iso3","year","pop"])
        # Suche Value-Spalte
        val_cols = [c for c in df.columns if c not in ("Entity","Code","Year")]
        if not val_cols:
            return pd.DataFrame(columns=["iso3","year","pop"])
        vcol = val_cols[0]
        out = df.rename(columns={"Code":"iso3", "Year":"year", vcol:"pop"})[["iso3","year","pop"]]
        out = out.dropna(subset=["iso3","year","pop"])
        out["year"] = out["year"].astype(int)
        return out[out["iso3"].astype(str).str.len()==3].reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["iso3","year","pop"])

def _asof_join_population_per_100k(fatal_df: pd.DataFrame, pop_df: pd.DataFrame, max_lag_years: int = 5) -> pd.DataFrame:
    """
    Für jede (iso3, year)-Fatalities-Zeile verwende die letzte verfügbare Population
    mit Jahr <= year (lag <= max_lag_years). Liefert value = fatalities_per_100k.
    """
    if fatal_df.empty or pop_df.empty:
        return pd.DataFrame(columns=["iso3","year","value"])

    fatal_df = fatal_df.sort_values(["iso3","year"]).copy()
    pop_df = pop_df.sort_values(["iso3","year"]).copy()

    out_rows = []
    for iso, grp in fatal_df.groupby("iso3"):
        vg = grp.sort_values("year")
        pg = pop_df[pop_df["iso3"] == iso].sort_values("year")
        if pg.empty:
            continue
        p_years = pg["year"].values
        p_vals  = pd.to_numeric(pg["pop"], errors="coerce").values
        for _, row in vg.iterrows():
            y = int(row["year"])
            idx = (p_years <= y).nonzero()[0]
            if len(idx) == 0:
                continue
            j = idx[-1]
            py, pv = int(p_years[j]), float(p_vals[j])
            if y - py <= max_lag_years and pv > 0:
                per_100k = (float(row["fatalities"]) / pv) * 100000.0
                out_rows.append({"iso3": iso, "year": y, "value": per_100k})
    if not out_rows:
        return pd.DataFrame(columns=["iso3","year","value"])
    return pd.DataFrame(out_rows)

def _request_with_backoff(url: str, params: dict) -> requests.Response:
    """GET mit Backoff bei 429/5xx."""
    for i, wait in enumerate([0] + BACKOFF_SEC):
        if wait:
            time.sleep(wait)
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            # Bei Cloudflare/ACLED kommt manchmal 520/522/524; retry
            if resp.status_code in (429, 500, 502, 503, 504, 520, 522, 524):
                if i < len(BACKOFF_SEC):
                    continue
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if i == len(BACKOFF_SEC):
                raise
            continue
    raise RuntimeError("unreachable")

def fetch_acled_fatalities() -> pd.DataFrame:
    """
    Zieht ACLED-Events und aggregiert Fatalities pro 100k:
      - auth via ACLED_EMAIL + ACLED_KEY
      - Zeitraum: 2010-01-01 bis zukünftig
      - Pagination bis kein Datensatz mehr
      - Aggregation: iso3, year -> sum(fatalities)
      - Asof-Join mit OWID population (<= year, max lag 5 Jahre)
    Rückgabe: DataFrame[iso3, year, value] mit value in 'fatalities per 100k'.
    """
    email = os.getenv("ACLED_EMAIL")
    key = os.getenv("ACLED_KEY")
    if not email or not key:
        # Kein Secret gesetzt → leer zurück, Pipeline bleibt robust
        return pd.DataFrame(columns=["iso3","year","value"])

    # 1) ACLED Paginated Fetch
    all_rows = []
    page = 1
    while page <= MAX_PAGES:
        params = {
            "email": email,
            "key": key,
            "event_date": "2010-01-01|2100-01-01",
            "fields": "iso3,event_date,fatalities",
            "limit": PAGE_SIZE,
            "page": page,
            "format": "json"
        }
        try:
            resp = _request_with_backoff(ACLED_BASE, params)
        except Exception:
            break

        try:
            js = resp.json()
        except Exception:
            break

        data = js.get("data", []) or []
        if not isinstance(data, list) or not data:
            break

        for row in data:
            iso3 = row.get("iso3")
            edate = row.get("event_date")
            fat   = row.get("fatalities")
            if not iso3 or not edate:
                continue
            # Sauber parsen
            try:
                year = int(str(edate)[:4])
            except Exception:
                continue
            try:
                fat = float(fat) if fat is not None else 0.0
            except Exception:
                fat = 0.0
            all_rows.append({"iso3": iso3, "year": year, "fatalities": fat})

        # Wenn weniger als PAGE_SIZE zurückkam → Ende
        if len(data) < PAGE_SIZE:
            break
        page += 1

    if not all_rows:
        return pd.DataFrame(columns=["iso3","year","value"])

    df = pd.DataFrame(all_rows)
    # Nur valide ISO3
    df = df[df["iso3"].astype(str).str.len() == 3]

    # 2) Aggregation pro Land/Jahr (Summe Fatalities)
    agg = df.groupby(["iso3","year"], as_index=False)["fatalities"].sum()

    # 3) OWID Population as-of join (<= year, max lag 5)
    pop = _fetch_owid_population()
    out = _asof_join_population_per_100k(agg, pop, max_lag_years=5)
    # Ausreißer / Negativwerte abfangen
    if not out.empty:
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["value"])
        out["value"] = out["value"].clip(lower=0)

    return out[["iso3","year","value"]] if not out.empty else pd.DataFrame(columns=["iso3","year","value"])
