import os
import io
import time
import math
import requests
import pandas as pd

# ---------------------------
# ACLED OAuth + Read Endpoints
# ---------------------------
ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"
ACLED_READ_URL  = "https://acleddata.com/api/acled/read"

REQUEST_TIMEOUT = 120
PAGE_LIMIT = 5000
MAX_PAGES = 500
BACKOFF_SEC = [2, 4, 8, 16]

# Nur die letzten N Jahre laden (Performance). Default 8.
YEARS_BACK = int(os.getenv("ACLED_YEARS_BACK", "8"))

def _request_with_backoff(method: str, url: str, **kwargs) -> requests.Response:
    """HTTP mit Backoff für 429/5xx/Cloudflare-Fehler."""
    headers = kwargs.pop("headers", {})
    headers.setdefault("Accept", "application/json")
    kwargs["headers"] = headers
    for i, wait in enumerate([0] + BACKOFF_SEC):
        if wait:
            time.sleep(wait)
        try:
            resp = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
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

def _get_access_token(username: str, password: str) -> str | None:
    """OAuth Password Grant → Bearer Token (24h gültig)."""
    data = {
        "username": username,
        "password": password,
        "grant_type": "password",
        "client_id": "acled",
    }
    try:
        resp = _request_with_backoff("POST", ACLED_TOKEN_URL, data=data)
        js = resp.json()
        token = js.get("access_token")
        if not token:
            print("ACLED: OAuth ohne access_token – Antwort:", js)
        return token
    except Exception as e:
        print("ACLED: OAuth-Fehler:", repr(e))
        return None

# ---------------------------
# Population (OWID) für Pro-Kopf-Berechnung
# ---------------------------
def _fetch_owid_population() -> pd.DataFrame:
    url = "https://ourworldindata.org/grapher/population.csv"
    try:
        r = _request_with_backoff("GET", url)
        df = pd.read_csv(io.BytesIO(r.content))
        if not set(["Code", "Year"]).issubset(df.columns):
            return pd.DataFrame(columns=["iso3","year","pop"])
        vcols = [c for c in df.columns if c not in ("Entity","Code","Year")]
        if not vcols:
            return pd.DataFrame(columns=["iso3","year","pop"])
        out = df.rename(columns={"Code":"iso3","Year":"year", vcols[0]:"pop"})[["iso3","year","pop"]]
        out = out.dropna(subset=["iso3","year","pop"])
        out["year"] = out["year"].astype(int)
        out = out[out["iso3"].astype(str).str.len()==3]
        return out.reset_index(drop=True)
    except Exception as e:
        print("ACLED: Konnte OWID population nicht laden:", repr(e))
        return pd.DataFrame(columns=["iso3","year","pop"])

def _asof_join_population_per_100k(fatal_df: pd.DataFrame, pop_df: pd.DataFrame, max_lag_years: int = 5) -> pd.DataFrame:
    if fatal_df.empty or pop_df.empty:
        return pd.DataFrame(columns=["iso3","year","value"])
    fatal_df = fatal_df.sort_values(["iso3","year"]).copy()
    pop_df = pop_df.sort_values(["iso3","year"]).copy()
    rows = []
    for iso, fg in fatal_df.groupby("iso3"):
        pg = pop_df[pop_df["iso3"] == iso]
        if pg.empty:
            continue
        py = pg["year"].values
        pv = pd.to_numeric(pg["pop"], errors="coerce").values
        for _, r in fg.iterrows():
            y = int(r["year"])
            idx = (py <= y).nonzero()[0]
            if len(idx) == 0:
                continue
            j = idx[-1]
            pop_year = int(py[j])
            pop_val = float(pv[j]) if pv[j] > 0 else math.nan
            if math.isnan(pop_val) or y - pop_year > max_lag_years:
                continue
            per_100k = (float(r["fatalities"]) / pop_val) * 100000.0
            rows.append({"iso3": iso, "year": y, "value": max(per_100k, 0.0)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["iso3","year","value"])

# ---------------------------
# ACLED Fetch
# ---------------------------
def fetch_acled_fatalities() -> pd.DataFrame:
    """
    Holt ACLED-Events via OAuth (Bearer), aggregiert Fatalities je iso3/Jahr
    und liefert Fatalities pro 100k (iso3, year, value).
    Lädt nur die letzten N Jahre (Env ACLED_YEARS_BACK, Default 8).
    Erwartete Secrets:
      - ACLED_USERNAME (oder ACLED_EMAIL)
      - ACLED_PASSWORD
    """
    username = os.getenv("ACLED_USERNAME") or os.getenv("ACLED_EMAIL")
    password = os.getenv("ACLED_PASSWORD")
    if not username or not password:
        print("ACLED: Secrets fehlen (ACLED_USERNAME/ACLED_PASSWORD).")
        return pd.DataFrame(columns=["iso3", "year", "value"])

    token = _get_access_token(username, password)
    if not token:
        return pd.DataFrame(columns=["iso3", "year", "value"])
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    this_year = pd.Timestamp.utcnow().year
    start_year = max(1997, this_year - YEARS_BACK)

    all_rows = []
    for year in range(start_year, this_year + 1):
        page = 1
        while page <= MAX_PAGES:
            # Robust: per event_date-Jahresfenster filtern (keine year_where-Speziallogik)
            date_from = f"{year}-01-01"
            date_to   = f"{year}-12-31"
            params = {
                # _format weglassen → JSON default, stabil
                "event_date": f"{date_from}|{date_to}",
                "limit": PAGE_LIMIT,
                "page": page,
            }
            try:
                resp = _request_with_backoff("GET", ACLED_READ_URL, headers=headers, params=params)
                js = resp.json()
            except Exception as e:
                print(f"ACLED: Fehler beim Lesen für {year}, page {page}: {repr(e)}")
                break

            # ACLED gibt bei Fehlern 'error' / 'message' zurück
            if isinstance(js, dict) and js.get("error"):
                print("ACLED: API-Fehler:", js.get("error"))
                break

            data = js.get("data", [])
            if not data:
                break

            for row in data:
                iso3 = row.get("iso3")
                fat  = row.get("fatalities")
                # Jahr aus event_date robuster ermitteln (falls 'year' nicht geliefert wird)
                ed   = row.get("event_date") or row.get("event_date2")
                try:
                    yr = int(str(ed)[:4]) if ed else year
                except Exception:
                    yr = year
                if not iso3:
                    continue
                try:
                    fat = float(fat) if fat is not None else 0.0
                except Exception:
                    fat = 0.0
                all_rows.append({"iso3": iso3, "year": yr, "fatalities": fat})

            if len(data) < PAGE_LIMIT:
                break
            page += 1

    if not all_rows:
        print("ACLED: Keine Daten empfangen (prüfe Account-Rechte und Zeitraum).")
        return pd.DataFrame(columns=["iso3","year","value"])

    df = pd.DataFrame(all_rows)
    df = df[df["iso3"].astype(str).str.len()==3]
    agg = df.groupby(["iso3","year"], as_index=False)["fatalities"].sum()

    pop = _fetch_owid_population()
    out = _asof_join_population_per_100k(agg, pop, max_lag_years=5)
    if out.empty:
        return pd.DataFrame(columns=["iso3","year","value"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce").clip(lower=0)
    return out[["iso3","year","value"]]


