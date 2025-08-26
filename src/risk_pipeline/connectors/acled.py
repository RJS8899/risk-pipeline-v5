import os
import io
import time
import math
import requests
import pandas as pd

# ACLED OAuth + API
ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"
ACLED_READ_URL  = "https://acleddata.com/api/acled/read"

REQUEST_TIMEOUT = 120
PAGE_LIMIT = 5000
MAX_PAGES = 500
BACKOFF_SEC = [2, 4, 8, 16]

# Wie viele Jahre rückwirkend laden? (Default 8)
# Du kannst das im Workflow setzen: ACLED_YEARS_BACK: 8
YEARS_BACK = int(os.getenv("ACLED_YEARS_BACK", "8"))

def _request_with_backoff(method: str, url: str, **kwargs) -> requests.Response:
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
    data = {
        "username": username,
        "password": password,
        "grant_type": "password",
        "client_id": "acled",
    }
    try:
        resp = _request_with_backoff("POST", ACLED_TOKEN_URL, data=data)
        js = resp.json()
        return js.get("access_token")
    except Exception:
        return None

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
    except Exception:
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

def fetch_acled_fatalities() -> pd.DataFrame:
    """
    Holt ACLED-Ereignisse via OAuth, aggregiert fatalities je iso3/jahr
    und gibt fatalities pro 100k zurück (iso3, year, value).
    Lädt nur die letzten N Jahre (YEARS_BACK), um Laufzeit & Traffic zu reduzieren.
    Erwartete Secrets:
      - ACLED_USERNAME (oder ACLED_EMAIL)
      - ACLED_PASSWORD
    Optional:
      - ACLED_YEARS_BACK (int)
    """
    username = os.getenv("ACLED_USERNAME") or os.getenv("ACLED_EMAIL")
    password = os.getenv("ACLED_PASSWORD")
    if not username or not password:
        return pd.DataFrame(columns=["iso3", "year", "value"])

    token = _get_access_token(username, password)
    if not token:
        return pd.DataFrame(columns=["iso3", "year", "value"])

    headers = {"Authorization": f"Bearer {token}"}
    this_year = pd.Timestamp.utcnow().year
    start_year = max(1997, this_year - YEARS_BACK)  # ACLED Startjahr 1997; wir begrenzen nach hinten

    all_rows = []
    for year in range(start_year, this_year + 1):
        page = 1
        while page <= MAX_PAGES:
            params = {
                "_format": "json",
                "fields": "iso3|year|fatalities",
                "year": str(year),
                "year_where": "%3D",
                "limit": PAGE_LIMIT,
                "page": page,
            }
            try:
                resp = _request_with_backoff("GET", ACLED_READ_URL, headers=headers, params=params)
                js = resp.json()
            except Exception:
                break

            data = js.get("data", [])
            if not data:
                break

            # zeilenweise light transform
            for row in data:
                iso3 = row.get("iso3")
                yr = row.get("year")
                fat = row.get("fatalities")
                if not iso3 or not yr:
                    continue
                try:
                    yr = int(yr)
                except Exception:
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
