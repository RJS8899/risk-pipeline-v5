import os, requests, pandas as pd

ACLED_BASE = "https://api.acleddata.com/acled/read"

def fetch_acled_fatalities() -> pd.DataFrame:
    email = os.getenv("ACLED_EMAIL")
    key = os.getenv("ACLED_KEY")
    if not email or not key:
        return pd.DataFrame(columns=["iso3","year","value"])
    try:
        params = {
            "email": email, "key": key,
            "event_date": "2022-01-01|2050-01-01",
            "fields": "iso3,event_date,fatalities",
            "limit": 20000
        }
        r = requests.get(ACLED_BASE, params=params, timeout=120)
        r.raise_for_status()
        js = r.json()
        data = js.get("data", [])
        if not data:
            return pd.DataFrame(columns=["iso3","year","value"])
        df = pd.DataFrame(data)
        if not set(["iso3","event_date","fatalities"]).issubset(df.columns):
            return pd.DataFrame(columns=["iso3","year","value"])
        df["year"] = pd.to_datetime(df["event_date"]).dt.year
        df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)
        agg = df.groupby(["iso3","year"])['fatalities'].sum().reset_index()
        agg = agg.rename(columns={"fatalities":"value"})
        return agg
    except Exception:
        return pd.DataFrame(columns=["iso3","year","value"])
