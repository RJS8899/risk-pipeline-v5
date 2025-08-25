import requests, time
from typing import List, Dict, Optional
import pandas as pd

WB_BASE = "https://api.worldbank.org/v2"

def fetch_indicator(code: str, countries: Optional[List[str]]=None, per_page=20000) -> pd.DataFrame:
    """Return DataFrame columns: iso3, year, value"""
    if not countries:
        countries = ["all"]
    url = f"{WB_BASE}/country/{';'.join(countries)}/indicator/{code}?format=json&per_page={per_page}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2 or data[1] is None:
            return pd.DataFrame(columns=["iso3","year","value"])
        rows = data[1]
        out = []
        for row in rows:
            country = row.get("countryiso3code") or (row.get("country") or {}).get("id")
            date = row.get("date")
            value = row.get("value")
            if country and date is not None:
                try:
                    out.append({"iso3": country, "year": int(date), "value": float(value) if value is not None else None})
                except Exception:
                    pass
        df = pd.DataFrame(out).dropna(subset=["value"]).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["iso3","year","value"])
