import pandas as pd
from .worldbank import fetch_indicator

def fetch_wgi(code: str) -> pd.DataFrame:
    df = fetch_indicator(code)
    return df
