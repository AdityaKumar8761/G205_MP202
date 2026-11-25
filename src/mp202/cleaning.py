import numpy as np
import pandas as pd


def clean_data(df):
    df = df.copy()
    df["service_records"] = df["service_records"].replace(-999, np.nan)
    df["date"] = pd.to_datetime(df["date"])
    mask = df["date"].between("2026-04-11", "2026-04-29")
    df.loc[mask, "accidents"] = np.nan
    df["brand_tier"] = (
        df["brand_tier"].str.strip().str.lower()
        .str.replace(".", "", regex=False)
        .replace({"bud": "budget", "pre": "premium"})
    )
    return df.drop(columns=["final_price_from_rc_transfer"], errors="ignore")
