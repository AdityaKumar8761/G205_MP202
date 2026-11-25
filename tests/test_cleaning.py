import numpy as np
import pandas as pd

from mp202.cleaning import clean_data


def make_df():
    return pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4"],
        "date": ["2026-01-05", "2026-04-15", "2026-02-01", "2026-03-01"],
        "age_years": [3.0, 5.0, 2.0, 8.0],
        "km_driven": [10000, 20000, 30000, 40000],
        "engine_cc": [150, 200, 350, 100],
        "owners": [1, 2, 1, 3],
        "service_records": [4, -999, 6, 2],
        "accidents": [0, 2, 1, 0],
        "brand_tier": [" Mid", "BUD.", "pre.", "PREMIUM"],
        "city_zone": ["North", "South", "Central", "North"],
        "final_price_from_rc_transfer": [50.0, 30.0, 90.0, 60.0],
        "resale_thousand_rs": [50.5, 30.1, 90.2, 60.3],
    })


def test_sentinel_replaced_with_nan():
    out = clean_data(make_df())
    assert (out["service_records"] == -999).sum() == 0
    assert out["service_records"].isna().sum() == 1


def test_brand_tier_has_three_clean_levels():
    out = clean_data(make_df())
    assert set(out["brand_tier"]) <= {"budget", "mid", "premium"}


def test_leaking_column_dropped():
    out = clean_data(make_df())
    assert "final_price_from_rc_transfer" not in out.columns


def test_stuck_sensor_window_set_to_nan():
    out = clean_data(make_df())
    inside = out[out["record_id"] == "R2"]["accidents"].iloc[0]
    outside = out[out["record_id"] == "R1"]["accidents"].iloc[0]
    assert np.isnan(inside)
    assert outside == 0
