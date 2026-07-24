"""
Data loading and caching for the UK rental listings dataset.

Swap load_rental_data() to pull from your PostgreSQL rental listings DB
instead of a flat file if that's your primary source - the caching
decorator behaves the same either way.
"""

import pandas as pd
import streamlit as st
import sqlalchemy


# ---- Option A: load from a local parquet/csv snapshot -------------------

@st.cache_data(ttl="6h", show_spinner="Loading latest rental listings...")
def load_rental_data(path: str = "data/uk_rental_ml_mvw.parquet") -> pd.DataFrame:
    """
    Load the rental listings dataset used both for training-time features
    and for the actual-price distribution plot shown to the user.

    Cached with a 6h TTL so the app doesn't re-read the file on every
    widget interaction, but still picks up periodic data refreshes.
    """
    df = pd.read_parquet(path)
    df = _clean_rental_data(df)
    return df


# ---- Option B: load from PostgreSQL --------------------------------------

# @st.cache_resource(show_spinner=False)
# def get_db_engine(connection_string: str) -> sqlalchemy.engine.Engine:
#     """
#     Cached as a resource (not data) because a DB engine/connection pool
#     shouldn't be re-created on every rerun.
#     """
#     return sqlalchemy.create_engine(connection_string, pool_pre_ping=True)


# @st.cache_data(ttl="1h", show_spinner="Querying latest rental listings...")
# def load_rental_data_from_db(_engine, query: str | None = None) -> pd.DataFrame:
#     """
#     _engine is prefixed with an underscore so Streamlit's cache doesn't
#     try to hash the unhashable SQLAlchemy engine object.
#     """
#     query = query or """
#         SELECT property_type, bedrooms, bathrooms, furnished,
#                postcode_district, latitude, longitude,
#                property_size_sqft, has_garden, has_parking,
#                listed_date, rental_price_pcm
#         FROM rental_listings
#         WHERE listed_date >= CURRENT_DATE - INTERVAL '12 months'
#     """
#     df = pd.read_sql(query, _engine)
#     df = _clean_rental_data(df)
#     return df


CATEGORICAL_FEATURES = [
    'postcode_district','INNER_OUTER_LONDON','REGION_LONDON','wardcode','PropertyType',
    #'Bedrooms',
    'furnishtype','btrflag','newbuild'
]


def _clean_rental_data(df: pd.DataFrame) -> pd.DataFrame:
    """Shared cleaning/filtering applied regardless of data source."""
    df = df.dropna(subset=["asking_price"])
    df = df[df["asking_price"].between(200, 60000)]  # strip obvious outliers
    df = df.drop_duplicates()

    # Ensure 'Bedrooms' is an integer type
    if "Bedrooms" in df.columns:
        df["Bedrooms"] = pd.to_numeric(df["Bedrooms"], errors='coerce').fillna(0).astype(int)

    # # Ensure 'btrflag' and 'newbuild' are boolean types
    # for col in ['btrflag', 'newbuild']:
    #     if col in df.columns:
    #         # Convert to boolean, coercing errors to NaN, then fill NaN with False (or appropriate default)
    #         df[col] = df[col].astype(bool) # Convert existing types to bool


    # Ensure log-transformed columns are float types
    # These columns are typically derived and should be numerical
    for col in ['num_listings_district_log10', 'num_listings_ward_log10']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)


    # LightGBM handles categoricals natively when given 'category' dtype —
    # this dataset also serves as the category_reference used by
    # build_input_frame() so live predictions use the exact same category
    # codes the model was trained on.
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def get_feature_options(df: pd.DataFrame) -> dict:
    """
    Derive the selectable options for each feature directly from the data,
    so the UI always reflects what's actually in the current dataset.
    """
    return {

    "postcode_district": sorted(df["postcode_district"].dropna().unique().tolist()),
    "wardcode": sorted(df["wardcode"].dropna().unique().tolist()),
    "INNER_OUTER_LONDON": sorted(df["INNER_OUTER_LONDON"].dropna().unique().tolist()),
    "REGION_LONDON": sorted(df["REGION_LONDON"].dropna().unique().tolist()),
    "PropertyType": sorted(df["PropertyType"].dropna().unique().tolist()),
    "Bedrooms": sorted(df["Bedrooms"].dropna().unique().astype(int).tolist()), # Convert to int here
    "furnishtype": sorted(df["furnishtype"].dropna().unique().tolist()),
    "btrflag":  df["btrflag"].unique().tolist(), # Will now sort properly as it's boolean
    "newbuild": df["newbuild"].unique().tolist(), # Will now sort properly as it's boolean


    }
