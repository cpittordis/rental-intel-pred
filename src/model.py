"""
Model loading and prediction — LightGBM.

Assumes a fitted LGBMRegressor (or a sklearn Pipeline wrapping one) saved
with joblib. LightGBM handles categorical features natively as long as
they're passed in as pandas 'category' dtype with the same categories
used at training time — build_input_frame() takes care of that.
"""

import joblib
import pandas as pd
import streamlit as st

CATEGORICAL_FEATURES = [
    "postcode_district",
    "INNER_OUTER_LONDON",
    "REGION_LONDON",
    "wardcode",
    "PropertyType",
    #"Bedrooms",
    "furnishtype",
    "btrflag",
    "newbuild",
]

from pathlib import Path
code_dir = Path(__file__).parent.resolve()

@st.cache_resource(show_spinner="Loading rental prediction model...")
def load_model(path: str = "models/final_lightgbm_model_uk_optuna_v20260703.pkl"):
    """
    Cached as a resource since the fitted LGBMRegressor should be loaded
    once per session/process, not re-deserialized on every prediction.
    """
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_quantile_models(
    lower_path: str = "models/rental_model_q10.joblib",
    upper_path: str = "models/rental_model_q90.joblib",
):
    """
    LightGBM doesn't expose a tree-spread interval the way RandomForest
    does — a quantile interval needs separate LGBMRegressor instances
    trained with objective='quantile' and alpha=0.1 / alpha=0.9
    respectively. Load them once and cache alongside the point model.

    Returns (lower_model, upper_model), or (None, None) if the files
    aren't present yet (interval reporting degrades gracefully).
    """
    try:
        lower_model = joblib.load(lower_path)
        upper_model = joblib.load(upper_path)
        return lower_model, upper_model
    except FileNotFoundError:
        return None, None


def build_input_frame(
    user_selection: dict,
    feature_order: list[str],
    category_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Convert the widget selections from app.py into a single-row DataFrame
    matching what the LightGBM model expects.

    category_reference (typically your training/full dataset) is used to
    set the categorical dtype's categories to match training time exactly
    — LightGBM encodes categories by their category codes internally, so
    a mismatch here silently corrupts predictions.
    """
    row = {feature: user_selection.get(feature) for feature in feature_order}
    input_frame = pd.DataFrame([row])

    for col in CATEGORICAL_FEATURES:
        if col not in input_frame.columns:
            continue
        if category_reference is not None and col in category_reference.columns:
            categories = category_reference[col].astype("category").cat.categories
            input_frame[col] = pd.Categorical(input_frame[col], categories=categories)
        else:
            input_frame[col] = input_frame[col].astype("category")

    return input_frame


def predict_rent(model, input_frame: pd.DataFrame) -> float:
    """Return a single predicted monthly rental value."""
    prediction = 10**(model.predict(input_frame))
    return float(prediction[0])


def predict_with_interval(model, input_frame: pd.DataFrame):
    """
    Point estimate from the main model, plus a [lower, upper] band from
    the two quantile LightGBM models if they've been loaded. Returns
    (point, lower, upper) — lower/upper are None if quantile models
    aren't available, so the UI can fall back to showing the point
    estimate alone.
    """
    point = predict_rent(model, input_frame)

    lower_model, upper_model = load_quantile_models()
    if lower_model is not None and upper_model is not None:
        lower = predict_rent(lower_model, input_frame)
        upper = predict_rent(upper_model, input_frame)
        return point, lower, upper

    return point, None, None
