"""
UK Rental Price Predictor
--------------------------
Single-page Streamlit app, backed by a LightGBM model:
  1. User selects property features in the sidebar
  2. LightGBM model predicts a rental value
  3. Distribution of actual rental prices for similar properties is shown,
     with the prediction marked on it
  4. SHAP (TreeExplainer) explanation shows which features drove the
     prediction, with a plain-English summary
"""

import streamlit as st
import plotly.express as px
import numpy as np
import joblib
import pandas as pd
import gc
from pathlib import Path

# from data_loader import load_rental_data, get_feature_options
# from model import load_model, build_input_frame, predict_rent , predict_with_interval
# from src.explainer import get_base_value , build_shap_explainer, explain_single_prediction, plot_local_contribution
# from src.text_generator import generate_explanation_text


# def get_memory_usage():
#     # Obtains resident set size (RSS) memory in bytes
#     process = psutil.Process(os.getpid())
#     mem_bytes = process.memory_info().rss
#     # Convert bytes to Megabytes for readability
#     return mem_bytes / (1024 * 1024)

# # Display a live metrics card in the Streamlit Sidebar
# current_mem = get_memory_usage()
# st.sidebar.metric(
#     label="RAM Usage", 
#     value=f"{current_mem:.2f} MB", 
#     delta=f"{1024 - current_mem:.2f} MB remaining",
#     delta_color="normal" if current_mem < 800 else "inverse"
# )

@st.cache_data(ttl="6h", show_spinner="Loading latest rental listings...")
#def load_rental_data(path: str | None = None) -> pd.DataFrame:
def load_rental_data(data_path: str = "data/inference_uk_rental_avm_v20261031.parquet") -> pd.DataFrame:
    """
    Load the rental listings dataset used both for training-time features
    and for the actual-price distribution plot shown to the user.

    Cached with a 6h TTL so the app doesn't re-read the file on every
    widget interaction, but still picks up periodic data refreshes.
    """
    # data_path = (
    #     Path(path)
    #     if path is not None
    #     else Path(__file__).resolve().parents[1] / "data" / "inference_uk_rental_avm_v20261031.parquet"
    # )
    df = pd.read_parquet(data_path)
    print(f"Loaded rental data with {len(df)} rows and {len(df.columns)} columns.")
    df = _clean_rental_data(df)
    print(f"Cleaned rental data has {len(df)} rows and {len(df.columns)} columns.")
    return df




def _clean_rental_data(df: pd.DataFrame) -> pd.DataFrame:
    """Shared cleaning/filtering applied regardless of data source."""

    df = df.dropna(subset=["asking_price"])
    df = df[df["asking_price"].between(100, 20000)]  # strip obvious outliers
    df = df.drop_duplicates()

    CATEGORICAL_FEATURES = [
    'postcode_district','INNER_OUTER_LONDON','REGION_LONDON','wardcode','PropertyType',
    #'Bedrooms',
    'furnishtype','btrflag','newbuild'
]

    # Ensure 'Bedrooms' is an integer type
    if "Bedrooms" in df.columns:
        df["Bedrooms"] = pd.to_numeric(df["Bedrooms"], errors='coerce').fillna(0).astype(int)

    # # Ensure 'btrflag' and 'newbuild' are boolean types
    # for col in ['btrflag', 'newbuild']:
    #     if col in df.columns:
    #         # Convert to boolean, coercing errors to NaN, then fill NaN with False (or appropriate default)
    #         df[col] = df[col].astype(bool) # Convert existing types to bool


    # LightGBM handles categoricals natively when given 'category' dtype —
    # this dataset also serves as the category_reference used by
    # build_input_frame() so live predictions use the exact same category
    # codes the model was trained on.
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # # final condensing 
    # df = condense_dtypes(df)

    return df

#     # 2. Downcast data types to use less memory
# def condense_dtypes(df):
#     for col in df.columns:
#         col_type = df[col].dtype

#         # Condense Floats (e.g., float64 -> float32)
#         if np.issubdtype(col_type, np.floating):
#             df[col] = pd.to_numeric(df[col], downcast="float")

#         # Condense Integers (e.g., int64 -> int8 or int16)
#         elif np.issubdtype(col_type, np.integer):
#             df[col] = pd.to_numeric(df[col], downcast="integer")

#         # # Condense Text objects to Categories if high repetition
#         # elif col_type == "object": #or col_type == "string":
#         #     if df[col].nunique() / len(df[col]) < 0.5:
#         #         df[col] = df[col].astype("category")
#     return df

    


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
    }


# from pathlib import Path
# code_dir = Path(__file__).parent.resolve()

@st.cache_resource(show_spinner="Loading rental prediction model...")
# def load_model(path: str | None = None):
def load_model(path: str = "models/lightgbm_optuna_uk_rental_avm_v20261031"):
    """
    Cached as a resource since the fitted LGBMRegressor should be loaded
    once per session/process, not re-deserialized on every prediction.
    """
    # model_path = (
    #     Path(path)
    #     if path is not None
    #     else Path(__file__).resolve().parent / "models" / "lightgbm_optuna_uk_rental_avm_v20261031"
    # )
    return joblib.load(path)


# @st.cache_resource(show_spinner=False)
# def load_quantile_models(
#     lower_path: str = "models/rental_model_q10.joblib",
#     upper_path: str = "models/rental_model_q90.joblib",
# ):
#     """
#     LightGBM doesn't expose a tree-spread interval the way RandomForest
#     does — a quantile interval needs separate LGBMRegressor instances
#     trained with objective='quantile' and alpha=0.1 / alpha=0.9
#     respectively. Load them once and cache alongside the point model.

#     Returns (lower_model, upper_model), or (None, None) if the files
#     aren't present yet (interval reporting degrades gracefully).
#     """
#     try:
#         lower_model = joblib.load(lower_path)
#         upper_model = joblib.load(upper_path)
#         return lower_model, upper_model
#     except FileNotFoundError:
#         return None, None


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

    CATEGORICAL_FEATURES = [
    "postcode_district",
    "INNER_OUTER_LONDON",
    "REGION_LONDON",
    "wardcode",
    "PropertyType",
    #"Bedrooms",
    "furnishtype",
    "btrflag",
    ]

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
    prediction = model.predict(input_frame)
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

    # lower_model, upper_model = load_quantile_models()
    # if lower_model is not None and upper_model is not None:
    #     lower = predict_rent(lower_model, input_frame)
    #     upper = predict_rent(upper_model, input_frame)
    #     return point, lower, upper
    return point#, None, None  # Return None for lower and upper if quantile models aren't available



st.set_page_config(page_title="UK Rental Price Predictor", layout="wide")

# Add logo to sidebar
st.sidebar.image("images/Rental_Intel_logo.png")

FEATURE_ORDER = [
    "postcode_district",
    "size_sq_ft",
    "INNER_OUTER_LONDON",
    "REGION_LONDON",
    "wardcode",
    "PropertyType",
    "Bedrooms",
    "furnishtype",
    "btrflag",
    "num_listings_district",
    "num_listings_ward",
    "num_occupants_district",
    "num_occupants_ward",
    "listings_per_occupant_district_percentage",
    "listings_per_occupant_ward_percentage",
    "size_sq_ft_missing",
    "mean_sqft_ward_bedroom_propertytype",
    "mean_sqft_ward_bedroom",
    "mean_sqft_bedroom_propertytype",
    "mean_sqft_global",
    "ward_mean_price",
    "ward_bedroom_mean_price",
    "ward_bedroom_type_price",
    "mean_rent_per_sqft_ward_bedroom_propertytype",
    "mean_rent_per_sqft_ward_bedroom",
    "mean_rent_per_sqft_ward",
]


# FEATURE_LABELS = {
#     "postcode_district": "Postcode district",
#     "INNER_OUTER_LONDON": "Inner/Outer London",
#     "REGION_LONDON": "London region",
#     "wardcode": "Ward code",
#     "PropertyType": "Property type",
#     "Bedrooms": "Bedrooms",
#     "furnishtype": "Furnished Type",
#     "btrflag": "BTR (Build to Rent) or Not",
#     "newbuild": "New Build",

#     "num_listings_district": "Number of listings in postcode district",
#     "num_listings_ward": "Number of listings in ward",
#     "num_occupants_district": "Number of occupants in postcode district",
#     "num_occupants_ward": "Number of occupants in ward",
#     "listings_per_occupant_district_percentage": "Listings per occupant in postcode district (%)",
#     "listings_per_occupant_ward_percentage": "Listings per occupant in ward (%)"
# }


# ---------------------------------------------------------------- load ----
df = load_rental_data()
model = load_model()  # LGBMRegressor (or a pipeline wrapping one)
options = get_feature_options(df)
if list(model.feature_name_) != FEATURE_ORDER:
    raise ValueError("The loaded model's feature schema does not match the prediction inputs.")

# st.dataframe(df.head(100), hide_index=True)

# print(df['btrflag'].unique())
# print(df['newbuild'].unique())

# Explainers built against a sample of the feature matrix (fast, still representative)
# x_train_sample = df[FEATURE_ORDER].sample(min(200, len(df)), random_state=42)
# shap_explainer = build_shap_explainer(model)
# base_value = get_base_value(shap_explainer)
# # smart_explainer = build_smart_explainer(model, x_train_sample, feature_labels=FEATURE_LABELS)


# ------------------------------------------------------------- sidebar ----
st.sidebar.header("Property features")

selected_postcode_district = st.sidebar.selectbox(
    "Postcode district", options["postcode_district"]
)
district_df = df[df["postcode_district"] == selected_postcode_district]

postcode_options = sorted(district_df["Postcode"].dropna().unique().tolist())
selected_postcode = st.sidebar.selectbox("Postcode", postcode_options)
postcode_df = district_df[district_df["Postcode"] == selected_postcode]
wardcode_options = postcode_df["wardcode"].dropna().unique().tolist()
if not wardcode_options:
    st.error(f"No ward code is available for postcode {selected_postcode}.")
    st.stop()
selected_wardcode = wardcode_options[0]
st.sidebar.caption(f"Ward code: {selected_wardcode}")

inner_outer_options = sorted(postcode_df["INNER_OUTER_LONDON"].dropna().unique().tolist())
selected_inner_outer = st.sidebar.selectbox("Inner/Outer London", inner_outer_options)
region_df = postcode_df[postcode_df["INNER_OUTER_LONDON"] == selected_inner_outer]
region_options = sorted(region_df["REGION_LONDON"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("London region", region_options)
selected_property_type = st.sidebar.selectbox("Property type", options["PropertyType"])
selected_bedrooms = st.sidebar.selectbox("Bedrooms", options["Bedrooms"])
selected_furnishtype = st.sidebar.selectbox("Furnished Type", options["furnishtype"])
selected_btrflag = st.sidebar.selectbox("BTR (Build to Rent) or Not", options["btrflag"])

available_sizes = pd.to_numeric(df["size_sq_ft"], errors="coerce")
available_sizes = available_sizes[np.isfinite(available_sizes) & (available_sizes > 0)]
if available_sizes.empty:
    st.error("No valid property sizes are available for the sqft slider.")
    st.stop()
size_min = int(np.floor(available_sizes.min()))
size_max = int(np.ceil(available_sizes.quantile(0.9)))
size_max = max(size_min, size_max)
size_default = int(np.clip(available_sizes.median(), size_min, size_max))
selected_size_sq_ft = st.sidebar.slider(
    "Size (sq ft)",
    min_value=size_min,
    max_value=size_max,
    value=size_default,
    step=1,
)

user_selection = {
    "postcode_district": selected_postcode_district,
    "size_sq_ft": selected_size_sq_ft,
    "INNER_OUTER_LONDON": selected_inner_outer,
    "REGION_LONDON": selected_region,
    "wardcode": selected_wardcode,
    "PropertyType": selected_property_type,
    "Bedrooms": selected_bedrooms,
    "furnishtype": selected_furnishtype,
    "btrflag": selected_btrflag,
}

area_df = df[df["wardcode"] == selected_wardcode]
if area_df.empty:
    area_df = district_df

selection_mask = (
    (area_df["PropertyType"] == selected_property_type)
    & (area_df["Bedrooms"] == selected_bedrooms)
    & (area_df["furnishtype"] == selected_furnishtype)
    & (area_df["btrflag"] == selected_btrflag)
)
context_df = area_df[selection_mask]
if context_df.empty:
    context_df = area_df

derived_feature_fallbacks = {
    "mean_sqft_ward_bedroom_propertytype": (
        "mean_sqft_ward_bedroom",
        "mean_sqft_bedroom_propertytype",
        "mean_sqft_global",
    ),
    "mean_sqft_ward_bedroom": (
        "mean_sqft_bedroom_propertytype",
        "mean_sqft_global",
    ),
    "mean_rent_per_sqft_ward_bedroom_propertytype": (
        "mean_rent_per_sqft_ward_bedroom",
        "mean_rent_per_sqft_ward",
    ),
    "mean_rent_per_sqft_ward_bedroom": ("mean_rent_per_sqft_ward",),
}


def get_context_value(dataframe: pd.DataFrame, column: str) -> float:
    missing_columns = set()
    for source_df in (dataframe, area_df, district_df, df):
        for candidate_column in (column, *derived_feature_fallbacks.get(column, ())):
            if candidate_column not in source_df.columns:
                missing_columns.add(candidate_column)
                continue
            values = pd.to_numeric(source_df[candidate_column], errors="coerce").dropna()
            if not values.empty:
                return float(values.median())
    if column in missing_columns:
        raise ValueError(
            f"Cannot retrieve model feature '{column}'; required data columns are unavailable: "
            f"{', '.join(sorted(missing_columns))}."
        )
    raise ValueError(f"No valid value for model feature '{column}' in the selected area.")


DERIVED_FEATURES = [
    "num_listings_district",
    "num_listings_ward",
    "num_occupants_district",
    "num_occupants_ward",
    "listings_per_occupant_district_percentage",
    "listings_per_occupant_ward_percentage",
    "size_sq_ft_missing",
    "mean_sqft_ward_bedroom_propertytype",
    "mean_sqft_ward_bedroom",
    "mean_sqft_bedroom_propertytype",
    "mean_sqft_global",
    "ward_mean_price",
    "ward_bedroom_mean_price",
    "ward_bedroom_type_price",
    "mean_rent_per_sqft_ward_bedroom_propertytype",
    "mean_rent_per_sqft_ward_bedroom",
    "mean_rent_per_sqft_ward",
]
for feature in DERIVED_FEATURES:
    user_selection[feature] = (
        0.0 if feature == "size_sq_ft_missing" else get_context_value(context_df, feature)
    )


predict_clicked = st.sidebar.button("Predict rent", type="primary")


# --------------------------------------------------------------- title ----
st.title("UK Rental Price Predictor")
st.caption("Select property features on the left, then predict to see the estimated monthly rent.")
st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        position: sticky;
        top: 0;
        z-index: 1000;
        padding: 0.75rem 1rem;
        background: var(--background-color);
        border-bottom: 1px solid var(--secondary-background-color);
    }
    [data-testid="stMetricValue"] {
        font-size: 3rem;
        line-height: 1.15;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def replace_na(x, value_str):

    if pd.isnull(x) == True:
        return value_str

    elif x == 'nan':
        return value_str

    elif str(x).lower().strip() == 'nan':
        return value_str

    elif x == np.nan:
        return value_str

    elif x == None:
        return value_str

    elif x == '':

        return value_str
    else:
        return x


# ---------------------------------------------------------- prediction ----
if predict_clicked:
    # df doubles as the category_reference so the live input row uses the
    # exact same category codes LightGBM was trained on.

    # df_filter = df[(df["Bedrooms"].astype(int) == 1)
    #             & (df["postcode_district"].astype(str) == r'B18')
    #             & (df['furnishtype'].astype(str) == r'furnished')
    #             &(df['btrflag'].astype(str) == 'nan')
    #             ]
    
    # st.dataframe(df_filter)

    # df['furnishtype'] = df['furnishtype'].astype(str).replace('nan','Not Specified')
    # df['btrflag'] = df['btrflag'].astype(str).replace('nan','NO')
    # df['newbuild'] = df['newbuild'].astype(str).replace('nan','NO')

    # df['furnishtype'] = df['furnishtype'].apply(lambda x: replace_na(x, 'Not Specified'))
    # df['btrflag'] = df['btrflag'].apply(lambda x: replace_na(x, 'NO'))
    # df['newbuild'] = df['newbuild'].apply(lambda x: replace_na(x, 'NO'))


    input_frame = build_input_frame(user_selection, FEATURE_ORDER, category_reference=df)
    # predicted_rent, lower, upper = predict_with_interval(model, input_frame)
    predicted_rent = predict_rent(model, input_frame)

    st.metric("Predicted monthly rent", f"£{predicted_rent:,.0f}")
    # if lower is not None and upper is not None:
    #     st.caption(f"Likely range: £{lower:,.0f} – £{upper:,.0f}")

    # ---- Distribution of actual prices for similar properties ----
    similar = context_df.copy()


    # similar['btrflag'] = similar['btrflag'].apply(lambda x: 'YES' if pd.isnull(x) == False else 'NO')
    # # similar['btrflag'] = similar['btrflag'].astype(str)
    # similar['newbuild'] = similar['newbuild'].apply(lambda x: 'YES' if pd.isnull(x) == False else 'NO')

    # similar['btrflag'] = similar['btrflag'].replace('nan', 'NO')  # Replace 'nan' with 'NO' for display purposes
    # similar['newbuild'] = similar['newbuild'].replace('nan', 'NO')



    # # # Fall back to a looser match if too few comparable listings exist
    if len(similar) < 10:

        similar = df[
                #(df["PropertyType"].astype(str) == rf'{user_selection["PropertyType"]}') # Corrected key
                #&
                (df["Bedrooms"].astype(int) == int(user_selection["Bedrooms"]))     # Corrected key
                & (df["postcode_district"].astype(str) == rf'{user_selection["postcode_district"]}')
                # & (df["wardcode"].astype(str) == rf'{user_selection["wardcode"]}')
                # & (df["INNER_OUTER_LONDON"].astype(str) == rf'{user_selection["INNER_OUTER_LONDON"]}')
                # & (df["REGION_LONDON"].astype(str) == rf'{user_selection["REGION_LONDON"]}')
                # & (df["furnishtype"].astype(str) == rf'{user_selection["furnishtype"]}')
                # & (df["btrflag"].astype(str) == rf'{user_selection["btrflag"]}')
                ]



    # similar['btrflag'] = similar['btrflag'].astype(str)
    # similar['newbuild'] = similar['newbuild'].astype(str)

    # similar['btrflag'] = similar['btrflag'].replace('nan', 'NO')  # Replace 'nan' with 'NO' for display purposes
    # similar['newbuild'] = similar['newbuild'].replace('nan', 'NO')

    # similar['furnishtype'] = similar['furnishtype'].apply(lambda x: 'Not Specified' if (pd.isnull(x) == True
    #                                                                                     or x == 'nan'
    #                                                                                     or x == ''
    #                                                                                     or x == None
    #                                                                                     or x == np.nan) else x)

    del df

    # Force Python to clear the unreferenced memory immediately
    gc.collect()

    similar['furnishtype'] =    similar['furnishtype'].apply(lambda x: replace_na(x, 'Not Specified'))
    similar['btrflag'] =        similar['btrflag'].apply(lambda x: replace_na(x, 'NO'))
    similar['newbuild'] =       similar['newbuild'].apply(lambda x: replace_na(x, 'NO'))

    st.dataframe(similar, hide_index=True, use_container_width=True)

    fig = px.histogram(
        similar, x="asking_price", nbins=30,
        title=f"Actual rental prices by property type (n={len(similar)})",
        labels={"asking_price": "Monthly rent (£)"},
        color="PropertyType"
    )
    fig.add_vline(
        x=predicted_rent, line_dash="dash", line_color="red",
        annotation_text="Your prediction", annotation_position="top",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    with st.container():
        fig = px.histogram(
            similar, x="asking_price", nbins=30,
            title=f"Actual rental prices for comparable properties (n={len(similar)})",
            labels={"asking_price": "Monthly rent (£)"},
            color="furnishtype"
        )
        fig.add_vline(
            x=predicted_rent, line_dash="dash", line_color="red",
            annotation_text="Your prediction", annotation_position="top",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        fig = px.histogram(
            similar, x="asking_price", nbins=30,
            title=f"Actual rental prices for comparable properties (n={len(similar)})",
            labels={"asking_price": "Monthly rent (£)"},
            color="btrflag"
        )
        fig.add_vline(
            x=predicted_rent, line_dash="dash", line_color="red",
            annotation_text="Your prediction", annotation_position="top",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        fig = px.histogram(
            similar, x="asking_price", nbins=30,
            title=f"Actual rental prices for comparable properties (n={len(similar)})",
            labels={"asking_price": "Monthly rent (£)"},
            color="newbuild"
        )
        fig.add_vline(
            x=predicted_rent, line_dash="dash", line_color="red",
            annotation_text="Your prediction", annotation_position="top",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

    # # ---- SHAP explainability ----
    # st.subheader("Why this prediction?")

    # contrib_df = explain_single_prediction(shap_explainer, input_frame, top_n=5)
    # explanation_text = generate_explanation_text(contrib_df, predicted_rent)

    # col_plot, col_text = st.columns([1, 1])

    # with col_plot:
    #     shap_fig = plot_local_contribution(smart_explainer, x_train_sample, input_frame)
    #     st.plotly_chart(shap_fig, use_container_width=True)

    # with col_text:
    #     st.write(explanation_text)
    #     st.dataframe(
    #         contrib_df.rename(columns={
    #             "feature": "Feature", "value": "Selected value", "contribution": "Impact on rent (£)"
    #         }),
    #         hide_index=True,
    #     )

else:
    st.info("Set your property features in the sidebar and click **Predict rent** to get started.")
