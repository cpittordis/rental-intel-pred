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

import pandas as pd

from src.data_loader import load_rental_data, get_feature_options
from src.model import load_model, build_input_frame, predict_rent, predict_with_interval
from src.explainer import get_base_value , build_shap_explainer, explain_single_prediction, plot_local_contribution
from src.text_generator import generate_explanation_text


st.set_page_config(page_title="UK Rental Price Predictor", layout="wide")

# Add logo to sidebar
st.sidebar.image("images/Rental_Intel_logo.png")

FEATURE_ORDER = [

## Categorical features : to select
'postcode_district',
 'INNER_OUTER_LONDON',
 'REGION_LONDON',
 'wardcode',
 'PropertyType',
 'Bedrooms',
 'furnishtype',
 'btrflag',
 'newbuild',

 # Calculated features : to be derived from the selected categorical features
 'num_occupants_district',
 'num_occupants_ward',
 'listings_per_occupant_district_percentage',
 'listings_per_occupant_ward_percentage',
 'num_listings_district_log10',
 'num_listings_ward_log10'
]

FEATURE_LABELS = {
    "postcode_district": "Postcode district",
    "INNER_OUTER_LONDON": "Inner/Outer London",
    "REGION_LONDON": "London region",
    "wardcode": "Ward code",
    "PropertyType": "Property type",
    "Bedrooms": "Bedrooms",
    "furnishtype": "Furnished Type",
    "btrflag": "BTR (Build to Rent) or Not",
    "newbuild": "New Build",

    "num_listings_district": "Number of listings in postcode district",
    "num_listings_ward": "Number of listings in ward",
    "num_occupants_district": "Number of occupants in postcode district",
    "num_occupants_ward": "Number of occupants in ward",
    "listings_per_occupant_district_percentage": "Listings per occupant in postcode district (%)",
    "listings_per_occupant_ward_percentage": "Listings per occupant in ward (%)"
}


# ---------------------------------------------------------------- load ----
df = load_rental_data()
model = load_model()  # LGBMRegressor (or a pipeline wrapping one)
options = get_feature_options(df)

# st.dataframe(df.head(100), hide_index=True)

# print(df['btrflag'].unique())
# print(df['newbuild'].unique())

# Explainers built against a sample of the feature matrix (fast, still representative)
# x_train_sample = df[FEATURE_ORDER].sample(min(200, len(df)), random_state=42)
# shap_explainer = build_shap_explainer(model)
# base_value = get_base_value(shap_explainer)
# # smart_explainer = build_smart_explainer(model, x_train_sample, feature_labels=FEATURE_LABELS)


# ------------------------------------------------------------- sidebar ----
# ------------------------------------------------------------- sidebar ----
st.sidebar.header("Property features")

# Get postcode_district selection first
selected_postcode_district = st.sidebar.selectbox("Postcode district", options["postcode_district"])

# Filter wardcodes based on selected postcode_district
filtered_wardcodes_df = df[df["postcode_district"] == selected_postcode_district]
filtered_wardcode_options = sorted(filtered_wardcodes_df["wardcode"].dropna().unique().tolist())

# Filter INNER/OUTER London Regions based on selected postcode_district
filtered_inner_outer_london_df = df[df["postcode_district"] == selected_postcode_district]
filtered_inner_outer_london_options = sorted(filtered_inner_outer_london_df["INNER_OUTER_LONDON"].dropna().unique().tolist())

# Filter London Regions based on selected postcode_district
filtered_london_regions_df = df[df["postcode_district"] == selected_postcode_district]
filtered_london_regions_options = sorted(filtered_london_regions_df["REGION_LONDON"].dropna().unique().tolist())

user_selection = {
    "postcode_district": selected_postcode_district,
    "INNER_OUTER_LONDON": st.sidebar.selectbox("Inner/Outer London", filtered_inner_outer_london_options),
    "REGION_LONDON": st.sidebar.selectbox("London region", filtered_london_regions_options),
    # Use the filtered wardcode options for the wardcode selectbox
    "wardcode": st.sidebar.selectbox("Ward code", filtered_wardcode_options),
    "PropertyType": st.sidebar.selectbox("Property type", options["PropertyType"]),
    "Bedrooms": st.sidebar.selectbox("Bedrooms", options["Bedrooms"]),
    "furnishtype": st.sidebar.selectbox("Furnished Type", options["furnishtype"]),
    "btrflag": st.sidebar.selectbox("BTR (Build to Rent) or Not", options["btrflag"]),
    "newbuild": st.sidebar.selectbox("New Build", options["newbuild"]),
}

# Now, calculate the dependent features using the selected values
# selected_postcode_district is already available
selected_wardcode = user_selection["wardcode"] # Get the selected wardcode from the filtered options

# Helper function to safely get pre-calculated values
def get_precalculated_value(dataframe, col_filter, filter_value, target_col, default_value=0):
    filtered_df = dataframe.loc[dataframe[col_filter] == filter_value, target_col]
    if not filtered_df.empty:
        # Ensure the returned value is numerical
        return pd.to_numeric(filtered_df.iloc[0], errors='coerce')#.fillna(default_value)
    return default_value

# Assign directly to the log-transformed feature names as expected by FEATURE_ORDER
num_listings_district_val = get_precalculated_value(df, "postcode_district", selected_postcode_district, "num_listings_district", 0)
user_selection["num_listings_district_log10"] = np.log10(num_listings_district_val + 1)

num_listings_ward_val = get_precalculated_value(df, "wardcode", selected_wardcode, "num_listings_ward", 0)
user_selection["num_listings_ward_log10"] = np.log10(num_listings_ward_val + 1)

user_selection["num_occupants_district"] = get_precalculated_value(df, "postcode_district", selected_postcode_district, "num_occupants_district", 0)
user_selection["num_occupants_ward"] = get_precalculated_value(df, "wardcode", selected_wardcode, "num_occupants_ward", 0)
user_selection["listings_per_occupant_district_percentage"] = get_precalculated_value(df, "postcode_district", selected_postcode_district, "listings_per_occupant_district_percentage", 0.0)
user_selection["listings_per_occupant_ward_percentage"] = get_precalculated_value(df, "wardcode", selected_wardcode, "listings_per_occupant_ward_percentage", 0.0)


predict_clicked = st.sidebar.button("Predict rent", type="primary")


# --------------------------------------------------------------- title ----
st.title("UK Rental Price Predictor")
st.caption("Select property features on the left, then predict to see the estimated monthly rent.")


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

    input_frame = build_input_frame(user_selection, FEATURE_ORDER, category_reference=df)
    predicted_rent, lower, upper = predict_with_interval(model, input_frame)

    col_pred, col_context = st.columns([1, 2])

    with col_pred:
        st.metric("Predicted monthly rent", f"£{predicted_rent:,.0f}")
        if lower is not None and upper is not None:
            st.caption(f"Likely range: £{lower:,.0f} – £{upper:,.0f}")

    # ---- Distribution of actual prices for similar properties ----
    with col_context:
        similar = df[
            (df["PropertyType"].astype(str) == rf'{user_selection["PropertyType"]}') # Corrected key
            & (df["Bedrooms"].astype(int) == int(user_selection["Bedrooms"]))     # Corrected key
            & (df["postcode_district"].astype(str) == rf'{user_selection["postcode_district"]}')
            & (df["wardcode"].astype(str) == rf'{user_selection["wardcode"]}')
            & (df["INNER_OUTER_LONDON"].astype(str) == rf'{user_selection["INNER_OUTER_LONDON"]}')
            & (df["REGION_LONDON"].astype(str) == rf'{user_selection["REGION_LONDON"]}')
            & (df["furnishtype"].astype(str) == rf'{user_selection["furnishtype"]}')
            & (df["btrflag"].astype(str) == rf'{user_selection["btrflag"]}')
            & (df["newbuild"].astype(str) == rf'{user_selection["newbuild"]}')
        ]



        # # # Fall back to a looser match if too few comparable listings exist
        if len(similar) < 10:
            
            similar = df[
                    (df["PropertyType"].astype(str) == rf'{user_selection["PropertyType"]}') # Corrected key
                    & (df["Bedrooms"].astype(int) == int(user_selection["Bedrooms"]))     # Corrected key
                    & (df["postcode_district"].astype(str) == rf'{user_selection["postcode_district"]}')
                    # & (df["wardcode"].astype(str) == rf'{user_selection["wardcode"]}')
                    # & (df["INNER_OUTER_LONDON"].astype(str) == rf'{user_selection["INNER_OUTER_LONDON"]}')
                    # & (df["REGION_LONDON"].astype(str) == rf'{user_selection["REGION_LONDON"]}')
                    # & (df["furnishtype"].astype(str) == rf'{user_selection["furnishtype"]}')
                    # & (df["btrflag"].astype(str) == rf'{user_selection["btrflag"]}')
                    # & (df["newbuild"].astype(str) == rf'{user_selection["newbuild"]}')
                    ]
        
        similar['btrflag'] = similar['btrflag'].astype(str)
        similar['newbuild'] = similar['newbuild'].astype(str)

        similar['btrflag'] = similar['btrflag'].replace('nan', 'NO')  # Replace 'nan' with 'NO' for display purposes
        similar['newbuild'] = similar['newbuild'].replace('nan', 'NO')

        st.dataframe(similar , hide_index=True)

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
