"""
Turns a table of SHAP contributions into a short, plain-English paragraph
explaining why the model predicted what it predicted.
"""

import pandas as pd


FEATURE_PHRASES = {
    "property_type": "the property type",
    "bedrooms": "the number of bedrooms",
    "bathrooms": "the number of bathrooms",
    "furnished": "whether it's furnished",
    "postcode_district": "the location",
    "property_size_sqft": "the property size",
    "has_garden": "having a garden",
    "has_parking": "having parking",
}


def _direction_phrase(contribution: float) -> str:
    return "increased" if contribution > 0 else "decreased"


def _format_value(feature: str, value) -> str:
    if feature == "has_garden" or feature == "has_parking":
        return "yes" if value else "no"
    return str(value)


def generate_explanation_text(contrib_df: pd.DataFrame, predicted_value: float) -> str:
    """
    contrib_df: output of explainer.explain_single_prediction()
    predicted_value: the model's point prediction (£ pcm)

    Produces something like:

    "The predicted rent of £1,850/month is mainly driven by the location
    (Zone 2, Islington), which increased the estimate, and the number of
    bedrooms (3), which also increased the estimate. Not having parking
    decreased the estimate slightly."
    """
    if contrib_df.empty:
        return "No feature contribution data is available for this prediction."

    sentences = []
    top_row = contrib_df.iloc[0]
    top_feature_phrase = FEATURE_PHRASES.get(top_row["feature"], top_row["feature"])
    top_direction = _direction_phrase(top_row["contribution"])
    top_value = _format_value(top_row["feature"], top_row["value"])

    sentences.append(
        f"The predicted rent of £{predicted_value:,.0f}/month is mainly driven by "
        f"{top_feature_phrase} ({top_value}), which {top_direction} the estimate."
    )

    for _, row in contrib_df.iloc[1:].iterrows():
        feature_phrase = FEATURE_PHRASES.get(row["feature"], row["feature"])
        direction = _direction_phrase(row["contribution"])
        value = _format_value(row["feature"], row["value"])
        magnitude = "significantly" if abs(row["contribution"]) > abs(top_row["contribution"]) * 0.5 else "slightly"
        sentences.append(
            f"{feature_phrase.capitalize()} ({value}) {direction} it {magnitude}."
        )

    return " ".join(sentences)
