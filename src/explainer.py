"""
Per-prediction explainability for a LightGBM model using SHAP's
TreeExplainer directly.

TreeExplainer is exact for tree ensembles and needs no background
dataset for LightGBM — contributions come straight from the tree
structure. Built once (cached), then queried per prediction with no
re-ingestion of data on each call. The plot is drawn straight from the
same contribution values computed for the table, so there's only one
source of truth and no duplicate SHAP computation.
"""

# import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# @st.cache_resource(show_spinner="Preparing explainability engine...")
# def build_shap_explainer(_model):
#     """
#     Cached as a resource — built once per session/process. _model is
#     prefixed with underscore so Streamlit doesn't try to hash it.
#     """
#     return shap.TreeExplainer(_model)


def get_base_value(shap_explainer) -> float:
    """
    The model's average prediction over its training data — the
    starting point that feature contributions are added to/subtracted
    from to reach the final predicted value.
    """
    base = shap_explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = base[0]
    return float(base)


def explain_single_prediction(shap_explainer, input_frame: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Compute SHAP contributions for a single input row and return the
    top_n features ranked by absolute contribution.

    Returns a DataFrame with columns: feature, value, contribution
    (signed — positive pushes rent up, negative pushes it down).

    input_frame must use the same categorical dtype/categories the
    model was trained on (see model.build_input_frame) — TreeExplainer
    reads LightGBM's internal category codes, so a mismatch here gives
    silently wrong contributions, not an error.
    """
    shap_values = shap_explainer.shap_values(input_frame)

    contrib_df = pd.DataFrame({
        "feature": input_frame.columns,
        "value": input_frame.iloc[0].astype(str).values,
        "contribution": shap_values[0],
    })
    contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
    contrib_df = contrib_df.sort_values("abs_contribution", ascending=False).head(top_n)
    return contrib_df.drop(columns="abs_contribution")


def plot_local_contribution(contrib_df: pd.DataFrame, predicted_value: float, base_value: float | None = None):
    """
    Draws a horizontal bar chart straight from contrib_df — the same
    values already computed by explain_single_prediction(). No second
    explainer, no recompiled background dataset, nothing re-ingested
    per prediction.
    """
    df_sorted = contrib_df.sort_values("contribution")
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in df_sorted["contribution"]]
    labels = [f"{feature} = {value}" for feature, value in zip(df_sorted["feature"], df_sorted["value"])]

    fig = go.Figure(go.Bar(
        x=df_sorted["contribution"],
        y=labels,
        orientation="h",
        marker_color=colors,
    ))

    subtitle = (
        f"Baseline £{base_value:,.0f} → Predicted £{predicted_value:,.0f}"
        if base_value is not None
        else f"Predicted £{predicted_value:,.0f}"
    )
    fig.update_layout(
        title=f"Feature impact on prediction<br><sup>{subtitle}</sup>",
        xaxis_title="Impact on rent (£)",
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )
    return fig