# UK Rental Price Predictor

Single-page Streamlit app: select property features → get a predicted
monthly rent → see it against the actual price distribution for similar
properties → see a SHAP-based explanation of the prediction in plain English.

## Structure

```
uk_rental_app/
├── app.py                  # Main Streamlit page — layout & orchestration
├── requirements.txt
├── src/
│   ├── data_loader.py       # Cached data loading (file or PostgreSQL)
│   ├── model.py              # Cached model loading + prediction
│   ├── explainer.py          # Shapash/SHAP per-prediction explainability
│   └── text_generator.py     # SHAP contributions → plain-English text
├── models/
│   └── rental_model.joblib   # Your trained sklearn-compatible pipeline
└── data/
    └── uk_rental_listings.parquet  # Your cleaned rental listings snapshot
```

## Setup

```bash
pip install -r requirements.txt
```

Drop your trained **LightGBM** (`LGBMRegressor`) model into
`models/rental_model.joblib`. Categorical features (`property_type`,
`furnished`, `postcode_district`, `has_garden`, `has_parking`) should be
trained as pandas `category` dtype so LightGBM's native categorical
handling is used — `src/model.py` and `src/data_loader.py` both cast to
`category` and rely on matching category codes between training and
live predictions.

For the prediction interval shown alongside the point estimate, train
two extra `LGBMRegressor`s with `objective='quantile'` — one with
`alpha=0.1`, one with `alpha=0.9` — and save them as
`models/rental_model_q10.joblib` and `models/rental_model_q90.joblib`.
This is optional: the app falls back to showing the point estimate
alone if those files aren't present.

Drop your cleaned listings dataset into
`data/uk_rental_listings.parquet`, or switch `app.py` to call
`load_rental_data_from_db()` instead if you're pulling live from
PostgreSQL.

## Run

```bash
streamlit run app.py
```

## Notes on the pieces

- **Caching**: `st.cache_data` for the dataset (invalidates on a TTL so
  refreshed data gets picked up), `st.cache_resource` for the model and
  the Shapash explainer (loaded once per session, not per rerun).
- **Distribution plot**: filters actual listings to the same property
  type + bedrooms + postcode district as the user's selection, falling
  back to a looser match if there are too few comparable listings.
  Swap the filtering logic in `app.py` for k-nearest-neighbours on the
  full feature vector if you want a tighter "similar properties" set.
- **Explainability**: `shap.TreeExplainer` is used directly against the
  LightGBM model for the contribution table (exact and fast for tree
  ensembles, unlike the generic `KernelExplainer` fallback). Shapash's
  `SmartExplainer` is layered on top purely for its ready-made local
  contribution plot — it also detects LightGBM as a tree model, so both
  paths agree on the numbers. `text_generator.py` turns the contribution
  table into a plain-English sentence.
- **Text generation**: currently rule-based (fast, deterministic, no
  extra API calls). If you want richer/more natural phrasing, swap
  `generate_explanation_text()` for a call to an LLM, passing the
  `contrib_df` as structured context.
