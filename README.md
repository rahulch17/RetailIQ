# RetailIQ — Demand Forecasting and Multi-Tool Business Assistant

## Project alignment
This implementation follows the CP-05 RetailIQ brief: reproducible data preparation, a star-schema analytical layer, executive dashboard, product-store-week forecasting with forward validation, a sequence-model comparison, a planner/executor assistant exposing SQL, forecast and retrieval tools, visible tool reasoning, and cited policy answers.

## Dataset
The code is configured for the Walmart Store Sales Forecasting dataset:
- `train.csv`
- `features.csv`
- `stores.csv`

Put them under `walmart/`.

Important source limitation: this Walmart dataset has weekly Store×Department sales, not transaction-level SKU sales. It has markdown fields but no actual selling price, inventory levels, product descriptions, or promotional calendar. The code therefore uses markdown fields as a promotion proxy and a neutral price placeholder. Optional `walmart/products.csv` can supply `Dept` + `description` for text-derived category features.

## Setup
```bash
python -m pip install -r requirements.txt
python pipeline.py
python train_models.py
python build_rag.py
python analysis.py
python -m streamlit run app.py
```

If TensorFlow is difficult to install on your Python version, the tree model and application can still run; the LSTM comparison will be skipped. For the assessed sequence-model requirement, run the LSTM on a supported TensorFlow/Python environment.

## Tool architecture
User → planner → one of:
1. SQL tool → SQLite star schema
2. Forecast tool → trained tree model
3. Retrieval tool → internal knowledge base
→ grounded answer + visible tool/reason

Gemini is optional for ambiguous planning and policy answer synthesis. Deterministic routing remains available so the demo does not fail just because an API key is absent.

## Forecasting
The tree model uses:
- lag 1, 2, 4, 8
- rolling 4 and 8
- promotion proxy
- calendar week and month
- neutral price placeholder

Validation is forward in time. MAPE is calculated only on meaningful non-zero actuals because ordinary MAPE becomes numerically meaningless around zero sales.

## Limitations to defend
1. Walmart is weekly Store×Department data, not transaction/SKU data.
2. No true item price is supplied, so price sensitivity is not estimable from this source.
3. No inventory table is supplied, so inventory cover is reported as unavailable rather than fabricated.
4. Markdown fields are a promotion proxy, not a true campaign flag.
5. Product-description text features require an optional product master with descriptions.

## Submission checklist
- Public Git repository
- README and architecture diagram
- Executed analysis/model notebooks
- SQL schema and query files
- Streamlit application
- Dashboard export PDF
- 8–10 presentation slides
- Live deployment

Do not commit `.env`, database files, model artifacts, or API keys.
