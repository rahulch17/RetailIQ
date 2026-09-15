# RetailIQ — Demand Forecasting and Multi-Tool Business Assistant

Runnable MVP based on Project CP-05:
- reproducible retail data pipeline
- star-schema analytical database
- product-store-week demand forecasting
- tree-based model + LSTM sequence model
- executive dashboard
- forecast explorer
- retrieval over internal policy documents
- planner/executor assistant selecting SQL, forecast, or retrieval tools

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python generate_demo_data.py
python pipeline.py
python train_models.py
streamlit run app.py
```

The demo generator creates synthetic data so the application can be tested end-to-end.
For the actual project, replace the demo files in `data/` with the approved dataset.

## Architecture

User -> Streamlit UI -> Planner/Executor Agent
                    -> SQL Analytics Tool -> SQLite star schema
                    -> Forecast Tool -> trained models
                    -> Retrieval Tool -> policy documents

The agent displays the selected tool and a concise selection explanation.
