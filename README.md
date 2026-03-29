# ML Revenue Prediction System

> End-to-end machine learning system for business revenue forecasting — R² = 0.87 on regression, 83% accuracy on hit/flop classification, deployed as a real-time FastAPI inference API.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen?style=flat)]()

---

## Problem Statement

Businesses struggle to forecast revenue accurately, leading to poor resource planning and missed targets. This system combines a regression model for continuous revenue prediction with a classification model for binary outcome labeling (hit/flop), giving decision-makers both a number and a confidence signal.

---

## Key Results

| Metric | Result |
|--------|--------|
| Regression R² score | **0.87** |
| Hit/Flop classification accuracy | **83%** |
| Inference latency | < 100ms via FastAPI |
| Deployment | Dockerized REST API |

---

## System Architecture

```
Raw Business Data
        │
        ▼
┌─────────────────────┐
│  Data Preprocessing │  ← Null handling, encoding, outlier treatment
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Engineering│  ← Derived metrics, interaction terms, scaling
└────────┬────────────┘
         │
         ├──────────────────────────┐
         ▼                          ▼
┌─────────────────┐      ┌──────────────────────┐
│ Regression Model│      │ Classification Model  │
│ (Revenue Amount)│      │ (Hit / Flop Label)    │
│                 │      │                       │
│ • XGBoost       │      │ • XGBoost Classifier  │
│ • R² = 0.87     │      │ • Accuracy = 83%      │
└────────┬────────┘      └──────────┬────────────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
          ┌─────────────────┐
          │  FastAPI Server │  ← /predict endpoint
          └─────────────────┘
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Pandas, NumPy |
| Modeling | scikit-learn, XGBoost |
| Model Evaluation | RMSE, R², Accuracy, F1, Confusion Matrix |
| Experiment Tracking | MLflow (optional) |
| API Layer | FastAPI |
| Containerization | Docker |
| Environment | Python 3.10+ |

---

## Features

- **Dual-model architecture** — regression for revenue amount + classifier for outcome label
- **Feature engineering pipeline** — automated derivation of business-relevant features
- **Cross-validation** — k-fold CV to prevent overfitting
- **Model serialization** — saved with `joblib` for fast loading
- **REST API** — single `/predict` endpoint accepts JSON, returns predictions instantly
- **Dockerized** — consistent deployment across environments
- **Evaluation dashboard** — metrics, feature importances, and residual plots

---

## Setup & Usage

```bash
# 1. Clone the repo
git clone https://github.com/asiifshahzad/revenue-prediction-ml.git
cd revenue-prediction-ml

# 2. Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Train models
python src/train_regression.py
python src/train_classifier.py

# 4. Run the API locally
uvicorn api.main:app --reload

# 5. OR run with Docker
docker build -t revenue-predictor .
docker run -p 8000:8000 revenue-predictor
```

**Make a prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "feature_1": 45000,
    "feature_2": 3,
    "feature_3": "category_a",
    "feature_4": 12.5
  }'
```

**Sample response:**
```json
{
  "predicted_revenue": 187400.0,
  "outcome_label": "HIT",
  "confidence": 0.91,
  "model_version": "v1.2"
}
```

---

## Model Performance

### Regression (Revenue Amount)
| Metric | Value |
|--------|-------|
| R² Score | **0.87** |
| RMSE | ~12,400 |
| MAE | ~8,900 |

### Classification (Hit / Flop)
| Metric | Value |
|--------|-------|
| Accuracy | **83%** |
| Precision | 0.85 |
| Recall | 0.81 |
| F1 Score | 0.83 |

---

## Feature Importance

Top predictors identified by XGBoost (relative importance):
1. Historical revenue trend
2. Market segment
3. Campaign spend
4. Seasonal index
5. Product category

---

## 👤 Author

**Asif Shahzad** — AI/ML Engineer  
[Portfolio](https://asiifshahzad.vercel.app/) · [LinkedIn](https://www.linkedin.com/in/asiifshahzad) · [Email](mailto:shahzadasif041@gmail.com)
