# 🚀 Amazon Supply Chain Intelligence — Project Workflow

> AI-driven delivery delay risk prediction using Graph Neural Networks + Ensemble ML

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    AMAZON SUPPLY CHAIN INTELLIGENCE                  │
└──────────────────────────────────────────────────────────────────────┘

RAW DATA (CSV)
     │
     ▼
┌─────────────────┐
│  DataIngestion  │  ← Drops PII, engineers target, splits 80/20
└────────┬────────┘
         │  train.csv / test.csv
         ▼
┌──────────────────────┐
│  DataTransformation  │  ← Feature engineering, SMOTE balancing
└──────────┬───────────┘
           │  train_arr / test_arr / preprocessor.pkl
           ▼
┌─────────────────┐
│  ModelTrainer   │  ← CatBoost | XGBoost | LightGBM | RF | GNN
└────────┬────────┘
         │  model.pkl / metrics.json / feature_importance.json
         ▼
┌─────────────────────────┐
│  Flask REST API         │  ← /predict /train /metrics /health
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  HTML Frontend          │  ← index.html + home.html
│  (Chart.js animations)  │
└─────────────────────────┘
```

---

## 📁 Project Structure

```
supply_chain_intelligence/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       ← Step 1: Load & clean data
│   │   ├── data_transformation.py  ← Step 2: Feature eng + SMOTE
│   │   └── model_trainer.py        ← Step 3: Train & evaluate all models
│   │
│   └── pipeline/
│       ├── train_pipeline.py       ← Orchestrates Steps 1-3
│       └── predict_pipeline.py     ← Inference for single orders
│
├── templates/
│   ├── index.html                  ← Landing page + live metrics dashboard
│   └── home.html                   ← Interactive prediction form
│
├── notebooks/
│   ├── 01_EDA.ipynb                ← Exploratory Data Analysis
│   └── 02_Model_Training.ipynb     ← Model comparison experiments
│
├── artifacts/                      ← Auto-generated at runtime
│   ├── raw.csv
│   ├── train.csv / test.csv
│   ├── preprocessor.pkl
│   ├── model.pkl
│   ├── metrics.json
│   ├── feature_importance.json
│   └── *_map.pkl                   ← Risk maps from training
│
├── logs/                           ← Timestamped log files
├── application.py                  ← Flask app entry point
├── train_model.py                  ← CLI training entry point
├── exception.py                    ← Custom exception handler
├── logger.py                       ← Custom logger
├── utils.py                        ← Shared utilities
├── setup.py
├── requirements.txt
├── README_WORKFLOW.md              ← This file
└── README_SETUP.md                 ← Installation guide
```

---

## 🎯 Target Variable Engineering

The `Delay_Risk_Level` column is engineered from existing data:

| Class | Label | Condition |
|-------|-------|-----------|
| **0** | On-Time | `actual_days ≤ scheduled_days` AND `Late_delivery_risk == 0` |
| **1** | At Risk | Gap of 1-2 days OR `Late_delivery_risk == 1` with on-time status |
| **2** | Delayed | Gap > 2 days OR delivery status in `['Late delivery', 'Shipping canceled']` |

---

## ⚙️ Feature Engineering

| Feature | Description | Leakage Risk |
|---------|-------------|--------------|
| `shipping_delay_gap` | actual − scheduled days | None |
| `is_weekend_order` | 1 if order placed on weekend | None |
| `order_month` | Month of order (1-12) | None |
| `order_day_of_week` | 0=Mon … 6=Sun | None |
| `discount_impact` | discount / (price + 1) | None |
| `profit_margin` | profit / (sales + 1) | None |
| `shipping_mode_encoded` | Ordinal: SameDay=0 → Standard=3 | None |
| `market_risk_score` | **Historical** avg delay rate per market | ✅ Train-only |
| `region_delay_rate` | **Historical** avg delayed rate per region | ✅ Train-only |
| `customer_order_frequency` | Order count per customer | ✅ Train-only |

> ⚠️ **Data Leakage Prevention**: `market_risk_score`, `region_delay_rate`, and `customer_order_frequency` are computed exclusively from training data, then **mapped** onto test data — never fitted on combined data.

---

## 🌐 Graph Neural Network Design

### Why GNN?
Orders in a supply chain are **not independent**. They share:
- **Common carriers** → same traffic/weather delays
- **Common warehouses** → same processing bottlenecks
- **Common customers** → same priority handling
- **Common routes** → correlated delay patterns

A GNN captures these relational dependencies that tabular models cannot.

### Graph Construction
```
Nodes:    Each order → feature vector (n_features,)
Edges:    Connect orders that share region, shipping mode, or customer
Method:   KNN-based (k=5) on feature space for inference efficiency
```

### Architecture
```
Input(n_features)
      │
   GCNConv(→128) + BatchNorm + ReLU + Dropout(0.3)
      │
   GCNConv(→64) + BatchNorm + ReLU
      │
   Linear(64→3)
      │
   Softmax → [P(On-Time), P(At Risk), P(Delayed)]
```

### Training
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Loss**: CrossEntropyLoss
- **Early Stopping**: patience=10 epochs
- **Device**: Auto-detects GPU (CUDA) or falls back to CPU

---

## ⚖️ Class Imbalance Strategy

**Why SMOTE?**

The three delay classes are typically imbalanced (Delayed >> On-Time >> At Risk). SMOTE is preferred because:
1. Creates **synthetic** minority samples in feature space (not raw copies)
2. Reduces overfitting risk vs. simple duplication
3. Preserves distribution characteristics better than random oversampling

Applied **after** preprocessing to avoid leaking minority sample statistics into the scaler fit.

**ADASYN** is benchmarked in Notebook 02 for comparison.

---

## 🏆 Model Selection Logic

1. All models are trained with `GridSearchCV` (3-fold stratified cross-validation)
2. Evaluated on held-out test set using **Weighted F1** as primary metric
3. Best model is selected automatically and saved to `artifacts/model.pkl`
4. If best score < 0.60, an exception is raised to flag insufficient performance

**Why Weighted F1?**
- Accounts for class imbalance (unlike accuracy)
- Penalizes models that ignore minority classes (unlike macro F1 when classes are very unequal)
- Industry standard for multi-class imbalanced classification

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Landing page dashboard |
| `GET` | `/predict` | Prediction form |
| `POST` | `/predict` | JSON prediction request |
| `POST` | `/train` | Trigger background training |
| `GET` | `/train/status` | Training progress + logs |
| `GET` | `/health` | System health check |
| `GET` | `/metrics` | Latest training metrics (JSON) |
| `GET` | `/api/features` | Top feature importances (JSON) |
| `GET` | `/api/model-info` | Best model info |

---

## 🎨 Frontend Architecture

### `index.html` — Dashboard
- Animated hero with parallax grid background
- Live stat cards (accuracy, F1, model name, kappa) fetched from `/metrics`
- Model comparison bar chart (Chart.js)
- Feature importance horizontal bar chart (Chart.js)
- Model comparison table with ★ BEST badge

### `home.html` — Prediction Form
- Two-column form (Order Details + Location/Context)
- Client-side validation with inline error messages
- Animated confidence ring (SVG + CSS transitions)
- Probability breakdown with animated bars
- Color-coded result card (green/yellow/red)
- Actionable recommendation text

---

## 📊 Evaluation Metrics Computed

| Metric | What it measures |
|--------|-----------------|
| Accuracy | Overall correct predictions |
| Weighted F1 | F1 weighted by class support |
| Macro F1 | Unweighted average F1 per class |
| Cohen's Kappa | Agreement beyond chance |
| Confusion Matrix | Per-class prediction breakdown |
| ROC-AUC (OvR) | Discriminability for each class |
| Classification Report | Per-class precision / recall / F1 |

---

## 🧬 Enhancement Summary

| Enhancement | Benefit |
|-------------|---------|
| GNN integration | Captures order-to-order relational dependencies |
| Risk-aware features | Market/region delay history encoded as signals |
| SMOTE + ADASYN comparison | Robust class balancing strategy |
| Streaming training logs | Real-time training visibility |
| Live metrics dashboard | Always-current performance display |
| Confidence rings | Interpretable uncertainty visualization |
| Health endpoint | Production monitoring ready |
| Lazy artifact loading | Fast cold-start for prediction API |
