# 🚀 Master Project Prompt: Amazon Supply Chain Intelligence — End-to-End ML Delivery Delay Predictor

---

## 🧠 Project Overview

Build a **production-ready, end-to-end machine learning system** called **"Amazon Supply Chain Intelligence"** — an AI-driven platform that predicts **delivery delay risk levels** (`On-Time`, `At Risk`, `Delayed`) for e-commerce orders. This is a **multi-class classification** problem enhanced with **Graph Neural Networks (GNN)**, served through a **Flask backend**, and presented via an **interactive HTML/CSS/JS frontend**.

---

## 📁 Exact Project File Structure to Generate

```
project_root/
│
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── predict_pipeline.py
│       └── train_pipeline.py
│
├── __init__.py
├── exception.py
├── logger.py
├── utils.py
│
├── templates/
│   ├── home.html
│   └── index.html
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Model_Training.ipynb
│
├── artifacts/                  # Auto-created at runtime
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── catboost_info/
│
├── logs/                       # Auto-created at runtime
│
├── application.py              # Flask entry point
├── train_model.py
├── setup.py
├── requirements.txt
├── requirments.txt             # Keep both (as in screenshot)
├── .gitignore
├── README_WORKFLOW.md
└── README_SETUP.md
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| ML Models | CatBoost, XGBoost, RandomForest, LightGBM, GNN (PyTorch Geometric) |
| Deep Learning | PyTorch + PyTorch Geometric (for GNN) |
| Backend | Flask |
| Frontend | HTML5, CSS3 (with animations), Vanilla JavaScript (Charts.js, Fetch API) |
| Data | Pandas, NumPy, Scikit-learn |
| Imbalance | imbalanced-learn (SMOTE, ADASYN) |
| Serialization | dill / pickle |
| EDA | Matplotlib, Seaborn, Plotly |
| Logging | Python logging module (custom) |
| Exception | Custom exception handler with traceback |

---

## 📊 Dataset Columns Available

```python
['Type', 'Days for shipping (real)', 'Days for shipment (scheduled)',
 'Benefit per order', 'Sales per customer', 'Delivery Status',
 'Late_delivery_risk', 'Category Id', 'Category Name', 'Customer City',
 'Customer Country', 'Customer Email', 'Customer Fname', 'Customer Id',
 'Customer Lname', 'Customer Password', 'Customer Segment',
 'Customer State', 'Customer Street', 'Customer Zipcode',
 'Department Id', 'Department Name', 'Latitude', 'Longitude', 'Market',
 'Order City', 'Order Country', 'Order Customer Id',
 'order date (DateOrders)', 'Order Id', 'Order Item Cardprod Id',
 'Order Item Discount', 'Order Item Discount Rate', 'Order Item Id',
 'Order Item Product Price', 'Order Item Profit Ratio',
 'Order Item Quantity', 'Sales', 'Order Item Total',
 'Order Profit Per Order', 'Order Region', 'Order State', 'Order Status',
 'Order Zipcode', 'Product Card Id', 'Product Category Id',
 'Product Description', 'Product Image', 'Product Name', 'Product Price',
 'Product Status', 'shipping date (DateOrders)', 'Shipping Mode']
```

**Target Variable (Engineer This):**
Create a new column `Delay_Risk_Level` with 3 classes:
- `0 = On-Time`: `Days for shipping (real)` <= `Days for shipment (scheduled)` AND `Late_delivery_risk == 0`
- `1 = At Risk`: difference of 1-2 days OR `Late_delivery_risk == 1` with `Delivery Status == 'Shipping on Time'`
- `2 = Delayed`: `Days for shipping (real)` > `Days for shipment (scheduled)` + 2 OR `Delivery Status` in `['Late delivery', 'Shipping canceled']`

---

## 🔧 Implementation Instructions — File by File

---

### `exception.py`
```python
# Custom exception class that captures:
# - File name where error occurred
# - Line number
# - Error message
# Uses sys.exc_info() for traceback details
# Format: "Error occurred in python script name [{file}] line number [{line}] error message [{msg}]"
```

### `logger.py`
```python
# Custom logger that:
# - Creates a logs/ directory automatically
# - Names log files with timestamp: MM_DD_YYYY_HH_MM_SS.log
# - Uses logging.basicConfig with format: "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
```

### `utils.py`
```python
# Utility functions:
# save_object(file_path, obj) → uses dill to serialize
# load_object(file_path) → uses dill to deserialize
# evaluate_models(X_train, y_train, X_test, y_test, models, params) →
#     runs GridSearchCV for each model, returns dict of {model_name: test_score}
# get_class_weights(y) → compute class weights for imbalanced data
```

---

### `src/components/data_ingestion.py`

```python
# DataIngestionConfig: dataclass with paths for raw, train, test CSV files in artifacts/
# DataIngestion class:
#   - initiate_data_ingestion():
#       1. Read CSV from data source (local path passed to class)
#       2. Drop irrelevant columns: ['Customer Email', 'Customer Password',
#          'Customer Fname', 'Customer Lname', 'Customer Street',
#          'Product Description', 'Product Image', 'Order Customer Id',
#          'Order Item Cardprod Id', 'Order Item Id']
#       3. Engineer target variable Delay_Risk_Level (3 classes as defined above)
#       4. Log all steps
#       5. Save raw data to artifacts/raw.csv
#       6. Train/test split (80/20, stratified on target)
#       7. Save to artifacts/train.csv and artifacts/test.csv
#       8. Return (train_path, test_path)
```

---

### `src/components/data_transformation.py`

```python
# DataTransformationConfig: path for preprocessor.pkl in artifacts/
# DataTransformation class:
#
# Feature Engineering (add all of these as new columns BEFORE preprocessing):
#   - shipping_delay_gap = Days_for_shipping_real - Days_for_shipment_scheduled
#   - is_weekend_order = 1 if order_date is Saturday/Sunday
#   - order_month = month from order date
#   - order_hour = hour from order date (if available)
#   - order_day_of_week = weekday number
#   - discount_impact = Order_Item_Discount / (Order_Item_Product_Price + 1)
#   - profit_margin = Order_Profit_Per_Order / (Sales + 1)
#   - shipping_mode_encoded (target encode or ordinal: Same Day=0, First Class=1, Second Class=2, Standard=3)
#   - market_risk_score = historical average delay rate per market (from train set)
#   - region_delay_rate = historical average delay rate per region (from train set)
#   - customer_order_frequency = count of orders per customer (from train set)
#
# Numerical Features Pipeline:
#   - SimpleImputer(strategy='median')
#   - StandardScaler()
#
# Categorical Features Pipeline:
#   - SimpleImputer(strategy='most_frequent')
#   - OneHotEncoder(handle_unknown='ignore', sparse=False)
#
# Class Imbalance Handling:
#   - Apply SMOTE after transformation on training data only
#   - Log class distribution before and after SMOTE
#   - Compare with ADASYN in notebook
#
# get_data_transformer_object() → returns ColumnTransformer pipeline
# initiate_data_transformation(train_path, test_path) → 
#     returns (train_arr, test_arr, preprocessor_path)
```

---

### `src/components/model_trainer.py`

```python
# ModelTrainerConfig: path for model.pkl in artifacts/
# ModelTrainer class:
#
# Models to compare:
# models = {
#     "CatBoost": CatBoostClassifier(verbose=0),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
#     "Random Forest": RandomForestClassifier(),
#     "LightGBM": LGBMClassifier(),
#     "GNN": <custom GNN wrapper — see GNN section below>
# }
#
# Hyperparameter Grids (define for each model):
# CatBoost: iterations, learning_rate, depth
# XGBoost: n_estimators, max_depth, learning_rate, subsample
# RandomForest: n_estimators, max_depth, min_samples_split
# LightGBM: n_estimators, learning_rate, num_leaves
#
# Evaluation Metrics (compute ALL of these):
#   - Accuracy
#   - Weighted F1-Score
#   - Macro F1-Score
#   - Cohen's Kappa
#   - Classification Report (per class precision/recall/f1)
#   - Confusion Matrix
#   - ROC-AUC (one-vs-rest)
#
# initiate_model_trainer(train_arr, test_arr) →
#     1. Split features/target
#     2. Run evaluate_models() from utils
#     3. Select best model (by weighted F1)
#     4. If best score < 0.60, raise CustomException
#     5. Save best model as artifacts/model.pkl
#     6. Return full metrics report
```

---

### 🌐 GNN Implementation (Graph Neural Network)

```python
# Location: src/components/model_trainer.py (or src/gnn_model.py)
# Use PyTorch Geometric (torch_geometric)
#
# Graph Construction Strategy:
#   - Each ORDER is a NODE in the graph
#   - EDGES connect orders that share: same customer, same region, same shipping mode, same product category
#   - Node features = transformed feature vector for each order
#   - Edge features (optional) = similarity weights
#
# GNN Architecture:
# class DeliveryGNN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)  # or SAGEConv
#         self.conv2 = GCNConv(hidden_channels, out_channels)
#         self.classifier = Linear(out_channels, num_classes)
#         self.dropout = Dropout(0.3)
#
#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.dropout(x)
#         x = self.conv2(x, edge_index)
#         return self.classifier(x)
#
# GNNWrapper class (sklearn-compatible):
#   - fit(X, y) → builds graph, trains GNN for N epochs with Adam optimizer
#   - predict(X) → builds inference graph, returns class predictions
#   - predict_proba(X) → returns softmax probabilities
#   - Training loop: CrossEntropyLoss, early stopping on val loss
#   - Save/load model state dict separately
#
# Graph Building Function:
# def build_graph(X_df):
#     # Build edge_index using KNN or shared attribute matching
#     # Return torch_geometric.data.Data(x=node_feats, edge_index=edge_index)
```

---

### `src/pipeline/train_pipeline.py`

```python
# TrainPipeline class:
#   - run_pipeline(data_path):
#       1. DataIngestion → get train/test paths
#       2. DataTransformation → get transformed arrays + preprocessor path
#       3. ModelTrainer → train all models, return best model report
#       4. Log summary of best model and metrics
#       5. Return metrics dict for display
```

### `src/pipeline/predict_pipeline.py`

```python
# PredictPipeline class:
#   - predict(features_dict):
#       1. Load preprocessor from artifacts/preprocessor.pkl
#       2. Load model from artifacts/model.pkl
#       3. Convert input dict → DataFrame
#       4. Apply feature engineering (same as training)
#       5. Transform with preprocessor
#       6. Predict class and probabilities
#       7. Return: {'prediction': label, 'confidence': float, 'probabilities': dict}
#       Label mapping: {0: 'On-Time', 1: 'At Risk', 2: 'Delayed'}
#
# CustomData class:
#   - __init__(**kwargs) → stores all form fields
#   - get_data_as_dataframe() → returns single-row DataFrame
```

---

### `application.py` (Flask Backend)

```python
# Flask app with the following routes:
#
# GET  /              → renders index.html (landing page)
# GET  /predict       → renders home.html (prediction form)
# POST /predict       → calls PredictPipeline, returns JSON response
# POST /train         → triggers TrainPipeline.run_pipeline(), streams logs
# GET  /health        → returns {"status": "ok", "model_loaded": bool}
# GET  /metrics       → returns last training metrics as JSON (stored in artifacts/metrics.json)
# GET  /api/features  → returns feature importance from best model as JSON
#
# CORS enabled for all routes
# Error handlers for 404, 500
# Use threading for training endpoint (non-blocking)
```

---

### `templates/index.html` — Landing Page

```
Design requirements:
- Dark theme with Amazon-inspired color palette (orange #FF9900, dark navy #131921)
- Animated hero section with moving background particles or gradient animation
- "How It Works" section with 3 steps (icons + descriptions)
- Model performance stats cards (accuracy, F1, etc.) fetched from /metrics
- Feature importance bar chart rendered with Chart.js (data from /api/features)
- Navigation bar with links to Predict and GitHub
- Smooth scroll behavior
- Animated counter for stats (orders processed, accuracy %)
- Responsive grid layout
- Footer with tech stack badges
```

### `templates/home.html` — Prediction Form Page

```
Design requirements:
- Dark theme consistent with index.html
- Two-column form layout:
    LEFT COLUMN (Order Details):
      - Shipping Mode (dropdown): Same Day / First Class / Second Class / Standard
      - Days for Shipping Real (number input)
      - Days for Shipment Scheduled (number input)
      - Order Item Quantity (number)
      - Order Item Discount Rate (slider 0-1)
      - Order Item Product Price (number)
      - Sales per Customer (number)

    RIGHT COLUMN (Location & Context):
      - Market (dropdown): Europe, LATAM, Pacific Asia, USCA, Africa
      - Order Region (dropdown: pre-filled regions)
      - Customer Segment (dropdown): Consumer / Corporate / Home Office
      - Department Name (dropdown from dataset)
      - Category Name (text input with datalist)
      - Order Status (dropdown)
      - Type (dropdown)

- Real-time form validation with inline error messages
- Animated PREDICT button with loading spinner
- Result section (initially hidden, shown after prediction):
    - Large animated risk badge: 🟢 On-Time / 🟡 At Risk / 🔴 Delayed
    - Confidence score as circular progress ring (CSS/JS animated)
    - Probability breakdown as horizontal bar chart (Chart.js)
    - Recommendation text based on prediction
- Ability to reset form and predict again
- Smooth CSS transitions for result reveal
```

---

### `notebooks/01_EDA.ipynb`

**Structure and required cells:**
```
1. Import Libraries & Load Data
2. Dataset Overview (shape, dtypes, describe, head)
3. Missing Value Analysis (heatmap with seaborn)
4. Target Variable Engineering
   - Define Delay_Risk_Level logic
   - Show class distribution with count + pie chart
5. Univariate Analysis
   - Histograms for all numerical columns
   - Value counts for all categorical columns
6. Bivariate Analysis
   - Delay_Risk_Level vs Shipping Mode (grouped bar)
   - Delay_Risk_Level vs Market (grouped bar)
   - Delay_Risk_Level vs Customer Segment
   - Delay vs Days gap (box plots)
7. Correlation Heatmap (numerical features)
8. Outlier Detection (IQR method + box plots)
9. Geographic Analysis
   - Scatter plot of Latitude/Longitude colored by delay class
10. Time-based Analysis
    - Orders by month, day of week
    - Delay rate over time
11. Feature Engineering Preview
    - Show all engineered features
    - Check distributions of new features
12. Class Imbalance Visualization
    - Before SMOTE: class bar chart
    - Simulated After SMOTE: expected distribution
13. Key Insights Summary (markdown cell)
```

---

### `notebooks/02_Model_Training.ipynb`

**Structure and required cells:**
```
1. Imports & Config
2. Load Preprocessed Data (from artifacts/)
3. Apply SMOTE + ADASYN (compare both)
4. Train/evaluate each model:
   a. CatBoost
   b. XGBoost  
   c. Random Forest
   d. LightGBM
   e. GNN (with training loss curve plot)
5. Model Comparison Table (all metrics side by side)
6. Confusion Matrix Subplots (one per model)
7. ROC Curves (one-vs-rest, all models on same plot)
8. Feature Importance (from best model)
9. Cross-Validation (5-fold stratified)
10. Hyperparameter Tuning Results (best params per model)
11. Final Model Selection Justification (markdown)
12. Save best model + preprocessor
```

---

### `README_WORKFLOW.md`

```markdown
# Amazon Supply Chain Intelligence — Project Workflow

## Architecture Diagram (ASCII)
[Data] → [DataIngestion] → [DataTransformation + SMOTE] → [ModelTrainer]
         ↓                         ↓                              ↓
      raw.csv              preprocessor.pkl               model.pkl + metrics.json
                                                                  ↓
                                                    [Flask API + HTML Frontend]
                                                                  ↓
                                                         [User Prediction]

## Component Descriptions
(describe each component's role)

## Data Flow
(step-by-step data transformation journey)

## Model Selection Logic
(explain how best model is selected)

## GNN Design Rationale
(explain why GNN, how graph is constructed, what edges mean)

## Feature Engineering Decisions
(why each feature was engineered)

## Class Imbalance Strategy
(why SMOTE over other techniques)
```

---

### `README_SETUP.md`

```markdown
# Setup Guide

## Prerequisites
- Python 3.10
- pip
- Git

## Installation Steps
1. Clone repo
2. Create virtual environment: `python3.10 -m venv venv`
3. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Install PyTorch: `pip install torch torchvision torchaudio`
6. Install PyTorch Geometric: (exact commands for Python 3.10 + CPU/GPU)

## Dataset
- Download dataset (provide source)
- Place at: `data/supply_chain_data.csv`

## Training
`python train_model.py --data_path data/supply_chain_data.csv`

## Running the App
`python application.py`
Visit: http://localhost:5000

## Running Notebooks
`jupyter notebook notebooks/`

## Environment Variables
- PORT (default 5000)
- DEBUG (default True)
- MODEL_PATH (default artifacts/model.pkl)
```

---

### `requirements.txt`

```
flask==3.0.0
flask-cors==4.0.0
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
catboost==1.2.2
xgboost==2.0.0
lightgbm==4.1.0
imbalanced-learn==0.11.0
dill==0.3.7
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0
torch==2.1.0
torch_geometric==2.4.0
jupyter==1.0.0
ipykernel==6.25.2
python-dotenv==1.0.0
gunicorn==21.2.0
```

---

### `setup.py`

```python
from setuptools import find_packages, setup

setup(
    name='amazon-supply-chain-intelligence',
    version='1.0.0',
    author='Your Name',
    author_email='your@email.com',
    packages=find_packages(),
    install_requires=[...]  # read from requirements.txt
)
```

---

### `train_model.py`

```python
# Entry point for training:
# import argparse
# Parse --data_path argument
# Call TrainPipeline().run_pipeline(data_path)
# Print metrics summary table
# Save metrics to artifacts/metrics.json
```

---

## 🧬 Enhancements I've Added Beyond Requirements

### 1. Graph Neural Network (GNN) for Relational Learning
Orders in supply chains are **not independent** — they share routes, warehouses, customers, and carriers. A GNN captures these relationships by modeling orders as graph nodes and connections (shared region, shipping mode, customer) as edges. This gives the model relational context that tabular classifiers cannot access.

### 2. Risk-Aware Feature Engineering
Beyond basic features, we add:
- `market_risk_score` and `region_delay_rate` — historical delay rates per geography, encoding systemic risk
- `customer_order_frequency` — high-frequency customers may get priority handling
- `shipping_delay_gap` — the core signal, explicitly computed
- Time-based features — weekends and month-end often have higher delays

### 3. Dual Imbalance Strategy
We apply SMOTE (Synthetic Minority Oversampling) but also benchmark ADASYN (Adaptive Synthetic) in the notebook, letting you choose based on validation performance.

### 4. Streaming Training Logs
The `/train` endpoint uses Flask's `Response` with `stream_with_context` to stream real-time training logs to the browser — no page refresh needed.

### 5. Interactive Metrics Dashboard
The landing page (`index.html`) fetches live metrics from `/metrics` and renders a feature importance chart with Chart.js, so the dashboard updates every time you retrain.

### 6. Confidence + Probability Visualization
The prediction page shows not just the class but the model's confidence (as an animated ring) and all three class probabilities (as a bar chart) — critical for a decision-support system.

### 7. Health Check Endpoint
`/health` enables deployment monitoring and integration with container orchestration tools like Docker/Kubernetes.

---

## ⚠️ Important Implementation Notes

1. **Python 3.10 compatibility**: Ensure all type hints use `Union[X, Y]` instead of `X | Y` (the `|` union syntax requires Python 3.10+, but torch_geometric compatibility should be verified).

2. **PyTorch Geometric installation**: For Python 3.10, use:
   ```bash
   pip install torch==2.1.0
   pip install torch_geometric
   pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
   ```

3. **GNN fallback**: If GPU is not available, GNN trains on CPU — it will be slower but functional. Add a config flag `USE_GNN=True/False` to skip GNN if resources are constrained.

4. **CatBoost artifacts**: CatBoost creates a `catboost_info/` directory during training — this is expected and should be in `.gitignore`.

5. **Data leakage prevention**: `market_risk_score` and `region_delay_rate` must be computed from TRAINING data only and then mapped onto test data — never fit on combined data.

---

## 🎯 Success Criteria

| Metric | Target |
|--------|--------|
| Weighted F1-Score | ≥ 0.80 |
| Macro F1-Score | ≥ 0.72 |
| Cohen's Kappa | ≥ 0.70 |
| All 3 classes recall | ≥ 0.65 |
| GNN vs baseline improvement | ≥ +2% F1 |

---

*Prompt version: 1.0 | Python 3.10 | Flask + GNN + Multi-class Classification*
