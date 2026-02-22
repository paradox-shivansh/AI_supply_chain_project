# 🛠️ Setup Guide — Amazon Supply Chain Intelligence

> Complete installation, configuration, and run instructions for Python 3.10

---

## 📋 Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | **3.10** (exact) |
| pip | ≥ 22.0 |
| Git | Any recent version |
| RAM | ≥ 8 GB (16 GB recommended for GNN) |
| Disk | ≥ 3 GB free |
| GPU | Optional (CUDA 11.8+ for faster GNN training) |

---

## ⚡ Quick Start (5 Minutes)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/supply-chain-intelligence.git
cd supply-chain-intelligence

# 2. Create virtual environment with Python 3.10
python3.10 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate          # Linux / macOS
# OR
venv\Scripts\activate             # Windows

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install core requirements
pip install -r requirements.txt

# 6. Install PyTorch (CPU version)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# 7. Install PyTorch Geometric (GNN support)
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# 8. Place dataset
mkdir -p data
# Download and place your CSV at: data/supply_chain_data.csv

# 9. Train the model
python train_model.py --data_path data/supply_chain_data.csv

# 10. Start the web app
python application.py
```

Visit: **http://localhost:5000**

---

## 🗄️ Dataset

**Source**: DataCo Smart Supply Chain Dataset  
**Kaggle**: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

### Download Instructions
1. Create a Kaggle account at kaggle.com
2. Go to the dataset page above
3. Click "Download" to get `DataCoSupplyChainDataset.csv`
4. Rename to `supply_chain_data.csv` and place in `data/` folder:

```
supply_chain_intelligence/
└── data/
    └── supply_chain_data.csv   ← Place here
```

The dataset contains ~180,000 order records with 53 features covering order details,
customer information, product data, shipping modes, and delivery outcomes.

---

## 🐍 Python 3.10 Installation

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

### macOS (Homebrew)
```bash
brew install python@3.10
echo 'export PATH="/opt/homebrew/opt/python@3.10/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Windows
Download from: https://www.python.org/downloads/release/python-31011/
During installation, check "Add Python to PATH"

---

## 🔥 GPU Support (Optional, Recommended for GNN)

For NVIDIA GPU with CUDA 11.8:
```bash
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

For CUDA 12.1:
```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU detection:
```python
import torch
print(torch.cuda.is_available())  # Should print: True
```

---

## 🔧 Training Options

### Basic Training
```bash
python train_model.py --data_path data/supply_chain_data.csv
```

### Disable GNN (faster, less RAM)
Open `src/components/model_trainer.py` and set at the top:
```python
HAS_TORCH = False  # Force-disable GNN
```

### Skip GridSearchCV (faster prototype)
In `utils.py`, set `cv_folds=2` or reduce parameter grids in `model_trainer.py`.

---

## 🌐 Running the Web Application

```bash
# Development (auto-reload)
python application.py

# Production (Gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 application:app

# With custom port
PORT=8080 python application.py
```

### Environment Variables

Create a `.env` file in the project root:

```env
PORT=5000
DEBUG=True
MODEL_PATH=artifacts/model.pkl
PREPROCESSOR_PATH=artifacts/preprocessor.pkl
```

---

## 📓 Running Notebooks

```bash
# Install Jupyter (already in requirements.txt)
pip install jupyter

# Start Jupyter
jupyter notebook notebooks/

# Or with JupyterLab
pip install jupyterlab
jupyter lab notebooks/
```

Open:
- `01_EDA.ipynb` — Full exploratory data analysis
- `02_Model_Training.ipynb` — Model training experiments

**Before running notebooks**, ensure you have trained the model at least once so `artifacts/` exists.

---

## 📁 Project Installation (as package)

```bash
pip install -e .
```

This installs the package in editable mode and makes the `train-supply-chain` CLI command available:

```bash
train-supply-chain --data_path data/supply_chain_data.csv
```

---

## 🔍 Verify Installation

```bash
# Check Python version
python --version  # Should show Python 3.10.x

# Check key packages
python -c "import flask; print('Flask:', flask.__version__)"
python -c "import sklearn; print('Sklearn:', sklearn.__version__)"
python -c "import catboost; print('CatBoost: OK')"
python -c "import xgboost; print('XGBoost: OK')"
python -c "import lightgbm; print('LightGBM: OK')"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"

# Check health endpoint (after starting app)
curl http://localhost:5000/health
```

---

## ❗ Common Issues & Fixes

### Issue: `ModuleNotFoundError: No module named 'torch_geometric'`
```bash
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```
If still failing, the GNN will automatically fall back to an MLP architecture. The app still works.

### Issue: `UnicodeDecodeError` when loading dataset
The dataset uses latin-1 encoding. The code handles this automatically with `encoding='latin-1'`.

### Issue: `OSError: [Errno 98] Address already in use`
```bash
# Kill the process using port 5000
lsof -ti:5000 | xargs kill -9
python application.py
```

### Issue: Low model performance (F1 < 0.60)
- Ensure the dataset is complete (not a sample)
- Check target class distribution in the logs
- Try increasing `max_iter` in CatBoost or `n_estimators` in other models

### Issue: `SMOTE` errors with too few minority samples
```bash
# In data_transformation.py, try k_neighbors=3 instead of 5
smote = SMOTE(random_state=42, k_neighbors=3)
```

### Issue: Out of Memory during GNN training
Reduce batch size in `model_trainer.py`:
```python
"GNN": GNNWrapper(batch_size=2000, epochs=30)
```

---

## 📊 Expected Training Time

| Component | Time (CPU) | Time (GPU) |
|-----------|-----------|-----------|
| Data Ingestion | ~30s | Same |
| Data Transformation + SMOTE | ~2-3 min | Same |
| CatBoost | ~5-10 min | ~2 min |
| XGBoost | ~8-15 min | ~3 min |
| LightGBM | ~3-5 min | ~1 min |
| Random Forest | ~10-20 min | Same |
| GNN | ~15-30 min | ~5 min |
| **Total** | **~45-90 min** | **~15 min** |

Times assume full dataset (~180K rows) with GridSearchCV.

---

## 🚀 Deployment (Production)

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "application:app"]
```

### Fly.io / Railway / Render
Set environment variable: `PORT=8080`  
Entry command: `gunicorn -w 2 -b 0.0.0.0:$PORT application:app`

---

## 📞 Support

- Review `logs/` directory for detailed error traces
- Each run creates a timestamped log file: `logs/MM_DD_YYYY_HH_MM_SS.log`
- Health check: `GET /health` returns system status
