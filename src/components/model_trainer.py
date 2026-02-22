import os
import sys
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from logger import logger
from exception import CustomException
from utils import save_object, load_object, evaluate_models, save_json, get_feature_importance

# Optional imports with graceful fallback
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("CatBoost not available")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    try:
        from torch_geometric.nn import GCNConv, SAGEConv
        from torch_geometric.data import Data
        HAS_TORCH_GEOMETRIC = True
    except ImportError:
        HAS_TORCH_GEOMETRIC = False
        logger.warning("PyTorch Geometric not available — GNN will use fallback")
except ImportError:
    HAS_TORCH = False
    HAS_TORCH_GEOMETRIC = False
    logger.warning("PyTorch not available — GNN disabled")


# ─────────────────────────────────────────────────────────────────────────────
# GNN IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

class DeliveryGNN(nn.Module):
    """
    Graph Neural Network for delivery delay prediction.
    Uses GCNConv layers to aggregate neighborhood information.
    Each order is a node; edges connect orders with shared attributes.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_classes: int):
        super(DeliveryGNN, self).__init__()
        if HAS_TORCH_GEOMETRIC:
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        else:
            # Fallback: use linear layers (MLP)
            self.conv1 = nn.Linear(in_channels, hidden_channels)
            self.conv2 = nn.Linear(hidden_channels, out_channels)
        self.classifier = nn.Linear(out_channels, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index=None):
        if HAS_TORCH_GEOMETRIC and edge_index is not None:
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
        return self.classifier(x)


def build_graph(X: np.ndarray, batch_size: int = 5000):
    """
    Build a graph from feature matrix using KNN-based edge construction.
    Returns edge_index tensor for PyTorch Geometric.
    Limits to batch_size nodes for memory efficiency.
    """
    if not HAS_TORCH:
        return None

    n = min(len(X), batch_size)
    X_sub = X[:n]

    from sklearn.neighbors import NearestNeighbors
    k = min(5, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nbrs.fit(X_sub)
    indices = nbrs.kneighbors(X_sub, return_distance=False)

    # Build edge list
    sources, targets = [], []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self
            sources.append(i)
            targets.append(j)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return edge_index


class GNNWrapper:
    """
    Sklearn-compatible wrapper for the DeliveryGNN model.
    Implements fit(), predict(), predict_proba() interfaces.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_classes: int = 3,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 5000,
        patience: int = 10,
    ):
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.model = None
        self.device = torch.device("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
        self.training_losses = []
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not HAS_TORCH:
            logger.warning("PyTorch not available — GNN skipped")
            return self

        logger.info(f"Training GNN on {self.device} | Samples: {len(X)}")

        n = min(len(X), self.batch_size)
        X_sub = X[:n].astype(np.float32)
        y_sub = y[:n].astype(np.int64)

        x_tensor = torch.tensor(X_sub, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_sub, dtype=torch.long).to(self.device)

        edge_index = build_graph(X_sub, batch_size=n)
        if edge_index is not None:
            edge_index = edge_index.to(self.device)

        in_channels = X_sub.shape[1]
        self.model = DeliveryGNN(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_classes=self.num_classes,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        patience_counter = 0
        self.training_losses = []

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out = self.model(x_tensor, edge_index)
            loss = criterion(out, y_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            self.training_losses.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                # Save best weights
                self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(f"GNN Epoch [{epoch + 1}/{self.epochs}] Loss: {loss.item():.4f}")

        # Load best weights
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

        logger.info(f"GNN training complete. Best loss: {best_loss:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X), dtype=int)

        self.model.eval()
        with torch.no_grad():
            n = min(len(X), self.batch_size)
            X_sub = X[:n].astype(np.float32)
            x_tensor = torch.tensor(X_sub, dtype=torch.float32).to(self.device)
            edge_index = build_graph(X_sub, batch_size=n)
            if edge_index is not None:
                edge_index = edge_index.to(self.device)
            out = self.model(x_tensor, edge_index)
            preds = out.argmax(dim=1).cpu().numpy()

        # Handle remaining samples
        if len(X) > n:
            remaining = np.zeros(len(X) - n, dtype=int)
            preds = np.concatenate([preds, remaining])

        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            proba = np.zeros((len(X), self.num_classes))
            proba[:, 0] = 1.0
            return proba

        self.model.eval()
        with torch.no_grad():
            n = min(len(X), self.batch_size)
            X_sub = X[:n].astype(np.float32)
            x_tensor = torch.tensor(X_sub, dtype=torch.float32).to(self.device)
            edge_index = build_graph(X_sub, batch_size=n)
            if edge_index is not None:
                edge_index = edge_index.to(self.device)
            out = self.model(x_tensor, edge_index)
            proba = F.softmax(out, dim=1).cpu().numpy()

        if len(X) > n:
            pad = np.zeros((len(X) - n, self.num_classes))
            pad[:, 0] = 1.0
            proba = np.concatenate([proba, pad], axis=0)

        return proba


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    metrics_file_path: str = os.path.join("artifacts", "metrics.json")
    best_model_name_path: str = os.path.join("artifacts", "best_model_name.txt")


class ModelTrainer:
    """
    Trains, evaluates, and saves the best classification model.
    Compares CatBoost, XGBoost, RandomForest, LightGBM, and GNN.
    """

    CLASS_NAMES = {0: "On-Time", 1: "At Risk", 2: "Delayed"}

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def _build_models(self) -> Dict:
        models = {}

        if HAS_CATBOOST:
            models["CatBoost"] = CatBoostClassifier(
                verbose=0, random_state=42, eval_metric="TotalF1"
            )

        if HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )

        models["Random Forest"] = RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced"
        )

        if HAS_LIGHTGBM:
            models["LightGBM"] = LGBMClassifier(
                random_state=42, n_jobs=-1, verbose=-1, class_weight="balanced"
            )

        if HAS_TORCH:
            models["GNN"] = GNNWrapper(
                hidden_channels=128,
                out_channels=64,
                num_classes=3,
                epochs=60,
                lr=0.001,
                patience=10,
            )

        return models

    def _build_params(self) -> Dict:
        params = {}

        if HAS_CATBOOST:
            params["CatBoost"] = {
                "iterations": [200, 300],
                "learning_rate": [0.05, 0.1],
                "depth": [6, 8],
            }

        if HAS_XGBOOST:
            params["XGBoost"] = {
                "n_estimators": [200, 300],
                "max_depth": [6, 8],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
            }

        params["Random Forest"] = {
            "n_estimators": [100, 200],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5],
        }

        if HAS_LIGHTGBM:
            params["LightGBM"] = {
                "n_estimators": [200, 300],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [31, 63],
            }

        # GNN has no sklearn-compatible grid search
        params["GNN"] = {}

        return params

    def _compute_full_metrics(
        self, model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str
    ) -> Dict:
        """Compute comprehensive evaluation metrics."""
        y_pred = model.predict(X_test)

        metrics = {
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=["On-Time", "At Risk", "Delayed"],
                output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        # ROC-AUC (needs probability scores)
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
                metrics["roc_auc_weighted"] = float(roc_auc)
        except Exception as e:
            logger.warning(f"ROC-AUC computation failed for {model_name}: {e}")
            metrics["roc_auc_weighted"] = None

        logger.info(f"\n{'='*40}")
        logger.info(f"MODEL: {model_name}")
        logger.info(f"  Accuracy:       {metrics['accuracy']:.4f}")
        logger.info(f"  Weighted F1:    {metrics['weighted_f1']:.4f}")
        logger.info(f"  Macro F1:       {metrics['macro_f1']:.4f}")
        logger.info(f"  Cohen's Kappa:  {metrics['cohen_kappa']:.4f}")
        if metrics.get("roc_auc_weighted"):
            logger.info(f"  ROC-AUC:        {metrics['roc_auc_weighted']:.4f}")

        return metrics

    def initiate_model_trainer(
        self, train_array: np.ndarray, test_array: np.ndarray
    ) -> Dict:
        """
        Full model training pipeline:
        1. Split features and target
        2. Train all models
        3. Select best by weighted F1
        4. Compute full metrics
        5. Save best model
        6. Return metrics report
        """
        logger.info("=" * 60)
        logger.info("MODEL TRAINING STARTED")
        logger.info("=" * 60)

        try:
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1].astype(int)
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1].astype(int)

            logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

            models = self._build_models()
            params = self._build_params()

            if not models:
                raise ValueError("No models available for training!")

            logger.info(f"Models to train: {list(models.keys())}")

            # ── Train all models ───────────────────────────────────────────
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            # ── Find best model by weighted F1 ─────────────────────────────
            best_model_name = max(
                model_report, key=lambda k: model_report[k]["weighted_f1"]
            )
            best_model = model_report[best_model_name]["model"]
            best_f1 = model_report[best_model_name]["weighted_f1"]

            logger.info(f"\n{'='*60}")
            logger.info(f"BEST MODEL: {best_model_name}")
            logger.info(f"Weighted F1 Score: {best_f1:.4f}")

            if best_f1 < 0.60:
                raise CustomException(
                    Exception(
                        f"Best model ({best_model_name}) F1={best_f1:.4f} is below threshold 0.60"
                    ),
                    sys,
                )

            # ── Compute detailed metrics for all models ────────────────────
            all_metrics = {}
            for name, info in model_report.items():
                all_metrics[name] = self._compute_full_metrics(
                    info["model"], X_test, y_test, name
                )
                all_metrics[name]["best_params"] = info.get("best_params", {})

            # ── Feature importance ─────────────────────────────────────────
            try:
                feature_names_path = os.path.join("artifacts", "feature_names.pkl")
                if os.path.exists(feature_names_path):
                    from utils import load_object
                    feature_names = load_object(feature_names_path)
                    importance = get_feature_importance(best_model, feature_names)
                    if importance:
                        save_json(
                            os.path.join("artifacts", "feature_importance.json"),
                            importance,
                        )
                        logger.info("Feature importance saved")
            except Exception as fe:
                logger.warning(f"Feature importance skipped: {fe}")

            # ── Save best model ────────────────────────────────────────────
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logger.info(
                f"Best model saved at: {self.model_trainer_config.trained_model_file_path}"
            )

            # Save best model name
            os.makedirs(os.path.dirname(self.model_trainer_config.best_model_name_path), exist_ok=True)
            with open(self.model_trainer_config.best_model_name_path, "w") as f:
                f.write(best_model_name)

            # ── Prepare and save final metrics report ──────────────────────
            final_report = {
                "best_model": best_model_name,
                "best_weighted_f1": best_f1,
                "model_comparison": {
                    name: {
                        "accuracy": info["accuracy"],
                        "weighted_f1": info["weighted_f1"],
                        "macro_f1": info["macro_f1"],
                        "cohen_kappa": info["cohen_kappa"],
                        "roc_auc_weighted": info.get("roc_auc_weighted"),
                    }
                    for name, info in all_metrics.items()
                },
                "best_model_full_report": all_metrics[best_model_name],
            }

            save_json(self.model_trainer_config.metrics_file_path, final_report)
            logger.info(f"Metrics saved at: {self.model_trainer_config.metrics_file_path}")
            logger.info("MODEL TRAINING COMPLETED")

            return final_report

        except Exception as e:
            raise CustomException(e, sys)
