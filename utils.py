import os
import sys
import json
import dill
import numpy as np
from typing import Any, Dict, Optional

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from logger import logger
from exception import CustomException


def save_object(file_path: str, obj: Any) -> None:
    """Serialize and save an object to disk using dill."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logger.info(f"Object saved successfully at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    """Load and deserialize an object from disk using dill."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def save_json(file_path: str, data: Dict) -> None:
    """Save a dictionary as a JSON file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4, default=str)
        logger.info(f"JSON saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_json(file_path: str) -> Dict:
    """Load a JSON file and return as dictionary."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def get_class_weights(y: np.ndarray) -> Dict:
    """Compute class weights to handle imbalanced data."""
    try:
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
        logger.info(f"Class weights computed: {class_weight_dict}")
        return class_weight_dict
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict,
    params: Dict,
    cv_folds: int = 3,
) -> Dict:
    """
    Train and evaluate multiple models using GridSearchCV.
    Returns a dictionary of {model_name: weighted_f1_score}.
    """
    try:
        report = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for model_name, model in models.items():
            logger.info(f"Training model: {model_name}")
            param_grid = params.get(model_name, {})

            if param_grid:
                gs = GridSearchCV(
                    model,
                    param_grid,
                    cv=cv,
                    scoring="f1_weighted",
                    n_jobs=-1,
                    verbose=0,
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                logger.info(f"{model_name} best params: {gs.best_params_}")
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")
            report[model_name] = {
                "model": best_model,
                "weighted_f1": weighted_f1,
                "best_params": gs.best_params_ if param_grid else {},
            }
            logger.info(f"{model_name} → Weighted F1: {weighted_f1:.4f}")

        return report
    except Exception as e:
        raise CustomException(e, sys)


def get_feature_importance(model: Any, feature_names: list) -> Dict:
    """Extract feature importance from a trained model."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            return {}

        importance_dict = {
            name: float(imp)
            for name, imp in zip(feature_names, importances)
        }
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        return sorted_importance
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        return {}
