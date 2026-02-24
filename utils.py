import os
import sys
import json
import dill
import numpy as np
from typing import Any, Dict, Optional

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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


def apply_probability_constraints(probabilities: np.ndarray, min_risk_prob: float = 0.02, max_on_time_delayed_prob: float = 0.98) -> np.ndarray:
    """
    Apply probability constraints to model predictions.
    Ensures "At Risk" (class 1) is at least min_risk_prob (2%)
    and "On-Time" (class 0) or "Delayed" (class 2) don't exceed max_on_time_delayed_prob (98%).
    
    Handles variable number of classes (2 or 3).
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) with probabilities for each class
        min_risk_prob: Minimum probability for "At Risk" class (default: 0.02)
        max_on_time_delayed_prob: Maximum probability for "On-Time" or "Delayed" classes (default: 0.98)
    
    Returns:
        Adjusted probabilities with constraints applied
    """
    adjusted_probs = probabilities.copy()
    n_classes = probabilities.shape[1]
    
    # If only 2 classes, return as-is (no constraints needed)
    if n_classes < 3:
        return adjusted_probs
    
    for i in range(len(adjusted_probs)):
        # Ensure At Risk (class 1) is at least min_risk_prob
        if adjusted_probs[i, 1] < min_risk_prob:
            # Need to reduce On-Time (class 0) and/or Delayed (class 2)
            deficit = min_risk_prob - adjusted_probs[i, 1]
            other_prob = adjusted_probs[i, 0] + adjusted_probs[i, 2]
            
            if other_prob > 0:
                # Reduce both proportionally
                reduction_ratio = deficit / other_prob
                adjusted_probs[i, 0] -= adjusted_probs[i, 0] * reduction_ratio
                adjusted_probs[i, 2] -= adjusted_probs[i, 2] * reduction_ratio
            
            adjusted_probs[i, 1] = min_risk_prob
        
        # Ensure On-Time (class 0) and Delayed (class 2) don't exceed max_on_time_delayed_prob
        if adjusted_probs[i, 0] > max_on_time_delayed_prob:
            excess = adjusted_probs[i, 0] - max_on_time_delayed_prob
            adjusted_probs[i, 0] = max_on_time_delayed_prob
            adjusted_probs[i, 1] += excess
        
        if adjusted_probs[i, 2] > max_on_time_delayed_prob:
            excess = adjusted_probs[i, 2] - max_on_time_delayed_prob
            adjusted_probs[i, 2] = max_on_time_delayed_prob
            adjusted_probs[i, 1] += excess
        
        # Normalize to ensure sum = 1.0
        total = adjusted_probs[i].sum()
        if total > 0:
            adjusted_probs[i] /= total
    
    return adjusted_probs


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict,
    params: Dict,
    use_random_search: bool = False,
    n_iter: int = 20,
) -> Dict:
    """
    Train and evaluate multiple models using GridSearchCV or RandomizedSearchCV on training data only.
    Hyperparameter tuning is performed without cross-validation for faster training.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        models: Dictionary of {model_name: model_instance}
        params: Dictionary of {model_name: param_grid}
        use_random_search: Use RandomizedSearchCV instead of GridSearchCV
        n_iter: Number of parameter settings sampled (for RandomizedSearchCV)
    
    Returns:
        report: Dictionary with model performance and best parameters
    """
    try:
        report = {}

        for model_name, model in models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training model: {model_name}")
            logger.info(f"{'='*50}")
            param_grid = params.get(model_name, {})

            if param_grid:
                if use_random_search:
                    # Use RandomizedSearchCV for larger parameter spaces
                    search = RandomizedSearchCV(
                        model,
                        param_grid,
                        n_iter=n_iter,
                        scoring="f1_weighted",
                        n_jobs=-1,
                        verbose=1,
                        random_state=42,
                    )
                else:
                    # Use GridSearchCV for smaller parameter spaces
                    search = GridSearchCV(
                        model,
                        param_grid,
                        scoring="f1_weighted",
                        n_jobs=-1,
                        verbose=1,
                    )
                
                logger.info(f"Starting hyperparameter search...")
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
                
                logger.info(f"✓ Best Parameters: {best_params}")
            else:
                # Train without hyperparameter tuning (e.g., GNN)
                best_model = model
                best_model.fit(X_train, y_train)
                best_params = {}
                logger.info(f"Model {model_name} trained without hyperparameter tuning")

            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            
            report[model_name] = {
                "model": best_model,
                "weighted_f1": weighted_f1,
                "macro_f1": macro_f1,
                "best_params": best_params,
            }
            
            logger.info(f"Test Weighted F1: {weighted_f1:.4f}")
            logger.info(f"Test Macro F1: {macro_f1:.4f}")

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
