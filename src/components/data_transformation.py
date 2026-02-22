import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from logger import logger
from exception import CustomException
from utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    market_risk_map_path: str = os.path.join("artifacts", "market_risk_map.pkl")
    region_delay_map_path: str = os.path.join("artifacts", "region_delay_map.pkl")
    customer_freq_map_path: str = os.path.join("artifacts", "customer_freq_map.pkl")


class DataTransformation:
    """
    Advanced feature engineering and preprocessing pipeline.
    Handles temporal features, risk scores, and class imbalance via SMOTE.
    """

    SHIPPING_MODE_MAP = {
        "Same Day": 0,
        "First Class": 1,
        "Second Class": 2,
        "Standard Class": 3,
    }

    TARGET_COL = "Delay_Risk_Level"

    NUMERICAL_FEATURES = [
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Benefit per order",
        "Sales per customer",
        "Late_delivery_risk",
        "Order Item Discount",
        "Order Item Discount Rate",
        "Order Item Product Price",
        "Order Item Profit Ratio",
        "Order Item Quantity",
        "Sales",
        "Order Item Total",
        "Order Profit Per Order",
        "Product Price",
        "Latitude",
        "Longitude",
        # Engineered
        "shipping_delay_gap",
        "is_weekend_order",
        "order_month",
        "order_day_of_week",
        "discount_impact",
        "profit_margin",
        "shipping_mode_encoded",
        "market_risk_score",
        "region_delay_rate",
        "customer_order_frequency",
    ]

    CATEGORICAL_FEATURES = [
        "Type",
        "Customer Segment",
        "Market",
        "Order Region",
        "Order Status",
        "Category Name",
        "Department Name",
        "Product Status",
    ]

    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        self._market_risk_map = {}
        self._region_delay_map = {}
        self._customer_freq_map = {}

    def _engineer_features(
        self, df: pd.DataFrame, is_train: bool = True
    ) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        df = df.copy()

        # ── 1. Shipping delay gap ──────────────────────────────────────────
        df["shipping_delay_gap"] = (
            df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
        )

        # ── 2. Date-based features ─────────────────────────────────────────
        date_col = "order date (DateOrders)"
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df["is_weekend_order"] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
            df["order_month"] = df[date_col].dt.month.fillna(6).astype(int)
            df["order_day_of_week"] = df[date_col].dt.dayofweek.fillna(0).astype(int)
        else:
            df["is_weekend_order"] = 0
            df["order_month"] = 6
            df["order_day_of_week"] = 0

        # ── 3. Discount impact ─────────────────────────────────────────────
        if "Order Item Discount" in df.columns and "Order Item Product Price" in df.columns:
            df["discount_impact"] = (
                df["Order Item Discount"] / (df["Order Item Product Price"] + 1)
            )
        else:
            df["discount_impact"] = 0.0

        # ── 4. Profit margin ───────────────────────────────────────────────
        if "Order Profit Per Order" in df.columns and "Sales" in df.columns:
            df["profit_margin"] = (
                df["Order Profit Per Order"] / (df["Sales"] + 1)
            )
        else:
            df["profit_margin"] = 0.0

        # ── 5. Shipping mode encoded ───────────────────────────────────────
        if "Shipping Mode" in df.columns:
            df["shipping_mode_encoded"] = (
                df["Shipping Mode"].map(self.SHIPPING_MODE_MAP).fillna(3)
            )
        else:
            df["shipping_mode_encoded"] = 3

        # ── 6. Market risk score (train: compute; test: map) ───────────────
        if "Market" in df.columns:
            if is_train:
                self._market_risk_map = (
                    df.groupby("Market")[self.TARGET_COL]
                    .apply(lambda x: (x > 0).mean())
                    .to_dict()
                )
            df["market_risk_score"] = (
                df["Market"].map(self._market_risk_map).fillna(0.5)
            )
        else:
            df["market_risk_score"] = 0.5

        # ── 7. Region delay rate (train: compute; test: map) ───────────────
        if "Order Region" in df.columns:
            if is_train:
                self._region_delay_map = (
                    df.groupby("Order Region")[self.TARGET_COL]
                    .apply(lambda x: (x == 2).mean())
                    .to_dict()
                )
            df["region_delay_rate"] = (
                df["Order Region"].map(self._region_delay_map).fillna(0.5)
            )
        else:
            df["region_delay_rate"] = 0.5

        # ── 8. Customer order frequency (train: compute; test: map) ────────
        if "Customer Id" in df.columns:
            if is_train:
                self._customer_freq_map = (
                    df["Customer Id"].value_counts().to_dict()
                )
            df["customer_order_frequency"] = (
                df["Customer Id"].map(self._customer_freq_map).fillna(1)
            )
        else:
            df["customer_order_frequency"] = 1

        return df

    def get_data_transformer_object(self) -> ColumnTransformer:
        """Build and return the sklearn preprocessing pipeline."""
        try:
            # Filter to only use numerical features that exist
            num_features = [f for f in self.NUMERICAL_FEATURES]
            cat_features = [f for f in self.CATEGORICAL_FEATURES]

            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_pipeline, num_features),
                    ("cat", categorical_pipeline, cat_features),
                ],
                remainder="drop",
            )

            logger.info(f"Numerical features: {num_features}")
            logger.info(f"Categorical features: {cat_features}")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Main transformation pipeline:
        1. Load train/test data
        2. Engineer features
        3. Apply preprocessor (fit on train, transform both)
        4. Apply SMOTE to training data
        5. Save preprocessor
        6. Return (train_arr, test_arr, preprocessor_path)
        """
        logger.info("=" * 60)
        logger.info("DATA TRANSFORMATION STARTED")
        logger.info("=" * 60)

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # Feature engineering (is_train=True computes risk maps)
            logger.info("Engineering features for training set...")
            train_df = self._engineer_features(train_df, is_train=True)

            logger.info("Engineering features for test set...")
            test_df = self._engineer_features(test_df, is_train=False)

            # Save risk maps
            save_object(self.transformation_config.market_risk_map_path, self._market_risk_map)
            save_object(self.transformation_config.region_delay_map_path, self._region_delay_map)
            save_object(self.transformation_config.customer_freq_map_path, self._customer_freq_map)

            # Separate features and target
            target_col = self.TARGET_COL
            X_train = train_df.drop(columns=[target_col], errors="ignore")
            y_train = train_df[target_col].values

            X_test = test_df.drop(columns=[target_col], errors="ignore")
            y_test = test_df[target_col].values

            # Drop columns not needed for model
            drop_cols = [
                "order date (DateOrders)", "shipping date (DateOrders)",
                "Customer Id", "Order Id", "Product Card Id",
                "Product Category Id", "Category Id", "Department Id",
                "Product Name", "Customer City", "Customer Country",
                "Customer State", "Order City", "Order Country",
                "Order State", "Customer Zipcode", "Order Zipcode",
                "Shipping Mode",  # already encoded
            ]
            X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
            X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

            # Build and fit preprocessor
            logger.info("Fitting preprocessing pipeline...")
            preprocessor = self.get_data_transformer_object()

            # Filter feature lists to only columns that exist
            num_feats = [f for f in self.NUMERICAL_FEATURES if f in X_train.columns]
            cat_feats = [f for f in self.CATEGORICAL_FEATURES if f in X_train.columns]

            # Rebuild preprocessor with only existing columns
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer

            numerical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            preprocessor = ColumnTransformer([
                ("num", numerical_pipeline, num_feats),
                ("cat", categorical_pipeline, cat_feats),
            ], remainder="drop")

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logger.info(f"Transformed train shape: {X_train_transformed.shape}")
            logger.info(f"Transformed test shape: {X_test_transformed.shape}")

            # Apply SMOTE for class imbalance
            logger.info("Class distribution BEFORE SMOTE:")
            unique, counts = np.unique(y_train, return_counts=True)
            for cls, cnt in zip(unique, counts):
                label = {0: "On-Time", 1: "At Risk", 2: "Delayed"}[cls]
                logger.info(f"  Class {cls} ({label}): {cnt} samples")

            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train_transformed, y_train
            )

            logger.info("Class distribution AFTER SMOTE:")
            unique, counts = np.unique(y_train_resampled, return_counts=True)
            for cls, cnt in zip(unique, counts):
                label = {0: "On-Time", 1: "At Risk", 2: "Delayed"}[cls]
                logger.info(f"  Class {cls} ({label}): {cnt} samples")

            # Concatenate features and target
            train_arr = np.c_[X_train_resampled, y_train_resampled]
            test_arr = np.c_[X_test_transformed, y_test]

            # Save preprocessor
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )
            logger.info(
                f"Preprocessor saved at: {self.transformation_config.preprocessor_obj_file_path}"
            )

            # Store feature names for later use
            try:
                cat_feature_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(cat_feats).tolist()
                all_feature_names = num_feats + cat_feature_names
                save_object(os.path.join("artifacts", "feature_names.pkl"), all_feature_names)
                logger.info(f"Total features after encoding: {len(all_feature_names)}")
            except Exception:
                pass

            logger.info("DATA TRANSFORMATION COMPLETED")
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
