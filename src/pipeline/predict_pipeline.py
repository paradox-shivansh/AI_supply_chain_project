import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from logger import logger
from exception import CustomException
from utils import load_json, load_object, apply_probability_constraints


# Label mapping
LABEL_MAP = {0: "On-Time", 1: "At Risk", 2: "Delayed"}
LABEL_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}
LABEL_ICONS = {0: "✅", 1: "⚠️", 2: "🚨"}

RECOMMENDATIONS = {
    0: "Order is on track for timely delivery. No action needed.",
    1: (
        "This order is at risk of delay. Consider upgrading shipping mode "
        "or proactively notifying the customer about potential delays."
    ),
    2: (
        "High likelihood of delivery delay detected. Recommend immediate "
        "carrier escalation, customer notification, and consider expedited "
        "replacement shipment if SLA breach is imminent."
    ),
}

SHIPPING_MODE_MAP = {
    "Same Day": 0,
    "First Class": 1,
    "Second Class": 2,
    "Standard Class": 3,
}


class PredictPipeline:
    """
    Handles end-to-end prediction for a single order.
    Loads the trained preprocessor and model, applies feature engineering,
    and returns structured prediction results.
    """

    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.market_risk_path = os.path.join("artifacts", "market_risk_map.pkl")
        self.region_delay_path = os.path.join("artifacts", "region_delay_map.pkl")
        self.customer_freq_path = os.path.join("artifacts", "customer_freq_map.pkl")
        self.label_mapping_path = os.path.join("artifacts", "label_mapping.json")

        self._preprocessor = None
        self._model = None
        self._market_risk_map = {}
        self._region_delay_map = {}
        self._customer_freq_map = {}
        self._label_mapping = {0: 0, 1: 1, 2: 2}
        self._inverse_label_mapping = {0: 0, 1: 1, 2: 2}

    def _load_artifacts(self):
        """Lazy-load all artifacts."""
        if self._model is None:
            logger.info("Loading model artifacts...")
            self._preprocessor = load_object(self.preprocessor_path)
            self._model = load_object(self.model_path)

            if os.path.exists(self.market_risk_path):
                self._market_risk_map = load_object(self.market_risk_path)
            if os.path.exists(self.region_delay_path):
                self._region_delay_map = load_object(self.region_delay_path)
            if os.path.exists(self.customer_freq_path):
                self._customer_freq_map = load_object(self.customer_freq_path)
            if os.path.exists(self.label_mapping_path):
                mapping_payload = load_json(self.label_mapping_path)
                self._label_mapping = {
                    int(k): int(v) for k, v in mapping_payload.get("label_mapping", {}).items()
                } or self._label_mapping
                self._inverse_label_mapping = {
                    int(k): int(v) for k, v in mapping_payload.get("inverse_label_mapping", {}).items()
                } or self._inverse_label_mapping

    def _decode_label(self, encoded_label: int) -> int:
        return self._inverse_label_mapping.get(int(encoded_label), int(encoded_label))

    def _encoded_index(self, original_label: int) -> int:
        return self._label_mapping.get(int(original_label), None)

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering as training."""
        df = df.copy()

        # Shipping delay gap
        df["shipping_delay_gap"] = (
            df.get("Days for shipping (real)", pd.Series([0])).values -
            df.get("Days for shipment (scheduled)", pd.Series([0])).values
        )

        # Date-based features — use defaults for prediction
        df["is_weekend_order"] = 0
        df["order_month"] = 6
        df["order_day_of_week"] = 2

        # Discount impact
        disc = df.get("Order Item Discount", pd.Series([0])).fillna(0)
        price = df.get("Order Item Product Price", pd.Series([1])).fillna(1)
        df["discount_impact"] = disc / (price + 1)

        # Profit margin
        profit = df.get("Order Profit Per Order", pd.Series([0])).fillna(0)
        sales = df.get("Sales", pd.Series([1])).fillna(1)
        df["profit_margin"] = profit / (sales + 1)

        # Shipping mode encoded
        shipping_mode = df.get("Shipping Mode", pd.Series(["Standard Class"])).fillna("Standard Class")
        df["shipping_mode_encoded"] = shipping_mode.map(SHIPPING_MODE_MAP).fillna(3)

        # Market risk score
        market = df.get("Market", pd.Series(["USCA"])).fillna("USCA")
        df["market_risk_score"] = market.map(self._market_risk_map).fillna(0.5)

        # Region delay rate
        region = df.get("Order Region", pd.Series(["Unknown"])).fillna("Unknown")
        df["region_delay_rate"] = region.map(self._region_delay_map).fillna(0.5)

        # Customer order frequency
        df["customer_order_frequency"] = 5  # Default for new customers

        return df

    def predict(self, features: Dict[str, Any]) -> Dict:
        """
        Make a prediction for a single order.

        Args:
            features: Dict of feature name → value

        Returns:
            Dict with prediction, confidence, probabilities, recommendation
        """
        try:
            self._load_artifacts()

            # Convert to DataFrame
            df = pd.DataFrame([features])

            # Apply feature engineering
            df = self._apply_feature_engineering(df)

            # Transform with preprocessor
            X_transformed = self._preprocessor.transform(df)

            # Predict class
            encoded_pred = int(self._model.predict(X_transformed)[0])
            decoded_pred = self._decode_label(encoded_pred)
            pred_label = LABEL_MAP.get(decoded_pred, f"Class {decoded_pred}")

            # Predict probabilities
            num_encoded_classes = len(self._inverse_label_mapping)
            if hasattr(self._model, "predict_proba"):
                encoded_proba = self._model.predict_proba(X_transformed)[0]
                # Pad if we have fewer predictions than expected classes
                if len(encoded_proba) < num_encoded_classes:
                    pad_width = num_encoded_classes - len(encoded_proba)
                    encoded_proba = np.pad(encoded_proba, (0, pad_width), constant_values=0.0)
            else:
                encoded_proba = np.zeros(num_encoded_classes, dtype=float)
            
            # Apply probability constraints only if we have 3 classes (98% max for On-Time/Delayed, 2% min for At Risk)
            encoded_proba_reshaped = encoded_proba.reshape(1, -1)
            constrained_proba = apply_probability_constraints(encoded_proba_reshaped)[0]
            
            # Update prediction based on constrained probabilities
            encoded_pred = int(np.argmax(constrained_proba))
            decoded_pred = self._decode_label(encoded_pred)
            pred_label = LABEL_MAP.get(decoded_pred, f"Class {decoded_pred}")
            
            confidence = float(
                constrained_proba[encoded_pred] if encoded_pred < len(constrained_proba) else 0.0
            )

            probabilities = {}
            for cls_code, label_name in LABEL_MAP.items():
                encoded_idx = self._encoded_index(cls_code)
                if encoded_idx is not None and encoded_idx < len(constrained_proba):
                    prob = constrained_proba[encoded_idx]
                else:
                    prob = 0.0
                probabilities[label_name] = round(float(prob) * 100, 2)

            result = {
                "prediction": pred_label,
                "prediction_code": decoded_pred,
                "confidence": round(confidence * 100, 2),
                "probabilities": probabilities,
                "color": LABEL_COLORS.get(decoded_pred, "#6b7280"),
                "icon": LABEL_ICONS.get(decoded_pred, "ℹ️"),
                "recommendation": RECOMMENDATIONS.get(
                    decoded_pred,
                    "Prediction returned an out-of-distribution class. Inspect the model outputs.",
                ),
            }

            logger.info(
                f"Prediction: {pred_label} (confidence: {confidence:.2%})"
            )
            return result

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Captures form input and converts to a DataFrame row for prediction.
    Mirrors the feature set expected by the pipeline.
    """

    def __init__(
        self,
        shipping_mode: str,
        days_for_shipping_real: float,
        days_for_shipment_scheduled: float,
        order_item_quantity: int,
        order_item_discount_rate: float,
        order_item_product_price: float,
        sales_per_customer: float,
        market: str,
        order_region: str,
        customer_segment: str,
        department_name: str,
        category_name: str,
        order_status: str,
        order_type: str,
        order_item_discount: float = 0.0,
        sales: float = 0.0,
        order_item_total: float = 0.0,
        order_profit_per_order: float = 0.0,
        benefit_per_order: float = 0.0,
        late_delivery_risk: int = 0,
        order_item_profit_ratio: float = 0.0,
        product_price: float = 0.0,
        product_status: int = 0,
        latitude: float = 0.0,
        longitude: float = 0.0,
    ):
        self.shipping_mode = shipping_mode
        self.days_for_shipping_real = days_for_shipping_real
        self.days_for_shipment_scheduled = days_for_shipment_scheduled
        self.order_item_quantity = order_item_quantity
        self.order_item_discount_rate = order_item_discount_rate
        self.order_item_product_price = order_item_product_price
        self.sales_per_customer = sales_per_customer
        self.market = market
        self.order_region = order_region
        self.customer_segment = customer_segment
        self.department_name = department_name
        self.category_name = category_name
        self.order_status = order_status
        self.order_type = order_type
        self.order_item_discount = order_item_discount
        self.sales = sales
        self.order_item_total = order_item_total
        self.order_profit_per_order = order_profit_per_order
        self.benefit_per_order = benefit_per_order
        self.late_delivery_risk = late_delivery_risk
        self.order_item_profit_ratio = order_item_profit_ratio
        self.product_price = product_price
        self.product_status = product_status
        self.latitude = latitude
        self.longitude = longitude

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """Convert the input data into a single-row DataFrame."""
        data_dict = {
            "Type": [self.order_type],
            "Days for shipping (real)": [self.days_for_shipping_real],
            "Days for shipment (scheduled)": [self.days_for_shipment_scheduled],
            "Benefit per order": [self.benefit_per_order],
            "Sales per customer": [self.sales_per_customer],
            "Late_delivery_risk": [self.late_delivery_risk],
            "Category Name": [self.category_name],
            "Customer Segment": [self.customer_segment],
            "Department Name": [self.department_name],
            "Latitude": [self.latitude],
            "Longitude": [self.longitude],
            "Market": [self.market],
            "Order Region": [self.order_region],
            "Order Item Discount": [self.order_item_discount],
            "Order Item Discount Rate": [self.order_item_discount_rate],
            "Order Item Product Price": [self.order_item_product_price],
            "Order Item Profit Ratio": [self.order_item_profit_ratio],
            "Order Item Quantity": [self.order_item_quantity],
            "Sales": [self.sales],
            "Order Item Total": [self.order_item_total],
            "Order Profit Per Order": [self.order_profit_per_order],
            "Order Status": [self.order_status],
            "Product Price": [self.product_price],
            "Product Status": [self.product_status],
            "Shipping Mode": [self.shipping_mode],
        }
        return pd.DataFrame(data_dict)
