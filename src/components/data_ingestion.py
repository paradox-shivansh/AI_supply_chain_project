import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from logger import logger
from exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    """
    Handles data loading, cleaning, target engineering,
    and train/test splitting.
    """

    COLUMNS_TO_DROP = [
        "Customer Email",
        "Customer Password",
        "Customer Fname",
        "Customer Lname",
        "Customer Street",
        "Product Description",
        "Product Image",
        "Order Customer Id",
        "Order Item Cardprod Id",
        "Order Item Id",
        "Customer Zipcode",
        "Order Zipcode",
    ]

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def _engineer_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer the Delay_Risk_Level target variable with 3 classes:
          0 = On-Time
          1 = At Risk
          2 = Delayed
        """
        logger.info("Engineering target variable: Delay_Risk_Level")

        real_col = "Days for shipping (real)"
        sched_col = "Days for shipment (scheduled)"
        late_risk_col = "Late_delivery_risk"
        delivery_status_col = "Delivery Status"

        delay_gap = df[real_col] - df[sched_col]

        conditions_delayed = (
            (delay_gap > 2) |
            (df[delivery_status_col].isin(["Late delivery", "Shipping canceled"]))
        )

        conditions_on_time = (
            (delay_gap <= 0) & (df[late_risk_col] == 0)
        )

        conditions_at_risk = (
            ((delay_gap >= 1) & (delay_gap <= 2)) |
            (
                (df[late_risk_col] == 1) &
                (df[delivery_status_col] == "Shipping on time")
            )
        )

        # Assign: default to Delayed, then override with better matches
        df["Delay_Risk_Level"] = 2  # Default: Delayed

        # Apply in order of specificity
        df.loc[conditions_at_risk, "Delay_Risk_Level"] = 1
        df.loc[conditions_on_time, "Delay_Risk_Level"] = 0
        # Delayed overrides at_risk if truly delayed
        df.loc[conditions_delayed, "Delay_Risk_Level"] = 2

        class_dist = df["Delay_Risk_Level"].value_counts()
        logger.info(f"Target class distribution:\n{class_dist}")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop irrelevant columns and handle obvious data quality issues."""
        logger.info(f"Original shape: {df.shape}")

        # Drop irrelevant columns (only those that exist)
        cols_to_drop = [c for c in self.COLUMNS_TO_DROP if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        logger.info(f"Dropped {len(cols_to_drop)} irrelevant columns")

        # Drop duplicates
        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {before - len(df)} duplicate rows")

        # Drop rows where critical columns are null
        critical_cols = ["Days for shipping (real)", "Days for shipment (scheduled)"]
        critical_cols = [c for c in critical_cols if c in df.columns]
        df = df.dropna(subset=critical_cols)

        logger.info(f"Cleaned shape: {df.shape}")
        return df

    def initiate_data_ingestion(self, data_path: str):
        """
        Main method to ingest data:
        1. Load CSV
        2. Clean data
        3. Engineer target variable
        4. Save raw data
        5. Split into train/test
        6. Return (train_path, test_path)
        """
        logger.info("=" * 60)
        logger.info("DATA INGESTION STARTED")
        logger.info("=" * 60)

        try:
            # Load data
            logger.info(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path, encoding="latin-1")
            logger.info(f"Data loaded successfully. Shape: {df.shape}")

            # Clean data
            df = self._clean_data(df)

            # Engineer target
            df = self._engineer_target(df)

            # Save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logger.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

            # Stratified train/test split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["Delay_Risk_Level"],
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logger.info(f"Train set shape: {train_set.shape}")
            logger.info(f"Test set shape: {test_set.shape}")
            logger.info(f"Train data saved at: {self.ingestion_config.train_data_path}")
            logger.info(f"Test data saved at: {self.ingestion_config.test_data_path}")
            logger.info("DATA INGESTION COMPLETED")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion(
        r"D:\AI_supply_chain_project\DataCoSupplyChainDataset.csv\DataCoSupplyChainDataset.csv"
    )
    print(f"Train: {train_path}, Test: {test_path}")
