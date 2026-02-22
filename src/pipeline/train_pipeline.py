import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from logger import logger
from exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    """
    Orchestrates the full training pipeline:
    DataIngestion → DataTransformation → ModelTrainer
    """

    def run_pipeline(self, data_path: str) -> dict:
        """
        Execute the complete ML training pipeline.

        Args:
            data_path: Path to the raw CSV dataset

        Returns:
            Dictionary containing metrics and best model info
        """
        logger.info("=" * 70)
        logger.info("AMAZON SUPPLY CHAIN INTELLIGENCE — TRAINING PIPELINE START")
        logger.info("=" * 70)

        try:
            # ── Step 1: Data Ingestion ─────────────────────────────────────
            logger.info("\n[STEP 1/3] DATA INGESTION")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(
                data_path=data_path
            )
            logger.info(f"✓ Ingestion complete. Train: {train_data_path}, Test: {test_data_path}")

            # ── Step 2: Data Transformation ────────────────────────────────
            logger.info("\n[STEP 2/3] DATA TRANSFORMATION")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = (
                data_transformation.initiate_data_transformation(
                    train_path=train_data_path,
                    test_path=test_data_path,
                )
            )
            logger.info(f"✓ Transformation complete. Preprocessor: {preprocessor_path}")

            # ── Step 3: Model Training ─────────────────────────────────────
            logger.info("\n[STEP 3/3] MODEL TRAINING")
            model_trainer = ModelTrainer()
            metrics_report = model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr,
            )
            logger.info(f"✓ Training complete. Best model: {metrics_report['best_model']}")
            logger.info(f"  Best Weighted F1: {metrics_report['best_weighted_f1']:.4f}")

            logger.info("=" * 70)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)

            return metrics_report

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise CustomException(e, sys)
