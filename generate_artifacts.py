"""
Quick script to generate missing preprocessor and supporting artifacts
"""
import os
import sys
import dill
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from logger import logger
from src.components.data_transformation import DataTransformation

def main():
    """Generate all missing artifacts from existing train data"""
    
    # Paths
    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"
    artifacts_dir = "artifacts"
    
    os.makedirs(artifacts_dir, exist_ok=True)
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found at {train_path}")
        sys.exit(1)
    
    # Create preprocessor from training data
    logger.info("Creating preprocessor from training data...")
    
    try:
        transformer_obj = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformer_obj.initiate_data_transformation(
            train_path, test_path
        )
        logger.info(f"✓ Preprocessor saved at: {preprocessor_path}")
        
        # Load the preprocessor to verify
        with open(preprocessor_path, 'rb') as f:
            preprocessor = dill.load(f)
        print(f"✓ Preprocessor successfully created: {preprocessor_path}")
        
        # Also create supporting maps for predictions
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Market risk map (average delay rate by market)
        market_risk_map = train_df.groupby('Market').apply(
            lambda x: (x.get('Delivery Status', pd.Series([0])) == 'Late').mean()
        ).to_dict()
        market_risk_path = os.path.join(artifacts_dir, "market_risk_map.pkl")
        with open(market_risk_path, 'wb') as f:
            dill.dump(market_risk_map, f)
        logger.info(f"✓ Market risk map saved: {market_risk_path}")
        
        # Region delay rate
        region_delay_map = train_df.groupby('Order Region').apply(
            lambda x: (x.get('Delivery Status', pd.Series([0])) == 'Late').mean()
        ).to_dict()
        region_delay_path = os.path.join(artifacts_dir, "region_delay_map.pkl")
        with open(region_delay_path, 'wb') as f:
            dill.dump(region_delay_map, f)
        logger.info(f"✓ Region delay map saved: {region_delay_path}")
        
        # Customer frequency (default 5 in predictions, this is just for reference)
        customer_freq_map = train_df.groupby('Customer Id').size().to_dict()
        customer_freq_path = os.path.join(artifacts_dir, "customer_freq_map.pkl")
        with open(customer_freq_path, 'wb') as f:
            dill.dump(customer_freq_map, f)
        logger.info(f"✓ Customer frequency map saved: {customer_freq_path}")
        
        print("\n" + "="*60)
        print("  ARTIFACTS GENERATION COMPLETE")
        print("="*60)
        print("✓ preprocessor.pkl")
        print("✓ market_risk_map.pkl")
        print("✓ region_delay_map.pkl")  
        print("✓ customer_freq_map.pkl")
        print("✓ model.pkl (already exists)")
        print("="*60)
        print("\nYou can now run: python application.py\n")
        
    except Exception as e:
        logger.error(f"Error creating preprocessor: {e}")
        print(f"❌ Failed to create preprocessor: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
