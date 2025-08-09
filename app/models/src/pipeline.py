import pandas as pd
import pickle
import os
import warnings
import sys

warnings.filterwarnings('ignore')

# Add root to path
sys.path.append('/home/antonios/Desktop/Practica_de_vara/data-science-internship')

try:
    from scripts.data_cleaning import (
        clean_missing_values, clean_duplicates, clean_statistical_outliers,
        clean_data_errors_and_logical, clean_infrequent_values, 
        clean_context_dependent_outliers, clean_dtypes, clean_target_leakage,
        clean_multicollinearity
    )
    from scripts.feature_engineering import (
        extract_temporal_features, extract_customer_behavior_features,
        extract_market_features, extract_deposit_features,
        extract_room_features, extract_country_features
    )
except ImportError as e:
    print(f"Warning: Could not import from scripts: {e}")


class HotelDataPipeline:
    """
    Simple production pipeline for hotel booking data preprocessing.
    Includes data cleaning and feature engineering steps.
    """
    
    def __init__(self, output_dir: str = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/app/models/trained_models'):
        self.output_dir = output_dir
        self.is_fitted = False
        os.makedirs(self.output_dir, exist_ok=True)
        
    def fit_transform(self, data: pd.DataFrame, target_column: str = 'is_canceled') -> pd.DataFrame:
        print("HOTEL DATA PIPELINE STARTING...")
        print(f"Input data shape: {data.shape}")
        
        self.target_column = target_column
        
        # Data Cleaning
        cleaned_data = self._run_cleaning_pipeline(data.copy())

        # Save cleaned data as CSV only
        cleaned_csv_path = os.path.join(self.output_dir, 'cleaned_data.csv')
        cleaned_data.to_csv(cleaned_csv_path, index=False)
        print(f"Cleaned data saved to: {cleaned_csv_path}")
        
        # Save cleaned data as PKL for feature engineering
        cleaned_pkl_path = os.path.join(self.output_dir, 'cleaned_data.pkl')
        cleaned_data.to_pickle(cleaned_pkl_path)

        # Feature Engineering (reads from PKL)
        final_data = self._run_feature_engineering(cleaned_pkl_path)

        # Save engineered data as CSV
        engineered_csv_path = os.path.join(self.output_dir, 'engineered_data.csv')
        final_data.to_csv(engineered_csv_path, index=False)
        print(f"Engineered data saved to: {engineered_csv_path}")
        
        self.is_fitted = True
        print("PIPELINE COMPLETED!")
        print(f"Final data shape: {final_data.shape}")
        
        return final_data
    
    def _run_cleaning_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Running data cleaning...")

        data = clean_missing_values(data)
        data = clean_duplicates(data)
        data = clean_statistical_outliers(data)
        data = clean_data_errors_and_logical(data)
        data = clean_infrequent_values(data)
        data = clean_context_dependent_outliers(data)
        data = clean_dtypes(data)

        if hasattr(self, 'target_column') and self.target_column in data.columns:
            data = clean_target_leakage(data)

        data = clean_multicollinearity(data)

        print("Data cleaning completed.")
        return data
    
    def _run_feature_engineering(self, cleaned_pkl_path: str) -> pd.DataFrame:
        print("Running feature engineering...")

        try:
            data = pd.read_pickle(cleaned_pkl_path)
            data = extract_temporal_features(data)
            data = extract_customer_behavior_features(data)
            data = extract_market_features(data)
            data = extract_deposit_features(data)
            data = extract_room_features(data)
            data = extract_country_features(data)

            print("Feature engineering completed.")
            return data

        except Exception as e:
            print(f"Feature engineering failed: {e}")
            print("Returning cleaned data without feature engineering.")
            return data


# Simple function to run the complete pipeline
def run_hotel_pipeline(data_path: str = None, 
                       output_dir: str = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/app/models/trained_models') -> pd.DataFrame:
    if data_path is None:
        data_path = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/raw/hotel_booking_cancellation_prediction.csv'
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    
    pipeline = HotelDataPipeline(output_dir=output_dir)
    processed_data = pipeline.fit_transform(data)
    
    return processed_data


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("HOTEL DATA PIPELINE - MAIN EXECUTION")
    print("=" * 80)

    try:
        processed_data = run_hotel_pipeline()
        print("=" * 80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Final processed data shape: {processed_data.shape}")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
