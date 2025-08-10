import pandas as pd
import os
import warnings
import sys

warnings.filterwarnings('ignore')

sys.path.append('/home/antonios/Desktop/Practica_de_vara/data-science-internship')

try:
    from scripts.data_cleaning import (
        clean_missing_values, clean_duplicates, clean_statistical_outliers,
        clean_data_errors_and_logical, clean_infrequent_values, 
        clean_context_dependent_outliers, clean_dtypes, clean_target_leakage,
        clean_multicollinearity
    )
    from scripts.feature_engineering import HotelFeatureExtractor
except ImportError as e:
    print(f"Warning: Could not import from scripts: {e}")


class HotelDataPipeline:
    """
    Simple production pipeline for hotel booking data preprocessing.
    Includes data cleaning and feature engineering steps.
    """
    
    def __init__(self, output_dir: str = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/app/models/trained_models'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def fit_transform(self, data: pd.DataFrame, target_column: str = 'is_canceled') -> pd.DataFrame:
        print("HOTEL DATA PIPELINE STARTING...")
        print(f"Input data shape: {data.shape}")
        print(f"Input columns: {list(data.columns)}")
        
        self.target_column = target_column
        
        # Data Cleaning
        cleaned_data = self._run_cleaning_pipeline(data.copy())
        print(f"After cleaning shape: {cleaned_data.shape}")
        print(f"After cleaning columns: {list(cleaned_data.columns)}")

        cleaned_csv_path = os.path.join(self.output_dir, 'cleaned_data.csv')
        cleaned_data.to_csv(cleaned_csv_path, index=False)
        print(f"Cleaned data saved to: {cleaned_csv_path}")
        
        
        cleaned_pkl_path = os.path.join(self.output_dir, 'cleaned_data.pkl')
        cleaned_data.to_pickle(cleaned_pkl_path)

        # Feature Engineering 
        final_data = self._run_feature_engineering_fixed(cleaned_data)

       
        engineered_csv_path = os.path.join(self.output_dir, 'engineered_data.csv')
        final_data.to_csv(engineered_csv_path, index=False)
        print(f"Engineered data saved to: {engineered_csv_path}")
        
        engineered_pkl_path = os.path.join(self.output_dir, 'engineered_data.pkl')
        final_data.to_pickle(engineered_pkl_path)
        print(f"Engineered data saved to: {engineered_pkl_path}")
        
        print(f"Final data shape: {final_data.shape}")
        print(f"Final columns: {list(final_data.columns)}")
        
        return final_data
    
    def _run_cleaning_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Running data cleaning...")

        data = clean_missing_values(data)
        print(f"After clean_missing_values: {data.shape}")
        
        data = clean_duplicates(data)
        print(f"After clean_duplicates: {data.shape}")
        
        data = clean_statistical_outliers(data)
        print(f"After clean_statistical_outliers: {data.shape}")
        
        data = clean_data_errors_and_logical(data)
        print(f"After clean_data_errors_and_logical: {data.shape}")
        
        data = clean_infrequent_values(data)
        print(f"After clean_infrequent_values: {data.shape}")
        
        data = clean_context_dependent_outliers(data)
        print(f"After clean_context_dependent_outliers: {data.shape}")
        
        data = clean_dtypes(data)
        print(f"After clean_dtypes: {data.shape}")

        
        if self.target_column in data.columns:
            print(f"Target column '{self.target_column}' found, running target leakage cleaning...")
            data = clean_target_leakage(data)
            print(f"After clean_target_leakage: {data.shape}")
        else:
            print(f"Warning: Target column '{self.target_column}' not found in data. Skipping target leakage cleaning.")

        data = clean_multicollinearity(data)
        print(f"After clean_multicollinearity: {data.shape}")

        print("Data cleaning completed.")
        return data
    
    def _run_feature_engineering_fixed(self, data: pd.DataFrame) -> pd.DataFrame:
        
        print("Running feature engineering...")
        print(f"Input to feature engineering shape: {data.shape}")
        print(f"Input to feature engineering columns: {list(data.columns)}")

        try:
            feature_extractor = HotelFeatureExtractor(output_dir=self.output_dir)
            
            # Run the complete feature engineering pipeline
            # This uses the class-based approach which maintains state properly
            final_data = feature_extractor.run_complete_feature_engineering(
                data=data, 
                apply_scaling=True, 
                scaling_method='standard'
            )
            
            print("Feature engineering completed successfully.")
            print(f"Output from feature engineering shape: {final_data.shape}")
            print(f"Output from feature engineering columns: {list(final_data.columns)}")
            
            return final_data

        except Exception as e:
            print(f"Feature engineering failed: {e}")
            import traceback
            traceback.print_exc()
            print("Returning cleaned data without feature engineering.")
            return data


def run_hotel_pipeline(data_path: str = None, 
                       output_dir: str = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/app/models/trained_models') -> pd.DataFrame:
    if data_path is None:
        data_path = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/raw/hotel_booking_cancellation_prediction.csv'
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"Loaded data shape: {data.shape}")
    print(f"Loaded data columns: {list(data.columns)}")
    
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
        print(f"Final processed data columns: {list(processed_data.columns)}")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()