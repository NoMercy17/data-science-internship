
import sys
import os
import pandas as pd

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Now import the functions directly from the data_cleaning.py file
sys.path.insert(0, os.path.join(parent_dir, 'scripts'))
import data_cleaning as dc

def test_your_raw_data():
    """
    Test your cleaning pipeline on your raw dataset.
    Replace the file path with your actual data file.
    """
    print("🚀 TESTING DATA CLEANING PIPELINE ON YOUR RAW DATA")
    print("="*60)
    
    
    print("📁 Loading your raw data...")
    
  
    try:
        file_path = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/raw/hotel_booking_cancellation_prediction.csv'  
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
    except FileNotFoundError:
        print("File not found. Please update the file_path variable with your actual data file path.")
        print(" Supported formats: CSV, Excel, JSON")
        return
    except Exception as e:
        print(f" Error loading data: {e}")
        return
    
    # STEP 3: RUN ANALYSIS FUNCTIONS (DETECTION ONLY)
    print("\n" + "="*60)
    print(" RUNNING ANALYSIS FUNCTIONS (DETECTION ONLY)")
    print("="*60)
    
    # Test missing values analysis
    dc.analyze_missing_values(df.copy())
    
    # Test duplicate analysis
    dc.analyze_duplicates(df.copy())
    
    # Test outlier analysis
    dc.analyze_statistical_outliers(df.copy())
    
    # Test logical errors analysis
    dc.analyze_data_errors_and_logical(df.copy())
    
    # RUN CLEANING FUNCTIONS (ACTUAL CLEANING)
    print("\n" + "="*60)
    print(" RUNNING CLEANING FUNCTIONS (ACTUAL CLEANING)")
    print("="*60)
    
    # Make a copy for cleaning
    df_clean = df.copy()
    original_shape = df_clean.shape
    
    print("\n Step 1: Cleaning Missing Values")
    print(f"Before: {df_clean.shape}")
    df_clean = dc.clean_missing_values(df_clean)
    print("After: {df_clean.shape}")
    
    print("\n Step 2: Cleaning Duplicates")
    print(f"Before: {df_clean.shape}")
    df_clean = dc.clean_duplicates(df_clean)
    print(f"After: {df_clean.shape}")
    
    print("\n Step 3: Cleaning Statistical Outliers")
    print(f"Before: {df_clean.shape}")
    df_clean = dc.clean_statistical_outliers(df_clean)
    print(f"After: {df_clean.shape}")
    
    print("\n Step 4: Cleaning Data Errors and Logical Issues")
    print(f"Before: {df_clean.shape}")
    df_clean = dc.clean_data_errors_and_logical(df_clean)
    print(f"After: {df_clean.shape}")
    
    
    print("\n📋 FINAL CLEANING SUMMARY")
    print("="*60)
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {df_clean.shape}")
    

    print("\n Cleaned data sample:")
    print(df_clean.head())
    

    # SAVE CLEANED DATA 
    output_dir = "/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/cleaned"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename based on original filename or use default
    if 'file_path' in locals():
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_name}_cleaned.csv"
    else:
        output_filename = "cleaned_data.csv"
    
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        df_clean.to_csv(output_path, index=False)
        print(f" Cleaned data saved to: {output_path}")
    except Exception as e:
        print(f" Error saving file: {e}")
    
    return df_clean

if __name__ == "__main__":
    cleaned_data = test_your_raw_data()

    if cleaned_data is not None:
        print("\n✅Data cleaning pipeline completed successfully!")
        print(" Cleaned data is available at: /home/antonios/Desktop/Practica_de_vara/data-science-internship/data/cleaned/")