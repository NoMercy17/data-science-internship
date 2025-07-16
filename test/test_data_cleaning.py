import os
from scripts.data_cleaning import hotel_data,output_dir
from scripts.data_cleaning import clean_missing_values,clean_duplicates,clean_statistical_outliers, clean_data_errors_and_logical, clean_infrequent_values, clean_context_dependent_outliers, clean_dtypes, clean_target_leakage

def run_cleaning_pipeline(data):
    """Run the complete cleaning pipeline in correct order"""
    
    print("="*60)
    print("STARTING DATA CLEANING PIPELINE")
    print("="*60)
    print(f"Original data shape: {data.shape}")
    
    # Step 1: Clean missing values
    print("\n" + "="*50)
    print("STEP 1: CLEANING MISSING VALUES")
    print("="*50)
    data = clean_missing_values(data)
    print(f"After missing values cleaning: {data.shape}")
    
    # Step 2: Clean duplicates
    print("\n" + "="*50)
    print("STEP 2: CLEANING DUPLICATES")
    print("="*50)
    data = clean_duplicates(data)
    print(f"After duplicate cleaning: {data.shape}")
    
    # Step 3: Clean statistical outliers
    print("\n" + "="*50)
    print("STEP 3: CLEANING STATISTICAL OUTLIERS")
    print("="*50)
    data = clean_statistical_outliers(data)
    print(f"After statistical outlier cleaning: {data.shape}")
    
    # Step 4: Clean data errors and logical issues
    print("\n" + "="*50)
    print("STEP 4: CLEANING DATA ERRORS & LOGICAL ISSUES")
    print("="*50)
    data = clean_data_errors_and_logical(data)
    print(f"After data error cleaning: {data.shape}")
    
    # Step 5: Clean infrequent values
    print("\n" + "="*50)
    print("STEP 5: CLEANING INFREQUENT VALUES")
    print("="*50)
    data = clean_infrequent_values(data)
    print(f"After infrequent values cleaning: {data.shape}")
    
    # Step 6: Clean context-dependent outliers
    print("\n" + "="*50)
    print("STEP 6: CLEANING CONTEXT-DEPENDENT OUTLIERS")
    print("="*50)
    data = clean_context_dependent_outliers(data)
    print(f"After context-dependent outlier cleaning: {data.shape}")
    
    # Step 7: Clean data types
    print("\n" + "="*50)
    print("STEP 7: CLEANING DATA TYPES")
    print("="*50)
    data = clean_dtypes(data)
    print(f"After data type cleaning: {data.shape}")
    
    # Step 8: Clean target leakage
    print("\n" + "="*50)
    print("STEP 8: CLEANING TARGET LEAKAGE")
    print("="*50)
    data = clean_target_leakage(data)
    print(f"After target leakage cleaning: {data.shape}")

    # Step 9: Cler multicollinearity
    print("\n" + "="*50)
    print("STEP 9: CLEANING MULTICOLLINEARITY")
    print("="*50)
    data = clean_target_leakage(data)
    print(f"After MULTICOLLINEARITY: {data.shape}")
    
    # Final save
    print("\n" + "="*50)
    print("FINAL SAVE")
    print("="*50)
    final_output_path = os.path.join(output_dir, 'final_cleaned_data.csv')
    data.to_csv(final_output_path, index=False)
    print(f"Final cleaned data saved to: {final_output_path}")

    final_output_path = os.path.join(output_dir, 'final_cleaned_data.pkl')
    data.to_pickle(final_output_path)
    print(f"Data with preserved dtypes saved to: {final_output_path}")
    
    print("\n" + "="*60)
    print("DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Final data shape: {data.shape}")
    print(f"All intermediate files saved in: {output_dir}")
    
    return data

run_cleaning_pipeline(hotel_data)