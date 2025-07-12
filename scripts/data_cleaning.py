import pandas as pd
import numpy as np

# CLEANING FUNCTIONS (Silent - RETURNS DATA)

def clean_missing_values(data):
    """
    Clean missing values without printing. Use this in your main EDA pipeline.
    Modifies the DataFrame in-place and returns the same DataFrame with missing values handled.
    """
    drop_threshold = 0.5 
    columns_to_drop = []
    
    for column in data.columns:
        dtype = data[column].dtype
        missing_pct = data[column].isnull().mean()
        
        # Drop columns with excessive missing values
        if missing_pct > drop_threshold:
            columns_to_drop.append(column)
        elif np.issubdtype(dtype, np.number):  # Numeric columns
            # Calculate median for imputation
            median_val = data[column].median()
            # Fill NaNs with median
            if not pd.isna(median_val):
                data[column].fillna(median_val, inplace=True)
        else:  # Categorical columns
            if not data[column].empty and data[column].notna().any():
                mode_values = data[column].mode()
                if len(mode_values) > 0:
                    mode_val = mode_values[0]
                    data[column].fillna(mode_val, inplace=True)
    
    # Drop columns with excessive missing values
    if columns_to_drop:
        data.drop(columns_to_drop, axis=1, inplace=True)
    
    return data

def clean_duplicates(data):
    """Clean duplicate rows without printing."""
    nr_duplicates = data.duplicated().sum()
    if nr_duplicates: 
        data = data.drop_duplicates()
    return data

def clean_statistical_outliers(data):
    """
    Clean statistical outliers by capping extreme values to mild bounds.
    Values that are valid but extreme/mildly too big/small.
    """
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        # Calculate IQR and bounds
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Different layers of outliers
        mild_lower = Q1 - 1.5 * IQR
        mild_upper = Q3 + 1.5 * IQR
        extreme_lower = Q1 - 3 * IQR
        extreme_upper = Q3 + 3 * IQR
        
        # Count different types of outliers
        all_outliers = ((data[col] < mild_lower) | (data[col] > mild_upper))
        extreme_outliers = ((data[col] < extreme_lower) | (data[col] > extreme_upper))
        mild_outliers = all_outliers & ~extreme_outliers
        
        extreme_count = extreme_outliers.sum()
        mild_count = mild_outliers.sum()
        total_outliers = extreme_count + mild_count
        
        if total_outliers:
            if extreme_count:
                data.loc[extreme_outliers & (data[col] < extreme_lower), col] = mild_lower
                data.loc[extreme_outliers & (data[col] > extreme_upper), col] = mild_upper
            if mild_count > 0:
                data.loc[mild_outliers & (data[col] < mild_lower), col] = mild_lower
                data.loc[mild_outliers & (data[col] > mild_upper), col] = mild_upper
    
    return data

def clean_data_errors_and_logical(data):
    """
    Function to fix data quality issues, data entry errors + logical errors
    """
    
    # Define columns that should not have negative values
    non_negative_columns = [
        'adults', 'children', 'babies', 'adr', 'previous_cancellations',
        'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
        'required_car_parking_spaces', 'total_of_special_requests'
    ]
    
    # Handle duplicate columns issue (stays_in_weeks_nights vs stays_in_week_nights)
    if 'stays_in_weeks_nights' in data.columns and 'stays_in_week_nights' in data.columns:
        if data['stays_in_weeks_nights'].equals(data['stays_in_week_nights']):
            data.drop('stays_in_weeks_nights', axis=1, inplace=True)
    
    # Remove the unnamed index column
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Remove bookings with 0 total guests (impossible booking)
    if all(col in data.columns for col in ['adults', 'children', 'babies']):
        mask = (data["adults"] + data["children"] + data["babies"]) > 0
        data.drop(data[~mask].index, inplace=True)
    
    # Set negative lead time to 0 (could be same-day booking)
    if 'lead_time' in data.columns:
        negative_lead_mask = data['lead_time'] < 0
        if negative_lead_mask.any():
            data.loc[negative_lead_mask, 'lead_time'] = 0
    
    # Fix invalid repeated guest values (set to 0, not a repeated guest)
    if 'is_repeated_guest' in data.columns:
        invalid_mask = ~data['is_repeated_guest'].isin([0, 1])
        if invalid_mask.any():
            data.loc[invalid_mask, 'is_repeated_guest'] = 0
    
    # Handle date-related logical issues
    date_columns = ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']
    if all(col in data.columns for col in date_columns):
        invalid_years = (data['arrival_date_year'] < 2010) | (data['arrival_date_year'] > 2035)
        if invalid_years.any():
            data.drop(data[invalid_years].index, inplace=True)
        
        invalid_days = (data['arrival_date_day_of_month'] < 1) | (data['arrival_date_day_of_month'] > 31)
        if invalid_days.any():
            data.drop(data[invalid_days].index, inplace=True)
    
    # Handle agent and company ID issues (NaN is OK)
    for col in ['agent', 'company']:
        if col in data.columns:
            negative_mask = data[col] < 0
            if negative_mask.any():
                data.loc[negative_mask, col] = float('nan')
    
    # Handle week number
    if 'arrival_date_week_number' in data.columns:
        invalid_weeks = (data['arrival_date_week_number'] < 1) | (data['arrival_date_week_number'] > 52)
        if invalid_weeks.any():
            data.drop(data[invalid_weeks].index, inplace=True)
    
    # Handle extreme ADR values 
    if 'adr' in data.columns:
        extreme_adr_mask = data['adr'] > 1000  
        if extreme_adr_mask.any():
            data.drop(data[extreme_adr_mask].index, inplace=True)
    
    # Handle negative values in non-negative columns
    for col in non_negative_columns:
        if col in data.columns:
            negative_mask = data[col] < 0
            if negative_mask.any():
                if col in ['adults', 'children', 'babies', 'adr']:
                    # For guest counts and ADR, remove the entire booking 
                    data.drop(data[negative_mask].index, inplace=True)
                else:
                    # For other columns, value = 0
                    data.loc[negative_mask, col] = 0
    
    return data

def handle_context_dependent_outliers(data):
    """Placeholder for context-dependent outlier handling"""
    return data

def handle_infrequent_categories(data):
    """Placeholder for infrequent category handling"""
    return data

def handle_target_leakage_outliers(data):
    """Placeholder for target leakage outlier handling"""
    return data


# ANALYSIS FUNCTIONS (PRINTS)

def analyze_missing_values(data):
    """
    Analyze and print detailed information about missing values.
    Use this for exploration and understanding your data.
    """
    print("\n=== MISSING VALUES ANALYSIS ===")
    drop_threshold = 0.5
    columns_dropped = []
    columns_processed = []
    
    for column in data.columns:
        print(f"\nProcessing column: {column}")
        
        dtype = data[column].dtype
        missing_count = data[column].isnull().sum()
        missing_pct = data[column].isnull().mean()
        
        print(f"Data type: {dtype}")
        print(f"Missing values: {missing_count} ({missing_pct:.1%})")
        
        # For numeric columns, show additional stats
        if np.issubdtype(dtype, np.number):
            zero_count = (data[column] == 0).sum()
            print(f"Zeros count: {zero_count}")
            if not data[column].empty and data[column].notna().any():
                print(f"Stats: mean={data[column].mean():.2f}, median={data[column].median():.2f}")
        
        # Check what would happen to this column
        if missing_pct > drop_threshold:
            print(f"WOULD DROP COLUMN '{column}' (over {drop_threshold*100:.0f}% missing values)")
            columns_dropped.append(column)
        elif np.issubdtype(dtype, np.number):  # Numeric columns
            median_val = data[column].median()
            if not pd.isna(median_val):
                print(f"Would fill {missing_count} missing values with median: {median_val:.2f}")
            columns_processed.append(column)
        else:  # Categorical columns
            if not data[column].empty and data[column].notna().any():
                mode_values = data[column].mode()
                if len(mode_values) > 0:
                    mode_val = mode_values[0]
                    print(f"Would fill {missing_count} missing values with mode: {mode_val}")
                columns_processed.append(column)
            else:
                print(f"Warning: Could not determine mode for column '{column}'")
    
    print(f"\nColumns processed: {len(columns_processed)}")
    print(f"Columns dropped: {len(columns_dropped)}")
    if columns_dropped:
        print(f"Dropped columns: {columns_dropped}")

def analyze_duplicates(data):
    """Analyze and print information about duplicate rows."""
    print("\n=== DUPLICATE ANALYSIS ===")
    nr_duplicates = data.duplicated().sum()
    print(f"Number of duplicate rows: {nr_duplicates}")
    if nr_duplicates: 
        print(f"Duplicates would be removed. New shape would be: {data.drop_duplicates().shape}")
    else:
        print("No duplicates found.")

def analyze_statistical_outliers(data):
    """
    Analyze and print information about statistical outliers using IQR method.
    """
    print("\n=== Analyzing Extreme/Mildly Outliers using IQR ===")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    outlier_summary = {}
    
    for col in numeric_columns:
        print(f"\n Processing {col}")
        
        # Calculate IQR and bounds
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Different layers of outliers
        mild_lower = Q1 - 1.5 * IQR
        mild_upper = Q3 + 1.5 * IQR
        extreme_lower = Q1 - 3 * IQR
        extreme_upper = Q3 + 3 * IQR
        
        # Count different types of outliers
        all_outliers = ((data[col] < mild_lower) | (data[col] > mild_upper))
        extreme_outliers = ((data[col] < extreme_lower) | (data[col] > extreme_upper))
        mild_outliers = all_outliers & ~extreme_outliers
        
        extreme_count = extreme_outliers.sum()
        mild_count = mild_outliers.sum()
        total_outliers = extreme_count + mild_count
        
        print(f"  Extreme outliers (>3*IQR): {extreme_count}")
        print(f"  Mild outliers (1.5-3*IQR): {mild_count}")
        print(f"  Bounds - Mild: [{mild_lower:.2f}, {mild_upper:.2f}]")
        print(f"  Bounds - Extreme: [{extreme_lower:.2f}, {extreme_upper:.2f}]")
        
        outlier_summary[col] = {
            'extreme_count': extreme_count,
            'mild_count': mild_count,
            'total_outliers': total_outliers,
            'extreme_bounds': (extreme_lower, extreme_upper),
            'mild_bounds': (mild_lower, mild_upper),
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'action': 'analyzed'
        }
    
    print("\nSTATISTICAL OUTLIERS SUMMARY")
    print("="*50)
    summary_data = pd.DataFrame(outlier_summary).T
    summary_data = summary_data[['extreme_count', 'mild_count', 'total_outliers', 'Q1', 'Q3', 'IQR']]
    print(summary_data)
    
    print("\nIndividual column summary:")
    for col, info in outlier_summary.items():
        if info['total_outliers'] > 0:
            print(f"  {col}: {info['extreme_count']} extreme + {info['mild_count']} mild = {info['total_outliers']} total outliers found")
        else:
            print(f"  {col}: No outliers found")

def analyze_data_errors_and_logical(data):
    """
    Function to detect and report data errors and logical issues 
    """
    print("\n=== DATA QUALITY DETECTION REPORT ===")
    logical_errors = {}
    
    print(f"Dataset size: {len(data)}")
    
    required_logical_columns = [
        "adults", "children", "babies", "stays_in_weekend_nights", 
        "stays_in_week_nights", "lead_time", "is_repeated_guest", "adr",
        "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"
    ]
    
    check_for_wrong_columns = list(set(required_logical_columns) - set(data.columns))
    if check_for_wrong_columns:
        print(f"Warning: Missing columns for logical error detection: {check_for_wrong_columns}")

    print("\n📊 Logical Constraints")
    print("-" * 30)
    
    # Total guests > 0
    if all(col in data.columns for col in ['adults', 'children', 'babies']):
        zero_guests = data[(data["adults"] + data["children"] + data["babies"]) == 0]
        logical_errors['zero_guests'] = len(zero_guests)
        print(f"Bookings with 0 total guests: {len(zero_guests)}")

    # Stay duration > 0
    if all(col in data.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
        zero_nights = data[(data["stays_in_weekend_nights"] + data["stays_in_week_nights"]) == 0]
        logical_errors['zero_nights'] = len(zero_nights)
        print(f"Bookings with 0 nights stay: {len(zero_nights)}")

    # Lead time >= 0
    if 'lead_time' in data.columns:
        negative_lead = data[data["lead_time"] < 0]
        logical_errors['negative_lead_time'] = len(negative_lead)
        print(f"Bookings with negative lead time: {len(negative_lead)}")

    # Valid repeated guest values (0 or 1)
    if 'is_repeated_guest' in data.columns:
        invalid_repeated = data[~data['is_repeated_guest'].isin([0, 1])]
        logical_errors['invalid_repeated_guest'] = len(invalid_repeated)
        print(f"Invalid is_repeated_guest values: {len(invalid_repeated)}")

    print("\n📊 Additional Hotel-Specific Checks")
    print("-" * 30)
    
    # Check for impossible date combinations
    if 'arrival_date_day_of_month' in data.columns:
        invalid_days = data[(data['arrival_date_day_of_month'] < 1) | (data['arrival_date_day_of_month'] > 31)]
        logical_errors['invalid_days'] = len(invalid_days)
        print(f"Bookings with invalid day of month: {len(invalid_days)}")
    
    # impossible week numbers
    if 'arrival_date_week_number' in data.columns:
        invalid_weeks = data[(data['arrival_date_week_number'] < 1) | (data['arrival_date_week_number'] > 53)]
        logical_errors['invalid_weeks'] = len(invalid_weeks)
        print(f"Bookings with invalid week numbers: {len(invalid_weeks)}")
    
    # unrealistic years
    if 'arrival_date_year' in data.columns:
        invalid_years = data[(data['arrival_date_year'] < 2010) | (data['arrival_date_year'] > 2025)]
        logical_errors['invalid_years'] = len(invalid_years)
        print(f"Bookings with unrealistic years: {len(invalid_years)}")
    
    # high ADR 
    if 'adr' in data.columns:
        extreme_adr = data[data['adr'] > 2000]  
        logical_errors['extreme_adr'] = len(extreme_adr)
        print(f"Bookings with extremely high ADR (>$5000): {len(extreme_adr)}")
    
    # Check for inconsistent stay duration 
    if all(col in data.columns for col in ['stays_in_weeks_nights', 'stays_in_week_nights']):
        inconsistent_stays = data[data['stays_in_weeks_nights'] != data['stays_in_week_nights']]
        logical_errors['inconsistent_stay_columns'] = len(inconsistent_stays)
        print(f"Bookings with inconsistent stay duration columns: {len(inconsistent_stays)}")
    
    if 'Unnamed: 0' in data.columns:
        logical_errors['unnecessary_index_column'] = 1
        print("Found unnecessary 'Unnamed: 0' column (should be removed)")
    
    # Check agent/company negative IDs
    for col in ['agent', 'company']:
        if col in data.columns:
            negative_ids = data[data[col] < 0]
            logical_errors[f'negative_{col}_ids'] = len(negative_ids)
            print(f"Bookings with negative {col} IDs: {len(negative_ids)}")

    print("\n📊 Impossible Values")
    print("-" * 30)
    
    non_negative_columns = [
        'adults', 'children', 'babies', 'adr', 'previous_cancellations',
        'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
        'required_car_parking_spaces', 'total_of_special_requests'
    ]

    # Check for negative values in non-negative columns
    for col in non_negative_columns:
        if col in data.columns:
            negative_values = data[data[col] < 0]
            logical_errors[f'negative_{col}'] = len(negative_values)
            print(f"Bookings with negative {col}: {len(negative_values)}")

    # Summary of detection
    print("\n📋 DETECTION SUMMARY")
    print("-" * 30)
    total_errors = sum(logical_errors.values())
    print(f"Total logical errors found: {total_errors}")

    if total_errors > 0:
        print("\nDetailed breakdown:")
        for error_type, count in logical_errors.items():
            if count > 0:
                print(f"  {error_type}: {count} records")
    else:
        print("No logical errors detected!")


# PIPELINE FUNCTIONS

def clean_outliers(data):
    """Pipeline for comprehensive outlier handling"""
    print("Starting Comprehensive Outlier Handling")
    print("=" * 50)
    original_shape = data.shape
    print(f"Original data shape: {original_shape}")

    
    print("\n" + "="*60)
    print("STEP 1: Cleaning Statistical Outliers")
    data = clean_statistical_outliers(data)
    print(f"After statistical outlier cleaning: {data.shape}")

    
    print("\n" + "="*60)
    print("STEP 2: Cleaning Data Errors and Logical Issues")
    data = clean_data_errors_and_logical(data)
    print(f"After data error cleaning: {data.shape}")

    
    print("\n" + "="*60)
    print("STEP 3: Handling Context-Dependent Outliers")
    data = handle_context_dependent_outliers(data)
    print(f"After context-dependent outlier handling: {data.shape}")

   
    print("\n" + "="*60)
    print("STEP 4: Handling Infrequent Categories")
    data = handle_infrequent_categories(data)
    print(f"After infrequent category handling: {data.shape}")

    
    print("\n" + "="*60)
    print("STEP 5: Handling Target Leakage Outliers")
    data = handle_target_leakage_outliers(data)
    print(f"After target leakage outlier handling: {data.shape}")

    print(f"\nFinal shape: {data.shape}")
    print(f"Rows removed: {original_shape[0] - data.shape[0]}")
    
    return data



def clean_data(data):
    """Main cleaning pipeline"""
    print("Starting Data Cleaning Pipeline")
    print("=" * 50)
    original_shape = data.shape
    print(f"Original data shape: {original_shape}")
    
    # Step 1: Clean missing values
    print("\n" + "="*60)
    print("STEP 1: Cleaning Missing Values")
    data = clean_missing_values(data)
    print(f"After missing value cleaning: {data.shape}")
    
    # Step 2: Clean duplicates
    print("\n" + "="*60)
    print("STEP 2: Cleaning Duplicates")
    data = clean_duplicates(data)
    print(f"After duplicate cleaning: {data.shape}")
    
    # Step 3: Clean outliers (comprehensive)
    print("\n" + "="*60)
    print("STEP 3: Comprehensive Outlier Cleaning")
    data = clean_outliers(data)
    print(f"After outlier cleaning: {data.shape}")
    
    print(f"\nFinal cleaned data shape: {data.shape}")
    print(f"Total rows removed: {original_shape[0] - data.shape[0]}")
    
    return data