import pandas as pd
import numpy as np
import os

# Load data
data = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/raw/hotel_booking_cancellation_prediction.csv'
hotel_data = pd.read_csv(data)

# Output directory
output_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/cleaned'
os.makedirs(output_dir, exist_ok=True)

def clean_missing_values(data):
    """1. Clean missing values and save result"""
    print(f"Initial data shape: {data.shape}")
    
    drop_threshold = 0.3
    
    # Drop columns with excessive missing values
    columns_to_drop = [col for col in data.columns if data[col].isnull().mean() > drop_threshold]
    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)
        print(f"Dropped {len(columns_to_drop)} columns with >30% missing values: {columns_to_drop}")
    
    # Fill missing values
    fill_values = {}
    for column in data.columns:
        if data[column].isnull().any():
            if np.issubdtype(data[column].dtype, np.number):
                median_val = data[column].median()
                if not pd.isna(median_val):
                    fill_values[column] = median_val
            else:
                if not data[column].empty and data[column].notna().any():
                    mode_values = data[column].mode()
                    if len(mode_values) > 0:
                        fill_values[column] = mode_values[0]
    
    if fill_values:
        data = data.fillna(fill_values)
        print(f"Filled missing values in {len(fill_values)} columns")
    
    print(f"After missing values: {data.shape}")
    
    # Save to file
    output_path = os.path.join(output_dir, '01_missing_values_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data

def clean_duplicates(data):
    """2. Clean duplicate rows and save result"""
    print(f"Before duplicates: {data.shape}")
    
    original_shape = data.shape[0]
    data = data.drop_duplicates()
    removed_duplicates = original_shape - data.shape[0]
    
    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates} duplicate rows")
    
    print(f"After duplicates: {data.shape}")
    
    # Save to file
    output_path = os.path.join(output_dir, '02_duplicates_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data

def clean_statistical_outliers(data):
    """3. Clean statistical outliers (extreme/mild) and save result"""
    print(f"Before outliers: {data.shape}")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        original_dtype = data[col].dtype
        
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Only process if IQR > 0 (avoid division by zero)
        if IQR > 0:
            mild_lower = Q1 - 1.5 * IQR
            mild_upper = Q3 + 1.5 * IQR
            extreme_lower = Q1 - 3 * IQR
            extreme_upper = Q3 + 3 * IQR
            
            all_outliers = ((data[col] < mild_lower) | (data[col] > mild_upper))
            extreme_outliers = ((data[col] < extreme_lower) | (data[col] > extreme_upper))
            mild_outliers = all_outliers & ~extreme_outliers
            
            if (extreme_outliers.sum() + mild_outliers.sum()) > 0:
                if pd.api.types.is_integer_dtype(original_dtype):
                    data[col] = data[col].astype('float64')
                
                if extreme_outliers.sum() > 0:
                    data.loc[extreme_outliers & (data[col] < extreme_lower), col] = mild_lower
                    data.loc[extreme_outliers & (data[col] > extreme_upper), col] = mild_upper
                    print(f"Capped {extreme_outliers.sum()} extreme outliers in {col}")
                    
                if mild_outliers.sum() > 0:
                    data.loc[mild_outliers & (data[col] < mild_lower), col] = mild_lower
                    data.loc[mild_outliers & (data[col] > mild_upper), col] = mild_upper
                    print(f"Capped {mild_outliers.sum()} mild outliers in {col}")
                
                if pd.api.types.is_integer_dtype(original_dtype):
                    data[col] = data[col].round().astype(original_dtype)
    
    print(f"After outliers: {data.shape}")
    
    # Save to file
    output_path = os.path.join(output_dir, '03_statistical_outliers_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data

def clean_data_errors_and_logical(data):
    """4. Clean data errors and logical inconsistencies and save result"""
    print(f"Before data errors: {data.shape}")
    
    # Handle duplicate columns
    if 'stays_in_weeks_nights' in data.columns and 'stays_in_week_nights' in data.columns:
        if data['stays_in_weeks_nights'].equals(data['stays_in_week_nights']):
            data.drop('stays_in_weeks_nights', axis=1, inplace=True)
            print("Removed duplicate column 'stays_in_weeks_nights'")
    
    # Remove unnamed index column
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', axis=1, inplace=True)
        print("Removed 'Unnamed: 0' column")
    
    # Remove bookings with 0 total guests (FIXED: More lenient condition)
    if all(col in data.columns for col in ['adults', 'children', 'babies']):
        original_count = len(data)
        # Only remove if ALL guest counts are 0 or negative
        mask = (data["adults"] + data["children"] + data["babies"]) > 0
        data = data[mask]
        removed_count = original_count - len(data)
        if removed_count > 0:
            print(f"Removed {removed_count} bookings with 0 total guests")
    
    # Fix negative lead time (FIXED: Don't remove, just cap)
    if 'lead_time' in data.columns:
        negative_count = (data['lead_time'] < 0).sum()
        if negative_count > 0:
            data.loc[data['lead_time'] < 0, 'lead_time'] = 0
            print(f"Fixed {negative_count} negative lead time values")
    
    # Fix invalid repeated guest values (FIXED: More lenient)
    if 'is_repeated_guest' in data.columns:
        # Don't remove rows, just fix invalid values
        invalid_mask = ~data['is_repeated_guest'].isin([0, 1])
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            data.loc[invalid_mask, 'is_repeated_guest'] = 0
            print(f"Fixed {invalid_count} invalid repeated guest values")
    
    # Handle date-related logical issues (FIXED: More reasonable ranges)
    date_columns = ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']
    if all(col in data.columns for col in date_columns):
        original_count = len(data)
        # More reasonable year range
        data = data[(data['arrival_date_year'] >= 2015) & (data['arrival_date_year'] <= 2025)]
        data = data[(data['arrival_date_day_of_month'] >= 1) & (data['arrival_date_day_of_month'] <= 31)]
        removed_count = original_count - len(data)
        if removed_count > 0:
            print(f"Removed {removed_count} rows with invalid dates")
    
    # Handle negative agent and company IDs
    for col in ['agent', 'company']:
        if col in data.columns:
            negative_count = (data[col] < 0).sum()
            if negative_count > 0:
                data.loc[data[col] < 0, col] = float('nan')
                print(f"Set {negative_count} negative {col} IDs to NaN")
    
    # Handle week number
    if 'arrival_date_week_number' in data.columns:
        original_count = len(data)
        data = data[(data['arrival_date_week_number'] >= 1) & (data['arrival_date_week_number'] <= 53)]
        removed_count = original_count - len(data)
        if removed_count > 0:
            print(f"Removed {removed_count} rows with invalid week numbers")
    
    # Handle extreme ADR values (FIXED: More reasonable threshold)
    if 'adr' in data.columns:
        original_count = len(data)
        # More reasonable ADR threshold
        data = data[data['adr'] <= 2000]
        removed_count = original_count - len(data)
        if removed_count > 0:
            print(f"Removed {removed_count} rows with ADR > 2000")
    
    # Handle negative values in specific columns (FIXED: More selective)
    # Only fix truly problematic negatives, don't remove rows
    fix_negative_columns = [
        'previous_cancellations', 'previous_bookings_not_canceled', 
        'booking_changes', 'days_in_waiting_list',
        'required_car_parking_spaces', 'total_of_special_requests'
    ]
    
    for col in fix_negative_columns:
        if col in data.columns:
            negative_count = (data[col] < 0).sum()
            if negative_count > 0:
                data.loc[data[col] < 0, col] = 0
                print(f"Set {negative_count} negative {col} values to 0")
    
    # Only remove rows for critical negatives
    critical_negative_columns = ['adults', 'children', 'babies']
    for col in critical_negative_columns:
        if col in data.columns:
            original_count = len(data)
            data = data[data[col] >= 0]
            removed_count = original_count - len(data)
            if removed_count > 0:
                print(f"Removed {removed_count} rows with negative {col}")
    
    # Reset index after row deletions
    data = data.reset_index(drop=True)
    
    print(f"After data errors: {data.shape}")
    
    # Save to file
    output_path = os.path.join(output_dir, '04_data_errors_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data

def clean_infrequent_values(data):
    """5. Clean infrequent/extreme values based on business logic and save result"""
    print(f"Before infrequent values: {data.shape}")
    
    print("Cleaning infrequent values...")
    
    # FIXED: More conservative capping for continuous variables
    # Days in waiting list
    if 'days_in_waiting_list' in data.columns:
        current_max = data['days_in_waiting_list'].max()
        if current_max > 500:  # Increased threshold
            cap_value = data['days_in_waiting_list'].quantile(0.99)  # 99th percentile instead of 95th
            mask = data['days_in_waiting_list'] > cap_value
            if mask.sum() > 0:
                data.loc[mask, 'days_in_waiting_list'] = cap_value
                print(f"Capped {mask.sum()} extreme values in days_in_waiting_list at 99th percentile ({cap_value})")
    
    # Lead time  
    if 'lead_time' in data.columns:
        current_max = data['lead_time'].max()
        if current_max > 500:  # Increased threshold
            cap_value = data['lead_time'].quantile(0.99)  # 99th percentile
            mask = data['lead_time'] > cap_value
            if mask.sum() > 0:
                data.loc[mask, 'lead_time'] = cap_value
                print(f"Capped {mask.sum()} extreme values in lead_time at 99th percentile ({cap_value})")
    
    # Previous cancellations
    if 'previous_cancellations' in data.columns:
        current_max = data['previous_cancellations'].max()
        if current_max > 50:  # Increased threshold
            cap_value = data['previous_cancellations'].quantile(0.99)  # 99th percentile
            mask = data['previous_cancellations'] > cap_value
            if mask.sum() > 0:
                data.loc[mask, 'previous_cancellations'] = cap_value
                print(f"Capped {mask.sum()} extreme values in previous_cancellations at 99th percentile ({cap_value})")
    
    # FIXED: More conservative categorical cleaning
    categorical_columns_to_clean = ['country']  # Removed 'agent' to preserve variation
    
    for col in categorical_columns_to_clean:
        if col in data.columns:
            if col == 'country':
                # More conservative threshold for country grouping
                min_frequency = len(data) * 0.0005  # 0.05% threshold instead of 0.1%
                value_counts = data[col].value_counts()
                frequent_countries = value_counts[value_counts >= min_frequency].index
                mask = ~data[col].isin(frequent_countries)
                if mask.sum() > 0:
                    data.loc[mask, col] = 'Other'
                    print(f"Grouped {mask.sum()} infrequent countries into 'Other' category")
    
    print(f"After infrequent values: {data.shape}")
    
    # Save to file
    output_path = os.path.join(output_dir, '05_infrequent_values_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data

def clean_context_dependent_outliers(data):
    """6. Clean context-dependent outliers and save result"""
    print(f"Before context outliers: {data.shape}")
    
    print("Cleaning context-dependent outliers...")
    
    # FIXED: More conservative context-dependent cleaning
    # High ADR in budget segments
    if 'adr' in data.columns and 'market_segment' in data.columns:
        budget_segments = ['Online TA', 'Groups']
        for segment in budget_segments:
            if segment in data['market_segment'].values:
                segment_data = data[data['market_segment'] == segment]
                if len(segment_data) > 0:
                    # Use 99th percentile instead of 95th
                    high_adr_threshold = segment_data['adr'].quantile(0.99)
                    high_adr_mask = (data['market_segment'] == segment) & (data['adr'] > high_adr_threshold)
                    affected_count = high_adr_mask.sum()
                    if affected_count > 0:
                        data.loc[high_adr_mask, 'adr'] = high_adr_threshold
                        print(f"Capped {affected_count} high ADR values in {segment} segment")
    
    # Walk-in bookings with deposit requirements (FIXED: More lenient)
    if 'lead_time' in data.columns and 'deposit_type' in data.columns:
        # Only fix if lead_time is exactly 0 AND deposit is not 'No Deposit'
        walk_in_with_deposit = (data['lead_time'] == 0) & (data['deposit_type'] != 'No Deposit')
        affected_count = walk_in_with_deposit.sum()
        if affected_count > 0 and affected_count < len(data) * 0.1:  # Only if < 10% of data
            data.loc[walk_in_with_deposit, 'deposit_type'] = 'No Deposit'
            print(f"Fixed {affected_count} walk-in bookings with deposit requirements")
    
    # Previous cancellations but not marked as repeated guest (FIXED: More lenient)
    if all(col in data.columns for col in ['previous_cancellations', 'is_repeated_guest']):
        # Only fix if previous_cancellations > 0 AND is_repeated_guest is 0
        prev_cancel_not_repeat = (data['previous_cancellations'] > 0) & (data['is_repeated_guest'] == 0)
        affected_count = prev_cancel_not_repeat.sum()
        if affected_count > 0 and affected_count < len(data) * 0.1:  # Only if < 10% of data
            data.loc[prev_cancel_not_repeat, 'is_repeated_guest'] = 1
            print(f"Fixed {affected_count} guests with previous cancellations not marked as repeated")
    
    # REMOVED: Parking spaces exceed adults (too aggressive)
    # REMOVED: Booking changes exceed lead time (too aggressive)
    
    print(f"After context outliers: {data.shape}")
    
    # Save to file
    output_path = os.path.join(output_dir, '06_context_outliers_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data

def clean_dtypes(data):
    """7. Clean data types with better preservation"""
    print(f"Before dtypes: {data.shape}")
    print("Cleaning data types...")
    
    # Handle date-related columns
    if 'arrival_date_month' in data.columns:
        data['arrival_date_month'] = data['arrival_date_month'].astype(str).str.title()
    
    # Handle reservation_status_date
    if 'reservation_status_date' in data.columns:
        data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'], errors='coerce')
    
    # Binary columns
    binary_columns = ['is_repeated_guest', 'is_canceled']
    for col in binary_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype('int8')
    
    # Integer columns
    integer_columns = [
        'lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
        'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes',
        'days_in_waiting_list', 'required_car_parking_spaces', 'total_of_special_requests'
    ]
    
    for col in integer_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype('int64')
    
    # Float columns
    float_columns = ['adr']
    for col in float_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')
    
    # Nullable integer columns
    nullable_int_columns = ['agent', 'company']
    for col in nullable_int_columns:
        if col in data.columns:
            numeric_series = pd.to_numeric(data[col], errors='coerce')
            non_null_mask = numeric_series.notna()
            if non_null_mask.any():
                numeric_series.loc[non_null_mask] = numeric_series.loc[non_null_mask].round()
                        
            try:
                data[col] = numeric_series.astype('Int64')
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not convert {col} to Int64, keeping as float64. Error: {e}")
                data[col] = numeric_series.astype('float64')
    
    # Categorical columns processing
    categorical_columns = {
        'hotel': lambda x: x.astype(str).str.title(),
        'meal': lambda x: x.astype(str).str.upper(),
        'country': lambda x: x.astype(str).str.upper(),
        'market_segment': lambda x: x.astype(str),
        'distribution_channel': lambda x: x.astype(str),
        'reserved_room_type': lambda x: x.astype(str).str.upper(),
        'assigned_room_type': lambda x: x.astype(str).str.upper(),
        'deposit_type': lambda x: x.astype(str).str.title(),
        'customer_type': lambda x: x.astype(str).str.title(),
        'reservation_status': lambda x: x.astype(str).str.title()
    }
    
    for col, transform_func in categorical_columns.items():
        if col in data.columns:
            try:
                data[col] = transform_func(data[col])
            except Exception as e:
                print(f"Warning: Could not transform {col}. Error: {e}")
                data[col] = data[col].astype(str)
    
    # Convert to category for memory efficiency 
    category_columns = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel',
                       'reserved_room_type', 'assigned_room_type', 'deposit_type',
                       'customer_type', 'reservation_status', 'arrival_date_month']
    
    for col in category_columns:
        if col in data.columns:
            try:
                # First ensure it's string type, then convert to category
                data[col] = data[col].astype(str).astype('category')
                print(f"✓ Converted {col} to category with {data[col].nunique()} unique values")
            except Exception as e:
                print(f"Warning: Could not convert {col} to category. Error: {e}")
    
    print(f"After dtypes: {data.shape}")
    
    # Print final dtypes to verify
    print("\nFinal data types:")
    print(data.dtypes)
    
    # preserve data types
    pickle_path = os.path.join(output_dir, '07_dtypes_cleaned.pkl')
    data.to_pickle(pickle_path)
    print(f"Data with preserved dtypes saved to: {pickle_path}")
    
    return data

def clean_target_leakage(data):
    """8. Detect and remove target leakage with better column variation checking"""
    print(f"Before target leakage: {data.shape}")
    
    high_correlation_threshold = 0.5  # Remove these (likely leakage)
    low_correlation_threshold = 0.2   # Remove these (not relevant)
    
    
    if 'is_canceled' not in data.columns:
        print("Warning: Target variable 'is_canceled' not found in data")
        output_path = os.path.join(output_dir, '08_target_leakage_cleaned.csv')
        data.to_csv(output_path, index=False)
        return data
    
    target = data['is_canceled']
    columns_to_remove = []
    correlation_results = []
    
    # Check each column for correlation with target
    for col in data.columns:
        if col == 'is_canceled':
            continue
            
        try:
            dtype_str = str(data[col].dtype)
            
            # FIXED: Better variation checking
            # Check if column has any variation
            if dtype_str in ['int64', 'float64', 'Int64', 'Float64'] or any(num_type in dtype_str.lower() for num_type in ['int', 'float']):
                col_filled = data[col].fillna(data[col].median() if data[col].notna().any() else 0)
                col_filled = col_filled.astype('float64')
                
                # Check for variation
                if col_filled.nunique() <= 1:
                    print(f"Skipping {col} - all values are identical (nunique={col_filled.nunique()})")
                    continue
                    
                if col_filled.std() == 0:
                    print(f"Skipping {col} - no variation (std=0)")
                    continue
                    
                target_filled = target.astype('float64')
                correlation = np.corrcoef(col_filled, target_filled)[0, 1]
                
                if not np.isnan(correlation):
                    abs_corr = abs(correlation)
                    
                    if abs_corr >= high_correlation_threshold or abs_corr < low_correlation_threshold:
                        columns_to_remove.append(col)
                        reason = "leakage" if abs_corr >= high_correlation_threshold else "not_relevant"
                        correlation_results.append({
                            'column': col,
                            'type': 'numeric',
                            'correlation': correlation,
                            'abs_correlation': abs_corr,
                            'reason': reason
                        })
                        print(f"Will remove {col} - correlation {correlation:.6f} ({reason})")
                    else:
                        print(f"Keeping {col} - correlation {correlation:.6f} (useful feature)")
            
            # For categorical columns
            elif 'category' in dtype_str.lower() or dtype_str == 'object':
                try:
                    # Check for variation first
                    if data[col].nunique() <= 1:
                        print(f"Skipping {col} - all values are identical (nunique={data[col].nunique()})")
                        continue
                    
                    dummies = pd.get_dummies(data[col], prefix=col, dummy_na=True)
                    max_correlation = 0
                    best_category = None
                    
                    for dummy_col in dummies.columns:
                        # Check for variation in dummy column
                        if dummies[dummy_col].nunique() <= 1:
                            continue
                            
                        dummy_correlation = np.corrcoef(dummies[dummy_col], target)[0, 1]
                        if not np.isnan(dummy_correlation) and abs(dummy_correlation) > abs(max_correlation):
                            max_correlation = dummy_correlation
                            best_category = dummy_col
                    
                    if best_category is not None:
                        if abs(max_correlation) >= high_correlation_threshold or abs(max_correlation) < low_correlation_threshold:
                            columns_to_remove.append(col)
                            reason = "leakage" if abs(max_correlation) >= high_correlation_threshold else "not_relevant"
                            correlation_results.append({
                                'column': col,
                                'type': 'categorical',
                                'correlation': max_correlation,
                                'abs_correlation': abs(max_correlation),
                                'best_category': best_category,
                                'reason': reason
                            })
                            print(f"Will remove {col} - max correlation {max_correlation:.6f} via {best_category} ({reason})")
                        else:
                            print(f"Keeping {col} - max correlation {max_correlation:.6f} (useful feature)")
                    else:
                        print(f"Skipping {col} - no valid correlations found")
                
                except Exception as e:
                    print(f"Warning: Could not calculate categorical correlation for {col}: {e}")
            
            # Special check for reservation_status
            if col == 'reservation_status':
                unique_values = data[col].astype(str).unique()
                if any('cancel' in str(val).lower() for val in unique_values):
                    canceled_values = [val for val in unique_values if 'cancel' in str(val).lower()]
                    
                    for canceled_val in canceled_values:
                        canceled_status_mask = data[col].astype(str) == str(canceled_val)
                        if canceled_status_mask.sum() > 0:
                            status_correlation = np.corrcoef(canceled_status_mask.astype(int), target)[0, 1]
                            abs_status_corr = abs(status_correlation)
                            
                            if abs_status_corr >= high_correlation_threshold:
                                if col not in columns_to_remove:
                                    columns_to_remove.append(col)
                                    correlation_results.append({
                                        'column': col,
                                        'type': 'special_reservation_status',
                                        'correlation': status_correlation,
                                        'abs_correlation': abs_status_corr,
                                        'leaky_value': canceled_val,
                                        'reason': 'leakage'
                                    })
                                print(f"Will remove {col} - '{canceled_val}' status has correlation {status_correlation:.6f} (leakage)")
                                break
        
        except Exception as e:
            print(f"Warning: Could not check correlation for column {col}: {e}")
            continue
    
    # Remove columns
    if columns_to_remove:
        columns_to_remove = list(dict.fromkeys(columns_to_remove))
        data = data.drop(columns=columns_to_remove)
        print(f"\n✅ Removed {len(columns_to_remove)} columns")
        print(f"Removed columns: {columns_to_remove}")
    else:
        print("\n✅ No columns to remove")
    
    print(f"After target leakage: {data.shape}")
    
    # Save to file
    pickle_path = os.path.join(output_dir, '08_target_leakage_cleaned.pkl')
    data.to_pickle(pickle_path)
    print(f"Data with preserved dtypes saved to: {pickle_path}")
    
    return data


def clean_multicollinearity(data):
    """
    Clean multicollinearity among features using correlation matrix and VIF analysis
    """
    print("\n=== MULTICOLLINEARITY CLEANING ===")
    print(f"Input data shape: {data.shape}")
    
    # Local parameters
    correlation_threshold = 0.8
    vif_threshold = 10
    
    data_clean = data.copy()
    
    # Remove target variable for multicollinearity analysis
    if 'is_canceled' in data_clean.columns:
        target = data_clean['is_canceled']
        data_clean = data_clean.drop(columns=['is_canceled'])
    else:
        target = None
    
    # Only work with numeric columns for multicollinearity
    numeric_columns = data_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        print("Not enough numeric columns for multicollinearity analysis")
        if target is not None:
            data_clean['is_canceled'] = target
        return data_clean
    
    print(f"Checking {len(numeric_columns)} numeric features for multicollinearity")
    
    # === STEP 1: CORRELATION MATRIX ANALYSIS ===
    print(f"\n1. Correlation Matrix Analysis (threshold: {correlation_threshold})")
    
    # Fill any NaN values for correlation calculation
    data_numeric = data_clean[numeric_columns].fillna(0)
    correlation_matrix = data_numeric.corr().abs()
    
    # Find highly correlated pairs
    highly_correlated_pairs = []
    features_to_remove = set()
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            
            if corr_value > correlation_threshold:
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                
                highly_correlated_pairs.append((feature1, feature2, corr_value))
                
                # Remove the feature with less variance (or second one if equal)
                var1 = data_numeric[feature1].var()
                var2 = data_numeric[feature2].var()
                
                if var1 < var2:
                    features_to_remove.add(feature1)
                    print(f"  • {feature1} vs {feature2}: r={corr_value:.3f} → Remove {feature1} (lower variance: {var1:.6f})")
                else:
                    features_to_remove.add(feature2)
                    print(f"  • {feature1} vs {feature2}: r={corr_value:.3f} → Remove {feature2} (lower variance: {var2:.6f})")
    
    if not highly_correlated_pairs:
        print("  ✅ No highly correlated feature pairs found")
    else:
        print(f"  ⚠️  Found {len(highly_correlated_pairs)} highly correlated pairs")
    
    # === STEP 2: VIF ANALYSIS ===
    print(f"\n2. Variance Inflation Factor Analysis (threshold: {vif_threshold})")
    
    # Remove highly correlated features first
    data_for_vif = data_numeric.drop(columns=list(features_to_remove))
    vif_numeric_columns = data_for_vif.columns.tolist()
    
    if len(vif_numeric_columns) < 2:
        print("  Not enough features remaining for VIF analysis")
        vif_features_to_remove = set()
    else:
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            # Calculate VIF iteratively (without scaling)
            vif_features_to_remove = set()
            max_iterations = 10
            iteration = 0
            
            while iteration < max_iterations:
                current_features = [col for col in data_for_vif.columns if col not in vif_features_to_remove]
                
                if len(current_features) < 2:
                    break
                
                vif_data = data_for_vif[current_features]
                
                # Calculate VIF for remaining features
                vif_scores = []
                for i, feature in enumerate(current_features):
                    try:
                        vif_score = variance_inflation_factor(vif_data.values, i)
                        if not np.isnan(vif_score) and not np.isinf(vif_score):
                            vif_scores.append((feature, vif_score))
                    except:
                        continue
                
                if not vif_scores:
                    break
                
                # Sort by VIF score
                vif_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Check if any feature has VIF > threshold
                highest_vif_feature, highest_vif_score = vif_scores[0]
                
                if highest_vif_score > vif_threshold:
                    vif_features_to_remove.add(highest_vif_feature)
                    print(f"  • Iteration {iteration + 1}: Remove {highest_vif_feature} (VIF={highest_vif_score:.2f})")
                    iteration += 1
                else:
                    break
            
            if not vif_features_to_remove:
                print("  ✅ No features with high VIF found")
            else:
                print(f"  ⚠️  Removed {len(vif_features_to_remove)} features due to high VIF")
        
        except ImportError:
            print("  ⚠️  statsmodels not available. Skipping VIF analysis.")
            print("  Install with: pip install statsmodels")
            vif_features_to_remove = set()
        except Exception as e:
            print(f"  ⚠️  Error in VIF calculation: {e}")
            vif_features_to_remove = set()
    
    # === STEP 3: REMOVE MULTICOLLINEAR FEATURES ===
    all_features_to_remove = features_to_remove.union(vif_features_to_remove)
    
    if all_features_to_remove:
        print(f"\n3. Removing {len(all_features_to_remove)} multicollinear features:")
        for feature in sorted(all_features_to_remove):
            print(f"  • {feature}")
        
        data_clean = data_clean.drop(columns=list(all_features_to_remove))
    else:
        print("\n3. ✅ No multicollinear features to remove")
    
    # Add target back if it existed
    if target is not None:
        data_clean['is_canceled'] = target
    
    # === STEP 4: FINAL SUMMARY ===
    print("\n=== MULTICOLLINEARITY SUMMARY ===")
    print(f"Original numeric features: {len(numeric_columns)}")
    print(f"Removed due to correlation: {len(features_to_remove)}")
    print(f"Removed due to VIF: {len(vif_features_to_remove)}")
    print(f"Final features: {data_clean.shape[1]}")
    print(f"Final data shape: {data_clean.shape}")
    
    # Save to file
    pickle_path = os.path.join(output_dir, '09_multicollinearity_cleaned.pkl')
    data_clean.to_pickle(pickle_path)
    print(f"Data after multicollinearity check saved to: {pickle_path}")
    
    return data_clean

# Main execution
if __name__ == "__main__":
    print("Starting data cleaning pipeline...")