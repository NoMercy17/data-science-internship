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
    
    # Save to file
    output_path = os.path.join(output_dir, '01_missing_values_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data

def clean_duplicates(data):
    """2. Clean duplicate rows and save result"""
    original_shape = data.shape[0]
    data = data.drop_duplicates()
    removed_duplicates = original_shape - data.shape[0]
    
    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates} duplicate rows")
    
    # Save to file
    output_path = os.path.join(output_dir, '02_duplicates_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data

def clean_statistical_outliers(data):
    """3. Clean statistical outliers (extreme/mild) and save result"""
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        original_dtype = data[col].dtype
        
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
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
    
    # Save to file
    output_path = os.path.join(output_dir, '03_statistical_outliers_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data

def clean_data_errors_and_logical(data):
    """4. Clean data errors and logical inconsistencies and save result"""
    
    # Handle duplicate columns
    if 'stays_in_weeks_nights' in data.columns and 'stays_in_week_nights' in data.columns:
        if data['stays_in_weeks_nights'].equals(data['stays_in_week_nights']):
            data.drop('stays_in_weeks_nights', axis=1, inplace=True)
            print("Removed duplicate column 'stays_in_weeks_nights'")
    
    # Remove unnamed index column
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', axis=1, inplace=True)
        print("Removed 'Unnamed: 0' column")
    
    # Remove bookings with 0 total guests
    if all(col in data.columns for col in ['adults', 'children', 'babies']):
        original_count = len(data)
        mask = (data["adults"] + data["children"] + data["babies"]) > 0
        data = data[mask]
        removed_count = original_count - len(data)
        if removed_count > 0:
            print(f"Removed {removed_count} bookings with 0 total guests")
    
    # Fix negative lead time
    if 'lead_time' in data.columns:
        negative_count = (data['lead_time'] < 0).sum()
        if negative_count > 0:
            data.loc[data['lead_time'] < 0, 'lead_time'] = 0
            print(f"Fixed {negative_count} negative lead time values")
    
    # Fix invalid repeated guest values
    if 'is_repeated_guest' in data.columns:
        invalid_count = (~data['is_repeated_guest'].isin([0, 1])).sum()
        if invalid_count > 0:
            data.loc[~data['is_repeated_guest'].isin([0, 1]), 'is_repeated_guest'] = 0
            print(f"Fixed {invalid_count} invalid repeated guest values")
    
    # Handle date-related logical issues
    date_columns = ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']
    if all(col in data.columns for col in date_columns):
        original_count = len(data)
        data = data[(data['arrival_date_year'] >= 2010) & (data['arrival_date_year'] <= 2035)]
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
        data = data[(data['arrival_date_week_number'] >= 1) & (data['arrival_date_week_number'] <= 52)]
        removed_count = original_count - len(data)
        if removed_count > 0:
            print(f"Removed {removed_count} rows with invalid week numbers")
    
    # Handle extreme ADR values
    if 'adr' in data.columns:
        original_count = len(data)
        data = data[data['adr'] <= 1000]
        removed_count = original_count - len(data)
        if removed_count > 0:
            print(f"Removed {removed_count} rows with ADR > 1000")
    
    # Handle negative values in specific columns
    non_negative_columns = [
        'adults', 'children', 'babies', 'adr', 'previous_cancellations',
        'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
        'required_car_parking_spaces', 'total_of_special_requests'
    ]
    
    for col in non_negative_columns:
        if col in data.columns:
            if col in ['adults', 'children', 'babies', 'adr']:
                original_count = len(data)
                data = data[data[col] >= 0]
                removed_count = original_count - len(data)
                if removed_count > 0:
                    print(f"Removed {removed_count} rows with negative {col}")
            else:
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    data.loc[data[col] < 0, col] = 0
                    print(f"Set {negative_count} negative {col} values to 0")
    
    # Reset index after row deletions
    data = data.reset_index(drop=True)
    
    # Save to file
    output_path = os.path.join(output_dir, '04_data_errors_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data

def clean_infrequent_values(data):
    """5. Clean infrequent/extreme values based on business logic and save result"""
    
    print("Cleaning infrequent values...")
    
    # Cap extreme values in continuous variables
    # Days in waiting list
    if 'days_in_waiting_list' in data.columns:
        current_max = data['days_in_waiting_list'].max()
        if current_max > 300:
            cap_value = data['days_in_waiting_list'].quantile(0.95)
            mask = data['days_in_waiting_list'] > cap_value
            if mask.sum() > 0:
                data.loc[mask, 'days_in_waiting_list'] = cap_value
                print(f"Capped {mask.sum()} extreme values in days_in_waiting_list at 95th percentile ({cap_value})")
    
    # Lead time  
    if 'lead_time' in data.columns:
        current_max = data['lead_time'].max()
        if current_max > 300:  
            cap_value = data['lead_time'].quantile(0.95)
            mask = data['lead_time'] > cap_value
            if mask.sum() > 0:
                data.loc[mask, 'lead_time'] = cap_value
                print(f"Capped {mask.sum()} extreme values in lead_time at 95th percentile ({cap_value})")
    
    # Previous cancellations
    if 'previous_cancellations' in data.columns:
        current_max = data['previous_cancellations'].max()
        if current_max > 20: 
            cap_value = data['previous_cancellations'].quantile(0.95)
            mask = data['previous_cancellations'] > cap_value
            if mask.sum() > 0:
                data.loc[mask, 'previous_cancellations'] = cap_value
                print(f"Capped {mask.sum()} extreme values in previous_cancellations at 95th percentile ({cap_value})")
    
    # Handle categorical columns with too many infrequent categories
    categorical_columns_to_clean = ['country', 'agent']  
    
    for col in categorical_columns_to_clean:
        if col in data.columns:
            if col == 'country':
                # Group infrequent countries into 'Other'
                min_frequency = len(data) * 0.001  # 0.1% threshold
                value_counts = data[col].value_counts()
                frequent_countries = value_counts[value_counts >= min_frequency].index
                mask = ~data[col].isin(frequent_countries)
                if mask.sum() > 0:
                    data.loc[mask, col] = 'Other'
                    print(f"Grouped {mask.sum()} infrequent countries into 'Other' category")
            
            elif col == 'agent':
                # For agent IDs, set infrequent ones to NaN
                if data[col].notna().sum() > 0:
                    value_counts = data[col].value_counts()
                    cumulative_freq = value_counts.cumsum() / value_counts.sum()
                    frequent_values = cumulative_freq[cumulative_freq <= 0.95].index
                    mask = data[col].notna() & ~data[col].isin(frequent_values)
                    if mask.sum() > 0:
                        data.loc[mask, col] = np.nan
                        print(f"Set {mask.sum()} infrequent values in {col} to NaN")
    
    # Save to file
    output_path = os.path.join(output_dir, '05_infrequent_values_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data

def clean_context_dependent_outliers(data):
    """6. Clean context-dependent outliers and save result"""
    
    print("Cleaning context-dependent outliers...")
    
    # High ADR in budget segments
    if 'adr' in data.columns and 'market_segment' in data.columns:
        budget_segments = ['Online TA', 'Groups']
        for segment in budget_segments:
            if segment in data['market_segment'].values:
                segment_data = data[data['market_segment'] == segment]
                if len(segment_data) > 0:
                    high_adr_threshold = segment_data['adr'].quantile(0.95)
                    high_adr_mask = (data['market_segment'] == segment) & (data['adr'] > high_adr_threshold)
                    affected_count = high_adr_mask.sum()
                    if affected_count > 0:
                        data.loc[high_adr_mask, 'adr'] = high_adr_threshold
                        print(f"Capped {affected_count} high ADR values in {segment} segment")
    
    # Walk-in bookings with deposit requirements
    if 'lead_time' in data.columns and 'deposit_type' in data.columns:
        walk_in_with_deposit = (data['lead_time'] == 0) & (data['deposit_type'] != 'No Deposit')
        affected_count = walk_in_with_deposit.sum()
        if affected_count > 0:
            data.loc[walk_in_with_deposit, 'deposit_type'] = 'No Deposit'
            print(f"Fixed {affected_count} walk-in bookings with deposit requirements")
    
    # Previous cancellations but not marked as repeated guest
    if all(col in data.columns for col in ['previous_cancellations', 'is_repeated_guest']):
        prev_cancel_not_repeat = (data['previous_cancellations'] > 0) & (data['is_repeated_guest'] == 0)
        affected_count = prev_cancel_not_repeat.sum()
        if affected_count > 0:
            data.loc[prev_cancel_not_repeat, 'is_repeated_guest'] = 1
            print(f"Fixed {affected_count} guests with previous cancellations not marked as repeated")
    
    # Parking spaces exceed adults
    if all(col in data.columns for col in ['required_car_parking_spaces', 'adults']):
        parking_exceeds_adults = data['required_car_parking_spaces'] > data['adults']
        affected_count = parking_exceeds_adults.sum()
        if affected_count > 0:
            data.loc[parking_exceeds_adults, 'required_car_parking_spaces'] = data.loc[parking_exceeds_adults, 'adults']
            print(f"Fixed {affected_count} bookings where parking spaces exceeded adults")
    
    # Booking changes exceed lead time
    if 'booking_changes' in data.columns and 'lead_time' in data.columns:
        changes_exceed_lead = (data['booking_changes'] > data['lead_time']) & (data['lead_time'] > 0)
        affected_count = changes_exceed_lead.sum()
        if affected_count > 0:
            data.loc[changes_exceed_lead, 'booking_changes'] = data.loc[changes_exceed_lead, 'lead_time']
            print(f"Fixed {affected_count} bookings where changes exceeded lead time")
    
    # Save to file
    output_path = os.path.join(output_dir, '06_context_outliers_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data

def clean_dtypes(data):
    """
    IMPROVED dtype cleaning function with proper categorical preservation
    """
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
    
    # Print final dtypes to verify
    print("\nFinal data types:")
    print(data.dtypes)
    
    # preserve data types
    pickle_path = os.path.join(output_dir, '07_dtypes_cleaned.pkl')
    data.to_pickle(pickle_path)
    print(f"Data with preserved dtypes saved to: {pickle_path}")
    
    return data


def clean_target_leakage(data):
    """
    Detect and remove columns that have correlation with target variable (is_canceled) 
    above the specified threshold (default 0.2)
    """
    

    correlation_threshold=0.2
    
    print(f"Detecting target leakage with correlation threshold: {correlation_threshold}...")
    
    if 'is_canceled' not in data.columns:
        print("Warning: Target variable 'is_canceled' not found in data")
        output_path = os.path.join(output_dir, '08_target_leakage_cleaned.csv')
        data.to_csv(output_path, index=False)
        return data
    
    target = data['is_canceled']
    columns_to_remove = []
    correlation_results = []  # Store results for reporting
    
    # Check each column for correlation with target
    for col in data.columns:
        if col == 'is_canceled':
            continue
            
        try:
            # Get dtype as string to handle all pandas dtypes safely
            dtype_str = str(data[col].dtype)
            
            # For numerical columns (including nullable integers)
            if (dtype_str in ['int64', 'float64', 'Int64', 'Float64'] or 
                any(num_type in dtype_str.lower() for num_type in ['int', 'float'])):
                
                # Handle missing values by filling with a neutral value
                col_filled = data[col].fillna(data[col].median() if data[col].notna().any() else 0)
                
                # Convert to float64 to ensure compatibility with corrcoef
                col_filled = col_filled.astype('float64')
                target_filled = target.astype('float64')
                
                # Skip if all values are the same (would cause division by zero)
                if col_filled.std() == 0:
                    print(f"Skipping {col} - all values are identical")
                    continue
                    
                correlation = np.corrcoef(col_filled, target_filled)[0, 1]
                
                # Check for correlation above threshold
                if not np.isnan(correlation) and abs(correlation) >= correlation_threshold:
                    columns_to_remove.append(col)
                    correlation_results.append({
                        'column': col,
                        'type': 'numeric',
                        'correlation': correlation,
                        'abs_correlation': abs(correlation)
                    })
                    print(f"Found high correlation ({correlation:.6f}) between {col} and target")
            
            # For categorical columns
            elif 'category' in dtype_str.lower() or dtype_str == 'object':
                # Calculate correlation using point-biserial correlation for categorical vs binary
                try:
                    # Create dummy variables for categorical column
                    dummies = pd.get_dummies(data[col], prefix=col, dummy_na=True)
                    
                    # Calculate correlation for each dummy variable
                    max_correlation = 0
                    best_category = None
                    
                    for dummy_col in dummies.columns:
                        dummy_correlation = np.corrcoef(dummies[dummy_col], target)[0, 1]
                        if not np.isnan(dummy_correlation) and abs(dummy_correlation) > abs(max_correlation):
                            max_correlation = dummy_correlation
                            best_category = dummy_col
                    
                    # Check if maximum correlation exceeds threshold
                    if abs(max_correlation) >= correlation_threshold:
                        columns_to_remove.append(col)
                        correlation_results.append({
                            'column': col,
                            'type': 'categorical',
                            'correlation': max_correlation,
                            'abs_correlation': abs(max_correlation),
                            'best_category': best_category
                        })
                        print(f"Found high categorical correlation ({max_correlation:.6f}) between {col} and target (via {best_category})")
                
                except Exception as e:
                    print(f"Warning: Could not calculate categorical correlation for {col}: {e}")
                    
                    # Fallback: Check for perfect mapping (original logic)
                    if data[col].notna().sum() > 0:
                        # Group by categorical column and calculate mean target for each category
                        grouped = data.groupby(data[col].astype(str), dropna=False)['is_canceled'].agg(['mean', 'count'])
                        
                        # Check if any category has mean >= threshold or <= (1-threshold)
                        # This indicates strong predictive power
                        extreme_mapping = (grouped['mean'] >= (1 - correlation_threshold)) | (grouped['mean'] <= correlation_threshold)
                        
                        if extreme_mapping.any():
                            # Calculate a pseudo-correlation based on the most extreme category
                            most_extreme_mean = grouped['mean'].iloc[np.argmax(np.abs(grouped['mean'] - 0.5))]
                            pseudo_correlation = abs(most_extreme_mean - 0.5) * 2  # Scale to [0,1]
                            
                            if pseudo_correlation >= correlation_threshold:
                                columns_to_remove.append(col)
                                correlation_results.append({
                                    'column': col,
                                    'type': 'categorical_fallback',
                                    'correlation': most_extreme_mean,
                                    'abs_correlation': pseudo_correlation,
                                    'extreme_categories': grouped[extreme_mapping].index.tolist()
                                })
                                print(f"Found high categorical mapping between {col} and target (pseudo-correlation: {pseudo_correlation:.6f})")
            
            # Special check for reservation_status which is likely to be leaky
            if col == 'reservation_status':
                # Check if 'Canceled' status has high correlation
                unique_values = data[col].astype(str).unique()
                if any('cancel' in str(val).lower() for val in unique_values):
                    # Find the canceled status value
                    canceled_values = [val for val in unique_values if 'cancel' in str(val).lower()]
                    
                    for canceled_val in canceled_values:
                        canceled_status_mask = data[col].astype(str) == str(canceled_val)
                        if canceled_status_mask.sum() > 0:
                            # Calculate correlation between this status and target
                            status_correlation = np.corrcoef(canceled_status_mask.astype(int), target)[0, 1]
                            
                            if abs(status_correlation) >= correlation_threshold:
                                if col not in columns_to_remove:  # Avoid duplicates
                                    columns_to_remove.append(col)
                                    correlation_results.append({
                                        'column': col,
                                        'type': 'special_reservation_status',
                                        'correlation': status_correlation,
                                        'abs_correlation': abs(status_correlation),
                                        'leaky_value': canceled_val
                                    })
                                print(f"Found high leakage in {col} - '{canceled_val}' status has correlation {status_correlation:.6f} with target")
                                break
        
        except Exception as e:
            print(f"Warning: Could not check correlation for column {col}: {e}")
            continue
    
    # ============================================================================
    # DETAILED REPORTING
    # ============================================================================
    print("\n" + "="*80)
    print("TARGET LEAKAGE DETECTION RESULTS")
    print("="*80)
    
    if correlation_results:
        print(f"\nFound {len(correlation_results)} columns with correlation >= {correlation_threshold}:")
        
        # Sort by absolute correlation (highest first)
        correlation_results.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        print(f"\n{'Column':<25} {'Type':<20} {'Correlation':<12} {'Details':<30}")
        print("-" * 87)
        
        for result in correlation_results:
            details = ""
            if result['type'] == 'categorical' and 'best_category' in result:
                details = f"via {result['best_category']}"
            elif result['type'] == 'categorical_fallback' and 'extreme_categories' in result:
                details = f"extreme cats: {len(result['extreme_categories'])}"
            elif result['type'] == 'special_reservation_status' and 'leaky_value' in result:
                details = f"leaky value: {result['leaky_value']}"
            
            print(f"{result['column']:<25} {result['type']:<20} {result['correlation']:<12.6f} {details:<30}")
        
        # Group by correlation strength
        perfect_corr = [r for r in correlation_results if r['abs_correlation'] >= 0.99]
        very_high_corr = [r for r in correlation_results if 0.8 <= r['abs_correlation'] < 0.99]
        high_corr = [r for r in correlation_results if 0.5 <= r['abs_correlation'] < 0.8]
        moderate_corr = [r for r in correlation_results if correlation_threshold <= r['abs_correlation'] < 0.5]
        
        print("\n📊 Correlation Strength Distribution:")
        if perfect_corr:
            print(f"  • Perfect (>=0.99): {len(perfect_corr)} columns")
        if very_high_corr:
            print(f"  • Very High (0.8-0.99): {len(very_high_corr)} columns")
        if high_corr:
            print(f"  • High (0.5-0.8): {len(high_corr)} columns")
        if moderate_corr:
            print(f"  • Moderate ({correlation_threshold}-0.5): {len(moderate_corr)} columns")
        
        # Show feature types
        numeric_features = [r for r in correlation_results if r['type'] == 'numeric']
        categorical_features = [r for r in correlation_results if r['type'] in ['categorical', 'categorical_fallback']]
        special_features = [r for r in correlation_results if r['type'] == 'special_reservation_status']
        
        print("\n🔍 Feature Type Distribution:")
        if numeric_features:
            print(f"  • Numeric: {len(numeric_features)} columns")
        if categorical_features:
            print(f"  • Categorical: {len(categorical_features)} columns")
        if special_features:
            print(f"  • Special (reservation_status): {len(special_features)} columns")
    
    # Remove leaky columns
    if columns_to_remove:
        # Remove duplicates while preserving order
        columns_to_remove = list(dict.fromkeys(columns_to_remove))
        
        data = data.drop(columns=columns_to_remove)
        print(f"\n✅ Removed {len(columns_to_remove)} columns with correlation >= {correlation_threshold}")
        print(f"Removed columns: {columns_to_remove}")
    else:
        print(f"\n✅ No columns with correlation >= {correlation_threshold} found")
    
    print(f"\n📊 Data shape after target leakage removal: {data.shape}")
    
    # Save to file
    pickle_path = os.path.join(output_dir, '08_target_leakage_cleaned.pkl')
    data.to_pickle(pickle_path)
    print(f"📁 Data with preserved dtypes saved to: {pickle_path}")
    
    return data