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
    drop_threshold = 0.5
    
    # Drop columns with excessive missing values
    columns_to_drop = [col for col in data.columns if data[col].isnull().mean() > drop_threshold]
    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)
        print(f"Dropped {len(columns_to_drop)} columns with >50% missing values: {columns_to_drop}")
    
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
    """7. Handle data types and save result"""
    
    print("Cleaning data types...")
    
    # Handle date-related columns
    if 'arrival_date_month' in data.columns:
        data['arrival_date_month'] = data['arrival_date_month'].str.title()
    
    # Handle reservation_status_date
    if 'reservation_status_date' in data.columns:
        data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'], errors='coerce')
    
    # Binary columns
    binary_columns = ['is_repeated_guest', 'is_canceled']
    for col in binary_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype('int8')
    
    # Categorical columns
    categorical_columns = {
        'hotel': lambda x: x.str.title(),
        'meal': lambda x: x.str.upper(),
        'country': lambda x: x.str.upper(),
        'market_segment': lambda x: x,
        'distribution_channel': lambda x: x,
        'reserved_room_type': lambda x: x.str.upper(),
        'assigned_room_type': lambda x: x.str.upper(),
        'deposit_type': lambda x: x.str.title(),
        'customer_type': lambda x: x.str.title(),
        'reservation_status': lambda x: x.str.title()
    }
    
    for col, transform_func in categorical_columns.items():
        if col in data.columns:
            data[col] = transform_func(data[col])
    
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
    
    # Convert to category for memory efficiency
    category_columns = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel',
                       'reserved_room_type', 'assigned_room_type', 'deposit_type',
                       'customer_type', 'reservation_status']
    for col in category_columns:
        if col in data.columns:
            data[col] = data[col].astype('category')
    
    # Save to file
    output_path = os.path.join(output_dir, '07_dtypes_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data

def clean_target_leakage(data):
    """8. Detect and remove columns that have perfect correlation with target variable (is_canceled)"""
    
    print("Detecting target leakage...")
    
    if 'is_canceled' not in data.columns:
        print("Warning: Target variable 'is_canceled' not found in data")
        output_path = os.path.join(output_dir, '08_target_leakage_cleaned.csv')
        data.to_csv(output_path, index=False)
        return data
    
    target = data['is_canceled']
    columns_to_remove = []
    
    # Check each column for perfect correlation with target
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
                
                # Check for perfect correlation (both exact and floating point precision)
                if not np.isnan(correlation) and (abs(correlation) == 1.0 or abs(correlation) > 0.999):
                    columns_to_remove.append(col)
                    print(f"Found perfect correlation ({correlation:.6f}) between {col} and target")
            
            # For categorical columns
            elif 'category' in dtype_str.lower() or dtype_str == 'object':
                # Check if there's a perfect one-to-one mapping
                if data[col].notna().sum() > 0:
                    # Group by categorical column and calculate mean target for each category
                    grouped = data.groupby(data[col].astype(str), dropna=False)['is_canceled'].agg(['mean', 'count'])
                    
                    # If mean is 0.0 or 1.0, it means ALL instances of that 
                    # category have the same target value (perfect predictability)
                    perfect_mapping = grouped['mean'].isin([0.0, 1.0]).all()
                    
                    # Also check if the mapping is deterministic in both directions
                    if perfect_mapping:
                        # Check if each target value maps to unique categories
                        reverse_grouped = data.groupby('is_canceled')[col].nunique()
                        total_unique_values = data[col].nunique()
                        
                        # If sum of unique values per target class equals 
                        # total unique values, there's no overlap (perfect separation)
                        if reverse_grouped.sum() == total_unique_values:
                            columns_to_remove.append(col)
                            print(f"Found perfect categorical mapping between {col} and target")
            
            # Special check for reservation_status which is likely to be leaky
            if col == 'reservation_status':
                # Check if 'Canceled' status perfectly predicts cancellation
                unique_values = data[col].astype(str).unique()
                if any('cancel' in str(val).lower() for val in unique_values):
                    # Find the canceled status value
                    canceled_values = [val for val in unique_values if 'cancel' in str(val).lower()]
                    
                    for canceled_val in canceled_values:
                        canceled_status_mask = data[col].astype(str) == str(canceled_val)
                        if canceled_status_mask.sum() > 0:
                            # Check if all 'Canceled' reservations have is_canceled = 1
                            if (data.loc[canceled_status_mask, 'is_canceled'] == 1).all():
                                # And check if all is_canceled = 1 have 'Canceled' status
                                canceled_target_mask = data['is_canceled'] == 1
                                if (data.loc[canceled_target_mask, col].astype(str) == str(canceled_val)).all():
                                    columns_to_remove.append(col)
                                    print(f"Found perfect leakage in {col} - '{canceled_val}' status perfectly predicts target")
                                    break
        
        except Exception as e:
            print(f"Warning: Could not check correlation for column {col}: {e}")
            continue
    
    # Remove leaky columns
    if columns_to_remove:
        data = data.drop(columns=columns_to_remove)
        print(f"Removed {len(columns_to_remove)} columns with target leakage: {columns_to_remove}")
    else:
        print("No columns with perfect target correlation found")
    
    # Save to file
    output_path = os.path.join(output_dir, '08_target_leakage_cleaned.csv')
    data.to_csv(output_path, index=False)
    
    return data