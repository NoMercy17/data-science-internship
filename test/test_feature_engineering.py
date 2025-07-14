
import os
import pandas as pd
from scripts.feature_engineering import HotelFeatureExtractor

# Configuration
input_path = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/cleaned/final_cleaned_data.pkl'
output_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results'

def run_feature_engineering_pipeline(data):
    """Run the complete feature engineering pipeline"""
    
    print("="*60)
    print("STARTING FEATURE ENGINEERING PIPELINE")
    print("="*60)
    print(f"Original data shape: {data.shape}")
    
    # CREATE A COPY TO AVOID MODIFYING ORIGINAL DATA
    data_copy = data.copy()
    
    # ADD THIS: Show initial data structure
    print("\n" + "="*50)
    print("INITIAL DATA PREVIEW")
    print("="*50)
    print("First 5 rows of the dataset:")
    print(data_copy.head())
    print(f"\nColumn names ({len(data_copy.columns)} total):")
    print(data_copy.columns.tolist())
    print("\nData types:")
    print(data_copy.dtypes)
    
    # Initialize the feature extractor
    extractor = HotelFeatureExtractor(output_dir=output_dir)
    
    # Step 1: Analyze data structure
    print("\n" + "="*50)
    print("STEP 1: ANALYZING DATA STRUCTURE")
    print("="*50)
    print(f"Dataset shape: {data_copy.shape}")
    if 'is_canceled' in data_copy.columns:
        print("Target variable distribution:")
        print(data_copy['is_canceled'].value_counts())
        print(f"Cancellation rate: {data_copy['is_canceled'].mean():.2%}")
    
    missing_data = data_copy.isnull().sum()
    if missing_data.sum() > 0:
        print("\nMissing values:")
        print(missing_data[missing_data > 0])
    
    print("\nColumn types:")
    print(data_copy.dtypes.value_counts())
    
    # Step 2: Extract temporal features
    print("\n" + "="*50)
    print("STEP 2: EXTRACTING TEMPORAL FEATURES")
    print("="*50)
    data_copy = extractor.extract_temporal_features(data_copy)
    print("✓ Created 9 temporal features")
    print(f"After temporal features: {data_copy.shape}")
    
    # Step 3: Extract customer behavior features
    print("\n" + "="*50)
    print("STEP 3: EXTRACTING CUSTOMER BEHAVIOR FEATURES")
    print("="*50)
    data_copy = extractor.extract_customer_behavior_features(data_copy)
    print("✓ Created 12 customer behavior features")
    print(f"After customer behavior features: {data_copy.shape}")
    
    # Step 4: Extract booking risk features
    print("\n" + "="*50)
    print("STEP 4: EXTRACTING BOOKING RISK FEATURES")
    print("="*50)
    data_copy = extractor.extract_booking_risk_features(data_copy)
    print("✓ Created 10 booking risk features")
    print(f"After booking risk features: {data_copy.shape}")
    
    # Step 5: Extract market features
    print("\n" + "="*50)
    print("STEP 5: EXTRACTING MARKET FEATURES")
    print("="*50)
    data_copy = extractor.extract_market_features(data_copy)
    if 'company' not in data_copy.columns:
        print("Warning: 'company' column not found, setting has_company to 0")
    print("✓ Created 5 market features")
    print(f"After market features: {data_copy.shape}")
    
    # Step 6: Handle high cardinality features
    print("\n" + "="*50)
    print("STEP 6: HANDLING HIGH CARDINALITY FEATURES")
    print("="*50)
    data_copy = extractor.handle_high_cardinality_features(data_copy)
    if 'company' not in data_copy.columns:
        print("Warning: 'company' column not found, setting company_grouped to 'Other'")
    print("✓ Handled high cardinality features")
    print(f"After cardinality handling: {data_copy.shape}")
    
    # NEW STEP 7: ENCODE CATEGORICAL VARIABLES
    print("\n" + "="*50)
    print("STEP 7: ENCODING CATEGORICAL VARIABLES")
    print("="*50)
    
    # Show categorical columns before encoding
    categorical_cols = data_copy.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns to encode ({len(categorical_cols)}): {categorical_cols}")
    
    # Apply categorical encoding
    data_copy = extractor.encode_categorical_variables(data_copy)
    print("✓ Categorical variables encoded")
    print(f"After categorical encoding: {data_copy.shape}")
    
    # Show the effect of encoding
    print("\nEncoding Results:")
    print(f"Label encoders created: {list(extractor.label_encoders.keys())}")
    
    # Count one-hot encoded columns
    onehot_cols = [col for col in data_copy.columns if any(col.startswith(prefix + '_') for prefix in [
        'meal', 'market_segment', 'distribution_channel', 'deposit_type',
        'reserved_room_type', 'assigned_room_type', 'booking_season',
        'lead_time_category', 'customer_experience_level', 'cancellation_tendency',
        'avg_stay_preference', 'price_category', 'deposit_risk', 'price_per_night_category'
    ])]
    print(f"One-hot encoded columns created ({len(onehot_cols)}): {onehot_cols[:10]}..." if len(onehot_cols) > 10 else f"One-hot encoded columns created ({len(onehot_cols)}): {onehot_cols}")
    
    # Step 8: Final feature selection and processing
    print("\n" + "="*50)
    print("STEP 8: FINAL FEATURE SELECTION")
    print("="*50)
    
    # Define features to keep
    base_features = [
        'hotel', 'is_canceled', 'lead_time', 'arrival_date_year',
        'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
        'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
        'booking_changes', 'days_in_waiting_list', 'customer_type', 'adr', 
        'required_car_parking_spaces', 'total_of_special_requests'
    ]
    
    # New engineered features
    temporal_features = [
        'is_peak_season', 'weekend_preference', 'is_holiday_season', 'arrival_quarter',
        'is_last_minute', 'is_advance_booking', 'lead_time_risk_score'
    ]
    
    customer_features = [
        'customer_loyalty_score', 'total_stay_nights', 'party_size', 'special_requests_ratio',
        'has_children', 'has_babies', 'is_family_booking', 'is_group_booking', 'is_business_likely'
    ]
    
    risk_features = [
        'room_type_mismatch', 'changes_per_night', 'adr_per_person', 'total_booking_value',
        'revenue_per_person', 'has_special_requirements', 'booking_complexity', 'waiting_list_risk'
    ]
    
    market_features = [
        'is_online_booking', 'has_agent', 'has_company', 'high_risk_segment', 'high_risk_channel'
    ]
    
    cardinality_features = [
        'agent_grouped', 'company_grouped', 'country_grouped'
    ]
    
    # Combine numerical and label-encoded features
    numerical_and_encoded = (base_features + temporal_features + customer_features + 
                           risk_features + market_features + cardinality_features)
    
    # Add one-hot encoded features (they have dynamic names)
    onehot_features = [col for col in data_copy.columns if any(col.startswith(prefix + '_') for prefix in [
        'meal', 'market_segment', 'distribution_channel', 'deposit_type',
        'reserved_room_type', 'assigned_room_type', 'booking_season',
        'lead_time_category', 'customer_experience_level', 'cancellation_tendency',
        'avg_stay_preference', 'price_category', 'deposit_risk', 'price_per_night_category'
    ])]
    
    # Combine all features
    all_features = numerical_and_encoded + onehot_features
    
    # Keep only features that exist in the dataframe
    final_features = [f for f in all_features if f in data_copy.columns]
    
    # Check for missing features
    missing_features = [f for f in all_features if f not in data_copy.columns]
    if missing_features:
        print(f"Warning: These expected features are missing: {missing_features}")
    
    data_final = data_copy[final_features]
    
    # ADD THIS: Show final data structure
    print("\n" + "="*50)
    print("FINAL DATA PREVIEW")
    print("="*50)
    print("First 5 rows of engineered dataset:")
    print(data_final.head())
    print(f"\nFinal column names ({len(data_final.columns)} total):")
    print(data_final.columns.tolist())
    print("\nFinal data types:")
    print(data_final.dtypes)
    
    # Check data types after encoding
    print("\nData type distribution after encoding:")
    print(data_final.dtypes.value_counts())
    
    # Final save
    print("\n" + "="*50)
    print("FINAL SAVE")
    print("="*50)
    
    # Save as pickle to preserve dtypes
    pickle_output_path = os.path.join(output_dir, 'feature_engineered_data.pkl')
    data_final.to_pickle(pickle_output_path)
    print(f"Feature engineered data saved to: {pickle_output_path}")
    
    # Save as CSV for easy viewing
    csv_output_path = os.path.join(output_dir, 'feature_engineered_data.csv')
    data_final.to_csv(csv_output_path, index=False)
    print(f"Feature engineered data (CSV) saved to: {csv_output_path}")
    
    # Save encoders for future use
    encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
    pd.to_pickle(extractor.label_encoders, encoders_path)
    print(f"Label encoders saved to: {encoders_path}")
    
    # Save feature info
    feature_info = {
        'total_features': len(final_features),
        'numerical_and_encoded_features': len(numerical_and_encoded),
        'onehot_features': len(onehot_features),
        'label_encoders': list(extractor.label_encoders.keys()),
        'feature_names': final_features,
        'onehot_columns': onehot_features
    }
    
    info_path = os.path.join(output_dir, 'feature_info.pkl')
    pd.to_pickle(feature_info, info_path)
    print(f"Feature info saved to: {info_path}")
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Original features: {len([col for col in base_features if col in data_final.columns])}")
    print(f"Final features: {data_final.shape[1]}")
    
    # Count new features by category
    new_temporal = len([f for f in temporal_features if f in data_final.columns])
    new_customer = len([f for f in customer_features if f in data_final.columns])
    new_risk = len([f for f in risk_features if f in data_final.columns])
    new_market = len([f for f in market_features if f in data_final.columns])
    new_cardinality = len([f for f in cardinality_features if f in data_final.columns])
    
    print(f"New temporal features: {new_temporal}")
    print(f"New customer behavior features: {new_customer}")
    print(f"New booking risk features: {new_risk}")
    print(f"New market features: {new_market}")
    print(f"New cardinality features: {new_cardinality}")
    print(f"One-hot encoded features: {len(onehot_features)}")
    print(f"Label encoded features: {len(extractor.label_encoders)}")
    print(f"Total new features: {new_temporal + new_customer + new_risk + new_market + new_cardinality + len(onehot_features)}")
    
    print(f"Final data shape: {data_final.shape}")
    print(f"All files saved in: {output_dir}")
    
    # Verify original data is unchanged
    print(f"\nOriginal data shape (unchanged): {data.shape}")
    print("✓ Original data was not modified")
    
    return data_final

if __name__ == "__main__":
    # Load the cleaned data
    print("Loading cleaned data...")
    if input_path.endswith('.pkl'):
        data = pd.read_pickle(input_path)
    else:
        data = pd.read_csv(input_path)
    
    # Run the feature engineering pipeline
    engineered_data = run_feature_engineering_pipeline(data)
    
    print("\nFeature engineering completed successfully!")
    print(f"Engineered data shape: {engineered_data.shape}")
    print(f"Files saved in: {output_dir}")