
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
    
    # Run the feature engineering pipelineimport pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

class HotelFeatureExtractor:
    """
    Complete hotel booking feature engineering pipeline - UPDATED VERSION
    """
    
    def __init__(self, output_dir='/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results'):
        self.feature_info = {}
        self.output_dir = output_dir
        self.label_encoders = {}
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_temporal_features(self, data):
        """Extract temporal features for cancellation prediction"""
        data_temp = data.copy()
        
        # 1. Booking Season
        def get_season(month):
            if month in ['December', 'January', 'February']:
                return 'Winter'
            elif month in ['March', 'April', 'May']:
                return 'Spring'
            elif month in ['June', 'July', 'August']:
                return 'Summer'
            else:
                return 'Fall'
        
        data_temp['booking_season'] = data_temp['arrival_date_month'].apply(get_season)
        
        # 2. Lead Time Categories
        def categorize_lead_time(lead_time):
            if lead_time <= 7:
                return 'Short'  
            elif lead_time <= 30:
                return 'Medium'   
            elif lead_time <= 90:
                return 'Long' 
            else:
                return 'Very_Long'
        
        data_temp['lead_time_category'] = data_temp['lead_time'].apply(categorize_lead_time)
        
        # 3. Peak Season Indicator
        peak_months = ['July', 'August', 'December']
        data_temp['is_peak_season'] = data_temp['arrival_date_month'].isin(peak_months).astype(int)
        
        # 4. Weekend Preference
        data_temp['weekend_preference'] = (
            data_temp['stays_in_weekend_nights'] / 
            (data_temp['stays_in_weekend_nights'] + data_temp['stays_in_week_nights'] + 0.001)
        )
        
        # 5. Holiday Season Indicator
        holiday_months = ['December', 'January', 'July', 'August']
        data_temp['is_holiday_season'] = data_temp['arrival_date_month'].isin(holiday_months).astype(int)
        
        # 6. Quarter mapping
        month_to_quarter = {
            'January': 1, 'February': 1, 'March': 1,
            'April': 2, 'May': 2, 'June': 2,
            'July': 3, 'August': 3, 'September': 3,
            'October': 4, 'November': 4, 'December': 4
        }
        data_temp['arrival_quarter'] = data_temp['arrival_date_month'].map(month_to_quarter)
        
        # 7. Last minute booking indicator
        data_temp['is_last_minute'] = (data_temp['lead_time'] <= 7).astype(int)
        
        # 8. Advance booking indicator
        data_temp['is_advance_booking'] = (data_temp['lead_time'] > 90).astype(int)
        
        # 9. Lead time risk score
        data_temp['lead_time_risk_score'] = np.where(
            data_temp['lead_time'] <= 7, 3,  # High risk
            np.where(data_temp['lead_time'] <= 30, 2,  # Medium risk
                     np.where(data_temp['lead_time'] <= 90, 1, 0))  # Low risk
        )
        
        return data_temp
    
    def extract_customer_behavior_features(self, data):
        """Extract customer behavior features"""
        df_behavior = data.copy()
        
        # 1. Customer Loyalty Score
        def calculate_loyalty_score(previous_bookings, previous_cancellations, is_repeated):
            if previous_bookings == 0:
                return 0
            
            base_score = previous_bookings * 10
            penalty = previous_cancellations * 5
            bonus = 20 if is_repeated else 0
            
            return max(0, base_score - penalty + bonus)
        
        df_behavior['customer_loyalty_score'] = df_behavior.apply(
            lambda row: calculate_loyalty_score(
                row['previous_bookings_not_canceled'],
                row['previous_cancellations'],
                row['is_repeated_guest']
            ), axis=1
        )
        
        # 2. Total Stay Duration
        df_behavior['total_stay_nights'] = (
            df_behavior['stays_in_weekend_nights'] + 
            df_behavior['stays_in_week_nights']
        )
        
        # 3. Party Size
        df_behavior['party_size'] = (
            df_behavior['adults'] + 
            df_behavior['children'] + 
            df_behavior['babies']
        )
        
        # 4. Special Requests Intensity
        df_behavior['special_requests_ratio'] = (
            df_behavior['total_of_special_requests'] / 
            (df_behavior['total_stay_nights'] + 1)
        )
        
        # 5. Customer Type Indicators
        df_behavior['has_children'] = (df_behavior['children'] > 0).astype(int)
        df_behavior['has_babies'] = (df_behavior['babies'] > 0).astype(int)
        df_behavior['is_family_booking'] = (
            (df_behavior['has_children'] == 1) | 
            (df_behavior['has_babies'] == 1) | 
            (df_behavior['adults'] >= 2)
        ).astype(int)
        df_behavior['is_group_booking'] = (df_behavior['party_size'] >= 4).astype(int)
        
        # 6. Business vs Leisure indicators
        market_segment_str = df_behavior['market_segment'].astype(str)
        distribution_channel_str = df_behavior['distribution_channel'].astype(str)
        
        df_behavior['is_business_likely'] = (
            (market_segment_str.str.contains('Corporate', case=False, na=False)) | 
            (distribution_channel_str.str.contains('Corporate', case=False, na=False)) |
            (df_behavior['stays_in_week_nights'] > df_behavior['stays_in_weekend_nights'])
        ).astype(int)
        
        # 7. Customer experience level
        df_behavior['customer_experience_level'] = np.where(
            df_behavior['is_repeated_guest'] == 1, 'Experienced',
            np.where(df_behavior['previous_bookings_not_canceled'] > 0, 'Returning', 'New')
        )
        
        # 8. Cancellation tendency
        df_behavior['cancellation_tendency'] = np.where(
            df_behavior['previous_cancellations'] == 0, 'Low',
            np.where(df_behavior['previous_cancellations'] <= 2, 'Medium', 'High')
        )
        
        # 9. Average stay preference
        df_behavior['avg_stay_preference'] = np.where(
            df_behavior['total_stay_nights'] <= 2, 'Short',
            np.where(df_behavior['total_stay_nights'] <= 7, 'Medium', 'Extended')
        )
        
        return df_behavior
    
    def extract_booking_risk_features(self, data):
        """Extract booking risk features"""
        data_risk = data.copy()
        
        # 1. Room Type Mismatch
        reserved_room_str = data_risk['reserved_room_type'].astype(str)
        assigned_room_str = data_risk['assigned_room_type'].astype(str)
        data_risk['room_type_mismatch'] = (reserved_room_str != assigned_room_str).astype(int)
        
        # 2. Booking Changes Intensity
        data_risk['changes_per_night'] = (
            data_risk['booking_changes'] / 
            (data_risk['total_stay_nights'] + 1)
        )
        
        # 3. Price Per Person
        data_risk['adr_per_person'] = (
            data_risk['adr'] / 
            (data_risk['party_size'] + 1)
        )
        
        # 4. Financial Features
        data_risk['total_booking_value'] = data_risk['adr'] * data_risk['total_stay_nights']
        data_risk['revenue_per_person'] = data_risk['total_booking_value'] / (data_risk['party_size'] + 1)
        
        # 5. Price Category (SINGLE VERSION - removed duplicate)
        data_risk['price_category'] = pd.cut(
            data_risk['adr'], 
            bins=[0, 75, 150, 300, float('inf')], 
            labels=['Budget', 'Standard', 'Premium', 'Luxury']
        )
        
        # 6. Special Requirements
        data_risk['has_special_requirements'] = (
            (data_risk['total_of_special_requests'] > 0) |
            (data_risk['required_car_parking_spaces'] > 0) |
            (data_risk['days_in_waiting_list'] > 0)
        ).astype(int)
        
        # 7. Booking Complexity Score
        data_risk['booking_complexity'] = (
            data_risk['booking_changes'] * 2 +
            data_risk['total_of_special_requests'] +
            (data_risk['days_in_waiting_list'] > 0).astype(int) * 3 +
            data_risk['room_type_mismatch'] * 2
        )
        
        # 8. Deposit risk indicator
        deposit_str = data_risk['deposit_type'].astype(str)
        data_risk['deposit_risk'] = np.where(
            deposit_str == 'No Deposit', 'High',
            np.where(deposit_str == 'Non Refund', 'Low', 'Medium')
        )
        
        return data_risk
    
    def handle_high_cardinality_features(self, data):
        """Handle high cardinality features by grouping rare categories"""
        data_processed = data.copy()
        
        # Handle agent column
        if 'agent' in data_processed.columns:
            # Group rare agents
            agent_counts = data_processed['agent'].value_counts()
            rare_agents = agent_counts[agent_counts < 10].index  # Agents with less than 10 bookings
            data_processed['agent_grouped'] = data_processed['agent'].apply(
                lambda x: 'Other' if x in rare_agents else x
            )
            print(f"Agent column: {len(agent_counts)} unique values -> {data_processed['agent_grouped'].nunique()} after grouping")
        else:
            data_processed['agent_grouped'] = 'No_Agent'
        
        # Handle company column
        if 'company' in data_processed.columns:
            # Group rare companies
            company_counts = data_processed['company'].value_counts()
            rare_companies = company_counts[company_counts < 5].index  # Companies with less than 5 bookings
            data_processed['company_grouped'] = data_processed['company'].apply(
                lambda x: 'Other' if x in rare_companies else x
            )
            print(f"Company column: {len(company_counts)} unique values -> {data_processed['company_grouped'].nunique()} after grouping")
        else:
            data_processed['company_grouped'] = 'No_Company'
        
        # Handle country column
        if 'country' in data_processed.columns:
            # Group rare countries
            country_counts = data_processed['country'].value_counts()
            rare_countries = country_counts[country_counts < 20].index  # Countries with less than 20 bookings
            data_processed['country_grouped'] = data_processed['country'].apply(
                lambda x: 'Other' if x in rare_countries else x
            )
            print(f"Country column: {len(country_counts)} unique values -> {data_processed['country_grouped'].nunique()} after grouping")
        else:
            data_processed['country_grouped'] = 'Unknown'
        
        # Add features based on grouped columns
        data_processed['has_agent'] = (data_processed['agent_grouped'] != 'No_Agent').astype(int)
        data_processed['has_company'] = (data_processed['company_grouped'] != 'No_Company').astype(int)
        
        # Drop original high cardinality columns
        columns_to_drop = ['agent', 'company', 'country']
        for col in columns_to_drop:
            if col in data_processed.columns:
                data_processed = data_processed.drop(columns=[col])
        
        return data_processed

    def extract_market_features(self, data):
        """Extract market and distribution features - UPDATED for your actual data"""
        data_market = data.copy()
        
        # Convert categorical columns to string for comparison
        distribution_channel_str = data_market['distribution_channel'].astype(str)
        market_segment_str = data_market['market_segment'].astype(str)
        
        # 1. Online vs Offline booking
        # Updated based on your actual distribution channels: TA/TO, Direct, Corporate
        online_channels = ['Direct']  # Direct bookings are typically online
        data_market['is_online_booking'] = (
            distribution_channel_str.isin(online_channels)
        ).astype(int)
        
        # 2. Market segment risk - UPDATED
        # Based on typical hotel industry knowledge:
        # - Groups tend to have higher cancellation rates
        # - Corporate can be more stable but depends on company policies
        # You should analyze your actual cancellation rates by segment to determine this
        print("Market segments found:", market_segment_str.unique())
        
        # Create a more flexible approach - you can adjust these based on your analysis
        high_risk_segments = ['Groups']  # Start with Groups as typically higher risk
        # Add 'Offline TA/TO' if it exists in market_segment
        if 'Offline TA/TO' in market_segment_str.unique():
            high_risk_segments.append('Offline TA/TO')
        
        data_market['high_risk_segment'] = (
            market_segment_str.isin(high_risk_segments)
        ).astype(int)
        
        # 3. Distribution channel risk - UPDATED
        # Based on your actual channels: TA/TO, Direct, Corporate
        # TA/TO (Travel Agent/Tour Operator) typically have higher cancellation risk
        high_risk_channels = ['TA/TO']
        data_market['high_risk_channel'] = (
            distribution_channel_str.isin(high_risk_channels)
        ).astype(int)
        
        # 4. Travel Agent booking indicator - UPDATED
        data_market['is_travel_agent_booking'] = (
            distribution_channel_str.str.contains('TA/TO', case=False, na=False)
        ).astype(int)
        
        # 5. Group booking indicator - UPDATED
        # Check both distribution channel and market segment for groups
        data_market['is_group_channel'] = (
            (distribution_channel_str.str.contains('Group', case=False, na=False)) |
            (market_segment_str.str.contains('Group', case=False, na=False))
        ).astype(int)
        
        # 6. Corporate booking indicator - NEW
        data_market['is_corporate_booking'] = (
            (distribution_channel_str.str.contains('Corporate', case=False, na=False)) |
            (market_segment_str.str.contains('Corporate', case=False, na=False))
        ).astype(int)
        
        # 7. Direct booking indicator - NEW
        data_market['is_direct_booking'] = (
            distribution_channel_str.str.contains('Direct', case=False, na=False)
        ).astype(int)
        
        return data_market
    
    def encode_categorical_variables(self, data):
        """
        Encode categorical variables using both label encoding and one-hot encoding
        """
        df = data.copy()
        
        # Binary categorical columns - label encoding (0, 1)
        binary_categorical = [
            'hotel', 'is_repeated_guest', 'customer_type'
        ]
        
        # Use one-hot encoding (create dummy variables)
        onehot_categorical = [
            'meal', 'market_segment', 'distribution_channel', 'deposit_type',
            'reserved_room_type', 'assigned_room_type', 'booking_season',
            'lead_time_category', 'customer_experience_level', 'cancellation_tendency',
            'avg_stay_preference', 'price_category', 'deposit_risk'
        ]
        
        # Label encoding for remaining categorical columns
        label_categorical = [
            'arrival_date_month'
        ]
        
        # Add grouped columns created by handle_high_cardinality_features
        if 'agent_grouped' in df.columns:
            label_categorical.append('agent_grouped')
        if 'company_grouped' in df.columns:
            label_categorical.append('company_grouped')
        if 'country_grouped' in df.columns:
            label_categorical.append('country_grouped')
        
        # Label Encoding 
        for col in binary_categorical + label_categorical:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"Label encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # One-hot encoding with dummies
        created_dummies = []
        for col in onehot_categorical:
            if col in df.columns:
                # Get unique values before encoding
                unique_values = df[col].unique()
                print(f"Creating dummy variables for {col}: {unique_values}")
                
                # Create dummy variables (drop_first=True to avoid multicollinearity)
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                created_dummies.extend(dummies.columns.tolist())
                
                # Add dummy columns to dataframe
                df = pd.concat([df, dummies], axis=1)
                
                # Drop original column
                df = df.drop(columns=[col])
                
                print(f"  Created columns: {dummies.columns.tolist()}")
        
        print(f"\nTotal dummy variables created: {len(created_dummies)}")
        
        return df
    
    def run_complete_feature_engineering(self, data):
        """
        Complete feature engineering pipeline - UPDATED VERSION
        """
        print("Starting feature engineering pipeline...")
        
        # First, let's examine the actual unique values in key columns
        print("\n=== DATA INSPECTION ===")
        key_columns = ['distribution_channel', 'market_segment']
        for col in key_columns:
            if col in data.columns:
                print(f"{col} unique values: {data[col].unique()}")
        
        # Step 1: Handle high cardinality features
        print("\n=== HANDLING HIGH CARDINALITY FEATURES ===")
        df = self.handle_high_cardinality_features(data)
        
        # Step 2: Extract temporal features
        print("\n=== TEMPORAL FEATURES ===")
        df = self.extract_temporal_features(df)
        
        # Step 3: Extract customer behavior features
        print("\n=== CUSTOMER BEHAVIOR FEATURES ===")
        df = self.extract_customer_behavior_features(df)
        
        # Step 4: Extract booking risk features
        print("\n=== BOOKING RISK FEATURES ===")
        df = self.extract_booking_risk_features(df)
        
        # Step 5: Extract market features
        print("\n=== MARKET FEATURES ===")
        df = self.extract_market_features(df)
        
        # Step 6: Encode categorical variables
        print("\n=== CATEGORICAL ENCODING ===")
        df = self.encode_categorical_variables(df)
        
        # Original numerical features
        numerical_features = [
            'lead_time', 'arrival_date_year', 'arrival_date_week_number', 
            'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 
            'adults', 'children', 'babies', 'previous_cancellations', 
            'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
            'adr', 'required_car_parking_spaces', 'total_of_special_requests'
        ]
        
        # Original categorical features (now encoded)
        encoded_categorical = [
            'hotel', 'is_repeated_guest', 'customer_type', 'arrival_date_month'
        ]
        
        # Add grouped columns if they exist
        if 'agent_grouped' in df.columns:
            encoded_categorical.append('agent_grouped')
        if 'company_grouped' in df.columns:
            encoded_categorical.append('company_grouped')
        if 'country_grouped' in df.columns:
            encoded_categorical.append('country_grouped')
        
        # Target variable
        target_feature = ['is_canceled'] if 'is_canceled' in df.columns else []
        
        # New engineered numerical features
        temporal_features = [
            'is_peak_season', 'weekend_preference', 'is_holiday_season', 
            'arrival_quarter', 'is_last_minute', 'is_advance_booking', 'lead_time_risk_score'
        ]
        
        customer_features = [
            'customer_loyalty_score', 'total_stay_nights', 'party_size', 'special_requests_ratio',
            'has_children', 'has_babies', 'is_family_booking', 'is_group_booking', 'is_business_likely'
        ]
        
        risk_features = [
            'room_type_mismatch', 'changes_per_night', 'adr_per_person', 'total_booking_value',
            'revenue_per_person', 'has_special_requirements', 'booking_complexity'
        ]
        
        # UPDATED market features
        market_features = [
            'is_online_booking', 'has_agent', 'has_company', 'high_risk_segment', 
            'high_risk_channel', 'is_travel_agent_booking', 'is_group_channel',
            'is_corporate_booking', 'is_direct_booking'  # Added new features
        ]
        
        # Combine all numerical features
        all_numerical_features = (numerical_features + encoded_categorical + temporal_features + 
                                customer_features + risk_features + market_features)
        
        # Add one-hot encoded features (they have dynamic names)
        onehot_features = [col for col in df.columns if any(col.startswith(prefix + '_') for prefix in [
            'meal', 'market_segment', 'distribution_channel', 'deposit_type',
            'reserved_room_type', 'assigned_room_type', 'booking_season',
            'lead_time_category', 'customer_experience_level', 'cancellation_tendency',
            'avg_stay_preference', 'price_category', 'deposit_risk'
        ])]
        
        # Combine all features
        final_features = target_feature + all_numerical_features + onehot_features
        
        # Keep only features that exist in the dataframe
        final_features = [f for f in final_features if f in df.columns]
        
        df_final = df[final_features]
        
        # Save feature engineering info
        self.feature_info = {
            'total_features': len(final_features),
            'numerical_features': len(all_numerical_features),
            'onehot_features': len(onehot_features),
            'label_encoders': list(self.label_encoders.keys()),
            'feature_names': final_features
        }
        
        # Save the processed dataset
        pickle_output_path = os.path.join(self.output_dir, 'feature_engineered_data.pkl')
        df_final.to_pickle(pickle_output_path)
        
        # Save encoders for future use
        encoders_path = os.path.join(self.output_dir, 'label_encoders.pkl')
        pd.to_pickle(self.label_encoders, encoders_path)
        
        # Save feature info
        info_path = os.path.join(self.output_dir, 'feature_info.pkl')
        pd.to_pickle(self.feature_info, info_path)
        
        print("\n=== FEATURE ENGINEERING COMPLETED ===")
        print(f"Total features: {self.feature_info['total_features']}")
        print(f"Numerical features: {self.feature_info['numerical_features']}")
        print(f"One-hot encoded features: {self.feature_info['onehot_features']}")
        print(f"Label encoded features: {len(self.feature_info['label_encoders'])}")
        
        return df_final, pickle_output_path
    engineered_data = run_feature_engineering_pipeline(data)
    
    print("\nFeature engineering completed successfully!")
    print(f"Engineered data shape: {engineered_data.shape}")
    print(f"Files saved in: {output_dir}")