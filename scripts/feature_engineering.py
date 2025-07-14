import pandas as pd
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
        
        # Handle agent column (already processed with NaN for unknown/rare)
        if 'agent' in data_processed.columns:
            # Check the data type and handle accordingly
            print(f"Agent column dtype: {data_processed['agent'].dtype}")
            print(f"Agent column unique values: {data_processed['agent'].unique()}")
            
            # Convert to string first to handle mixed types, then fill NaN
            data_processed['agent_grouped'] = data_processed['agent'].astype(str).replace('nan', 'No_Agent')
            
            # If there are actual NaN values, handle them
            if data_processed['agent'].isna().any():
                data_processed['agent_grouped'] = data_processed['agent_grouped'].fillna('No_Agent')
            
            print(f"Agent column: {data_processed['agent'].nunique()} unique values -> {data_processed['agent_grouped'].nunique()} after handling NaN")
        else:
            data_processed['agent_grouped'] = 'No_Agent'
        
        # Handle company column (not present in dataset)
        data_processed['company_grouped'] = 'No_Company'
        print("Company column: Not present in dataset")
        
        # Handle country column (group rare countries < 0.1% as 'Other')
        if 'country' in data_processed.columns:
            total_records = len(data_processed)
            country_counts = data_processed['country'].value_counts()
            country_percentages = country_counts / total_records * 100
            
            # Countries with less than 0.1% frequency
            rare_countries = country_percentages[country_percentages < 0.1].index
            
            data_processed['country_grouped'] = data_processed['country'].apply(
                lambda x: 'Other' if x in rare_countries else x
            )
            print(f"Country column: {len(country_counts)} unique values -> {data_processed['country_grouped'].nunique()} after grouping")
            print(f"Grouped {len(rare_countries)} countries with < 0.1% frequency as 'Other'")
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
        """Extract market features based on ACTUAL data, not assumptions"""
        data_market = data.copy()
        
        # Convert categorical columns to string for comparison
        distribution_channel_str = data_market['distribution_channel'].astype(str)
        market_segment_str = data_market['market_segment'].astype(str)
        
        # STEP 1: INSPECT YOUR ACTUAL DATA
        print("=== ACTUAL DATA INSPECTION ===")
        print("Distribution channels found:", distribution_channel_str.unique())
        print("Distribution channel counts:")
        print(distribution_channel_str.value_counts())
        print("\nMarket segments found:", market_segment_str.unique())
        print("Market segment counts:")
        print(market_segment_str.value_counts())
        
        # STEP 2: ANALYZE CANCELLATION RATES BY CATEGORY (if you have is_canceled)
        if 'is_canceled' in data_market.columns:
            print("\n=== CANCELLATION ANALYSIS ===")
            print("Cancellation rates by distribution channel:")
            channel_cancel_rates = data_market.groupby('distribution_channel')['is_canceled'].agg(['mean', 'count']).round(3)
            print(channel_cancel_rates)
            
            print("\nCancellation rates by market segment:")
            segment_cancel_rates = data_market.groupby('market_segment')['is_canceled'].agg(['mean', 'count']).round(3)
            print(segment_cancel_rates)
        
        # STEP 3: CREATE RISK CATEGORIES BASED ON YOUR ACTUAL DATA
        # For now, let's work with what we know exists:
        
        # Distribution Channel Risk (based on your actual channels)
        actual_channels = distribution_channel_str.unique()
        print(f"\nWorking with actual channels: {actual_channels}")
        
        def categorize_channel_risk_actual(channel):
            """Categorize based on your actual distribution channels"""
            if channel == 'TA/TO':
                return 'High_Risk'  # Travel agents typically higher risk
            elif channel == 'Direct':
                return 'Low_Risk'   # Direct bookings more committed
            elif channel == 'GDS':  # Only include if it actually exists
                return 'Medium_Risk'
            else:
                return 'Unknown_Risk'
        
        # Only create risk categories if we have the channel
        if 'GDS' in actual_channels:
            print("GDS found in data - including in risk categorization")
            data_market['distribution_channel_risk'] = distribution_channel_str.apply(categorize_channel_risk_actual)
        else:
            print("GDS NOT found in data - using simplified categorization")
            def categorize_channel_risk_simple(channel):
                if channel == 'TA/TO':
                    return 'High_Risk'
                elif channel == 'Direct':
                    return 'Low_Risk'
                else:
                    return 'Medium_Risk'  # Anything else gets medium risk
            
            data_market['distribution_channel_risk'] = distribution_channel_str.apply(categorize_channel_risk_simple)
        
        # Market Segment Risk (based on your actual segments)
        actual_segments = market_segment_str.unique()
        print(f"Working with actual segments: {actual_segments}")
        
        def categorize_market_segment_risk_actual(segment):
            """Categorize based on your actual market segments"""
            if segment == 'Groups':
                return 'High_Risk'
            elif segment in ['Offline TA/TO', 'Online TA']:
                return 'Medium_Risk'
            elif segment in ['Corporate', 'Direct', 'Complementary']:
                return 'Low_Risk'
            else:
                return 'Medium_Risk'  # Default for unknown segments
        
        data_market['market_segment_risk'] = market_segment_str.apply(categorize_market_segment_risk_actual)
        
        # Create binary indicators for backward compatibility
        data_market['high_risk_segment'] = (
            data_market['market_segment_risk'] == 'High_Risk'
        ).astype(int)
        
        data_market['high_risk_channel'] = (
            data_market['distribution_channel_risk'] == 'High_Risk'
        ).astype(int)
        
        # Other features...
        data_market['is_online_booking'] = (
            distribution_channel_str == 'Direct'
        ).astype(int)
        
        data_market['is_travel_agent_booking'] = (
            distribution_channel_str == 'TA/TO'
        ).astype(int)
        
        data_market['is_direct_booking'] = (
            distribution_channel_str == 'Direct'
        ).astype(int)
        
        data_market['is_corporate_booking'] = (
            (distribution_channel_str == 'Corporate') |
            (market_segment_str == 'Corporate')
        ).astype(int)
        
        data_market['is_group_booking'] = (
            market_segment_str == 'Groups'
        ).astype(int)
        
        # Print final risk distribution
        print("\n=== FINAL RISK CATEGORIZATION ===")
        print("Distribution Channel Risk:")
        print(data_market['distribution_channel_risk'].value_counts())
        print("\nMarket Segment Risk:")
        print(data_market['market_segment_risk'].value_counts())
        
        return data_market
    
    # Keep the original method name as an alias for backward compatibility
    def extract_market_features_data_driven(self, data):
        """Alias for extract_market_features - for backward compatibility"""
        return self.extract_market_features(data)
    
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
            'high_risk_channel', 'is_travel_agent_booking', 'is_group_booking',
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