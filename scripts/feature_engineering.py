import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

class HotelFeatureExtractor:
    """
    Complete hotel booking feature engineering pipeline
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
            (market_segment_str == 'Corporate') | 
            (distribution_channel_str == 'Corporate') |
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
        
        # 5. Price Category
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
        
        # 9. Waiting list risk
        data_risk['waiting_list_risk'] = (data_risk['days_in_waiting_list'] > 0).astype(int)
        
        # 10. Price per night category
        data_risk['price_per_night_category'] = np.where(
            data_risk['adr'] <= 50, 'Very_Low',
            np.where(data_risk['adr'] <= 100, 'Low',
                     np.where(data_risk['adr'] <= 200, 'Medium', 'High'))
        )
        
        return data_risk
    
    def extract_market_features(self, data):
        """Extract market and distribution features"""
        data_market = data.copy()
        
        # Convert categorical columns to string for comparison
        distribution_channel_str = data_market['distribution_channel'].astype(str)
        market_segment_str = data_market['market_segment'].astype(str)
        
        # 1. Online vs Offline booking
        data_market['is_online_booking'] = (
            distribution_channel_str.isin(['TA/TO', 'Direct', 'Online TA'])
        ).astype(int)
        
        # 2. Agent involvement
        if 'agent' in data_market.columns:
            data_market['has_agent'] = (~data_market['agent'].isna()).astype(int)
        else:
            data_market['has_agent'] = 0
        
        # 3. Company involvement
        if 'company' in data_market.columns:
            data_market['has_company'] = (~data_market['company'].isna()).astype(int)
        else:
            data_market['has_company'] = 0
        
        # 4. Market segment risk
        high_risk_segments = ['Online TA', 'Offline TA/TO']
        data_market['high_risk_segment'] = (
            market_segment_str.isin(high_risk_segments)
        ).astype(int)
        
        # 5. Distribution channel risk
        high_risk_channels = ['TA/TO']
        data_market['high_risk_channel'] = (
            distribution_channel_str.isin(high_risk_channels)
        ).astype(int)
        
        return data_market
    
    def handle_high_cardinality_features(self, data):
        """Handle high cardinality categorical features"""
        df = data.copy()
        
        # Handle 'agent' - group rare agents
        if 'agent' in df.columns:
            agent_counts = df['agent'].value_counts()
            top_agents = agent_counts.head(10).index
            df['agent_grouped'] = df['agent'].apply(
                lambda x: x if x in top_agents else 'Other'
            )
        else:
            df['agent_grouped'] = 'Other'
        
        # Handle 'company' - group rare companies
        if 'company' in df.columns:
            company_counts = df['company'].value_counts()
            top_companies = company_counts.head(10).index
            df['company_grouped'] = df['company'].apply(
                lambda x: x if x in top_companies else 'Other'
            )
        else:
            df['company_grouped'] = 'Other'
        
        # Handle 'country' - group by region or top countries
        if 'country' in df.columns:
            country_counts = df['country'].value_counts()
            top_countries = country_counts.head(15).index
            df['country_grouped'] = df['country'].apply(
                lambda x: x if x in top_countries else 'Other'
            )
        else:
            df['country_grouped'] = 'Other'
        
        return df
    
    def encode_categorical_variables(self, data):
        """
        Encode categorical variables using both label encoding and one-hot encoding
        
        One-hot encoding creates dummy variables:
        - meal column with values ['BB', 'HB', 'FB', 'SC'] becomes:
          meal_HB, meal_FB, meal_SC (meal_BB is dropped to avoid multicollinearity)
        - Each new column contains 0 or 1 indicating presence of that category
        """
        df = data.copy()
        
        # Binary categorical columns -label encoding (0, 1)
        binary_categorical = [
            'hotel', 'is_repeated_guest', 'customer_type'
        ]
        
        # Use one-hot encoding (create dummy variables)
        # Columns like: meal_HB, meal_FB, meal_SC, market_segment_Corporate
        onehot_categorical = [
            'meal', 'market_segment', 'distribution_channel', 'deposit_type',
            'reserved_room_type', 'assigned_room_type', 'booking_season',
            'lead_time_category', 'customer_experience_level', 'cancellation_tendency',
            'avg_stay_preference', 'price_category', 'deposit_risk',
            'price_per_night_category'
        ]
        
        # High cardinality categorical columns -label encoding (0, 1, 2, 3, ...)
        label_categorical = [
            'agent_grouped', 'company_grouped', 'country_grouped', 'arrival_date_month'
        ]
        
        # Label Encoding 
        # This converts categories to numbers: 'City Hotel' -> 0, 'Resort Hotel' -> 1
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
                
                # Create dummy variables (one-hot encoding)
                # drop_first=True removes one category to avoid multicollinearity
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
        Complete feature engineering pipeline
        """
        # Step 1: Extract temporal features
        df = self.extract_temporal_features(data)
        
        # Step 2: Extract customer behavior features
        df = self.extract_customer_behavior_features(df)
        
        # Step 3: Extract booking risk features
        df = self.extract_booking_risk_features(df)
        
        # Step 4: Extract market features
        df = self.extract_market_features(df)
        
        # Step 5: Handle high cardinality features
        df = self.handle_high_cardinality_features(df)
        
        # Step 6: Encode categorical variables
        df = self.encode_categorical_variables(df)
        
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
            'hotel', 'is_repeated_guest', 'customer_type', 'agent_grouped', 
            'company_grouped', 'country_grouped', 'arrival_date_month'
        ]
        
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
            'revenue_per_person', 'has_special_requirements', 'booking_complexity', 'waiting_list_risk'
        ]
        
        market_features = [
            'is_online_booking', 'has_agent', 'has_company', 'high_risk_segment', 'high_risk_channel'
        ]
        
        # Combine all numerical features
        all_numerical_features = (numerical_features + encoded_categorical + temporal_features + 
                                customer_features + risk_features + market_features)
        
        # Add one-hot encoded features (they have dynamic names)
        onehot_features = [col for col in df.columns if any(col.startswith(prefix + '_') for prefix in [
            'meal', 'market_segment', 'distribution_channel', 'deposit_type',
            'reserved_room_type', 'assigned_room_type', 'booking_season',
            'lead_time_category', 'customer_experience_level', 'cancellation_tendency',
            'avg_stay_preference', 'price_category', 'deposit_risk', 'price_per_night_category'
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
        
        print("Feature engineering completed!")
        print(f"Total features: {self.feature_info['total_features']}")
        print(f"Numerical features: {self.feature_info['numerical_features']}")
        print(f"One-hot encoded features: {self.feature_info['onehot_features']}")
        print(f"Label encoded features: {len(self.feature_info['label_encoders'])}")
        
        return df_final, pickle_output_path