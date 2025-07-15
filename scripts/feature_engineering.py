import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import KBinsDiscretizer
import pickle
warnings.filterwarnings('ignore')

class HotelFeatureExtractor:
    """
    Optimized hotel booking feature engineering pipeline with scaling and discretization
    """
    
    def __init__(self, output_dir='/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results'):
        self.feature_info = {}
        self.output_dir = output_dir
        self.label_encoders = {}
        self.scalers = {}
        self.discretizers = {}
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
        
        # 2. Lead Time Binary Categories 
        data_temp['is_last_minute_lead_time'] = (data_temp['lead_time'] <= 7).astype(int)
        data_temp['is_normal_lead_time'] = (
            (data_temp['lead_time'] > 7) & (data_temp['lead_time'] <= 90)
        ).astype(int)
        data_temp['is_advance_lead_time'] = (data_temp['lead_time'] > 90).astype(int)
        
        # 3. Lead Time Risk Score
        data_temp['lead_time_risk_score'] = (
            data_temp['is_last_minute_lead_time'] * 3 +  # High risk
            data_temp['is_normal_lead_time'] * 2 +       # Medium risk  
            data_temp['is_advance_lead_time'] * 1        # Low risk
        )
        
        # 4. Peak Season Indicator 
        peak_months = ['July', 'August', 'December']
        data_temp['is_peak_season'] = data_temp['arrival_date_month'].isin(peak_months).astype(int)
        
        # 5. Weekend Preference - USING ORIGINAL COLUMNS (will be deleted later)
        data_temp['weekend_preference'] = (
            data_temp['stays_in_weekend_nights'] / 
            (data_temp['stays_in_weekend_nights'] + data_temp['stays_in_week_nights'] + 0.001)
        )
        
        # 6. Holiday Season Indicator 
        holiday_months = ['December', 'January', 'July', 'August']
        data_temp['is_holiday_season'] = data_temp['arrival_date_month'].isin(holiday_months).astype(int)
        
        # Drop original lead_time and arrival_date_month (replaced with derived features)
        columns_to_drop = ['lead_time', 'arrival_date_month']
        for col in columns_to_drop:
            if col in data_temp.columns:
                data_temp = data_temp.drop(columns=[col])
                print(f"Dropped original column: {col} (replaced with derived features)")
        
        print("Created temporal features:")
    
        return data_temp
    
    def extract_customer_behavior_features(self, data):
        """Extract customer behavior features with optimized column usage"""
        data_behavior = data.copy()
        
        # 1. Total Stay Duration (CREATE FIRST - will replace individual stay columns)
        data_behavior['total_stay_nights'] = (
            data_behavior['stays_in_weekend_nights'] + 
            data_behavior['stays_in_week_nights']
        )
        
        # 2. Party Size (CREATE FIRST - will replace children/babies columns)
        data_behavior['party_size'] = (
            data_behavior['adults'] + 
            data_behavior['children'] + 
            data_behavior['babies']
        )
        
        # 3. Customer Type Indicators (CREATE BEFORE dropping children/babies)
        data_behavior['has_children'] = (data_behavior['children'] > 0).astype(int)
        data_behavior['has_babies'] = (data_behavior['babies'] > 0).astype(int)
        data_behavior['is_family_booking'] = (
            (data_behavior['has_children'] == 1) | 
            (data_behavior['has_babies'] == 1) | 
            (data_behavior['adults'] >= 2)
        ).astype(int)
        data_behavior['is_group_booking'] = (data_behavior['party_size'] >= 4).astype(int)
        
        # 4. Business vs Leisure indicators (CORRECTED - using TA/TO not Online TA)
        market_segment_str = data_behavior['market_segment'].astype(str)
        distribution_channel_str = data_behavior['distribution_channel'].astype(str)
        
        # Business likely: Corporate from both channels + TA/TO distribution + more weekdays than weekends
        data_behavior['is_business_likely'] = (
            (market_segment_str == 'Corporate') | 
            (distribution_channel_str == 'Corporate') |
            (distribution_channel_str == 'TA/TO') |  # TA/TO distribution, not Online TA
            (data_behavior['stays_in_week_nights'] > data_behavior['stays_in_weekend_nights'])
        ).astype(int)
        
        # 5. Customer Loyalty Score (using original columns before they're processed)
        def calculate_loyalty_score(previous_bookings, previous_cancellations, is_repeated):
            if previous_bookings == 0:
                return 0
            
            base_score = previous_bookings * 10
            penalty = previous_cancellations * 5
            bonus = 20 if is_repeated else 0
            
            return max(0, base_score - penalty + bonus)
        
        data_behavior['customer_loyalty_score'] = data_behavior.apply(
            lambda row: calculate_loyalty_score(
                row['previous_bookings_not_canceled'],
                row['previous_cancellations'],
                row['is_repeated_guest']
            ), axis=1
        )
        
        # 6. Special Requests Intensity (using new total_stay_nights)
        data_behavior['special_requests_ratio'] = (
            data_behavior['total_of_special_requests'] / 
            (data_behavior['total_stay_nights'] + 1)
        )
        
        # 7. Customer experience level (CREATE BEFORE dropping is_repeated_guest)
        data_behavior['customer_experience_level'] = np.where(
            data_behavior['is_repeated_guest'] == 1, 'Experienced',
            np.where(data_behavior['previous_bookings_not_canceled'] > 0, 'Returning', 'New')
        )
        
        # 8. Cancellation tendency (CREATE BEFORE dropping previous_cancellations)
        data_behavior['cancellation_tendency'] = np.where(
            data_behavior['previous_cancellations'] == 0, 'Low',
            np.where(data_behavior['previous_cancellations'] <= 2, 'Medium', 'High')
        )
        
        # 9. Average stay preference (KEEP BOTH - they serve different purposes)
        data_behavior['avg_stay_preference'] = np.where(
            data_behavior['total_stay_nights'] <= 2, 'Short',
            np.where(data_behavior['total_stay_nights'] <= 5, 'Medium', 'Extended')
        )
        
        # DROP REDUNDANT COLUMNS after creating derived features
        columns_to_drop = [
            'children',  # replaced by has_children and party_size
            'babies',    # replaced by has_babies and party_size
            'stays_in_weekend_nights',  # replaced by total_stay_nights and weekend_preference
            'stays_in_week_nights',     # replaced by total_stay_nights and weekend_preference
            'is_repeated_guest',        # replaced by customer_experience_level
            'previous_cancellations',   # replaced by cancellation_tendency
            'previous_bookings_not_canceled'  # replaced by customer_experience_level
        ]
        
        dropped_columns = []
        for col in columns_to_drop:
            if col in data_behavior.columns:
                data_behavior = data_behavior.drop(columns=[col])
                dropped_columns.append(col)
                print(f"Dropped {col} (replaced with derived features)")
        
        print(f"Dropped {len(dropped_columns)} redundant columns: {dropped_columns}")
        
        return data_behavior
    
    def extract_booking_risk_features(self, data):
        """Extract booking risk features with optimized column usage"""
        data_risk = data.copy()
        
        # 1. Room type mismatch (CREATE FIRST - will replace room type columns)
        reserved_room_str = data_risk['reserved_room_type'].astype(str)
        assigned_room_str = data_risk['assigned_room_type'].astype(str)
        
        data_risk['room_type_mismatch'] = (
            reserved_room_str != assigned_room_str
        ).astype(int)
        
        # 2. Changes per night (using total_stay_nights)
        data_risk['changes_per_night'] = (
            data_risk['booking_changes'] / 
            (data_risk['total_stay_nights'] + 1)
        )
        
        # 3. ADR per person (using party_size) - CREATE FIRST
        data_risk['adr_per_person'] = (
            data_risk['adr'] / 
            data_risk['party_size']
        )
        
        # 4. Total booking value (using total_stay_nights) - CREATE FIRST
        data_risk['total_booking_value'] = (
            data_risk['adr'] * data_risk['total_stay_nights']
        )
        
        # 5. Revenue per person (using party_size)
        data_risk['revenue_per_person'] = (
            data_risk['total_booking_value'] / 
            data_risk['party_size']
        )
        
        # 6. Special requirements indicator (CREATE BEFORE dropping total_of_special_requests)
        data_risk['has_special_requirements'] = (
            data_risk['total_of_special_requests'] > 0
        ).astype(int)
        
        # 7. Booking complexity score (CREATE BEFORE dropping columns)
        data_risk['booking_complexity'] = (
            data_risk['booking_changes'] + 
            data_risk['total_of_special_requests'] + 
            data_risk['room_type_mismatch']
        )
        
        # 8. Price category (using adr_per_person for better categorization)
        try:
            data_risk['price_category'] = pd.cut(
                data_risk['adr_per_person'],
                bins=[0, 25, 50, 100, float('inf')],
                labels=['Budget', 'Mid-range', 'Premium', 'Luxury']
            )
        except Exception as e:
            print(f"Warning: Could not create price_category due to: {e}")
            data_risk['price_category'] = pd.qcut(
                data_risk['adr_per_person'], 
                q=4, 
                labels=['Budget', 'Mid-range', 'Premium', 'Luxury'],
                duplicates='drop'
            )
        
        # 9. Deposit risk
        deposit_risk_mapping = {
            'No Deposit': 'High',
            'Refundable': 'Medium',
            'Non Refund': 'Low'
        }
        
        data_risk['deposit_risk'] = data_risk['deposit_type'].astype(str).map(deposit_risk_mapping)
        data_risk['deposit_risk'] = data_risk['deposit_risk'].fillna('Medium')
        
        # DROP REDUNDANT COLUMNS 
        columns_to_drop = [
            'reserved_room_type',  # room_type_mismatch
            'assigned_room_type',  # room_type_mismatch
            'adr',  # adr_per_person and total_booking_value
            'total_of_special_requests',  # special_requests_ratio, has_special_requirements, booking_complexity
            'booking_changes'  # changes_per_night and booking_complexity
        ]
        
        dropped_columns = []
        for col in columns_to_drop:
            if col in data_risk.columns:
                data_risk = data_risk.drop(columns=[col])
                dropped_columns.append(col)
                print(f"Dropped {col} (replaced with derived features)")
        
        print(f"Dropped {len(dropped_columns)} redundant columns: {dropped_columns}")
        
        return data_risk
    
    def handle_high_cardinality_features(self, data):
        """Handle high cardinality features by removing them"""
        data_processed = data.copy()
        
        columns_to_drop = ['agent', 'company', 'country']
        
        dropped_columns = []
        for col in columns_to_drop:
            if col in data_processed.columns:
                data_processed = data_processed.drop(columns=[col])
                dropped_columns.append(col)
                print(f"Dropped {col} column (high cardinality, low importance)")
        
        return data_processed

    def extract_market_features(self, data):
        """Extract market features with optimized categorical handling"""
        data_market = data.copy()
        
        # Convert to string for processing
        distribution_channel_str = data_market['distribution_channel'].astype(str)
        market_segment_str = data_market['market_segment'].astype(str)
        
        print("=== MARKET FEATURES ENGINEERING ===")
        print("Distribution channels:", distribution_channel_str.unique())
        print("Market segments:", market_segment_str.unique())
        
        # CREATE SUBCATEGORIES FOR MARKET SEGMENT
        # Only create specific categories for: Online TA, Offline TA/TO, Groups, Direct, Corporate
        # Group Complementary and Aviation as "Other"
        
        def create_market_segment_categories(segment):
            if segment == 'Online TA':
                return 'online_ta'
            elif segment == 'Offline TA/TO':
                return 'offline_ta_to'
            elif segment == 'Groups':
                return 'groups'
            elif segment == 'Direct':
                return 'direct'
            elif segment == 'Corporate':
                return 'corporate'
            else:  # Complementary, Aviation
                return 'other'
        
        data_market['market_segment_category'] = market_segment_str.apply(create_market_segment_categories)
        
        # Create binary indicators for main market segments
        data_market['market_segment_online_ta'] = (data_market['market_segment_category'] == 'online_ta').astype(int)
        data_market['market_segment_offline_ta_to'] = (data_market['market_segment_category'] == 'offline_ta_to').astype(int)
        data_market['market_segment_groups'] = (data_market['market_segment_category'] == 'groups').astype(int)
        data_market['market_segment_direct'] = (data_market['market_segment_category'] == 'direct').astype(int)
        data_market['market_segment_corporate'] = (data_market['market_segment_category'] == 'corporate').astype(int)
        data_market['market_segment_other'] = (data_market['market_segment_category'] == 'other').astype(int)
        
        # CREATE SUBCATEGORIES FOR DISTRIBUTION CHANNEL
        # Only create specific categories for: TA/TO, Direct, Corporate
        # Group GDS as "Other"
        
        def create_distribution_channel_categories(channel):
            if channel == 'TA/TO':
                return 'ta_to'
            elif channel == 'Direct':
                return 'direct'
            elif channel == 'Corporate':
                return 'corporate'
            else:  # GDS
                return 'other'
        
        data_market['distribution_channel_category'] = distribution_channel_str.apply(create_distribution_channel_categories)
        
        # Create binary indicators for main distribution channels
        data_market['distribution_channel_ta_to'] = (data_market['distribution_channel_category'] == 'ta_to').astype(int)
        data_market['distribution_channel_direct'] = (data_market['distribution_channel_category'] == 'direct').astype(int)
        data_market['distribution_channel_corporate'] = (data_market['distribution_channel_category'] == 'corporate').astype(int)
        data_market['distribution_channel_other'] = (data_market['distribution_channel_category'] == 'other').astype(int)
        
        # CREATE RISK CATEGORIES
        # Distribution Channel Risk
        def categorize_channel_risk(channel):
            if channel == 'TA/TO':
                return 'High'
            elif channel == 'Direct':
                return 'High'
            elif channel == 'Corporate':
                return 'Low'
            else:  # GDS
                return 'Medium'
        
        data_market['distribution_channel_risk'] = distribution_channel_str.apply(categorize_channel_risk)
        
        # Market Segment Risk
        def categorize_segment_risk(segment):
            if segment == 'Groups':
                return 'High'
            elif segment in ['Offline TA/TO', 'Online TA']:
                return 'Medium'
            elif segment in ['Corporate', 'Direct']:
                return 'Low'
            else:  # Complementary, Aviation
                return 'Medium'
        
        data_market['market_segment_risk'] = market_segment_str.apply(categorize_segment_risk)
        
        # Create binary high-risk indicators
        data_market['high_risk_segment'] = (data_market['market_segment_risk'] == 'High').astype(int)
        data_market['high_risk_channel'] = (data_market['distribution_channel_risk'] == 'High').astype(int)
        
        # Additional market indicators
        data_market['is_travel_agent_booking'] = (distribution_channel_str == 'TA/TO').astype(int)
        data_market['is_direct_booking'] = (distribution_channel_str == 'Direct').astype(int)
        data_market['is_corporate_booking'] = (
            (distribution_channel_str == 'Corporate') |
            (market_segment_str == 'Corporate')
        ).astype(int)
        data_market['is_group_booking_market'] = (market_segment_str == 'Groups').astype(int)
        
        # DROP ORIGINAL COLUMNS after creating subcategories
        columns_to_drop = ['distribution_channel', 'market_segment', 'distribution_channel_category', 'market_segment_category']
        
        dropped_columns = []
        for col in columns_to_drop:
            if col in data_market.columns:
                data_market = data_market.drop(columns=[col])
                dropped_columns.append(col)
                print(f"Dropped original column: {col} (replaced with subcategories)")
        
        print(f"Created market subcategories and dropped {len(dropped_columns)} original columns")
        
        return data_market
    
    def encode_categorical_variables(self, data):
        """Encode remaining categorical variables"""
        data = data.copy()
        
        # Binary categorical columns for label encoding
        binary_categorical = [
            'hotel', 'customer_type'
        ]
        
        # Multi-class categorical columns for one-hot encoding
        onehot_categorical = [
            'meal', 'deposit_type', 'booking_season', 'customer_experience_level', 
            'cancellation_tendency', 'avg_stay_preference', 'price_category', 
            'deposit_risk', 'distribution_channel_risk', 'market_segment_risk'
        ]
        
        # Label Encoding for binary categories
        for col in binary_categorical:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
                print(f"Label encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # One-hot encoding for multi-class categories
        created_dummies = []
        for col in onehot_categorical:
            if col in data.columns:
                unique_values = data[col].unique()
                print(f"Creating dummy variables for {col}: {unique_values}")
                
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                created_dummies.extend(dummies.columns.tolist())
                
                data = pd.concat([data, dummies], axis=1)
                data = data.drop(columns=[col])
                
                print(f"  Created columns: {dummies.columns.tolist()}")
        
        print(f"\nTotal dummy variables created: {len(created_dummies)}")
        
        return data
    
    def scale_features(self, data, scaling_method='standard'):
        """
        Scale features that require scaling based on analysis
        """
        print(f"\n=== FEATURE SCALING ({scaling_method.upper()}) ===")
        
        data_scaled = data.copy()
        
        # Columns that need scaling based on your analysis
        columns_to_scale = [
            'arrival_date_week_number',
            'arrival_date_day_of_month', 
            'total_stay_nights',
            'adr_per_person',
            'total_booking_value',
            'revenue_per_person'
        ]
        
        # Filter to only include columns that exist in the data
        existing_columns_to_scale = [col for col in columns_to_scale if col in data_scaled.columns]
        
        if not existing_columns_to_scale:
            print("No columns found that need scaling")
            return data_scaled
        
        print(f"Columns to scale: {existing_columns_to_scale}")
        
        # Choose scaler based on method
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            print(f"Unknown scaling method: {scaling_method}. Using StandardScaler.")
            scaler = StandardScaler()
        
        # Apply scaling
        try:
            # Fit and transform the selected columns
            data_scaled[existing_columns_to_scale] = scaler.fit_transform(
                data_scaled[existing_columns_to_scale]
            )
            
            # Store the scaler for later use
            self.scalers[scaling_method] = {
                'scaler': scaler,
                'columns': existing_columns_to_scale
            }
            
            print(f"✅ Successfully scaled {len(existing_columns_to_scale)} columns using {scaling_method}")
            
            # Print scaling statistics
            for col in existing_columns_to_scale:
                mean_val = data_scaled[col].mean()
                std_val = data_scaled[col].std()
                min_val = data_scaled[col].min()
                max_val = data_scaled[col].max()
                print(f"  {col}: mean={mean_val:.3f}, std={std_val:.3f}, min={min_val:.3f}, max={max_val:.3f}")
                
        except Exception as e:
            print(f"❌ Error during scaling: {e}")
            return data
        
        return data_scaled
    
    def discretize_features(self, data, discretization_strategy='uniform'):
        """
        Discretize continuous features where appropriate
        """
        print(f"\n=== FEATURE DISCRETIZATION ({discretization_strategy.upper()}) ===")
        
        data_discretized = data.copy()
        
        # Features that might benefit from discretization
        # Choose features with high ranges or outliers
        discretization_candidates = {
            'total_stay_nights': {'n_bins': 5, 'labels': ['Very Short', 'Short', 'Medium', 'Long', 'Extended']},
            'adr_per_person': {'n_bins': 4, 'labels': ['Budget', 'Economy', 'Premium', 'Luxury']},
            'total_booking_value': {'n_bins': 5, 'labels': ['Very Low', 'Low', 'Medium', 'High', 'Very High']},
            'revenue_per_person': {'n_bins': 4, 'labels': ['Low', 'Medium', 'High', 'Very High']},
            'special_requests_ratio': {'n_bins': 3, 'labels': ['None', 'Few', 'Many']},
            'weekend_preference': {'n_bins': 3, 'labels': ['Weekday', 'Mixed', 'Weekend']}
        }
        
        # Filter to only include columns that exist in the data
        existing_candidates = {
            col: config for col, config in discretization_candidates.items() 
            if col in data_discretized.columns
        }
        
        if not existing_candidates:
            print("No columns found for discretization")
            return data_discretized
        
        print(f"Columns to discretize: {list(existing_candidates.keys())}")
        
        discretized_columns = []
        
        for col, config in existing_candidates.items():
            try:
                discretized_col = f"{col}_discretized"
            
                # Check if column has enough unique values for discretization
                unique_values = data_discretized[col].nunique()
                if unique_values < config['n_bins']:
                    print(f"  ⚠️  Skipping {col}: only {unique_values} unique values (need >= {config['n_bins']})")
                    continue
                
                # Create discretizer
                if discretization_strategy == 'uniform':
                    discretizer = KBinsDiscretizer(
                        n_bins=config['n_bins'], 
                        encode='ordinal', 
                        strategy='uniform',
                        subsample=None
                    )
                elif discretization_strategy == 'quantile':
                    discretizer = KBinsDiscretizer(
                        n_bins=config['n_bins'], 
                        encode='ordinal', 
                        strategy='quantile',
                        subsample=None
                    )
                else:
                    discretizer = KBinsDiscretizer(
                        n_bins=config['n_bins'], 
                        encode='ordinal', 
                        strategy='kmeans',
                        subsample=None
                    )
                
                # Fit and transform
                discretized_values = discretizer.fit_transform(
                    data_discretized[[col]]
                ).flatten()
                
                # Convert to categorical with labels if provided
                if 'labels' in config and len(config['labels']) == config['n_bins']:
                    discretized_values = pd.Categorical(
                        discretized_values,
                        categories=range(config['n_bins']),
                        ordered=True
                    )
                    discretized_values = discretized_values.map(dict(enumerate(config['labels'])))
                
                # Add discretized column
                data_discretized[discretized_col] = discretized_values
                
                # Store discretizer
                self.discretizers[col] = {
                    'discretizer': discretizer,
                    'labels': config.get('labels', None),
                    'n_bins': config['n_bins']
                }
                
                discretized_columns.append(discretized_col)
                
                # Print discretization info
                print(f"  ✅ {col} → {discretized_col}")
                if 'labels' in config:
                    value_counts = data_discretized[discretized_col].value_counts().sort_index()
                    print(f"     Distribution: {dict(value_counts)}")
                
            except Exception as e:
                print(f"  ❌ Error discretizing {col}: {e}")
                continue
        
        print(f"✅ Successfully discretized {len(discretized_columns)} columns")
        
        return data_discretized
    
    def analyze_scaling_needs(self, data):
        """
        Analyze which features need scaling based on statistical properties
        """
        print("\n=== SCALING NEEDS ANALYSIS ===")
        
        # Get numeric columns only
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variable if present
        if 'is_canceled' in numeric_cols:
            numeric_cols.remove('is_canceled')
        
        scaling_recommendations = []
        
        for col in numeric_cols:
            if col in data.columns:
                # Calculate statistics
                col_min = data[col].min()
                col_max = data[col].max()
                col_range = col_max - col_min
                col_std = data[col].std()
                col_mean = data[col].mean()
                
                
                # Criteria: large range, high standard deviation, or values not in [0,1] range
                needs_scaling = (
                    col_range > 10 or  # Large range
                    col_std > 2 or     # High standard deviation
                    col_max > 100 or   # Large maximum value
                    (col_min < 0 and col_max > 1)  # Values span beyond [0,1]
                )
                
                if needs_scaling:
                    scaling_recommendations.append({
                        'column': col,
                        'range': col_range,
                        'std': col_std,
                        'mean': col_mean,
                        'min': col_min,
                        'max': col_max
                    })
        
        # Sort by range (descending)
        scaling_recommendations.sort(key=lambda x: x['range'], reverse=True)
        
        print(f"Found {len(scaling_recommendations)} columns that need scaling:")
        for rec in scaling_recommendations:
            print(f"  • {rec['column']}: range={rec['range']:.2f}, std={rec['std']:.2f}")
        
        return [rec['column'] for rec in scaling_recommendations]
    
    
def run_complete_feature_engineering(self, data, apply_scaling=True, apply_discretization=True,
                                     scaling_method='standard', discretization_strategy='uniform'):
    """
    Complete optimized feature engineering pipeline with scaling and discretization
    """
    print("Starting optimized feature engineering pipeline...")
    
    # Store original shape for reporting
    original_shape = data.shape
    
    # Step 1: Handle high cardinality features
    print("\n=== HANDLING HIGH CARDINALITY FEATURES ===")
    data = self.handle_high_cardinality_features(data)
    
    # Step 2: Extract temporal features
    print("\n=== EXTRACTING TEMPORAL FEATURES ===")
    data = self.extract_temporal_features(data)
    
    # Step 3: Extract customer behavior features
    print("\n=== EXTRACTING CUSTOMER BEHAVIOR FEATURES ===")
    data = self.extract_customer_behavior_features(data)
    
    # Step 4: Extract booking risk features
    print("\n=== EXTRACTING BOOKING RISK FEATURES ===")
    data = self.extract_booking_risk_features(data)
    
    # Step 5: Extract market features
    print("\n=== EXTRACTING MARKET FEATURES ===")
    data = self.extract_market_features(data)
    
    # Step 6: Encode categorical variables
    print("\n=== ENCODING CATEGORICAL VARIABLES ===")
    data = self.encode_categorical_variables(data)
    
    # Step 7: Apply scaling if requested
    if apply_scaling:
        print("\n=== ANALYZING SCALING NEEDS ===")
        scaling_candidates = self.analyze_scaling_needs(data)
        
        if scaling_candidates:
            print(f"Applying {scaling_method} scaling to {len(scaling_candidates)} columns...")
            data = self.scale_features(data, scaling_method=scaling_method)
        else:
            print("No columns require scaling")
    
    # Step 8: Apply discretization if requested
    if apply_discretization:
        print("\n=== APPLYING DISCRETIZATION ===")
        data = self.discretize_features(data, discretization_strategy=discretization_strategy)
    
    # Step 9: Final summary
    print("\n=== PIPELINE SUMMARY ===")
    print(f"Original dataset shape: {original_shape}")
    print(f"Final dataset shape: {data.shape}")
    print(f"Features added: {data.shape[1] - original_shape[1]}")
    print("Feature types summary:")
    print(f"  • Numeric features: {len(data.select_dtypes(include=[np.number]).columns)}")
    print(f"  • Categorical features: {len(data.select_dtypes(include=['object', 'category']).columns)}")
    print(f"  • Boolean features: {len(data.select_dtypes(include=['bool']).columns)}")
    
    # Store feature information for future use
    self.feature_info = {
        'original_shape': original_shape,
        'final_shape': data.shape,
        'feature_names': data.columns.tolist(),
        'numeric_features': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_features': data.select_dtypes(include=['object', 'category']).columns.tolist(),
        'scaling_applied': apply_scaling,
        'discretization_applied': apply_discretization,
        'scaling_method': scaling_method if apply_scaling else None,
        'discretization_strategy': discretization_strategy if apply_discretization else None,
        'label_encoders': list(self.label_encoders.keys()) if hasattr(self, 'label_encoders') else [],
        'scalers': list(self.scalers.keys()) if hasattr(self, 'scalers') else [],
        'discretizers': list(self.discretizers.keys()) if hasattr(self, 'discretizers') else []
    }
    
    # Save only the 2 essential files
    print("\n=== SAVING RESULTS ===")
    
    # 1. Save engineered data as pickle (preserves data types)
    pickle_output_path = os.path.join(self.output_dir, 'feature_engineered_data.pkl')
    data.to_pickle(pickle_output_path)
    print(f"✅ Engineered data saved to: {pickle_output_path}")
    
    # 2. Save CSV for quick inspection
    csv_output_path = os.path.join(self.output_dir, 'feature_engineered_data.csv')
    data.to_csv(csv_output_path, index=False)
    print(f"✅ CSV version saved to: {csv_output_path}")
    
    print("\n🎉 Feature engineering pipeline completed successfully!")
    print(f"📁 Files saved: {pickle_output_path}, {csv_output_path}")
    
    return data