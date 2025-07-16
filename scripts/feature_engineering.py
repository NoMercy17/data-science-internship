import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
warnings.filterwarnings('ignore')

class HotelFeatureExtractor:
    """
    Optimized hotel booking feature engineering pipeline with target leakage awareness
    """
    
    def __init__(self, output_dir='/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results'):
        self.feature_info = {}
        self.output_dir = output_dir
        self.label_encoders = {}
        self.scalers = {}
        self.discretizers = {}
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define which original columns to keep based on target leakage analysis
        self.columns_to_keep = [
            'lead_time',  # correlation 0.303848 
            'country',    # max correlation 0.353784   
            'market_segment',  # max correlation 0.236574 
            'assigned_room_type',  # max correlation 0.214921 
            'deposit_type',  # max correlation 0.491193 
            'total_of_special_requests'  # correlation -0.259646 
        ]
    
    def extract_temporal_features(self, data):
        """Extract temporal features only from kept columns"""
        data_temp = data.copy()
        
        # Only create features from lead_time (which is kept)
        if 'lead_time' in data_temp.columns:
            # Lead Time Binary Categories 
            data_temp['is_last_minute_lead_time'] = (data_temp['lead_time'] <= 7).astype(int)
            data_temp['is_normal_lead_time'] = (
                (data_temp['lead_time'] > 7) & (data_temp['lead_time'] <= 90)
            ).astype(int)
            data_temp['is_advance_lead_time'] = (data_temp['lead_time'] > 90).astype(int)
            
            # Lead Time Risk Score
            data_temp['lead_time_risk_score'] = (
                data_temp['is_last_minute_lead_time'] * 3 +  # High risk
                data_temp['is_normal_lead_time'] * 2 +       # Medium risk  
                data_temp['is_advance_lead_time'] * 1        # Low risk
            )
            
            # Remove original column at the end of this function
            data_temp = data_temp.drop(columns=['lead_time'])
            print("Created temporal features from lead_time and removed original column")
        
        return data_temp
    
    def extract_customer_behavior_features(self, data):
        """Extract customer behavior features only from kept columns"""
        data_behavior = data.copy()
        
        # Only create features from total_of_special_requests 
        if 'total_of_special_requests' in data_behavior.columns:
            # Special requests indicators
            data_behavior['has_special_requirements'] = (
                data_behavior['total_of_special_requests'] > 0
            ).astype(int)
            
            # Special requests categories
            data_behavior['special_requests_level'] = np.where(
                data_behavior['total_of_special_requests'] == 0, 'None',
                np.where(data_behavior['total_of_special_requests'] <= 1, 'Low',
                np.where(data_behavior['total_of_special_requests'] <= 3, 'Medium', 'High'))
            )
            print("Value counts:")
            print(data_behavior['special_requests_level'].value_counts())
            print("Unique values:", data_behavior['special_requests_level'].unique())
            # Remove original column at the end of this function
            data_behavior = data_behavior.drop(columns=['total_of_special_requests'])
            print("Created customer behavior features from total_of_special_requests and removed original column")
        
        return data_behavior
    
    def extract_market_features(self, data):
        """Extract market features only from kept columns"""
        data_market = data.copy()
        
        # Process market_segment 
        if 'market_segment' in data_market.columns:
            market_segment_str = data_market['market_segment'].astype(str)
            
            print("=== MARKET SEGMENT FEATURES ===")
            print("Market segments:", market_segment_str.unique())
            
            # Create market segment categories (intermediate variable)
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
            
            # Market Segment Risk (intermediate variable)
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
            data_market['medium_risk_segment'] = (data_market['market_segment_risk'] == 'Medium').astype(int)
            data_market['low_risk_segment'] = (data_market['market_segment_risk'] == 'Low').astype(int)
           
            # Remove intermediate variables and original column
            data_market = data_market.drop(columns=['market_segment_category', 'market_segment_risk', 'market_segment'])
            print("Created market segment features and removed original column and intermediate variables")
        
        return data_market
    
    def extract_deposit_features(self, data):
        """Extract deposit features from kept columns"""
        data_deposit = data.copy()
        
        # Process deposit_type (which is kept)
        if 'deposit_type' in data_deposit.columns:
            deposit_type_str = data_deposit['deposit_type'].astype(str)
            
            print("=== DEPOSIT FEATURES ===")
            print("Deposit types:", deposit_type_str.unique())
            
            # Create deposit risk categories (intermediate variable)
            deposit_risk_mapping = {
                'No Deposit': 'High',
                'Refundable': 'Medium', 
                'Non Refund': 'Low'
            }
            
            data_deposit['deposit_risk'] = deposit_type_str.map(deposit_risk_mapping)
            data_deposit['deposit_risk'] = data_deposit['deposit_risk'].fillna('Medium')
            
            # Create binary deposit indicators
            data_deposit['has_deposit'] = (deposit_type_str != 'No Deposit').astype(int)
            data_deposit['has_refundable_deposit'] = (deposit_type_str == 'Refundable').astype(int)
            data_deposit['has_non_refund_deposit'] = (deposit_type_str == 'Non Refund').astype(int)
            
            # Create risk level indicators
            data_deposit['deposit_risk_high'] = (data_deposit['deposit_risk'] == 'High').astype(int)
            data_deposit['deposit_risk_medium'] = (data_deposit['deposit_risk'] == 'Medium').astype(int)
            data_deposit['deposit_risk_low'] = (data_deposit['deposit_risk'] == 'Low').astype(int)
            
            # Remove intermediate variable and original column
            data_deposit = data_deposit.drop(columns=['deposit_risk', 'deposit_type'])
            print("Created deposit features and removed original column and intermediate variables")
        
        return data_deposit
    
    def extract_room_features(self, data):
        """Extract room features from kept columns"""
        data_room = data.copy()
        
        # Process assigned_room_type (which is kept)
        if 'assigned_room_type' in data_room.columns:
            room_type_str = data_room['assigned_room_type'].astype(str)
            
            print("=== ROOM FEATURES ===")
            print("Room types:", room_type_str.unique())
            
            # Group room types by frequency to avoid high cardinality
            room_counts = room_type_str.value_counts()
            
            # Keep top 7 room types, group others as 'Other'
            top_rooms = room_counts.head(7).index.tolist()
            
            def categorize_room(room):
                if room in top_rooms:
                    return room
                else:
                    return 'Other'
            
            data_room['room_type_grouped'] = room_type_str.apply(categorize_room)
            
            # Create binary indicators for top room types
            for room in top_rooms:
                data_room[f'room_type_{room}'] = (room_type_str == room).astype(int)
            
            data_room['room_type_other'] = (data_room['room_type_grouped'] == 'Other').astype(int)
            
            # Remove intermediate variable and original column
            data_room = data_room.drop(columns=['room_type_grouped', 'assigned_room_type'])
            print(f"Created room features for top {len(top_rooms)} room types and removed original column and intermediate variables")
        
        return data_room
    
    def extract_country_features(self, data):
        """Extract country features from kept columns"""
        data_country = data.copy()
        
        # Process country (which is kept)
        if 'country' in data_country.columns:
            country_str = data_country['country'].astype(str)
            
            print("=== COUNTRY FEATURES ===")
            print(f"Number of unique countries: {country_str.nunique()}")
            
            # Group countries by frequency to avoid high cardinality
            country_counts = country_str.value_counts()
            
            # Keep top 7 countries, group others as 'Other'
            top_countries = country_counts.head(7).index.tolist()
            
            def categorize_country(country):
                if country in top_countries:
                    return country
                else:
                    return 'Other'
            
            data_country['country_grouped'] = country_str.apply(categorize_country)
            
            # Create binary indicators for top countries
            for country in top_countries:
                data_country[f'country_{country}'] = (country_str == country).astype(int)
            
            data_country['country_other'] = (data_country['country_grouped'] == 'Other').astype(int)
            
            # Remove intermediate variable and original column
            data_country = data_country.drop(columns=['country_grouped', 'country'])
            print(f"Created country features for top {len(top_countries)} countries and removed original column and intermediate variables")
        
        return data_country
    
    def encode_categorical_variables(self, data):
        """Encode remaining categorical variables"""
        data = data.copy()
        
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable if present
        if 'is_canceled' in categorical_columns:
            categorical_columns.remove('is_canceled')
        
        print("=== ENCODING CATEGORICAL VARIABLES ===")
        print(f"Categorical columns to encode: {categorical_columns}")
        
        # for label encoding
        binary_categorical = []
        
        # for one-hot encoding
        onehot_categorical = []
        
        # Classify columns based on number of unique values
        for col in categorical_columns:
            unique_count = data[col].nunique()
            if unique_count == 2:
                binary_categorical.append(col)
            elif unique_count <= 5:  # limit for one-hot encoding
                onehot_categorical.append(col)
            else:
                print(f"Warning: {col} has {unique_count} unique values - may need special handling")
                onehot_categorical.append(col)  # Still try one-hot
        
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
                
                
                print(f"  Created columns: {dummies.columns.tolist()}")
        
        for col in onehot_categorical:
            if col in data.columns:
                data = data.drop(columns=[col])
                print(f"  Removed original column: {col}")
        
        print(f"\nTotal dummy variables created: {len(created_dummies)}")
        
        return data
    
    def scale_features(self, data, scaling_method='standard'):
        """Scale features that require scaling"""
        print(f"\n=== FEATURE SCALING ({scaling_method.upper()}) ===")
        
        data_scaled = data.copy()
        
        # Only scale numeric columns that need scaling
        columns_to_scale = []
        
        numeric_columns = data_scaled.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target variable if present
        if 'is_canceled' in numeric_columns:
            numeric_columns.remove('is_canceled')
        
        # Check which numeric columns need scaling
        for col in numeric_columns:
            col_min = data_scaled[col].min()
            col_max = data_scaled[col].max()
            col_range = col_max - col_min
            
            # Scale if range is large or max value is high
            if col_range > 10 or col_max > 100:
                columns_to_scale.append(col)
        
        if not columns_to_scale:
            print("No columns found that need scaling")
            return data_scaled
        
        print(f"Columns to scale: {columns_to_scale}")
        
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
            data_scaled[columns_to_scale] = scaler.fit_transform(data_scaled[columns_to_scale])
            
            # Store the scaler for later use
            self.scalers[scaling_method] = {
                'scaler': scaler,
                'columns': columns_to_scale
            }
            
            print(f"✅ Successfully scaled {len(columns_to_scale)} columns using {scaling_method}")
            
        except Exception as e:
            print(f" Error during scaling: {e}")
            return data
        
        return data_scaled
    
    
    def run_complete_feature_engineering(self, data, apply_scaling=True, scaling_method='standard'):
        """
        Complete feature engineering pipeline respecting target leakage analysis
        """
        print("Starting target leakage-aware feature engineering pipeline...")
        print(f"Columns to keep: {self.columns_to_keep}")
        
        
        original_shape = data.shape
        
      
        print("\n=== STEP 1: EXTRACTING TEMPORAL FEATURES ===")
        data = self.extract_temporal_features(data)
        
       
        print("\n=== STEP 2: EXTRACTING CUSTOMER BEHAVIOR FEATURES ===")
        data = self.extract_customer_behavior_features(data)
        

        print("\n=== STEP 3: EXTRACTING MARKET FEATURES ===")
        data = self.extract_market_features(data)
        
       
        print("\n=== STEP 4: EXTRACTING DEPOSIT FEATURES ===")
        data = self.extract_deposit_features(data)
        
        
        print("\n=== STEP 5: EXTRACTING ROOM FEATURES ===")
        data = self.extract_room_features(data)
        
        
        print("\n=== STEP 6: EXTRACTING COUNTRY FEATURES ===")
        data = self.extract_country_features(data)
        
      
        print("\n=== STEP 7: ENCODING CATEGORICAL VARIABLES ===")
        data = self.encode_categorical_variables(data)
        
        # Apply scaling if requested
        if apply_scaling:
            print("\n=== STEP 8: APPLYING SCALING ===")
            data = self.scale_features(data, scaling_method=scaling_method)
     
        # Final summary
        print("\n=== PIPELINE SUMMARY ===")
        print(f"Original dataset shape: {original_shape}")
        print(f"Final dataset shape: {data.shape}")
        print(f"Features after engineering: {data.shape[1]}")
        print("Feature types summary:")
        print(f"  • Numeric features: {len(data.select_dtypes(include=[np.number]).columns)}")
        print(f"  • Categorical features: {len(data.select_dtypes(include=['object', 'category']).columns)}")
        print(f"  • Boolean features: {len(data.select_dtypes(include=['bool']).columns)}")
        
        # Final column list
        print(f"\nFinal columns ({len(data.columns)}):")
        for i, col in enumerate(data.columns, 1):
            print(f"  {i:2d}. {col}")
        
        self.feature_info = {
            'original_shape': original_shape,
            'final_shape': data.shape,
            'feature_names': data.columns.tolist(),
            'kept_original_columns': self.columns_to_keep,
            'numeric_features': data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': data.select_dtypes(include=['object', 'category']).columns.tolist(),
            'scaling_applied': apply_scaling,
            'scaling_method': scaling_method if apply_scaling else None,
            'label_encoders': list(self.label_encoders.keys()) if hasattr(self, 'label_encoders') else [],
            'scalers': list(self.scalers.keys()) if hasattr(self, 'scalers') else []
        }
    
        print("\n=== SAVING RESULTS ===")

       
        pickle_output_path = os.path.join(self.output_dir, 'target_leakage_aware_features.pkl')
        data.to_pickle(pickle_output_path)
        print(f"✅ Engineered data saved to: {pickle_output_path}")
        
        
        csv_output_path = os.path.join(self.output_dir, 'target_leakage_aware_features.csv')
        data.to_csv(csv_output_path, index=False)
        print(f"✅ CSV version saved to: {csv_output_path}")
                
        return data