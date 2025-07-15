import os
import pandas as pd
import numpy as np
from scripts.feature_engineering import HotelFeatureExtractor
import warnings
warnings.filterwarnings('ignore')

def test_feature_engineering_pipeline():
    """
    Simple test to show before/after feature engineering transformations
    """
    
    # Configuration
    input_path = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/cleaned/final_cleaned_data.pkl'
    output_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results'
    
    print("="*80)
    print("FEATURE ENGINEERING PIPELINE TEST")
    print("="*80)
    
    # Load original data
    print("\n📁 Loading original data...")
    if input_path.endswith('.pkl'):
        original_data = pd.read_pickle(input_path)
    else:
        original_data = pd.read_csv(input_path)
    
    print(f"✅ Data loaded successfully!")
    
    # ============================================================================
    # BEFORE FEATURE ENGINEERING
    # ============================================================================
    print("\n" + "="*60)
    print("🔍 BEFORE FEATURE ENGINEERING")
    print("="*60)
    
    print(f"📊 Original Data Shape: {original_data.shape}")
    print(f"📝 Total Original Columns: {len(original_data.columns)}")
    
    print(f"\n📋 Original Column Names:")
    for i, col in enumerate(original_data.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n📝 Categorical Columns:")
    categorical_cols = original_data.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        unique_count = original_data[col].nunique()
        print(f"  • {col} ({unique_count} unique values)")
    
    # ============================================================================
    # RUN FEATURE ENGINEERING
    # ============================================================================
    print("\n" + "="*60)
    print("⚙️  RUNNING FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Initialize feature extractor
    extractor = HotelFeatureExtractor(output_dir=output_dir)
    
    # Create a copy to avoid modifying original
    data_copy = original_data.copy()
    
    # Step-by-step feature engineering
    print("\n🔄 Step 1: Handling high cardinality features...")
    data_copy = extractor.handle_high_cardinality_features(data_copy)
    
    print("🔄 Step 2: Extracting temporal features...")
    data_copy = extractor.extract_temporal_features(data_copy)
    
    print("🔄 Step 3: Extracting customer behavior features...")
    data_copy = extractor.extract_customer_behavior_features(data_copy)
    
    print("🔄 Step 4: Extracting booking risk features...")
    data_copy = extractor.extract_booking_risk_features(data_copy)
    
    print("🔄 Step 5: Extracting market features...")
    data_copy = extractor.extract_market_features(data_copy)
    
    print("🔄 Step 6: Encoding categorical variables...")
    engineered_data = extractor.encode_categorical_variables(data_copy)
    
    # ============================================================================
    # AFTER FEATURE ENGINEERING
    # ============================================================================
    print("\n" + "="*60)
    print("✨ AFTER FEATURE ENGINEERING")
    print("="*60)
    
    print(f"📊 Engineered Data Shape: {engineered_data.shape}")
    print(f"📝 Total Engineered Columns: {len(engineered_data.columns)}")
    print(f"📈 New Features Created: {len(engineered_data.columns) - len(original_data.columns)}")
    
    # ============================================================================
    # FEATURE ANALYSIS
    # ============================================================================
    print("\n" + "="*60)
    print("🔍 FEATURE TRANSFORMATION ANALYSIS")
    print("="*60)
    
    # Original features that remain
    original_features_kept = [col for col in original_data.columns if col in engineered_data.columns]
    print(f"\n✅ Original Features Kept ({len(original_features_kept)}):")
    for col in original_features_kept:
        print(f"  • {col}")
    
    # Features that were removed/transformed
    removed_features = [col for col in original_data.columns if col not in engineered_data.columns]
    print(f"\n❌ Original Features Removed/Transformed ({len(removed_features)}):")
    for col in removed_features:
        print(f"  • {col}")
    
    # New engineered features
    new_features = [col for col in engineered_data.columns if col not in original_data.columns]
    print(f"\n🆕 New Engineered Features ({len(new_features)}):")
    
    # Categorize new features
    temporal_features = [col for col in new_features if any(keyword in col.lower() for keyword in 
                        ['season', 'quarter', 'weekend', 'holiday', 'lead_time', 'advance', 'last_minute'])]
    
    customer_features = [col for col in new_features if any(keyword in col.lower() for keyword in 
                        ['loyalty', 'party', 'family', 'group', 'children', 'babies', 'business', 'experience', 'stay'])]
    
    risk_features = [col for col in new_features if any(keyword in col.lower() for keyword in 
                    ['risk', 'mismatch', 'changes', 'complexity', 'price', 'adr', 'revenue', 'booking_value'])]
    
    market_features = [col for col in new_features if any(keyword in col.lower() for keyword in 
                      ['online', 'agent', 'company', 'channel', 'segment', 'corporate', 'direct', 'travel'])]
    
    dummy_features = [col for col in new_features if '_' in col and any(col.startswith(prefix + '_') for prefix in 
                     ['meal', 'market_segment', 'distribution_channel', 'deposit_type', 'reserved_room_type', 
                      'assigned_room_type', 'booking_season', 'lead_time_category', 'customer_experience_level', 
                      'cancellation_tendency', 'avg_stay_preference', 'price_category', 'deposit_risk'])]
    
    grouping_features = [col for col in new_features if 'grouped' in col.lower()]
    
    print(f"\n🕒 Temporal Features ({len(temporal_features)}):")
    for col in temporal_features:
        print(f"  • {col}")
    
    print(f"\n👥 Customer Behavior Features ({len(customer_features)}):")
    for col in customer_features:
        print(f"  • {col}")
    
    print(f"\n⚠️  Risk Assessment Features ({len(risk_features)}):")
    for col in risk_features:
        print(f"  • {col}")
    
    print(f"\n🏢 Market Features ({len(market_features)}):")
    for col in market_features:
        print(f"  • {col}")
    
    if grouping_features:
        print(f"\n🔗 Grouped Features (High Cardinality Handling) ({len(grouping_features)}):")
        for col in grouping_features:
            print(f"  • {col}")
    
    print(f"\n🏷️  Dummy Variables (One-Hot Encoded) ({len(dummy_features)}):")
    # Group dummy variables by their prefix
    dummy_groups = {}
    for col in dummy_features:
        prefix = col.split('_')[0] + '_' + col.split('_')[1] if len(col.split('_')) > 1 else col
        if prefix not in dummy_groups:
            dummy_groups[prefix] = []
        dummy_groups[prefix].append(col)
    
    for prefix, cols in dummy_groups.items():
        print(f"  📋 {prefix} ({len(cols)} categories):")
        for col in cols[:3]:  # Show first 3 categories
            print(f"    • {col}")
        if len(cols) > 3:
            print(f"    ... and {len(cols) - 3} more")
    
    # ============================================================================
    # ENCODING INFORMATION
    # ============================================================================
    print("\n" + "="*60)
    print("🔧 ENCODING TRANSFORMATIONS")
    print("="*60)
    
    if hasattr(extractor, 'label_encoders') and extractor.label_encoders:
        print(f"\n📊 Label Encoding Applied to ({len(extractor.label_encoders)}) columns:")
        for col, encoder in extractor.label_encoders.items():
            classes = encoder.classes_
            print(f"  • {col}: {len(classes)} categories")
            if len(classes) <= 8:
                mapping = dict(zip(classes, encoder.transform(classes)))
                print(f"    Mapping: {mapping}")
            else:
                print(f"    Categories: {classes[:3]}... (showing first 3)")
    
    # ============================================================================
    # TRANSFORMATION SUMMARY
    # ============================================================================
    print("\n" + "="*60)
    print("📈 TRANSFORMATION SUMMARY")
    print("="*60)
    
    print(f"\n📋 Feature Engineering Results:")
    print(f"  • Original Features: {len(original_data.columns)}")
    print(f"  • Features After Engineering: {len(engineered_data.columns)}")
    print(f"  • New Features Created: {len(new_features)}")
    print(f"  • Features Removed/Transformed: {len(removed_features)}")
    print(f"  • Growth Factor: {len(engineered_data.columns) / len(original_data.columns):.2f}x")
    
    print(f"\n💾 Data Types After Engineering:")
    dtype_counts_after = engineered_data.dtypes.value_counts()
    for dtype, count in dtype_counts_after.items():
        print(f"  • {dtype}: {count} columns")
    
    # ============================================================================
    # SAVE RESULTS (ONLY 2 FILES)
    # ============================================================================
    print("\n" + "="*60)
    print("💾 SAVING RESULTS")
    print("="*60)
    
    # 1. Save engineered data as pickle (preserves data types)
    pickle_output_path = os.path.join(output_dir, 'feature_engineered_data.pkl')
    engineered_data.to_pickle(pickle_output_path)
    print(f"✅ Engineered data saved to: {pickle_output_path}")
    
    # 2. Save CSV for quick inspection
    csv_output_path = os.path.join(output_dir, 'feature_engineered_data.csv')
    engineered_data.to_csv(csv_output_path, index=False)
    print(f"✅ CSV version saved to: {csv_output_path}")
    
    print("\n" + "="*80)
    print("✨ FEATURE ENGINEERING PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print(f"📁 Files saved: {pickle_output_path}, {csv_output_path}")
    print("="*80)
    
    return engineered_data

if __name__ == "__main__":
    try:
        engineered_data = test_feature_engineering_pipeline()
        print(f"\n🎉 Success! Final engineered dataset shape: {engineered_data.shape}")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()