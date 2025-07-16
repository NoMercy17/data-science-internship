import pandas as pd
from scripts.feature_engineering import HotelFeatureExtractor
import warnings
warnings.filterwarnings('ignore')

def test_feature_engineering_pipeline():
    """
    Test the feature engineering pipeline with clear before/after comparison
    """
    
    # Configuration
    input_path = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/cleaned/final_cleaned_data.pkl'
    output_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results'
    
    print("="*80)
    print("FEATURE ENGINEERING PIPELINE TEST")
    print("="*80)
    
    # Load data
    try:
        if input_path.endswith('.pkl'):
            original_data = pd.read_pickle(input_path)
        else:
            original_data = pd.read_csv(input_path)
        print(f"📁 Data loaded: {original_data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # ============================================================================
    # RUN FEATURE ENGINEERING
    # ============================================================================
    print("\n⚙️  Running feature engineering...")
    
    try:
        # Initialize feature extractor
        extractor = HotelFeatureExtractor(output_dir=output_dir)
        
        # Run complete pipeline
        engineered_data = extractor.run_complete_feature_engineering(
            data=original_data.copy(),
            apply_scaling=True,
            scaling_method='standard'
        )
        
    except Exception as e:
        print(f"❌ Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ============================================================================
    # BEFORE vs AFTER COMPARISON
    # ============================================================================
    print("\n" + "="*60)
    print("📊 BEFORE vs AFTER COMPARISON")
    print("="*60)
    
    print("\n📈 Dataset Shape:")
    print(f"  • Original: {original_data.shape}")
    print(f"  • Final:    {engineered_data.shape}")
    print(f"  • Change:   +{len(engineered_data.columns) - len(original_data.columns)} features")
    
    print("\n📋 Feature Types:")
    orig_numeric = len(original_data.select_dtypes(include=['int64', 'float64']).columns)
    orig_categorical = len(original_data.select_dtypes(include=['object', 'category']).columns)
    final_numeric = len(engineered_data.select_dtypes(include=['int64', 'float64']).columns)
    final_categorical = len(engineered_data.select_dtypes(include=['object', 'category']).columns)
    
    print(f"  • Numeric:     {orig_numeric} → {final_numeric} (+{final_numeric - orig_numeric})")
    print(f"  • Categorical: {orig_categorical} → {final_categorical} ({final_categorical - orig_categorical:+d})")
    
    # ============================================================================
    # FEATURE TRANSFORMATION ANALYSIS
    # ============================================================================
    print("\n" + "="*60)
    print("🔍 FEATURE TRANSFORMATION ANALYSIS")
    print("="*60)
    
    # Analyze what happened to original features
    kept_features = [col for col in original_data.columns if col in engineered_data.columns]
    removed_features = [col for col in original_data.columns if col not in engineered_data.columns]
    new_features = [col for col in engineered_data.columns if col not in original_data.columns]
    
    print("\n📊 Feature Changes:")
    print(f"  • Kept:     {len(kept_features)} features")
    print(f"  • Removed:  {len(removed_features)} features")
    print(f"  • Created:  {len(new_features)} features")
    
    if removed_features:
        print(f"\n❌ Removed Features ({len(removed_features)}):")
        for col in removed_features:
            print(f"  • {col}")
    
    if new_features:
        print(f"\n🆕 New Features ({len(new_features)}):")
        # Categorize new features by type
        feature_categories = {
            'temporal': ['lead_time', 'advance', 'last_minute', 'normal'],
            'customer': ['special', 'requirements', 'requests'],
            'market': ['market_segment', 'online_ta', 'offline_ta', 'groups', 'direct', 'corporate', 'risk_segment'],
            'deposit': ['deposit', 'refundable', 'non_refund'],
            'room': ['room_', 'standard', 'premium', 'suite'],
            'country': ['country_']
        }
        
        categorized_features = {category: [] for category in feature_categories.keys()}
        uncategorized_features = []
        
        for feature in new_features:
            categorized = False
            for category, keywords in feature_categories.items():
                if any(keyword in feature.lower() for keyword in keywords):
                    categorized_features[category].append(feature)
                    categorized = True
                    break
            if not categorized:
                uncategorized_features.append(feature)
        
        for category, features in categorized_features.items():
            if features:
                print(f"  📂 {category.upper()}: {len(features)} features")
        
        if uncategorized_features:
            print(f"  📂 OTHER: {len(uncategorized_features)} features")
    
    # ============================================================================
    # SUMMARY STATISTICS
    # ============================================================================
    print("\n" + "="*60)
    print("📈 FINAL SUMMARY")
    print("="*60)
    
    print(f"\n📊 Transformation Results:")
    print(f"  • Original Features: {len(original_data.columns)}")
    print(f"  • Final Features: {len(engineered_data.columns)}")
    print(f"  • Net Change: +{len(engineered_data.columns) - len(original_data.columns)}")
    
    # Memory usage
    original_memory = original_data.memory_usage(deep=True).sum() / 1024**2  # MB
    engineered_memory = engineered_data.memory_usage(deep=True).sum() / 1024**2  # MB
    
    print("\n💾 Memory Usage:")
    print(f"  • Original: {original_memory:.2f} MB")
    print(f"  • Final:    {engineered_memory:.2f} MB")
    print(f"  • Change:   {engineered_memory - original_memory:.2f} MB ({((engineered_memory/original_memory - 1) * 100):+.1f}%)")
    
    # Validation
    validation_passed = True
    
    if engineered_data.empty:
        print("\n❌ Dataset is empty!")
        validation_passed = False
    elif len(engineered_data.columns) < 5:
        print("\n❌ Too few features!")
        validation_passed = False
    elif engineered_data.select_dtypes(include=['float64']).isin([float('inf'), float('-inf')]).any().any():
        print("\n❌ Infinite values detected!")
        validation_passed = False
    else:
        print("\n✅ Pipeline validation passed!")
    
    if validation_passed:
        print("\n🎉 Feature engineering completed successfully!")
    else:
        print("\n❌ Feature engineering failed validation!")
    
    return engineered_data, None

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    try:
        engineered_data, summary = test_feature_engineering_pipeline()
        
        if engineered_data is not None:
            print("\n🎊 Feature engineering pipeline completed!")
        else:
            print("\n❌ Feature engineering pipeline failed!")
            
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()