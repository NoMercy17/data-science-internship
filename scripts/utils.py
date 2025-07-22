import pandas as pd

# Used for checking on which columns i need to do the scalling, etc



# Load your data
data_path = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results/feature_engineered_data.pkl'

try:
    data = pd.read_pickle(data_path)
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ File not found. Please update the data_path variable.")
    exit()
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

print("=== FEATURE VALUE ANALYSIS ===")
print(f"Data shape: {data.shape}")
print()

# Get only numeric columns (excluding target and boolean columns)
numeric_cols = []
for col in data.columns:
    if col == 'is_canceled':
        continue
    if data[col].dtype in ['int64', 'float64']:
        # Skip binary columns (0/1 only)
        if not (data[col].nunique() == 2 and data[col].min() == 0 and data[col].max() == 1):
            numeric_cols.append(col)

print("=== NUMERIC COLUMNS ANALYSIS ===")
print(f"Found {len(numeric_cols)} numeric columns to analyze:")
print()

scaling_candidates = []

for col in numeric_cols:
    print(f"📊 {col}:")
    print(f"   Min: {data[col].min()}")
    print(f"   Max: {data[col].max()}")
    print(f"   Mean: {data[col].mean():.2f}")
    print(f"   Std: {data[col].std():.2f}")
    print(f"   Range: {data[col].max() - data[col].min()}")
    print(f"   Unique values: {data[col].nunique()}")
    
    # Check for outliers
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
    print(f"   Outliers: {outliers} ({outliers/len(data)*100:.1f}%)")
    
    # Show some sample values
    sample_values = data[col].head(10).tolist()
    print(f"   Sample values: {sample_values}")
    
    # Scaling recommendation
    col_range = data[col].max() - data[col].min()
    if col_range > 10:  # Arbitrary threshold for "needs scaling"
        scaling_candidates.append(col)
        print("   ✅ SCALING RECOMMENDED")
    else:
        print("   ❌ Scaling probably not needed")
    
    print("-" * 50)

print("\n=== SCALING SUMMARY ===")
print("Columns that should be scaled:")
for col in scaling_candidates:
    col_range = data[col].max() - data[col].min()
    print(f"  • {col} (range: {col_range})")

print(f"\nTotal columns to scale: {len(scaling_candidates)}")

# Check range ratios
if len(scaling_candidates) > 1:
    ranges = {col: data[col].max() - data[col].min() for col in scaling_candidates}
    max_range = max(ranges.values())
    min_range = min(ranges.values())
    ratio = max_range / min_range if min_range > 0 else float('inf')
    print(f"Range ratio (max/min): {ratio:.1f}")
    if ratio > 10:
        print("🚨 SCALING DEFINITELY NEEDED - Large range differences!")
    else:
        print("⚠️  Scaling recommended for consistency")

print("\n=== BOOLEAN/BINARY COLUMNS ===")
boolean_cols = []
for col in data.columns:
    if col == 'is_canceled':
        continue
    if data[col].dtype == 'bool' or (data[col].nunique() == 2 and data[col].min() == 0 and data[col].max() == 1):
        boolean_cols.append(col)

print(f"Found {len(boolean_cols)} boolean/binary columns (no scaling needed):")
for col in boolean_cols:
    print(f"  • {col}")

print("\n=== FINAL SCALING RECOMMENDATION ===")
print("Copy this list to your scaling function:")
print("columns_to_scale = [")
for col in scaling_candidates:
    print(f"    '{col}',")
print("]")

print("\n=== COMPLETE ANALYSIS FINISHED ===")
print(f"Total columns analyzed: {len(numeric_cols)}")
print(f"Columns needing scaling: {len(scaling_candidates)}")
print(f"Boolean/binary columns: {len(boolean_cols)}")
print("Ready for scaling implementation!")





