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









# import pandas as pd
# import numpy as np
# import os
# import pickle
# import warnings
# warnings.filterwarnings('ignore')

# # Machine Learning Libraries
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score, 
#     classification_report, confusion_matrix, roc_auc_score, roc_curve
# )
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline

# # Visualization
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set up paths
# input_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/cleaned'
# output_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/models'
# os.makedirs(output_dir, exist_ok=True)

# def load_cleaned_data():
#     """Load the final cleaned data"""
#     print("=== LOADING CLEANED DATA ===")
    
#     # Try to load the final cleaned pickle file
#     pickle_files = [
#         '09_multicollinearity_cleaned.pkl',
#         '08_target_leakage_cleaned.pkl', 
#         '07_dtypes_cleaned.pkl'
#     ]
    
#     for pickle_file in pickle_files:
#         pickle_path = os.path.join(input_dir, pickle_file)
#         if os.path.exists(pickle_path):
#             print(f"Loading data from: {pickle_file}")
#             data = pd.read_pickle(pickle_path)
#             break
#     else:
#         # Fallback to CSV if no pickle files found
#         csv_files = [
#             '06_context_outliers_cleaned.csv',
#             '05_infrequent_values_cleaned.csv',
#             '04_data_errors_cleaned.csv'
#         ]
        
#         for csv_file in csv_files:
#             csv_path = os.path.join(input_dir, csv_file)
#             if os.path.exists(csv_path):
#                 print(f"Loading data from: {csv_file}")
#                 data = pd.read_csv(csv_path)
#                 break
#         else:
#             raise FileNotFoundError("No cleaned data files found!")
    
#     print(f"Data loaded successfully: {data.shape}")
#     print(f"Target variable distribution:")
#     print(data['is_canceled'].value_counts())
#     print(f"Cancellation rate: {data['is_canceled'].mean():.2%}")
    
#     return data

# def prepare_features(data):
#     """Prepare features for modeling"""
#     print("\n=== FEATURE PREPARATION ===")
    
#     # Separate features and target
#     if 'is_canceled' not in data.columns:
#         raise ValueError("Target variable 'is_canceled' not found in data")
    
#     X = data.drop('is_canceled', axis=1)
#     y = data['is_canceled']
    
#     print(f"Features shape: {X.shape}")
#     print(f"Target shape: {y.shape}")
    
#     # Handle categorical variables
#     categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
#     numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
#     print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")
#     print(f"Numerical columns ({len(numerical_columns)}): {numerical_columns}")
    
#     # Encode categorical variables
#     label_encoders = {}
#     X_encoded = X.copy()
    
#     for col in categorical_columns:
#         le = LabelEncoder()
#         X_encoded[col] = le.fit_transform(X[col].astype(str))
#         label_encoders[col] = le
#         print(f"Encoded {col}: {X[col].nunique()} unique values")
    
#     # Handle any remaining missing values
#     missing_counts = X_encoded.isnull().sum()
#     if missing_counts.any():
#         print(f"Handling missing values:")
#         for col, count in missing_counts[missing_counts > 0].items():
#             print(f"  {col}: {count} missing values")
#             if X_encoded[col].dtype in ['int64', 'float64']:
#                 X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())
#             else:
#                 X_encoded[col] = X_encoded[col].fillna(X_encoded[col].mode()[0])
    
#     print(f"Final feature matrix shape: {X_encoded.shape}")
    
#     return X_encoded, y, label_encoders, categorical_columns, numerical_columns

# def split_data(X, y, test_size=0.2, random_state=42):
#     """Split data into train/test sets"""
#     print(f"\n=== DATA SPLITTING ===")
    
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y
#     )
    
#     print(f"Training set: {X_train.shape}, {y_train.shape}")
#     print(f"Test set: {X_test.shape}, {y_test.shape}")
    
#     print(f"Training set class distribution:")
#     print(y_train.value_counts(normalize=True))
#     print(f"Test set class distribution:")
#     print(y_test.value_counts(normalize=True))
    
#     return X_train, X_test, y_train, y_test

# def evaluate_model(model, X_test, y_test, model_name="Model"):
#     """Evaluate model performance"""
#     print(f"\n=== {model_name.upper()} EVALUATION ===")
    
#     # Predictions
#     y_pred = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)[:, 1]
    
#     # Metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, y_pred_proba)
    
#     print(f"Accuracy:  {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall:    {recall:.4f}")
#     print(f"F1-Score:  {f1:.4f}")
#     print(f"ROC-AUC:   {auc:.4f}")
    
#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print(f"\nConfusion Matrix:")
#     print(cm)
    
#     # Classification Report
#     print(f"\nClassification Report:")
#     print(classification_report(y_test, y_pred))
    
#     return {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'auc': auc,
#         'predictions': y_pred,
#         'probabilities': y_pred_proba
#     }

# def train_logistic_regression(X_train, y_train, X_test, y_test, handle_imbalance='balanced'):
#     """Train Logistic Regression model"""
#     print(f"\n=== LOGISTIC REGRESSION TRAINING ===")
    
#     # Scale features for logistic regression
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Handle class imbalance
#     if handle_imbalance == 'balanced':
#         print("Using class_weight='balanced'")
#         model = LogisticRegression(
#             class_weight='balanced',
#             random_state=42,
#             max_iter=1000
#         )
#     elif handle_imbalance == 'smote':
#         print("Using SMOTE for oversampling")
#         smote = SMOTE(random_state=42)
#         X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
#         model = LogisticRegression(random_state=42, max_iter=1000)
#     else:
#         print("No imbalance handling")
#         model = LogisticRegression(random_state=42, max_iter=1000)
    
#     # Train model
#     model.fit(X_train_scaled, y_train)
    
#     # Create a pipeline for predictions (includes scaling)
#     class LogisticRegressionPipeline:
#         def __init__(self, scaler, model):
#             self.scaler = scaler
#             self.model = model
        
#         def predict(self, X):
#             X_scaled = self.scaler.transform(X)
#             return self.model.predict(X_scaled)
        
#         def predict_proba(self, X):
#             X_scaled = self.scaler.transform(X)
#             return self.model.predict_proba(X_scaled)
    
#     pipeline = LogisticRegressionPipeline(scaler, model)
    
#     # Evaluate
#     results = evaluate_model(pipeline, X_test, y_test, "Logistic Regression")
    
#     # Feature importance (coefficients)
#     feature_importance = pd.DataFrame({
#         'feature': X_train.columns,
#         'coefficient': model.coef_[0],
#         'abs_coefficient': np.abs(model.coef_[0])
#     }).sort_values('abs_coefficient', ascending=False)
    
#     print(f"\nTop 10 Most Important Features:")
#     print(feature_importance.head(10))
    
#     return pipeline, results, feature_importance

# def train_random_forest(X_train, y_train, X_test, y_test, handle_imbalance='balanced'):
#     """Train Random Forest model"""
#     print(f"\n=== RANDOM FOREST TRAINING ===")
    
#     # Handle class imbalance
#     if handle_imbalance == 'balanced':
#         print("Using class_weight='balanced'")
#         model = RandomForestClassifier(
#             n_estimators=100,
#             class_weight='balanced',
#             random_state=42,
#             n_jobs=-1
#         )
#     elif handle_imbalance == 'smote':
#         print("Using SMOTE for oversampling")
#         smote = SMOTE(random_state=42)
#         X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
#         model = RandomForestClassifier(
#             n_estimators=100,
#             random_state=42,
#             n_jobs=-1
#         )
#         model.fit(X_train_resampled, y_train_resampled)
#     else:
#         print("No imbalance handling")
#         model = RandomForestClassifier(
#             n_estimators=100,
#             random_state=42,
#             n_jobs=-1
#         )
    
#     if handle_imbalance != 'smote':
#         model.fit(X_train, y_train)
    
#     # Evaluate
#     results = evaluate_model(model, X_test, y_test, "Random Forest")
    
#     # Feature importance
#     feature_importance = pd.DataFrame({
#         'feature': X_train.columns,
#         'importance': model.feature_importances_
#     }).sort_values('importance', ascending=False)
    
#     print(f"\nTop 10 Most Important Features:")
#     print(feature_importance.head(10))
    
#     return model, results, feature_importance

# def compare_models(models_results):
#     """Compare multiple models"""
#     print(f"\n=== MODEL COMPARISON ===")
    
#     comparison_df = pd.DataFrame(models_results).T
#     comparison_df = comparison_df.round(4)
    
#     print(comparison_df)
    
#     # Find best model for each metric
#     print(f"\nBest Models by Metric:")
#     for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
#         best_model = comparison_df[metric].idxmax()
#         best_score = comparison_df[metric].max()
#         print(f"{metric.title():10}: {best_model} ({best_score:.4f})")
    
#     return comparison_df

# def plot_results(models_results, y_test):
#     """Plot model comparison results"""
#     print(f"\n=== PLOTTING RESULTS ===")
    
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
#     # 1. Metrics comparison
#     metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
#     model_names = list(models_results.keys())
    
#     metric_data = []
#     for model_name in model_names:
#         for metric in metrics:
#             metric_data.append({
#                 'Model': model_name,
#                 'Metric': metric,
#                 'Score': models_results[model_name][metric]
#             })
    
#     metric_df = pd.DataFrame(metric_data)
    
#     ax1 = axes[0, 0]
#     sns.barplot(data=metric_df, x='Metric', y='Score', hue='Model', ax=ax1)
#     ax1.set_title('Model Performance Comparison')
#     ax1.set_ylim(0, 1)
#     ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     # 2. ROC Curves (if available)
#     ax2 = axes[0, 1]
#     for model_name, results in models_results.items():
#         if 'probabilities' in results:
#             fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
#             auc_score = results['auc']
#             ax2.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    
#     ax2.plot([0, 1], [0, 1], 'k--')
#     ax2.set_xlabel('False Positive Rate')
#     ax2.set_ylabel('True Positive Rate')
#     ax2.set_title('ROC Curves')
#     ax2.legend()
    
#     # 3. Confusion Matrix for best F1 model
#     best_f1_model = max(models_results.keys(), key=lambda x: models_results[x]['f1'])
#     cm = confusion_matrix(y_test, models_results[best_f1_model]['predictions'])
    
#     ax3 = axes[1, 0]
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
#     ax3.set_title(f'Confusion Matrix - {best_f1_model}')
#     ax3.set_xlabel('Predicted')
#     ax3.set_ylabel('Actual')
    
#     # 4. Class distribution
#     ax4 = axes[1, 1]
#     y_test.value_counts().plot(kind='bar', ax=ax4)
#     ax4.set_title('Test Set Class Distribution')
#     ax4.set_xlabel('Class')
#     ax4.set_ylabel('Count')
#     ax4.tick_params(axis='x', rotation=0)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
#     plt.show()

# def main():
#     """Main modeling pipeline"""
#     print("HOTEL BOOKING CANCELLATION PREDICTION - MODELING PIPELINE")
#     print("=" * 60)
    
#     # 1. Load data
#     data = load_cleaned_data()
    
#     # 2. Prepare features
#     X, y, label_encoders, categorical_columns, numerical_columns = prepare_features(data)
    
#     # 3. Split data
#     X_train, X_test, y_train, y_test = split_data(X, y)
    
#     # 4. Train models
#     models_results = {}
#     trained_models = {}
    
#     # Logistic Regression with balanced class weights
#     print("\n" + "="*60)
#     lr_model, lr_results, lr_importance = train_logistic_regression(
#         X_train, y_train, X_test, y_test, handle_imbalance='balanced'
#     )
#     models_results['Logistic Regression (Balanced)'] = lr_results
#     trained_models['Logistic Regression (Balanced)'] = lr_model
    
#     # Logistic Regression with SMOTE
#     print("\n" + "="*60)
#     lr_smote_model, lr_smote_results, lr_smote_importance = train_logistic_regression(
#         X_train, y_train, X_test, y_test, handle_imbalance='smote'
#     )
#     models_results['Logistic Regression (SMOTE)'] = lr_smote_results
#     trained_models['Logistic Regression (SMOTE)'] = lr_smote_model
    
#     # Random Forest with balanced class weights
#     print("\n" + "="*60)
#     rf_model, rf_results, rf_importance = train_random_forest(
#         X_train, y_train, X_test, y_test, handle_imbalance='balanced'
#     )
#     models_results['Random Forest (Balanced)'] = rf_results
#     trained_models['Random Forest (Balanced)'] = rf_model
    
#     # 5. Compare models
#     print("\n" + "="*60)
#     comparison_df = compare_models(models_results)
    
#     # 6. Plot results
#     plot_results(models_results, y_test)
    
#     # 7. Save results
#     results_summary = {
#         'comparison_df': comparison_df,
#         'models_results': models_results,
#         'trained_models': trained_models,
#         'feature_importance': {
#             'logistic_regression': lr_importance,
#             'logistic_regression_smote': lr_smote_importance,
#             'random_forest': rf_importance
#         },
#         'label_encoders': label_encoders,
#         'feature_columns': {
#             'categorical': categorical_columns,
#             'numerical': numerical_columns
#         }
#     }
    
#     # Save to pickle
#     with open(os.path.join(output_dir, 'modeling_results.pkl'), 'wb') as f:
#         pickle.dump(results_summary, f)
    
#     # Save comparison to CSV
#     comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
    
#     print(f"\nResults saved to: {output_dir}")
#     print("Files created:")
#     print("- modeling_results.pkl (complete results)")
#     print("- model_comparison.csv (performance comparison)")
#     print("- model_comparison.png (visualizations)")
    
#     return results_summary

# if __name__ == "__main__":
#     results = main()