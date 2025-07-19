import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

input_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results'
output_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/outputs/models'
os.makedirs(output_dir, exist_ok=True)

def load_cleaned_data():
    """
    Load the cleaned and feature-engineered dataset from either pickle or CSV format.
    """
    pickle_file = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results/feature_engineering_results.pkl'
    if os.path.exists(pickle_file):
        print(f"Loading data from {pickle_file}")
        data = pd.read_pickle(pickle_file)
    else:
        csv_file = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results/feature_engineering_results.csv'
        if os.path.exists(csv_file):
            print(f"Loading data from {csv_file}")
            data = pd.read_csv(csv_file)
        else:
            raise FileNotFoundError("No data files found!")
    
    print(f"Data loaded successfully: {data.shape}")
    print("Target variable distribution:")
    print(data['is_canceled'].value_counts())
    print(f"Cancellation rate: {data['is_canceled'].mean():.2%}")
    
    return data

def prepare_features(data):
    """
    Prepare features for machine learning by handling different data types and ensuring all features are numeric.
   
    """
    # Separate features and target
    if 'is_canceled' not in data.columns:
        raise ValueError("Target variable 'is_canceled' not found in data")
    
    X = data.drop('is_canceled', axis=1)
    y = data['is_canceled']
    
    print(f"Initial features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Check data types
    print("\nData types before processing:")
    print(X.dtypes.value_counts())
    
    # Handle datetime columns by extracting features
    datetime_cols = X.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        print(f"\nFound datetime columns: {list(datetime_cols)}")
        
        for col in datetime_cols:
            print(f"Processing datetime column: {col}")
            
            # For reservation_status_date
            if 'reservation_status_date' in col:
                X[f'{col}_year'] = X[col].dt.year
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_day'] = X[col].dt.day
                X[f'{col}_dayofweek'] = X[col].dt.dayofweek  # 0=Monday, 6=Sunday
                X[f'{col}_quarter'] = X[col].dt.quarter
                
                # Weekend indicator
                X[f'{col}_is_weekend'] = (X[col].dt.dayofweek >= 5).astype(int)
                
                # Month-related features
                X[f'{col}_is_month_start'] = X[col].dt.is_month_start.astype(int)
                X[f'{col}_is_month_end'] = X[col].dt.is_month_end.astype(int)
                
                # Seasonal features (very important for hotel bookings)
                def get_season(month):
                    if month in [12, 1, 2]:
                        return 1  # Winter
                    elif month in [3, 4, 5]:
                        return 2  # Spring
                    elif month in [6, 7, 8]:
                        return 3  # Summer
                    else:
                        return 4  # Fall
                
                X[f'{col}_season'] = X[col].dt.month.apply(get_season)
                
                # Peak season indicators (adjust based on your hotel's location)
                X[f'{col}_is_peak_season'] = X[col].dt.month.isin([6, 7, 8, 12]).astype(int)
                X[f'{col}_is_holiday_season'] = X[col].dt.month.isin([12, 1, 7, 8]).astype(int)
                
                # Cyclical features for seasonality (helps ML models understand cyclical nature)
                X[f'{col}_month_sin'] = np.sin(2 * np.pi * X[col].dt.month / 12)
                X[f'{col}_month_cos'] = np.cos(2 * np.pi * X[col].dt.month / 12)
                X[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * X[col].dt.dayofweek / 7)
                X[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * X[col].dt.dayofweek / 7)
                
                # Days since a reference date 
                reference_date = pd.Timestamp('2015-01-01')  
                X[f'{col}_days_since_ref'] = (X[col] - reference_date).dt.days
                
                # Year-relative features
                X[f'{col}_day_of_year'] = X[col].dt.dayofyear
                X[f'{col}_week_of_year'] = X[col].dt.isocalendar().week
                
                print(f"Extracted {len([c for c in X.columns if c.startswith(col)])} features from {col}")
            
            else:
                # For other datetime columns, use basic extraction
                X[f'{col}_year'] = X[col].dt.year
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_day'] = X[col].dt.day
                X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                print(f"Extracted {len([c for c in X.columns if c.startswith(col)])} features from {col}")
        
        # Drop original datetime columns
        X = X.drop(columns=datetime_cols)
        print(f"Converted datetime columns to numeric features. New shape: {X.shape}")
    
    # Handle object columns (categorical variables)
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"\nFound object columns: {list(object_cols)}")
        
        # Convert object columns to category and then to numeric codes
        for col in object_cols:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
        
        print("Converted object columns to numeric codes")
    
    # Ensure all columns are numeric
    non_numeric_cols = X.select_dtypes(exclude=[np.number, 'bool']).columns
    if len(non_numeric_cols) > 0:
        print(f"\nWarning: Still have non-numeric columns: {list(non_numeric_cols)}")
        # Drop any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number, 'bool'])
    
    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        print(f"\nConverting boolean columns to int: {list(bool_cols)}")
        X[bool_cols] = X[bool_cols].astype(int)
    
    print(f"\nFinal features shape: {X.shape}")
    print("Final data types:")
    print(X.dtypes.value_counts())
    
    # Check for any remaining issues
    print(f"\nChecking for missing values: {X.isnull().sum().sum()}")
    print(f"Checking for infinite values: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets with stratification to maintain class distribution in both sets.
    
    
    """
    print("\n=== DATA SPLITTING ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Testing set: {X_test.shape}, {y_test.shape}")

    print("Training set class distribution:")
    print(y_train.value_counts(normalize=True))
    print("Test set class distribution:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation using multiple metrics.
    
    We evaluate using:
    - Accuracy: Overall correctness
    - Precision: How many predicted cancellations were actually canceled
    - Recall: How many actual cancellations were correctly identified
    - F1-Score: Harmonic mean of precision and recall
    - ROC-AUC: Area under the ROC curve, measures discrimination ability
    
    """
    print(f"\n=== {model_name.upper()} EVALUATION ===")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) 

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def train_random_forest(X_train, X_test, y_train, y_test, handle_imbalance='balanced'):
    """
    Train Random Forest model with options for handling class imbalance.

    """
    print("\n=== RANDOM FOREST TRAINING ===")
    
    # Random Forest doesn't require feature scaling, but we'll track if we use SMOTE
    if handle_imbalance == 'balanced':
        print("Using class_weight='balanced'")
        model = RandomForestClassifier(
            n_estimators=100,           # Number of trees
            max_depth=10,               # Maximum depth to prevent overfitting
            min_samples_split=5,        # Minimum samples to split a node
            min_samples_leaf=2,         # Minimum samples in leaf node
            class_weight='balanced',    # Handle class imbalance
            random_state=42,
            n_jobs=-1                   # Use all available cores
        )
        X_train_processed = X_train
        y_train_processed = y_train
        
    elif handle_imbalance == 'smote':
        print("Using SMOTE for oversampling")
        smote = SMOTE(random_state=42)
        X_train_processed, y_train_processed = smote.fit_resample(X_train, y_train)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
    else:
        print("No imbalance handling")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        X_train_processed = X_train
        y_train_processed = y_train

    # Train the model
    print("Training Random Forest model...")
    model.fit(X_train_processed, y_train_processed)
    
    # Print training info
    print(f"Model trained with {len(model.estimators_)} trees")
    print(f"Training data shape: {X_train_processed.shape}")
    print(f"Training target distribution: {pd.Series(y_train_processed).value_counts().to_dict()}")
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test, "Random Forest")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Additional Random Forest specific metrics
    print("\nRandom Forest Specific Metrics:")
    print(f"Out-of-bag score: {model.oob_score_:.4f}" if hasattr(model, 'oob_score_') else "OOB score not available")
    print(f"Number of features used: {model.n_features_in_}")
    
    return model, results, feature_importance

def plot_rf_results(model_results, feature_importance, y_test, model_name="Random Forest"):
    """
    Create comprehensive visualizations for Random Forest model results.
    
    """
    print(f"\n=== PLOTTING {model_name.upper()} RESULTS ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Metrics bar plot
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    values = [model_results[metric] for metric in metrics]
    
    ax1 = axes[0, 0]
    bars = ax1.bar(metrics, values, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax1.set_title(f'{model_name} Performance Metrics')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 2. ROC Curve
    ax2 = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, model_results['probabilities'])
    ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {model_results["auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'{model_name} ROC Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, model_results['predictions'])
    ax3 = axes[1, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title(f'{model_name} Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # 4. Feature Importance
    ax4 = axes[1, 1]
    top_features = feature_importance.head(10)
    ax4.barh(range(len(top_features)), top_features['importance'])
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['feature'])
    ax4.set_title(f'{model_name} Top 10 Feature Importance')
    ax4.set_xlabel('Importance Score')
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/outputs/plots'
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'model_random_forest_results.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
 
    print("HOTEL BOOKING CANCELLATION PREDICTION - RANDOM FOREST MODELING")
    print("=" * 70)
    
    # Load data
    data = load_cleaned_data()

    # Prepare features
    X, y = prepare_features(data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train Random Forest model
    print("\n" + "="*70)
    rf_model, rf_results, rf_importance = train_random_forest(
        X_train, X_test, y_train, y_test, handle_imbalance='balanced'
    )

    # Plot results
    plot_rf_results(rf_results, rf_importance, y_test, "Random Forest")
    
    # Save results
    results_summary = {
        'model': rf_model,
        'results': rf_results,
        'feature_importance': rf_importance,
        'test_data': {
            'X_test': X_test,
            'y_test': y_test
        }
    }
    
    # Save to pickle
    with open(os.path.join(output_dir, 'random_forest_results.pkl'), 'wb') as f:
        pickle.dump(results_summary, f)
    
    # Save feature importance to CSV
    rf_importance.to_csv(os.path.join(output_dir, 'random_forest_feature_importance.csv'), index=False)
    
    print(f"\nResults saved to: {output_dir}")
    print("Files created:")
    print("- random_forest_results.pkl (complete results)")
    print("- random_forest_feature_importance.csv (feature importance)")
    print("- model_random_forest_results.png (visualizations)")
    
    return results_summary

if __name__ == "__main__":
    results = main()