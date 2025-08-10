import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Output directories
output_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/app/models/trained_models'
plots_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/models_plots'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)


def load_cleaned_data():
    """
    Load the cleaned and feature-engineered dataset from either pickle or CSV format.
    """
    pickle_file = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/app/models/preprocessors/engineered_data.pkl'
    if os.path.exists(pickle_file):
        print(f"Loading data from {pickle_file}")
        data = pd.read_pickle(pickle_file)
    else:
        csv_file = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/app/models/preprocessors/engineered_data.csv'
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
                X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                X[f'{col}_quarter'] = X[col].dt.quarter
                
                # Weekend indicator
                X[f'{col}_is_weekend'] = (X[col].dt.dayofweek >= 5).astype(int)
                
                # Month-related features
                X[f'{col}_is_month_start'] = X[col].dt.is_month_start.astype(int)
                X[f'{col}_is_month_end'] = X[col].dt.is_month_end.astype(int)
                
                # Seasonal features
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
                
                # Peak season indicators
                X[f'{col}_is_peak_season'] = X[col].dt.month.isin([6, 7, 8, 12]).astype(int)
                X[f'{col}_is_holiday_season'] = X[col].dt.month.isin([12, 1, 7, 8]).astype(int)
                
                # Cyclical features for seasonality
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
    Split data into training and testing sets with stratification.
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


def train_default_random_forest(X_train, y_train):
    """
    Train a default Random Forest model for comparison.
    """
    print("\n=== TRAINING DEFAULT RANDOM FOREST ===")
    
    # Default model with balanced class weights
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("Training default Random Forest model...")
    model.fit(X_train, y_train)
    
    print(f"Model trained with {len(model.estimators_)} trees")
    
    return model


def tune_random_forest(X_train, y_train, cv_folds=5):
    """
    Comprehensive hyperparameter tuning for Random Forest using GridSearchCV.
    """
    print("\n=== RANDOM FOREST HYPERPARAMETER TUNING ===")
    
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    print(f"Parameter grid size: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features']) * len(param_grid['bootstrap'])} combinations")
    
    # Base model with balanced class weights
    base_model = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # GridSearchCV with cross-validation
    print(f"Starting GridSearchCV with {cv_folds}-fold cross-validation...")
    print("This may take several minutes...")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\nBest cross-validation F1 score: {best_score:.4f}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Create results summary
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # Display top 5 parameter combinations
    print("\nTop 5 parameter combinations:")
    top_results = cv_results.nlargest(5, 'mean_test_score')[
        ['mean_test_score', 'std_test_score', 'params']
    ]
    for idx, row in top_results.iterrows():
        print(f"Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']*2:.4f}) - {row['params']}")
    
    return best_model, best_params, cv_results


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation using multiple metrics.
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


def plot_model_comparison(default_results, tuned_results, default_importance, tuned_importance, y_test, cv_results):
    """
    Create comprehensive comparison visualizations.
    """
    print("\n=== CREATING COMPARISON VISUALIZATIONS ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Model comparison metrics
    ax1 = axes[0, 0]
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Default RF': [
            default_results['accuracy'],
            default_results['precision'],
            default_results['recall'],
            default_results['f1'],
            default_results['auc']
        ],
        'Tuned RF': [
            tuned_results['accuracy'],
            tuned_results['precision'],
            tuned_results['recall'],
            tuned_results['f1'],
            tuned_results['auc']
        ]
    })
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax1.bar(x - width/2, comparison_df['Default RF'], width, label='Default RF', alpha=0.8)
    ax1.bar(x + width/2, comparison_df['Tuned RF'], width, label='Tuned RF', alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Default vs Tuned Random Forest')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df['Metric'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC Curves Comparison
    ax2 = axes[0, 1]
    
    # Default model ROC
    fpr_default, tpr_default, _ = roc_curve(y_test, default_results['probabilities'])
    ax2.plot(fpr_default, tpr_default, label=f'Default RF (AUC = {default_results["auc"]:.3f})', alpha=0.8)
    
    # Tuned model ROC
    fpr_tuned, tpr_tuned, _ = roc_curve(y_test, tuned_results['probabilities'])
    ax2.plot(fpr_tuned, tpr_tuned, label=f'Tuned RF (AUC = {tuned_results["auc"]:.3f})', alpha=0.8)
    
    ax2.plot([0, 1], [0, 1], 'k--', label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cross-validation scores distribution
    ax3 = axes[1, 0]
    ax3.hist(cv_results['mean_test_score'], bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(cv_results['mean_test_score'].max(), color='red', linestyle='--', 
                label=f'Best Score: {cv_results["mean_test_score"].max():.4f}')
    ax3.set_xlabel('Cross-Validation F1 Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of CV Scores')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature importance comparison (top 10)
    ax4 = axes[1, 1]
    
    # Get top 10 features from tuned model
    tuned_top10 = tuned_importance.head(10)
    
    # Get corresponding importance from default model
    default_importance_dict = dict(zip(default_importance['feature'], 
                                     default_importance['importance']))
    
    default_values = [default_importance_dict.get(feat, 0) for feat in tuned_top10['feature']]
    
    y_pos = np.arange(len(tuned_top10))
    
    ax4.barh(y_pos - 0.2, default_values, 0.4, label='Default RF', alpha=0.8)
    ax4.barh(y_pos + 0.2, tuned_top10['importance'], 0.4, label='Tuned RF', alpha=0.8)
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(tuned_top10['feature'], fontsize=8)
    ax4.set_xlabel('Importance Score')
    ax4.set_title('Feature Importance Comparison (Top 10)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(plots_dir, 'random_forest_comparison_results.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df


def main():
    """
    Main function to orchestrate the Random Forest pipeline with comparison.
    """
    print("=== HOTEL BOOKING CANCELLATION PREDICTION - RANDOM FOREST ===")
    print("Starting Random Forest model training and comparison pipeline...")
    
    try:
        # Load data
        print("\n1. Loading cleaned data...")
        data = load_cleaned_data()
        
        # Prepare features
        print("\n2. Preparing features...")
        X, y = prepare_features(data)
        
        # Split data
        print("\n3. Splitting data...")
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Train default model
        print("\n4. Training default Random Forest...")
        default_model = train_default_random_forest(X_train, y_train)
        
        # Get default model feature importance
        default_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': default_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate default model
        default_results = evaluate_model(default_model, X_test, y_test, "Default Random Forest")
        
        # Train tuned model
        print("\n5. Training tuned Random Forest...")
        tuned_model, best_params, cv_results = tune_random_forest(X_train, y_train)
        
        # Get tuned model feature importance
        tuned_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': tuned_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate tuned model
        tuned_results = evaluate_model(tuned_model, X_test, y_test, "Tuned Random Forest")
        
        # Create comparison visualizations
        print("\n6. Creating comparison visualizations...")
        comparison_df = plot_model_comparison(
            default_results, tuned_results, default_importance, tuned_importance, y_test, cv_results
        )
        
        # Calculate improvement
        comparison_df['Improvement'] = comparison_df['Tuned RF'] - comparison_df['Default RF']
        comparison_df['Improvement %'] = (comparison_df['Improvement'] / comparison_df['Default RF']) * 100
        
        # Save results - only tuned model feature importance and comparison
        print("\n7. Saving results...")
        
        # Save tuned model feature importance
        tuned_importance.to_csv(
            os.path.join(output_dir, 'tuned_random_forest_feature_importance.csv'), 
            index=False
        )
        print(f"Tuned feature importance saved to: {os.path.join(output_dir, 'tuned_random_forest_feature_importance.csv')}")
        
        # Save model comparison
        comparison_df.to_csv(
            os.path.join(output_dir, 'random_forest_model_comparison.csv'), 
            index=False
        )
        print(f"Model comparison saved to: {os.path.join(output_dir, 'random_forest_model_comparison.csv')}")
        
        # Save complete results as pickle (similar to linear regression)
        results_summary = {
            'tuned_model': tuned_model,
            'best_params': best_params,
            'tuned_feature_importance': tuned_importance,
            'tuned_metrics': tuned_results,
            'default_metrics': default_results,
            'comparison_df': comparison_df,
            'cv_results': cv_results,
            'test_data': {
                'X_test': X_test,
                'y_test': y_test
            }
        }
        
        with open(os.path.join(output_dir, 'tuned_random_forest.pkl'), 'wb') as f:
            pickle.dump(results_summary, f)
        
        # Final summary
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        print("Summary of Results:")
        print(f"- Default Random Forest F1-Score: {default_results['f1']:.4f}")
        print(f"- Tuned Random Forest F1-Score: {tuned_results['f1']:.4f}")
        
        best_model_name = "Tuned" if tuned_results['f1'] > default_results['f1'] else "Default"
        best_f1 = max(tuned_results['f1'], default_results['f1'])
        print(f"- Best performing model: {best_model_name} Random Forest (F1-Score: {best_f1:.4f})")
        
        improvement = tuned_results['f1'] - default_results['f1']
        improvement_pct = (improvement / default_results['f1']) * 100
        print(f"- Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        print("\nFiles saved:")
        print("- Tuned feature importance: tuned_random_forest_feature_importance.csv")
        print("- Model comparison: random_forest_model_comparison.csv")
        print("- Complete results: tuned_random_forest.pkl")
        print(f"- Visualization: {plots_dir}/random_forest_comparison_results.png")
        
    except Exception as e:
        print(f"Error occurred during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()