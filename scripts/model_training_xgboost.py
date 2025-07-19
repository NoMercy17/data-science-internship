import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class XGBoostHotelBookingPredictor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.model = None
        self.feature_importance = None
        
    def load_cleaned_data(self):
        """Load preprocessed data"""
        input_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results'
        
        pickle_file = os.path.join(input_dir, 'feature_engineering_results.pkl')
        if os.path.exists(pickle_file):
            print(f"Loading data from {pickle_file}")
            data = pd.read_pickle(pickle_file)
        else:
            csv_file = os.path.join(input_dir, 'feature_engineering_results.csv')
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
    
    def prepare_features(self, data):
        """Prepare features with improved preprocessing"""
        if 'is_canceled' not in data.columns:
            raise ValueError("Target variable 'is_canceled' not found in data")
        
        X = data.drop('is_canceled', axis=1)
        y = data['is_canceled']
        
        print(f"Initial features shape: {X.shape}")
        
        # Handle datetime columns more efficiently
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            for col in datetime_cols:
                # Extract essential datetime features
                X[f'{col}_year'] = X[col].dt.year
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_day'] = X[col].dt.day
                X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                X[f'{col}_is_weekend'] = (X[col].dt.dayofweek >= 5).astype(int)
                
                # Cyclical encoding for better ML performance
                X[f'{col}_month_sin'] = np.sin(2 * np.pi * X[col].dt.month / 12)
                X[f'{col}_month_cos'] = np.cos(2 * np.pi * X[col].dt.month / 12)
                X[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * X[col].dt.dayofweek / 7)
                X[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * X[col].dt.dayofweek / 7)
            
            X = X.drop(columns=datetime_cols)
        
        # Handle categorical variables
        object_cols = X.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            for col in object_cols:
                X[col] = pd.Categorical(X[col]).codes
        
        # Ensure all data is numeric
        X = X.select_dtypes(include=[np.number, 'bool']).astype(float)
        
        print(f"Final features shape: {X.shape}")
        return X, y
    
    def progressive_feature_selection(self, X_train, y_train, X_test, y_test):
        """
        Implement progressive feature selection approach using XGBoost
        Start with most important features and gradually add more
        """
        print("\n=== PROGRESSIVE FEATURE SELECTION WITH XGBOOST ===")
        
        # Step 1: Get initial feature importance using XGBoost
        xgb_selector = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        xgb_selector.fit(X_train, y_train)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 20 most important features:")
        print(importance_df.head(20))
        
        # Step 2: Test with different numbers of features
        feature_counts = [2, 5, 10, 20, 50, 100, min(200, len(X_train.columns))]
        results = []
        
        for n_features in feature_counts:
            if n_features > len(X_train.columns):
                continue
                
            # Select top N features
            top_features = importance_df.head(n_features)['feature'].tolist()
            X_train_subset = X_train[top_features]
            X_test_subset = X_test[top_features]
            
            # Apply SMOTE for balanced training
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_subset, y_train)
            
            # Train XGBoost model
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )
            model.fit(X_train_balanced, y_train_balanced)
            
            # Evaluate
            y_pred = model.predict(X_test_subset)  # Hard predictions
            y_pred_proba = model.predict_proba(X_test_subset)[:,1]  # Probability scores
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results.append({
                'n_features': n_features,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'features': top_features
            })
            
            print(f"Features: {n_features:3d} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Find optimal number of features (highest F1 score)
        best_result = results_df.loc[results_df['f1'].idxmax()]
        print(f"\nBest performance with {best_result['n_features']} features:")
        print(f"F1 Score: {best_result['f1']:.4f}")
        print(f"AUC: {best_result['auc']:.4f}")
        
        return results_df, best_result['features']
    
    def hyperparameter_tuning(self, X_train, y_train, selected_features):
        """Perform hyperparameter tuning for XGBoost"""
        print("\n=== HYPERPARAMETER TUNING ===")
        
        # Use only selected features
        X_train_final = X_train[selected_features]
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_final, y_train)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        # Create XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(X_train_balanced, y_train_balanced)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def train_final_model(self, X_train, y_train, selected_features):
        """Train final XGBoost model with selected features"""
        print("\n=== TRAINING FINAL XGBOOST MODEL ===")
        
        # Use only selected features
        X_train_final = X_train[selected_features]
        
        # Perform hyperparameter tuning
        best_model, best_params = self.hyperparameter_tuning(X_train, y_train, selected_features)
        
        # Create pipeline with SMOTE and best XGBoost model
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', best_model)
        ])
        
        # Fit pipeline
        pipeline.fit(X_train_final, y_train)
        
        # Extract feature importance from the trained model
        trained_xgb = pipeline.named_steps['classifier']
        
        self.feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': trained_xgb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features in Final XGBoost Model:")
        print(self.feature_importance.head(10))
        
        return pipeline, selected_features, best_params
    
    def evaluate_model(self, model, X_test, y_test, selected_features, model_name="Final XGBoost Model"):
        """Evaluate XGBoost model performance"""
        print(f"\n=== {model_name.upper()} EVALUATION ===")
        
        # Use only selected features
        X_test_final = X_test[selected_features]
        
        # Predictions
        y_pred = model.predict(X_test_final)
        y_pred_proba = model.predict_proba(X_test_final)[:,1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        for metric, value in metrics.items():
            if metric not in ['predictions', 'probabilities']:
                print(f"{metric.capitalize()}: {value:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def plot_feature_selection_results(self, results_df):
        """Plot feature selection results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Performance vs Number of Features
        ax1 = axes[0, 0]
        ax1.plot(results_df['n_features'], results_df['f1'], 'o-', label='F1 Score', color='red')
        ax1.plot(results_df['n_features'], results_df['auc'], 's-', label='AUC', color='blue')
        ax1.set_xlabel('Number of Features')
        ax1.set_ylabel('Score')
        ax1.set_title('XGBoost Performance vs Number of Features')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Detailed metrics
        ax2 = axes[0, 1]
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for metric, color in zip(metrics, colors):
            ax2.plot(results_df['n_features'], results_df[metric], 'o-', label=metric, color=color)
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Score')
        ax2.set_title('All XGBoost Metrics vs Number of Features')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feature importance
        ax3 = axes[1, 0]
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(15)
            bars = ax3.barh(range(len(top_features)), top_features['importance'], color='lightblue')
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'])
            ax3.set_title('Top 15 XGBoost Feature Importance')
            ax3.set_xlabel('Importance Score')
            ax3.invert_yaxis()  # Most important at top
        else:
            ax3.text(0.5, 0.5, 'Feature importance not available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Performance comparison
        ax4 = axes[1, 1]
        best_idx = results_df['f1'].idxmax()
        best_result = results_df.loc[best_idx]
        
        metrics_values = [best_result['accuracy'], best_result['precision'], 
                        best_result['recall'], best_result['f1'], best_result['auc']]
        bars = ax4.bar(metrics, metrics_values, color=['blue', 'green', 'orange', 'red', 'purple'])
        ax4.set_title(f'Best XGBoost Performance ({best_result["n_features"]} features)')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join('/home/antonios/Desktop/Practica_de_vara/data-science-internship/outputs/plots', 'model_xgboost_results.png'), 
                dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_xgboost_specific_analysis(self, model, X_test, y_test, selected_features):
        """Plot XGBoost-specific analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Feature importance (horizontal bar chart)
        ax1 = axes[0, 0]
        if self.feature_importance is not None:
            top_20 = self.feature_importance.head(20)
            ax1.barh(range(len(top_20)), top_20['importance'], color='lightgreen')
            ax1.set_yticks(range(len(top_20)))
            ax1.set_yticklabels(top_20['feature'])
            ax1.set_title('Top 20 XGBoost Feature Importance')
            ax1.set_xlabel('Importance Score')
            ax1.invert_yaxis()
        
        # Plot 2: ROC Curve
        ax2 = axes[0, 1]
        X_test_final = X_test[selected_features]
        y_pred_proba = model.predict_proba(X_test_final)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('XGBoost ROC Curve')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Prediction probability distribution
        ax3 = axes[1, 0]
        y_pred_proba = model.predict_proba(X_test_final)[:,1]
        
        # Separate probabilities by actual class
        prob_canceled = y_pred_proba[y_test == 1]
        prob_not_canceled = y_pred_proba[y_test == 0]
        
        ax3.hist(prob_not_canceled, bins=50, alpha=0.7, label='Not Canceled', color='blue', density=True)
        ax3.hist(prob_canceled, bins=50, alpha=0.7, label='Canceled', color='red', density=True)
        ax3.set_xlabel('Predicted Probability of Cancellation')
        ax3.set_ylabel('Density')
        ax3.set_title('XGBoost Prediction Probability Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Confusion matrix heatmap
        ax4 = axes[1, 1]
        y_pred = model.predict(X_test_final)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title('XGBoost Confusion Matrix')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join('/home/antonios/Desktop/Practica_de_vara/data-science-internship/outputs/plots', 'xgboost_detailed_analysis.png'), 
                dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_pipeline(self):
        """Run the complete XGBoost modeling pipeline"""
        print("HOTEL BOOKING CANCELLATION PREDICTION - XGBOOST PIPELINE")
        print("=" * 60)
        
        # Load and prepare data
        data = self.load_cleaned_data()
        X, y = self.prepare_features(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")
        
        # Progressive feature selection
        results_df, selected_features = self.progressive_feature_selection(
            X_train, y_train, X_test, y_test
        )
        
        # Train final model with hyperparameter tuning
        final_model, final_features, best_params = self.train_final_model(
            X_train, y_train, selected_features
        )
        
        # Evaluate final model
        final_metrics = self.evaluate_model(
            final_model, X_test, y_test, final_features
        )
        
        # Plot results
        self.plot_feature_selection_results(results_df)
        self.plot_xgboost_specific_analysis(final_model, X_test, y_test, final_features)
        
        # Save results
        results_summary = {
            'model': final_model,
            'selected_features': final_features,
            'best_hyperparameters': best_params,
            'feature_importance': self.feature_importance,
            'metrics': final_metrics,
            'feature_selection_results': results_df,
            'test_data': {
                'X_test': X_test[final_features],
                'y_test': y_test
            }
        }
        
        # Save to files
        with open(os.path.join(self.output_dir, 'xgboost_model.pkl'), 'wb') as f:
            pickle.dump(results_summary, f)
        
        self.feature_importance.to_csv(
            os.path.join(self.output_dir, 'xgboost_feature_importance.csv'), 
            index=False
        )
        
        # Save hyperparameters
        pd.DataFrame([best_params]).to_csv(
            os.path.join(self.output_dir, 'xgboost_best_hyperparameters.csv'), 
            index=False
        )
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Selected {len(final_features)} features for final model")
        print(f"Final F1 Score: {final_metrics['f1']:.4f}")
        print(f"Final AUC: {final_metrics['auc']:.4f}")
        print(f"Best hyperparameters: {best_params}")
        
        return results_summary

# Usage
if __name__ == "__main__":
    output_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/outputs/models'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots directory if it doesn't exist
    plots_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/outputs/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    predictor = XGBoostHotelBookingPredictor(output_dir)
    results = predictor.run_complete_pipeline()