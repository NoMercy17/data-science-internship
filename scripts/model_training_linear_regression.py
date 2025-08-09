import pandas as pd
import numpy as np
import os
import pickle
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')


# Visualization


class HotelBookingPredictor:
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
        Implement progressive feature selection approach
        Start with most important features and gradually add more
        """
        print("\n=== PROGRESSIVE FEATURE SELECTION ===")
        
        # Step 1: Get initial feature importance using Random Forest
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_selector.fit(X_train, y_train)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': X_train.columns,  # e.g. feature: price
            'importance': rf_selector.feature_importances_ # e.g. importance: 0.25
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
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_subset)
            X_test_scaled = scaler.transform(X_test_subset)
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            
            # Train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_balanced, y_train_balanced)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled) # hard guesses(YES/NO will theuy cancel?)
            y_pred_proba = model.predict_proba(X_test_scaled)[:,1] # confidence scores(70% they will cancel)
            
            accuracy = accuracy_score(y_test, y_pred) # How often was it right? (8/10)
            precision = precision_score(y_test, y_pred)#  When it said "YES, they'll cancel," how often was it right?
            recall = recall_score(y_test, y_pred)# Of all the people who actually canceled, how many did we catch?
            f1 = f1_score(y_test, y_pred) # like an overall score
            auc = roc_auc_score(y_test, y_pred_proba) # How good is it at ranking people by cancel risk?
            
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
            # "Features: 10 | Accuracy: 0.8432 | F1: 0.7651 | AUC: 0.8123"
        
        # Convert to DataFrame for easier analysis(table with all our results)
        results_df = pd.DataFrame(results)
        
        # Find optimal number of features (highest F1 score)
        best_result = results_df.loc[results_df['f1'].idxmax()]
        print(f"\nBest performance with {best_result['n_features']} features:")
        print(f"F1 Score: {best_result['f1']:.4f}")
        print(f"AUC: {best_result['auc']:.4f}")
        
        return results_df, best_result['features']
    
    def train_final_model(self, X_train, y_train, selected_features):
        """Train final model with selected features"""
        print("\n=== TRAINING FINAL MODEL ===")
        
        # Use only selected features
        X_train_final = X_train[selected_features]
        
        # Create pipeline with scaling and SMOTE using imblearn Pipeline

        # Create pipeline with scaling and SMOTE
        pipeline = ImbPipeline([
        ('scaler', StandardScaler()), # make clues the same size
        ('smote', SMOTE(random_state=42)), # balance the data
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

        # eg for scaler
        # Before: [price=150, lead_time=30, month=7, country=2, adults=2]
        # After:  [price=0.23, lead_time=0.45, month=0.58, country=0.12, adults=0.33]
        
        # eg for smote
        # Before: 800 "won't cancel" + 200 "will cancel" = UNFAIR!
        # After:  800 "won't cancel" + 800 "will cancel" = FAIR GAME!

        # eg for classifier
        # Input: Balanced, scaled data
        # Output: "I learned that high price + long lead_time = likely to cancel!"

        # Fit pipeline, runs the 3 step process
        pipeline.fit(X_train_final, y_train)
        
        # Extract feature importance
        coefficients = pipeline.named_steps['classifier'].coef_[0]
        abs_coefficients = np.abs(coefficients)

        # Convert to importance scores (normalized to sum to 1, like Random Forest)
        importance_scores = abs_coefficients / np.sum(abs_coefficients)

        self.feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        # eg of that dataframe
        #         self.feature_importance = pd.DataFrame({
        #     'feature': ['lead_time', 'price', 'month', 'country', 'adults'],
        #     'coefficient': [0.85, -0.73, 0.62, 0.45, -0.38],
        #     'abs_coefficient': [0.85, 0.73, 0.62, 0.45, 0.38]
        # })
        
        print("Top 10 Most Important Features in Final Model:")
        print(self.feature_importance.head(10))
        
        return pipeline, selected_features
    
    def evaluate_model(self, model, X_test, y_test, selected_features, model_name="Final Model"):
        """Evaluate model performance"""
        print(f"\n=== {model_name.upper()} EVALUATION ===")
        
        # Use only selected features
        X_test_final = X_test[selected_features]
        
        # Predictions
        y_pred = model.predict(X_test_final)# Hard answers: Will cancel? Yes, No, Yes

        y_pred_proba = model.predict_proba(X_test_final)[:,1]#[0.78, 0.23, 0.85]  # Confidence: 78%, 23%, 85%
        
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
        ax1.plot(results_df['n_features'], results_df['f1'], 'o-', label='F1 Score')
        ax1.plot(results_df['n_features'], results_df['auc'], 's-', label='AUC')
        ax1.set_xlabel('Number of Features')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance vs Number of Features')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Detailed metrics
        ax2 = axes[0, 1]
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        for metric in metrics:
            ax2.plot(results_df['n_features'], results_df[metric], 'o-', label=metric)
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Score')
        ax2.set_title('All Metrics vs Number of Features')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feature importance (FIXED - use 'importance' column)
        ax3 = axes[1, 0]
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(15)
            ax3.barh(range(len(top_features)), top_features['importance'])  # Changed from 'abs_coefficient' to 'importance'
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'])
            ax3.set_title('Top 15 Feature Importance')
            ax3.set_xlabel('Importance Score')  # Updated label
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
        ax4.set_title(f'Best Model Performance ({best_result["n_features"]} features)')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join('/home/antonios/Desktop/Practica_de_vara/data-science-internship/outputs/plots', 'model_linear_regression_results.png'), 
                dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_pipeline(self):
        """Run the complete modeling pipeline"""
        print("HOTEL BOOKING CANCELLATION PREDICTION - COMPLETE PIPELINE")
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
        
        # Train final model
        final_model, final_features = self.train_final_model(
            X_train, y_train, selected_features
        )
        
        # Evaluate final model
        final_metrics = self.evaluate_model(
            final_model, X_test, y_test, final_features
        )
        
        # Plot results
        self.plot_feature_selection_results(results_df)
        
        # Save results
        results_summary = {
            'model': final_model,
            'selected_features': final_features,
            'feature_importance': self.feature_importance,
            'metrics': final_metrics,
            'feature_selection_results': results_df,
            'test_data': {
                'X_test': X_test[final_features],
                'y_test': y_test
            }
        }
        
        # Save to files
        with open(os.path.join(self.output_dir, 'linear_regression.pkl'), 'wb') as f:
            pickle.dump(results_summary, f)
        
        self.feature_importance.to_csv(
            os.path.join(self.output_dir, 'linear_regression_feature_importance.csv'), 
            index=False
        )
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Selected {len(final_features)} features for final model")
        print(f"Final F1 Score: {final_metrics['f1']:.4f}")
        print(f"Final AUC: {final_metrics['auc']:.4f}")
        
        return results_summary

# Usage
if __name__ == "__main__":
    output_dir = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/app/models/trained_models'
    os.makedirs(output_dir, exist_ok=True)
    
    predictor = HotelBookingPredictor(output_dir)
    results = predictor.run_complete_pipeline()