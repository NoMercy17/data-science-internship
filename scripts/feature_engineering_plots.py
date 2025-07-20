import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

warnings.filterwarnings('ignore')

class HotelFeatureVisualizationPipeline:
    """
    Enhanced feature engineering pipeline with comprehensive visualization
    """
    
    def __init__(self, output_dir='/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results'):
        self.feature_info = {}
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'plots')
        self.label_encoders = {}
        self.scalers = {}
        self.step_data = {}  # Store data at each step for comparison
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Visualization settings
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Define which original columns to keep
        self.columns_to_keep = [
            'lead_time', 'country', 'market_segment', 'assigned_room_type',
            'deposit_type', 'total_of_special_requests'
        ]
    
    def save_step_data(self, data, step_name):
        """Save data at each step for comparison"""
        self.step_data[step_name] = data.copy()
    
    def plot_data_shape_evolution(self):
        """Plot how data shape evolves through the pipeline"""
        steps = list(self.step_data.keys())
        shapes = [self.step_data[step].shape for step in steps]
        rows = [shape[0] for shape in shapes]
        cols = [shape[1] for shape in shapes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot number of columns evolution
        ax1.plot(range(len(steps)), cols, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax1.set_title('Feature Count Evolution Through Pipeline', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Pipeline Steps')
        ax1.set_ylabel('Number of Features')
        ax1.set_xticks(range(len(steps)))
        ax1.set_xticklabels([step.replace('_', '\n') for step in steps], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add annotations for feature counts
        for i, (step, col_count) in enumerate(zip(steps, cols)):
            ax1.annotate(f'{col_count}', (i, col_count), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        
        # Plot rows (should remain constant)
        ax2.plot(range(len(steps)), rows, marker='s', linewidth=2, markersize=8, color='coral')
        ax2.set_title('Sample Count Through Pipeline', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Pipeline Steps')
        ax2.set_ylabel('Number of Samples')
        ax2.set_xticks(range(len(steps)))
        ax2.set_xticklabels([step.replace('_', '\n') for step in steps], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '1_data_shape_evolution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Data shape evolution plot saved")
    
    def plot_temporal_features_analysis(self, original_data, transformed_data):
        """Visualize temporal feature transformations"""
        if 'lead_time' not in original_data.columns:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Features Analysis', fontsize=16, fontweight='bold')
        
        # Original lead_time distribution
        axes[0,0].hist(original_data['lead_time'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Original Lead Time Distribution')
        axes[0,0].set_xlabel('Lead Time (days)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)
        
        # Lead time categories
        lead_time_features = ['is_last_minute_lead_time', 'is_normal_lead_time', 'is_advance_lead_time']
        categories = [col for col in lead_time_features if col in transformed_data.columns]
        
        if categories:
            category_counts = [transformed_data[col].sum() for col in categories]
            category_labels = [col.replace('is_', '').replace('_lead_time', '').title() for col in categories]
            
            colors = ['red', 'orange', 'green']
            axes[0,1].bar(category_labels, category_counts, color=colors, alpha=0.7, edgecolor='black')
            axes[0,1].set_title('Lead Time Categories Distribution')
            axes[0,1].set_ylabel('Count')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(category_counts):
                axes[0,1].text(i, v + max(category_counts)*0.01, str(v), ha='center', fontweight='bold')
        
        # Lead time risk score distribution
        if 'lead_time_risk_score' in transformed_data.columns:
            risk_counts = transformed_data['lead_time_risk_score'].value_counts().sort_index()
            axes[1,0].bar(risk_counts.index, risk_counts.values, color='purple', alpha=0.7, edgecolor='black')
            axes[1,0].set_title('Lead Time Risk Score Distribution')
            axes[1,0].set_xlabel('Risk Score')
            axes[1,0].set_ylabel('Count')
            
            # Add value labels
            for i, v in enumerate(risk_counts.values):
                axes[1,0].text(risk_counts.index[i], v + max(risk_counts.values)*0.01, 
                              str(v), ha='center', fontweight='bold')
        
        # Correlation heatmap of temporal features
        temporal_cols = [col for col in transformed_data.columns if 'lead_time' in col or 'advance' in col or 'last_minute' in col or 'normal' in col]
        if len(temporal_cols) > 1 and 'is_canceled' in transformed_data.columns:
            temporal_cols_with_target = temporal_cols + ['is_canceled']
            corr_matrix = transformed_data[temporal_cols_with_target].corr()
            
            im = axes[1,1].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            axes[1,1].set_xticks(range(len(corr_matrix.columns)))
            axes[1,1].set_yticks(range(len(corr_matrix.columns)))
            axes[1,1].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            axes[1,1].set_yticklabels(corr_matrix.columns)
            axes[1,1].set_title('Temporal Features Correlation Matrix')
            
            # Add correlation values
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix.columns)):
                    axes[1,1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                  ha='center', va='center', fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[1,1], shrink=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '2_temporal_features_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Temporal features analysis plot saved")
    
    def plot_customer_behavior_analysis(self, original_data, transformed_data):
        """Visualize customer behavior feature transformations"""
        if 'total_of_special_requests' not in original_data.columns:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Behavior Features Analysis', fontsize=16, fontweight='bold')
        
        # Original special requests distribution
        axes[0,0].hist(original_data['total_of_special_requests'], bins=range(0, original_data['total_of_special_requests'].max()+2), 
                      alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0,0].set_title('Original Special Requests Distribution')
        axes[0,0].set_xlabel('Number of Special Requests')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)
        
        # Special requests level categories
        if 'special_requests_level' in self.step_data.get('customer_behavior', pd.DataFrame()).columns:
            level_data = self.step_data['customer_behavior']['special_requests_level']
            level_counts = level_data.value_counts()
            
            colors = {'None': 'lightgray', 'Low': 'lightblue', 'Medium': 'orange', 'High': 'red'}
            bar_colors = [colors.get(level, 'gray') for level in level_counts.index]
            
            axes[0,1].bar(level_counts.index, level_counts.values, color=bar_colors, alpha=0.7, edgecolor='black')
            axes[0,1].set_title('Special Requests Level Categories')
            axes[0,1].set_ylabel('Count')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, v in enumerate(level_counts.values):
                axes[0,1].text(i, v + max(level_counts.values)*0.01, str(v), ha='center', fontweight='bold')
        
        # Has special requirements indicator
        if 'has_special_requirements' in transformed_data.columns:
            req_counts = transformed_data['has_special_requirements'].value_counts()
            labels = ['No Requirements', 'Has Requirements']
            colors = ['lightcoral', 'lightgreen']
            
            axes[1,0].pie(req_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1,0].set_title('Distribution of Special Requirements')
        
        # Correlation with target if available
        if 'is_canceled' in transformed_data.columns:
            behavior_cols = [col for col in transformed_data.columns if 'special' in col or 'requirements' in col]
            if behavior_cols:
                correlations = transformed_data[behavior_cols + ['is_canceled']].corr()['is_canceled'][:-1]
                
                colors = ['red' if corr > 0 else 'blue' for corr in correlations.values]
                axes[1,1].bar(range(len(correlations)), correlations.values, color=colors, alpha=0.7, edgecolor='black')
                axes[1,1].set_title('Customer Behavior Features vs Cancellation')
                axes[1,1].set_xlabel('Features')
                axes[1,1].set_ylabel('Correlation with is_canceled')
                axes[1,1].set_xticks(range(len(correlations)))
                axes[1,1].set_xticklabels([col.replace('_', '\n') for col in correlations.index], rotation=45, ha='right')
                axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1,1].grid(True, alpha=0.3)
                
                # Add correlation values
                for i, v in enumerate(correlations.values):
                    axes[1,1].text(i, v + 0.01 if v >= 0 else v - 0.01, f'{v:.3f}', 
                                  ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '3_customer_behavior_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Customer behavior analysis plot saved")
    
    def plot_market_segment_analysis(self, original_data, transformed_data):
        """Visualize market segment feature transformations"""
        if 'market_segment' not in original_data.columns:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Market Segment Features Analysis', fontsize=16, fontweight='bold')
        
        # Original market segment distribution
        segment_counts = original_data['market_segment'].value_counts()
        axes[0,0].bar(segment_counts.index, segment_counts.values, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Original Market Segment Distribution')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(segment_counts.values):
            axes[0,0].text(i, v + max(segment_counts.values)*0.01, str(v), ha='center', fontweight='bold')
        
        # Market segment binary features
        segment_binary_cols = [col for col in transformed_data.columns if col.startswith('market_segment_') and col != 'market_segment_risk']
        if segment_binary_cols:
            binary_counts = [transformed_data[col].sum() for col in segment_binary_cols]
            binary_labels = [col.replace('market_segment_', '').replace('_', ' ').title() for col in segment_binary_cols]
            
            axes[0,1].bar(binary_labels, binary_counts, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0,1].set_title('Market Segment Binary Features')
            axes[0,1].set_ylabel('Count')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            for i, v in enumerate(binary_counts):
                axes[0,1].text(i, v + max(binary_counts)*0.01, str(v), ha='center', fontweight='bold')
        
        # Risk level distribution (from intermediate data)
        if 'market_features' in self.step_data and 'market_segment_risk' in self.step_data['market_features'].columns:
            risk_data = self.step_data['market_features']['market_segment_risk']
            risk_counts = risk_data.value_counts()
            
            risk_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            bar_colors = [risk_colors.get(risk, 'gray') for risk in risk_counts.index]
            
            axes[1,0].bar(risk_counts.index, risk_counts.values, color=bar_colors, alpha=0.7, edgecolor='black')
            axes[1,0].set_title('Market Segment Risk Level Distribution')
            axes[1,0].set_ylabel('Count')
            
            for i, v in enumerate(risk_counts.values):
                axes[1,0].text(i, v + max(risk_counts.values)*0.01, str(v), ha='center', fontweight='bold')
        
        # Cancellation rate by market segment
        if 'is_canceled' in original_data.columns:
            cancellation_by_segment = original_data.groupby('market_segment')['is_canceled'].agg(['count', 'sum', 'mean']).sort_values('mean', ascending=False)
            
            bars = axes[1,1].bar(cancellation_by_segment.index, cancellation_by_segment['mean'], 
                               alpha=0.7, color='tomato', edgecolor='black')
            axes[1,1].set_title('Cancellation Rate by Market Segment')
            axes[1,1].set_ylabel('Cancellation Rate')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].set_ylim(0, 1)
            
            # Add percentage labels
            for bar, rate in zip(bars, cancellation_by_segment['mean']):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, rate + 0.01, 
                              f'{rate:.1%}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '4_market_segment_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Market segment analysis plot saved")
    
    def plot_deposit_analysis(self, original_data, transformed_data):
        """Visualize deposit feature transformations"""
        if 'deposit_type' not in original_data.columns:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Deposit Features Analysis', fontsize=16, fontweight='bold')
        
        # Original deposit type distribution
        deposit_counts = original_data['deposit_type'].value_counts()
        colors = {'No Deposit': 'lightcoral', 'Refundable': 'lightyellow', 'Non Refund': 'lightgreen'}
        bar_colors = [colors.get(deposit, 'gray') for deposit in deposit_counts.index]
        
        axes[0,0].bar(deposit_counts.index, deposit_counts.values, color=bar_colors, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Original Deposit Type Distribution')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(deposit_counts.values):
            axes[0,0].text(i, v + max(deposit_counts.values)*0.01, str(v), ha='center', fontweight='bold')
        
        # Deposit binary features
        deposit_binary_cols = [col for col in transformed_data.columns if col.startswith('has_') and 'deposit' in col]
        if deposit_binary_cols:
            binary_counts = [transformed_data[col].sum() for col in deposit_binary_cols]
            binary_labels = [col.replace('has_', '').replace('_', ' ').title() for col in deposit_binary_cols]
            
            axes[0,1].bar(binary_labels, binary_counts, alpha=0.7, color='lightblue', edgecolor='black')
            axes[0,1].set_title('Deposit Binary Features')
            axes[0,1].set_ylabel('Count')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            for i, v in enumerate(binary_counts):
                axes[0,1].text(i, v + max(binary_counts)*0.01, str(v), ha='center', fontweight='bold')
        
        # Deposit risk distribution
        risk_cols = [col for col in transformed_data.columns if col.startswith('deposit_risk_')]
        if risk_cols:
            risk_counts = [transformed_data[col].sum() for col in risk_cols]
            risk_labels = [col.replace('deposit_risk_', '').title() for col in risk_cols]
            risk_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            bar_colors = [risk_colors.get(label, 'gray') for label in risk_labels]
            
            axes[1,0].bar(risk_labels, risk_counts, color=bar_colors, alpha=0.7, edgecolor='black')
            axes[1,0].set_title('Deposit Risk Level Distribution')
            axes[1,0].set_ylabel('Count')
            
            for i, v in enumerate(risk_counts):
                axes[1,0].text(i, v + max(risk_counts)*0.01, str(v), ha='center', fontweight='bold')
        
        # Cancellation rate by deposit type
        if 'is_canceled' in original_data.columns:
            cancellation_by_deposit = original_data.groupby('deposit_type')['is_canceled'].agg(['count', 'sum', 'mean']).sort_values('mean', ascending=False)
            
            bars = axes[1,1].bar(cancellation_by_deposit.index, cancellation_by_deposit['mean'], 
                               color=bar_colors, alpha=0.7, edgecolor='black')
            axes[1,1].set_title('Cancellation Rate by Deposit Type')
            axes[1,1].set_ylabel('Cancellation Rate')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].set_ylim(0, 1)
            
            for bar, rate in zip(bars, cancellation_by_deposit['mean']):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, rate + 0.01, 
                              f'{rate:.1%}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '5_deposit_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Deposit analysis plot saved")
    
    def plot_scaling_analysis(self, before_scaling, after_scaling):
        """Visualize the effects of feature scaling"""
        # Find numeric columns that were scaled
        numeric_cols = before_scaling.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_canceled' in numeric_cols:
            numeric_cols.remove('is_canceled')
        
        # Select up to 6 features for visualization
        cols_to_plot = numeric_cols[:6] if len(numeric_cols) >= 6 else numeric_cols
        
        if not cols_to_plot:
            print("No numeric features found for scaling analysis")
            return
        
        fig, axes = plt.subplots(2, len(cols_to_plot), figsize=(4*len(cols_to_plot), 8))
        fig.suptitle('Feature Scaling Analysis - Before vs After', fontsize=16, fontweight='bold')
        
        if len(cols_to_plot) == 1:
            axes = axes.reshape(2, 1)
        
        for i, col in enumerate(cols_to_plot):
            # Before scaling
            axes[0, i].hist(before_scaling[col], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            axes[0, i].set_title(f'Before Scaling\n{col}')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].grid(True, alpha=0.3)
            
            # Add statistics text
            mean_before = before_scaling[col].mean()
            std_before = before_scaling[col].std()
            axes[0, i].text(0.7, 0.8, f'μ={mean_before:.2f}\nσ={std_before:.2f}', 
                           transform=axes[0, i].transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            
            # After scaling
            axes[1, i].hist(after_scaling[col], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, i].set_title(f'After Scaling\n{col}')
            axes[1, i].set_xlabel('Value')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].grid(True, alpha=0.3)
            
            # Add statistics text
            mean_after = after_scaling[col].mean()
            std_after = after_scaling[col].std()
            axes[1, i].text(0.7, 0.8, f'μ={mean_after:.2f}\nσ={std_after:.2f}', 
                           transform=axes[1, i].transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '6_scaling_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Scaling analysis plot saved")
    
    def plot_feature_importance_heatmap(self, final_data):
        """Create a correlation heatmap with the target variable"""
        if 'is_canceled' not in final_data.columns:
            print("Target variable 'is_canceled' not found for correlation analysis")
            return
        
        # Calculate correlations with target
        correlations = final_data.corr()['is_canceled'].drop('is_canceled').abs().sort_values(ascending=False)
        
        # Select top 20 features for visualization
        top_features = correlations.head(20)
        
        # Create correlation matrix for top features + target
        features_to_plot = top_features.index.tolist() + ['is_canceled']
        corr_matrix = final_data[features_to_plot].corr()
        
        plt.figure(figsize=(14, 12))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        
        plt.title('Top 20 Features Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '7_feature_importance_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a bar plot of feature importance
        plt.figure(figsize=(12, 8))
        
        colors = ['red' if corr > 0.3 else 'orange' if corr > 0.1 else 'lightblue' for corr in top_features.values]
        bars = plt.bar(range(len(top_features)), top_features.values, color=colors, alpha=0.7, edgecolor='black')
        
        plt.title('Top 20 Features by Absolute Correlation with Target', fontsize=14, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Absolute Correlation with is_canceled')
        plt.xticks(range(len(top_features)), [feat.replace('_', '\n') for feat in top_features.index], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add correlation values on bars
        for bar, corr in zip(bars, top_features.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{corr:.3f}', ha='center', fontweight='bold')
        
        # Add color legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='High (>0.3)'),
                          Patch(facecolor='orange', alpha=0.7, label='Medium (0.1-0.3)'),
                          Patch(facecolor='lightblue', alpha=0.7, label='Low (<0.1)')]
        plt.legend(handles=legend_elements, title='Correlation Level')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '8_feature_importance_ranking.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Feature importance analysis plots saved")
    
    def plot_pipeline_summary(self, original_data, final_data):
        """Create a comprehensive summary plot of the entire pipeline"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Overall pipeline summary
        ax1 = fig.add_subplot(gs[0, :2])
        steps = list(self.step_data.keys())
        feature_counts = [self.step_data[step].shape[1] for step in steps]
        
        ax1.plot(range(len(steps)), feature_counts, marker='o', linewidth=3, markersize=10, color='steelblue')
        ax1.set_title('Feature Engineering Pipeline Summary', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Pipeline Steps')
        ax1.set_ylabel('Number of Features')
        ax1.set_xticks(range(len(steps)))
        ax1.set_xticklabels([step.replace('_', '\n') for step in steps], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        for i, count in enumerate(feature_counts):
            ax1.annotate(f'{count}', (i, count), textcoords="offset points", 
                        xytext=(0,15), ha='center', fontweight='bold', fontsize=12)
        
        # Data type distribution
        ax2 = fig.add_subplot(gs[0, 2:])
        original_types = original_data.dtypes.value_counts()
        final_types = final_data.dtypes.value_counts()
        
        x = np.arange(len(original_types))
        width = 0.35
        
        ax2.bar(x - width/2, original_types.values, width, label='Original', color='lightcoral', alpha=0.7, edgecolor='black')
        ax2.bar(x + width/2, [final_types.get(dtype, 0) for dtype in original_types.index], 
                width, label='Final', color='lightblue', alpha=0.7, edgecolor='black')
        
        ax2.set_title('Data Types Distribution: Before vs After', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Data Types')
        ax2.set_ylabel('Count')
        ax2.set_xticks(x)
        ax2.set_xticklabels(original_types.index, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Feature categories distribution
        ax3 = fig.add_subplot(gs[1, :2])
        
        # Categorize features
        feature_categories = {
            'Temporal': len([col for col in final_data.columns if 'lead_time' in col or 'advance' in col or 'last_minute' in col]),
            'Market Segment': len([col for col in final_data.columns if 'market_segment' in col]),
            'Deposit': len([col for col in final_data.columns if 'deposit' in col]),
            'Customer Behavior': len([col for col in final_data.columns if 'special' in col or 'requirements' in col]),
            'Room': len([col for col in final_data.columns if 'room' in col]),
            'Country': len([col for col in final_data.columns if 'country' in col]),
            'Other': len([col for col in final_data.columns if not any(keyword in col for keyword in 
                         ['lead_time', 'advance', 'last_minute', 'market_segment', 'deposit', 'special', 'requirements', 'room', 'country'])])
        }
        
        # Remove categories with 0 features
        feature_categories = {k: v for k, v in feature_categories.items() if v > 0}
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(feature_categories)))
        wedges, texts, autotexts = ax3.pie(feature_categories.values(), labels=feature_categories.keys(), 
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Feature Categories Distribution', fontsize=14, fontweight='bold')
        
        # Make percentage labels bold
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        # Missing values comparison
        ax4 = fig.add_subplot(gs[1, 2:])
        original_missing = original_data.isnull().sum().sum()
        final_missing = final_data.isnull().sum().sum()
        
        categories = ['Original Data', 'Final Data']
        missing_counts = [original_missing, final_missing]
        colors = ['red' if count > 0 else 'green' for count in missing_counts]
        
        bars = ax4.bar(categories, missing_counts, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_title('Missing Values: Before vs After', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Number of Missing Values')
        
        for bar, count in zip(bars, missing_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(missing_counts)*0.01, 
                    str(count), ha='center', fontweight='bold', fontsize=12)
        
        # Top correlations with target
        if 'is_canceled' in final_data.columns:
            ax5 = fig.add_subplot(gs[2, :2])
            correlations = final_data.corr()['is_canceled'].drop('is_canceled').abs().sort_values(ascending=False).head(10)
            
            colors = ['red' if corr > 0.3 else 'orange' if corr > 0.1 else 'lightblue' for corr in correlations.values]
            bars = ax5.bar(range(len(correlations)), correlations.values, color=colors, alpha=0.7, edgecolor='black')
            
            ax5.set_title('Top 10 Features by Correlation with Target', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Features')
            ax5.set_ylabel('Absolute Correlation')
            ax5.set_xticks(range(len(correlations)))
            ax5.set_xticklabels([feat.replace('_', '\n')[:15] + '...' if len(feat) > 15 else feat.replace('_', '\n') 
                               for feat in correlations.index], rotation=45, ha='right', fontsize=10)
            ax5.grid(True, alpha=0.3)
            
            for bar, corr in zip(bars, correlations.values):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                        f'{corr:.2f}', ha='center', fontweight='bold', fontsize=9)
        
        # Pipeline statistics summary
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')
        
        # Calculate statistics
        original_shape = original_data.shape
        final_shape = final_data.shape
        feature_increase = final_shape[1] - original_shape[1]
        feature_increase_pct = (feature_increase / original_shape[1]) * 100
        
        # Create summary text
        summary_text = f"""
        PIPELINE STATISTICS SUMMARY
        
        Original Dataset:     {original_shape[0]:,} rows × {original_shape[1]} features
        Final Dataset:        {final_shape[0]:,} rows × {final_shape[1]} features
        
        Feature Engineering Results:
        • Features Added:     +{feature_increase} ({feature_increase_pct:.1f}% increase)
        • Data Integrity:     {final_shape[0]:,} rows preserved
        • Missing Values:     {final_data.isnull().sum().sum()} remaining
        
        Feature Categories Created:
        """
        
        for category, count in feature_categories.items():
            summary_text += f"        • {category}: {count} features\n"
        
        # Add pipeline steps summary
        summary_text += f"\n        Pipeline Steps Completed: {len(steps)}\n"
        for i, step in enumerate(steps, 1):
            summary_text += f"        {i}. {step.replace('_', ' ').title()}\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Hotel Booking Feature Engineering Pipeline - Complete Summary', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '9_pipeline_complete_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Complete pipeline summary plot saved")
    












    def generate_interactive_dashboard(self, final_data):
        """Generate an interactive HTML dashboard using Plotly"""
        if 'is_canceled' not in final_data.columns:
            print("Target variable 'is_canceled' not found for interactive dashboard")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Feature Correlation with Target', 'Feature Distribution by Type',
                          'Top Features Importance', 'Cancellation Rate Analysis',
                          'Feature Engineering Impact', 'Data Quality Summary'),
            specs=[
    [{"secondary_y": False}, {"type": "domain"}],   # row 1
    [{"secondary_y": False}, {"secondary_y": False}],  # row 2
    [{"secondary_y": False}, {"type": "domain"}]    #  row 3, col 2
]
        )
        
        # 1. Feature correlation with target
        correlations = final_data.corr()['is_canceled'].drop('is_canceled').abs().sort_values(ascending=False).head(15)
        fig.add_trace(
            go.Bar(x=correlations.index, y=correlations.values, 
                  name='Correlation', marker_color='steelblue'),
            row=1, col=1
        )
        
        # 2. Feature distribution by type (pie chart)
        feature_categories = {
            'Temporal': len([col for col in final_data.columns if 'lead_time' in col or 'advance' in col or 'last_minute' in col]),
            'Market Segment': len([col for col in final_data.columns if 'market_segment' in col]),
            'Deposit': len([col for col in final_data.columns if 'deposit' in col]),
            'Customer Behavior': len([col for col in final_data.columns if 'special' in col or 'requirements' in col]),
            'Room': len([col for col in final_data.columns if 'room' in col]),
            'Country': len([col for col in final_data.columns if 'country' in col]),
            'Other': len([col for col in final_data.columns if not any(keyword in col for keyword in 
                         ['lead_time', 'advance', 'last_minute', 'market_segment', 'deposit', 'special', 'requirements', 'room', 'country'])])
        }
        feature_categories = {k: v for k, v in feature_categories.items() if v > 0}
        
        fig.add_trace(
            go.Pie(labels=list(feature_categories.keys()), values=list(feature_categories.values()),
                  name="Feature Categories"),
            row=1, col=2
        )
        
        # 3. Top features importance
        top_10_corr = correlations.head(10)
        fig.add_trace(
            go.Scatter(x=list(range(len(top_10_corr))), y=top_10_corr.values,
                      mode='lines+markers', name='Top Features',
                      line=dict(color='red', width=2), marker=dict(size=8)),
            row=2, col=1
        )
        
        # 4. Cancellation rate analysis
        total_bookings = len(final_data)
        canceled_bookings = final_data['is_canceled'].sum()
        
        fig.add_trace(
            go.Bar(x=['Not Canceled', 'Canceled'], 
                  y=[total_bookings - canceled_bookings, canceled_bookings],
                  marker_color=['green', 'red'], name='Booking Status'),
            row=2, col=2
        )
        
        # 5. Feature engineering impact
        if hasattr(self, 'step_data'):
            steps = list(self.step_data.keys())
            feature_counts = [self.step_data[step].shape[1] for step in steps]
            
            fig.add_trace(
                go.Scatter(x=steps, y=feature_counts,
                          mode='lines+markers', name='Feature Count Evolution',
                          line=dict(color='purple', width=3), marker=dict(size=10)),
                row=3, col=1
            )
        
        # 6. Data quality summary
        missing_values = final_data.isnull().sum().sum()
        total_values = final_data.shape[0] * final_data.shape[1]
        quality_score = ((total_values - missing_values) / total_values) * 100
        
        fig.add_trace(
            go.Indicator(
                mode = "gauge+number+delta",
                value = quality_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Data Quality Score (%)"},
                delta = {'reference': 95},
                gauge = {'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "lightgreen"}],
                        'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 90}}),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Hotel Booking Feature Engineering - Interactive Dashboard",
            title_x=0.5,
            height=1000,
            showlegend=True
        )
        
        # Update subplot titles
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=2, col=1)
        fig.update_xaxes(tickangle=45, row=3, col=1)
        
        # Save as HTML
        dashboard_path = os.path.join(self.output_dir, 'interactive_dashboard.html')
        fig.write_html(dashboard_path)
        
        print(f"✅ Interactive dashboard saved to: {dashboard_path}")
        
        return fig
    
    def create_feature_summary_report(self, original_data, final_data):
        """Generate a comprehensive feature summary report"""
        report = []
        report.append("="*80)
        report.append("HOTEL BOOKING FEATURE ENGINEERING PIPELINE REPORT")
        report.append("="*80)
        report.append("")
        
        # Dataset overview
        report.append("DATASET OVERVIEW:")
        report.append("-" * 30)
        report.append(f"Original Dataset Shape: {original_data.shape[0]:,} rows × {original_data.shape[1]} columns")
        report.append(f"Final Dataset Shape: {final_data.shape[0]:,} rows × {final_data.shape[1]} columns")
        report.append(f"Features Added: {final_data.shape[1] - original_data.shape[1]}")
        report.append(f"Feature Increase: {((final_data.shape[1] - original_data.shape[1]) / original_data.shape[1]) * 100:.1f}%")
        report.append("")
        
        # Pipeline steps summary
        if hasattr(self, 'step_data'):
            report.append("PIPELINE EXECUTION SUMMARY:")
            report.append("-" * 30)
            for i, (step, data) in enumerate(self.step_data.items(), 1):
                report.append(f"{i}. {step.replace('_', ' ').title()}: {data.shape[1]} features")
            report.append("")
        
        # Feature categories analysis
        report.append("FEATURE CATEGORIES ANALYSIS:")
        report.append("-" * 30)
        
        feature_categories = {
            'Temporal Features': [col for col in final_data.columns if 'lead_time' in col or 'advance' in col or 'last_minute' in col],
            'Market Segment Features': [col for col in final_data.columns if 'market_segment' in col],
            'Deposit Features': [col for col in final_data.columns if 'deposit' in col],
            'Customer Behavior Features': [col for col in final_data.columns if 'special' in col or 'requirements' in col],
            'Room Features': [col for col in final_data.columns if 'room' in col],
            'Country Features': [col for col in final_data.columns if 'country' in col]
        }
        
        for category, features in feature_categories.items():
            if features:
                report.append(f"{category}: {len(features)} features")
                for feature in features[:5]:  # Show first 5 features
                    report.append(f"  • {feature}")
                if len(features) > 5:
                    report.append(f"  ... and {len(features) - 5} more")
                report.append("")
        
        # Top correlations with target
        if 'is_canceled' in final_data.columns:
            report.append("TOP FEATURES BY CORRELATION WITH TARGET:")
            report.append("-" * 30)
            correlations = final_data.corr()['is_canceled'].drop('is_canceled').abs().sort_values(ascending=False).head(15)
            for i, (feature, corr) in enumerate(correlations.items(), 1):
                report.append(f"{i:2d}. {feature:<40} {corr:.4f}")
            report.append("")
        
        # Data quality assessment
        report.append("DATA QUALITY ASSESSMENT:")
        report.append("-" * 30)
        original_missing = original_data.isnull().sum().sum()
        final_missing = final_data.isnull().sum().sum()
        report.append(f"Missing Values (Original): {original_missing:,}")
        report.append(f"Missing Values (Final): {final_missing:,}")
        report.append(f"Data Completeness: {((final_data.size - final_missing) / final_data.size) * 100:.2f}%")
        report.append("")
        
        # Feature types summary
        report.append("FEATURE TYPES SUMMARY:")
        report.append("-" * 30)
        dtype_counts = final_data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            report.append(f"{str(dtype):<15}: {count} features")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS FOR MODEL TRAINING:")
        report.append("-" * 30)
        
        if final_missing > 0:
            report.append("• Handle remaining missing values before model training")
        
        high_corr_features = correlations[correlations > 0.3] if 'is_canceled' in final_data.columns else []
        if len(high_corr_features) > 0:
            report.append(f"• Focus on top {len(high_corr_features)} highly correlated features for initial modeling")
        
        if final_data.shape[1] > 50:
            report.append("• Consider feature selection techniques to reduce dimensionality")
        
        report.append("• Validate feature engineering with cross-validation")
        report.append("• Monitor for data leakage in temporal features")
        
        report.append("")
        report.append("="*80)
        
        # Save report
        report_path = os.path.join(self.output_dir, 'feature_engineering_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Feature engineering report saved to: {report_path}")
        
        # Also print to console
        print("\n" + "\n".join(report))
        
        return report
    
    def run_complete_analysis(self, original_data, final_data):
        """Run the complete visualization and analysis pipeline"""
        print("\n🚀 Starting comprehensive feature engineering analysis...")
        print("="*80)
        
        # 1. Data shape evolution
        self.plot_data_shape_evolution()
        
        # 2. Temporal features analysis
        self.plot_temporal_features_analysis(original_data, final_data)
        
        # 3. Customer behavior analysis
        self.plot_customer_behavior_analysis(original_data, final_data)
        
        # 4. Market segment analysis
        self.plot_market_segment_analysis(original_data, final_data)
        
        # 5. Deposit analysis
        self.plot_deposit_analysis(original_data, final_data)
        
        # 6. Scaling analysis (if scaling was applied)
        if 'before_scaling' in self.step_data and 'after_scaling' in self.step_data:
            self.plot_scaling_analysis(self.step_data['before_scaling'], self.step_data['after_scaling'])
        
        # 7. Feature importance analysis
        self.plot_feature_importance_heatmap(final_data)
        
        # 8. Complete pipeline summary
        self.plot_pipeline_summary(original_data, final_data)
        
        # 9. Interactive dashboard
        self.generate_interactive_dashboard(final_data)
        
        # 10. Feature summary report
        self.create_feature_summary_report(original_data, final_data)
        
        print("\n" + "="*80)
        print("🎉 ANALYSIS COMPLETE!")
        print(f"📊 All visualizations saved to: {self.plots_dir}")
        print(f"📋 Reports saved to: {self.output_dir}")
        print("="*80)

if __name__ == "__main__":
    # Initialize the visualizer
    visualizer = HotelFeatureVisualizationPipeline()

    original_data_path = "/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/cleaned/final_cleaned_data.pkl"
    final_data_path = "/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/results/feature_engineering_results.pkl"

    # Load your original and final data (ensure they were saved before)
    with open(original_data_path, 'rb') as f:
        original_data = pickle.load(f)

    with open(final_data_path, 'rb') as f:
        final_data = pickle.load(f)
    
    # Generate all plots
    visualizer.run_complete_analysis(original_data, final_data)